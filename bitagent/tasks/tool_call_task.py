# The MIT License (MIT)
# Copyright © 2023 RogueTensor

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import ast
import json
import random
import datetime
import bittensor as bt
from huggingface_hub import dataset_info
from bitagent.protocol import QueryTask
from bitagent.tasks import Task
from bitagent.tasks import TASK_WEIGHTS
from bitagent.schemas.chat import messages_to_list
from bitagent.datasources.tools import ToolDataset
from bitagent.datasources.tools import ToolCallData
from bitagent.helpers.tool_parsing import validate_tool_call, find_msgs_before_tool_call, find_first_tool_call
from bitagent.criteria import default_criteria, tool_call_criteria, irrelevant_tool_call_criteria

class ToolCallTask(Task):
    def __init__(
        self,
        validator,
        name: str,
        desc: str = "",
        offline: bool = False,
    ):
        super().__init__(name=name, desc=desc)
        self.validator = validator
        self.timeout = 15.0
        self.name += " - Tool Call"
        self.weight = TASK_WEIGHTS["tool_call"]

        if offline:
            self.mode = "offline"
        messages = None
        for _ in range(10):
            try:
                messages, tools, data = self.generate_task_data()
                expected_messages = messages_to_list(data.messages)
                self.prompt = expected_messages[0]
                expected_tool_call_messages = [em for em in expected_messages if em['role'] == 'tool call']

                expected_tool_call = expected_tool_call_messages[0]['content']

                if (not expected_tool_call.get("name")):
                    self.criteria = irrelevant_tool_call_criteria()
                    self.correct_answer = "Irrelevant tool call found"
                else:
                    # Otherwise, normal scenario
                    self.criteria =tool_call_criteria(expected_response=expected_tool_call)
                    self.correct_answer = expected_tool_call

                break

            except Exception as e:
                print(f'Exception getting new task - {e} - you may need to CHECK YOUR vLLM docker instance')
                pass

        if not messages:
            raise Exception(f"Failed to generate task data 10 times")
        self.messages = messages
        self.synapse = QueryTask(messages=messages, tools=tools)

    def generate_task_data(self) -> ToolCallData:
    
        data: ToolCallData = next(self.validator.task_dataset)
        random.seed(self.validator.seed)

        first_call = find_first_tool_call(data.messages)
        if first_call.content == {}:
            # ----------------------------
            # CASE A: Irrelevance
            # ----------------------------

            new_tools = []
            # We'll keep collecting until we have at least 5 tools

            # handle param irrelevance
            if len(data.tools) == 1:
                while len(new_tools) < 4:
                    extra_data = next(self.validator.tool_dataset)
                    new_tools.extend(extra_data.tools)
                new_tools.extend(data.tools)
            # handle case for func irrelevance and no tool
            else:
                while len(new_tools) < 5:
                    extra_data = next(self.validator.tool_dataset)
                    new_tools.extend(extra_data.tools)
            
            random.shuffle(new_tools)
            all_tools = new_tools[:5]
            messages_before_call = find_msgs_before_tool_call(data.messages)
            data = ToolCallData(messages=data.messages, tools=all_tools)
            return messages_before_call, data.tools, data

        else:
            # ----------------------------
            # CASE B: We do have a tool call
            # ----------------------------

            for _ in range(4):
                # filter out any tool with a name already in data.tools
                new_batch = next(self.validator.tool_dataset)
                # pick only new tool names that we don't already have
                new_tools = [
                    t for t in new_batch.tools
                    if t.name not in [dt.name for dt in data.tools]
                ]
                data.tools.extend(new_tools)


            messages = data.messages
            filtered_msgs = []
            seen_tool_call = False
            for msg in messages:
                filtered_msgs.append(msg)
                if seen_tool_call:
                    break
                if msg.role == 'tool call':
                    seen_tool_call = True
            data.messages = filtered_msgs


            messages_before_call = find_msgs_before_tool_call(data.messages)

            all_tools = data.tools
            random.shuffle(all_tools)
            data = ToolCallData(messages=data.messages, tools=all_tools[:5])

            return messages_before_call, data.tools, data
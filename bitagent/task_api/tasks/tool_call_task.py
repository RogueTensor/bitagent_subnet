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
import json
import random
import bittensor as bt
from bitagent.protocol import QnATask
from bitagent.task_api.tasks import Task
from common.base.validator import BaseValidatorNeuron
from bitagent.task_api.criteria import default_criteria, tool_call_criteria, dataset_tool_call_criteria
from bitagent.task_api.datasources.tools import ToolCallData, split_dialogue
from bitagent.schemas.conversation import Conversation
from bitagent.task_api.helpers.convo_parsing import find_msgs_before_tool_call
from bitagent.task_api.postprocess import tool_call_postprocess
from bitagent.task_api.tasks import TASK_WEIGHTS
REWRITE_SYSTEM_PROMPT = """
You will be provided a conversation between a user and an assistant. Your job is to generate a variation of the conversation in which the user asks a different question, and as a result the assistant will respond differently. Do not continue the conversation, ONLY modify it. Remember to keep any json objects in the conversation intact.
Here is a list of actions that you have available to you:
{}"""


def messages_to_string(convo: Conversation):
    res = ""
    for message in convo.messages:
        if message.role == "tool call":
            res += f"{message.role.upper()}: ```json\n{message.content}``` \n\n"
        else:
            res += f"{message.role.upper()}: {message.content} \n\n"

    return res


class ToolCallTask(Task):
    def __init__(
        self,
        validator: BaseValidatorNeuron,
        name: str,
        sub_task_id_to_get: int = None,
        desc: str = "",
    ):
        super().__init__(name=name, desc=desc)
        self.validator = validator
        self.timeout = 12.0
        self.name += " - Tool Call"
        self.real_task = bool(random.random() < 0.9)
        if self.real_task:
            try:
                message_history, tools, data = self.generate_task_data()
                self.weight = TASK_WEIGHTS["tool_call"]
                self.criteria = default_criteria + tool_call_criteria(
                    expected_convo=data.convo
                )
            except Exception as e:
                # bt.logging.error(f'Exception getting real task {e}')
                pass
        else:
            try:
                message_history, tools, data = self.generate_dataset_task_data()
                self.criteria = default_criteria + dataset_tool_call_criteria() 
                self.postprocess = tool_call_postprocess()
                self.weight = TASK_WEIGHTS["tool_call_dataset"]
            except Exception as e:
                # bt.logging.error(f'Exception getting fake task {e}')
                pass
        self.message_history = message_history
        notes = """Tool Calling"""
        self.synapse = QnATask(
            urls=[], datas=[], tools=tools, notes=notes, message_history=message_history
        )

    
    def generate_dataset_task_data(self):
        try:
            data: ToolCallData = next(self.validator.local_tool_gen_dataset)
        except Exception as e:
            bt.logging.warning(f"Issue getting fake data {e}")
        messages_before_call = find_msgs_before_tool_call(data.convo)
        return Conversation(messages=messages_before_call), data.tools, data
    def generate_task_data(self):
        data: ToolCallData = self.generate_data()
        
        messages_before_call = find_msgs_before_tool_call(data.convo)
        return Conversation(messages=messages_before_call), data.tools, data
        
        
    
    def generate_data(self) -> ToolCallData:
        use_synth = bool(random.random() < 0.7)
        # use_synth = False
        if use_synth:
            data: ToolCallData = next(self.validator.local_tool_call_dataset)
        else:
            data: ToolCallData = next(self.validator.tool_dataset)
        system_prompt = REWRITE_SYSTEM_PROMPT.format("\n".join([json.dumps(dict(func)) for func in data.tools]))
        user_prompt = messages_to_string(data.convo)
        count = 0
        while count < 4:
            count += 1
            rewritten_history = self.validator.chat_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.6,
                max_new_tokens=4096,
            )
            
            rewritten_history = rewritten_history.replace("TOOL CALLS", "TOOL CALL").replace('\'','').replace("```json\n","").replace("```","").replace("```json","")
            if "TOOL CALL" in user_prompt and "TOOL CALL" not in rewritten_history:
                continue
            rewritten_messages = split_dialogue(rewritten_history)
            try: 
                for message in rewritten_messages.messages:
                    if message.role == "tool call":
                        json.loads(message.content)
            except Exception as e:
                raise ValueError(f'Failed to load tool call: {e}')
                
            return ToolCallData(convo=rewritten_messages, tools=data.tools)
        if "TOOL CALL" in user_prompt and "TOOL CALL" not in rewritten_history:
            raise ValueError("Unable to rewrite the system prompt")
            


def remove_special_characters(input_string):
    # Using a list comprehension to filter only alphabet letters
    filtered_characters = [char for char in input_string if char.isalpha()]
    # Joining the filtered characters back into a string
    filtered_string = "".join(filtered_characters)
    return filtered_string

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
import bittensor as bt
from bitagent.protocol import QnATask
from bitagent.task_api.tasks import Task
from bitagent.task_api.tasks import TASK_WEIGHTS
from common.base.validator import BaseValidatorNeuron
from bitagent.schemas.chat import messages_to_list
from bitagent.task_api.datasources.tools import ToolCallData
from bitagent.task_api.postprocess import tool_call_postprocess
from bitagent.task_api.helpers.tool_parsing import validate_tool_call
from bitagent.task_api.helpers.convo_parsing import find_msgs_before_tool_call, find_first_tool_call
from bitagent.task_api.criteria import default_criteria, tool_call_criteria, dataset_tool_call_criteria
REWRITE_PROPMT = """Please rewrite the following text, ensuring to maintain the original meaning and nuances but altering the sentence structures, vocabulary, and overall presentation. 
The goal is to produce a version of the text that conveys the same information and sentiments as the original, but in a fresh and distinct manner. 
Avoid summarizing or omitting any details; instead, focus on presenting the same concepts and messages in a new light.
            
Rewrite this text: {query}
            
Rewritten text: """

REWRITE_TOOL_PROMPT = "Modify the function call to have different arguments. Your response should only be the modified function. You should not use the same argument values. The arguments should be valid in reference to the other argument values\n Given the function call:\n{tool_call}. Modified function call: "

REWRITE_TOOL_USER_PROMPT = "You rewrite questions to make sense when paired with a function call. The rewritten question will need to be changed to match the arguments of the function call. You should change the phrasing of the question up. Your response should be the rewritten question.\nFunction call:\n{tool_call} \n Question: {user}\n Question:"

REWRITE_TOOL_ASSISTANT_PROMPT = """Input:
User Question: {user}
Tool Call: {tool_call}
Incorrect Answer: {assistant}
Task: Rewrite the incorrect answer to accurately reflect the result of the given Tool call. Also, modify the wording to ensure it is different from the original, it should be concise. Output only the revised answer."""



# TODO add handling for when theres multiple function calls in the messages 
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
        self.real_task = bool(random.random() < 0.99)
        if self.real_task:
            try:
                messages, tools, data = self.generate_task_data()
                self.weight = TASK_WEIGHTS["tool_call"]
                self.criteria = default_criteria + tool_call_criteria(
                    expected_convo=messages_to_list(data.messages)
                )
            except Exception as e:
                bt.logging.error(f'Exception getting real task {e}')
                pass
        else:
            try:
                messages, tools, data = self.generate_dataset_task_data()
                self.criteria = default_criteria + dataset_tool_call_criteria() 
                self.postprocess = tool_call_postprocess()
                self.name += " Dataset"
                self.weight = TASK_WEIGHTS["tool_call_dataset"]
            except Exception as e:
                bt.logging.error(f'Exception getting dataset task {e}')
                pass
        self.messages = messages
        self.synapse = QnATask(
            urls=[], datas=[], tools=tools, messages=messages
        )

    
    def generate_dataset_task_data(self):
        try:
            data: ToolCallData = next(self.validator.local_tool_gen_dataset)
        except Exception as e:
            bt.logging.warning(f"Issue getting fake data {e}")
        messages_before_call = find_msgs_before_tool_call(data.messages)
        if messages_before_call[-1].role == "assistant":
            messages_before_call = messages_before_call[:-1]
        all_tools = data.tools
        random.shuffle(all_tools)
        return messages_before_call, all_tools, data
    
    
    def generate_task_data(self) -> ToolCallData:
        # use_synth = bool(random.random() < 0.01)
        use_synth = False
        if use_synth:
            data: ToolCallData = next(self.validator.local_tool_call_dataset)
        else:
            data: ToolCallData = next(self.validator.tool_dataset)
            for _ in range(random.randint(2,6)):
                data.tools = data.tools + next(self.validator.tool_dataset).tools
        
        # remove all the messages after the first tool call, keeping the assistant
        # this reduces the number of messages needing rewording
        messages = data.messages
        filtered_msgs = []
        seen_tool_call = False
        for msg in messages:
            filtered_msgs.append(msg)
            if seen_tool_call: # want to do break after to include the assistant response
                break
            if msg.role == 'tool call':
                seen_tool_call = True
        data.messages = filtered_msgs
        
        user = data.messages[0].content
        assistant = data.messages[-1].content
        
        count = 0
        while count < 10:
            count += 1
            if find_first_tool_call(data.messages):
                tool_call = find_first_tool_call(data.messages).content
                rewritten_tool_call = self.validator.chat_llm([{"role": "user", "content": REWRITE_TOOL_PROMPT.format(tool_call=tool_call)}], max_new_tokens=1000, temperature=1.2)
                try: # check that the tool call can be loaded, and that it's valid
                    try:
                        new_tool_call = json.dumps(json.loads(rewritten_tool_call))
                        tool_call_dict = json.loads(rewritten_tool_call)
                    except:
                        new_tool_call = json.dumps(ast.literal_eval(rewritten_tool_call))
                        tool_call_dict = ast.literal_eval(rewritten_tool_call)
                    for tool in data.tools:
                        if tool.name == tool_call_dict['name']:
                            if not validate_tool_call(tool, tool_call_dict):
                                raise Exception('The tool call is not valid')
                except Exception as e:
                    bt.logging.warning(f'An error occured while rewriting the tool call {e}')
                    count = 11
                    continue
                
                new_user = self.validator.chat_llm([{"role": "user", "content": REWRITE_TOOL_USER_PROMPT.format(tool_call=new_tool_call, user=user)}], max_new_tokens=1000, temperature=1)
                if not self.check_rewrite_alignment(new_user, user):
                    raise Exception(f"User rewrite is not in alignment\nOriginal: {user}\n Rewrite: {new_user}")
                
                new_assistant = self.validator.chat_llm([{"role": "user", "content": REWRITE_TOOL_ASSISTANT_PROMPT.format(tool_call=new_tool_call, user=new_user, assistant=assistant)}], max_new_tokens=1000, temperature=1).split("(")[0] # sometimes it adds an explanation in paranthesis
                if not self.check_rewrite_alignment(new_assistant, assistant):
                    raise Exception(f"Assistant rewrite is not in alignment\nOriginal: {assistant}\n Rewrite: {new_assistant}")
                
                data.messages[0].content = new_user
                data.messages[-1].content = new_assistant
                for i, msg in enumerate(data.messages):
                    if msg.role == 'tool call':
                        data.messages[i].content = new_tool_call

                data = ToolCallData(messages=data.messages, tools=data.tools)
                messages_before_call = find_msgs_before_tool_call(data.messages)
                if messages_before_call[-1].role == "assistant":
                    messages_before_call = messages_before_call[:-1]
                
            else:
                new_user = self.validator.validator_llm(REWRITE_PROPMT.format(query=user))
                if not self.check_rewrite_alignment(new_user, user):
                    raise Exception(f"User rewrite is not in alignment\nOriginal: {user}\n Rewrite: {new_user}")
                
                new_assistant = self.validator.validator_llm(REWRITE_PROPMT.format(query=assistant))
                if not self.check_rewrite_alignment(new_assistant, assistant):
                    raise Exception(f"Assistant rewrite is not in alignment\nOriginal: {assistant}\n Rewrite: {new_assistant}")
                
                data.messages[0].content = new_user
                data.messages[-1].content = new_assistant
                if messages_before_call[-1].role == "assistant":
                    messages_before_call = messages_before_call[:-1]
                
            all_tools = data.tools
            random.shuffle(all_tools)
            return messages_before_call, all_tools, data

    def check_rewrite_alignment(self, original: str, rewrite: str) -> bool:
        score = self.validator.measure_relevance_of_texts(original, rewrite)
        
        if score > 0.98:
            return False
        
        if score < 0.2:
            return False

        if len(rewrite) > 2 * len(rewrite):
            return False
        
        if len(rewrite) < 0.25 * len(rewrite):
            return False
        
        return True

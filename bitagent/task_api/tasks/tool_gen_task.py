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
from bitagent.task_api.criteria import default_criteria, tool_gen_criteria, dataset_tool_gen_criteria
from bitagent.task_api.datasources.tools import ToolCallData
from bitagent.task_api.datasources.tool_constants import categories 
from bitagent.task_api.helpers.convo_parsing import find_first_tool_call
from bitagent.task_api.postprocess import tool_gen_postprocess
from bitagent.task_api.tasks import TASK_WEIGHTS

query_system_prompt = """
You are a IT automation assistant. 
Given a category you will generate a user query that will be given to an AI chatbot. The user's query should be very concise, specific, and require a single step to complete (simple). The query should include all the information needed to complete the task.

For example:
Category: Password Cracking 
Query: Can you crack the password '421huihdwa' for me?

Now let's give it a shot!
"""
# For example:
# Category: Currency Excahnge
# Query: How much is '100' USD in EUR?
class ToolGenTask(Task):
    def __init__(
        self,
        validator: BaseValidatorNeuron,
        name: str,
        sub_task_id_to_get: int = None,
        desc: str = "",
    ):
        super().__init__(name=name, desc=desc)
        self.validator = validator
        self.timeout = 17.0
        self.name += " - Tool Generation"
        self.real_task = bool(random.random() < 0.90) # 90% chance of it being a real task
        if self.real_task:
            prompt, tool = self.generate_real_task()

            self.criteria = default_criteria + tool_gen_criteria(
                expected_tool = tool
            )
            self.weight = TASK_WEIGHTS['tool_gen']
        else:
            prompt = self.generate_dataset_task()

            self.criteria = default_criteria + dataset_tool_gen_criteria() 
            self.postprocess = tool_gen_postprocess()
            self.name += " Dataset"
            self.weight = TASK_WEIGHTS['tool_gen_dataset']
        notes = """Tool Generation"""
        self.synapse = QnATask(
            prompt=prompt, urls=[], datas=[], notes=notes, 
        )

    def generate_real_task(self) -> ToolCallData:
        for _ in range(100):
            if  bool(random.random() < 0.50):
                data: ToolCallData = next(self.validator.tool_dataset)
            else:
                data: ToolCallData = next(self.validator.local_tool_gen_dataset)
            if not any(msg for msg in data.convo.messages if msg.role == 'tool call'):
                continue
            query = data.convo.messages[0].content
            
            rewrite_prompt = f"""Please rewrite the following text, ensuring to maintain the original meaning and nuances but altering the sentence structures, vocabulary, and overall presentation. 
            The goal is to produce a version of the text that conveys the same information and sentiments as the original, but in a fresh and distinct manner. 
            Avoid summarizing or omitting any details; instead, focus on presenting the same concepts and messages in a new light.
            
            Rewrite this text: {query}
            
            Rewritten text: """
            
            rewritten_query = self.validator.validator_llm(rewrite_prompt, max_new_tokens=200)
            try:
                first_tool_call = json.loads(find_first_tool_call(data.convo).content)
            except Exception as e:
                # bt.logging.error(f"first tool call error {e}")
                pass

            tool = [tool for tool in data.tools if tool.name == first_tool_call['name']][0]
            
            return rewritten_query, dict(tool)
        raise Exception("No good tool calls could be generated")
    
    def generate_dataset_task(self):
        category = random.choice(categories)
        query = self.validator.chat_llm([
            {"role": "user", "content": f"{query_system_prompt}\nCategory: {category}\nQuery: "}
        ], temperature=0.7)
        return query
  
def remove_special_characters(input_string):
    # Using a list comprehension to filter only alphabet letters
    filtered_characters = [char for char in input_string if char.isalpha()]
    # Joining the filtered characters back into a string
    filtered_string = "".join(filtered_characters)
    return filtered_string

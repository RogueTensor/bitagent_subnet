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

import yaml
import random
from typing import List
from bitagent.schemas.tool import Tool
from bitagent.protocol import QnATask
from bitagent.task_api.tasks import Task, TASK_WEIGHTS
from common.base.validator import BaseValidatorNeuron
from bitagent.task_api.criteria import default_criteria, gen_tool_selection_criteria

class GeneratedToolSelectionTask(Task):
    def __init__(self, validator: BaseValidatorNeuron, name: str, sub_task_id_to_get: int = None, desc: str = ""):
        super().__init__(name=name, desc=desc)
        self.validator=validator
        self.timeout=8.0
        self.name += " - Ansible Tools"
        self.weight = TASK_WEIGHTS["generated_tool_selection"]
        prompt, tools, answer = self.generate_random_tool_prompt_tools_and_answer(sub_task_id_to_get)
        self.correct_answer = answer

        self.criteria=default_criteria+gen_tool_selection_criteria(expected_answer=answer)
        notes = """The task combines a prompt and a list of tools. 
The task is to generate a query that would require the tools to be completed in order as a list.
The provided prompt does not include everything, it does not include the tools for example.
This means you will need to build a query combining the provided prompt and the tools.
For example, your query (new prompt) to your LLM may look like: 
    "From these tools: <tools>, please provide the correct tools in the correct order (as a JSON list) for this task: <prompt>."
The correct answer is a list of the tools in the correct order, with the right spelling and syntax/case."""
        self.synapse=QnATask(prompt=prompt, urls=[], datas=[], tools=tools, notes=notes)

    # we need 3 things - 
    # many task sets randomly fetched
    # a selected task set from the random task sets
    # and a query for the action we want performed from the selected task set
    def generate_random_tool_prompt_tools_and_answer(self, sub_task_id_to_get: int = None) -> [str, List[dict], List]:
        queries, tool_names, tools = self.get_random_task_sets()
        selected_idx = random.randrange(len(tool_names))
        selected_tool_names = tool_names[selected_idx]
        prompt = queries[selected_idx]

        # return all tools as a shuffled list of good and bad tools
        return [prompt, tools, selected_tool_names]

    # get ansible tasks/playbooks and fetch the tools
    def get_random_task_sets(self):
        task_sets = []
        # get 3-6 task sets at random
        while len(task_sets) < random.randint(3, 6):
            try:
                tasks = next(self.validator.ansible_dataset)
                # only grabbing the named ones
                cleaned_tasks = []
                for task in tasks:
                    if "name" in task.keys():
                        cleaned_tasks.append(task)

                # must have enough tasks to be added
                if len(cleaned_tasks) > 3:
                    task_sets.append(cleaned_tasks)
                else:
                    #bt.logging.debug("Skipping ... could not add tasks b/c not enough named items: ", cleaned_tasks)
                    pass
            except Exception as e:
                #bt.logging.debug(f"Failed to generate tool task: {e}")
                pass

        query_prompt = """You are to write a query that a customer may ask, you need to make sure you follow these rules:
 - The query must be short and concise.
 - The query must be able to be answered by the tools provided in the correct order.
 - The query should not give away any of the tools or their order.

Write a short (2 sentences or less) customer-like query that would require the following tasks to be completed in order: 
        
{}"""
        queries = []
        tool_names = []
        all_tools = []
        # for each task set, get queries and updated, meaninful tool names
        for task_set in task_sets:
            query = self.validator.validator_llm(query_prompt.format(task_set))
            queries.append(query.strip())
            tool_renames = []
            for task in task_set:
                task_name = task["name"].strip()
                prompt = f"Given this query: {query}; please provide a meaningful, very simple and short function name (using camelcase syntax, no spaces, no newlines, no symbols, no scripting, just consecutive characters for the function name) representing this task: {task_name} - ",
                unhappy_count = 0
                while True: # while a good name hasnt been generated
                    tool_rename = self.validator.validator_llm(
                        prompt,
                        max_new_tokens=10
                    )
                    if len(tool_rename.strip().split(' ')) == 0:
                        break
                    unhappy_count += 1
                    if unhappy_count > 3:
                        prompt = f"Given this query: {query}; please provide a meaningful, simple and short function name (camelcase syntax, no spaces, no newlines, no symbols, no scripting, just consecutive characters for the function name) (IT MUST BE SHORT) representing this task: {task_name} - ",
                    if unhappy_count > 5:
                        break
                        
                tool_rename = remove_special_characters(tool_rename).strip()
                tool_renames.append(tool_rename)
                # add the Tool with name and description
                all_tools.append(Tool(
                    name=tool_rename,
                    description=task["name"].strip(),
                    arguments={}
                ))
            tool_names.append(tool_renames)

        return [queries, tool_names, all_tools]

def remove_special_characters(input_string):
    # Using a list comprehension to filter only alphabet letters
    filtered_characters = [char for char in input_string if char.isalpha()]
    # Joining the filtered characters back into a string
    filtered_string = ''.join(filtered_characters)
    return filtered_string

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
import random
import traceback
import bittensor as bt
from typing import List
from pprint import pformat
from bitagent.protocol import QueryTask
from bitagent.schemas.tool import Tool
from common.utils.uids import get_uid_rank
from bitagent.tasks import TASK_FREQUENCY
from common.base.validator import BaseValidatorNeuron
from bitagent.criteria import Criterion, default_criteria
from bitagent.schemas.chat import ChatMessage, messages_from_list, messages_to_list

# Task()
# combines criterion/criteria with the QnATask (prompt,data) for eval to form a task for the miner
class Task():
    criteria: List[Criterion]
    synapse: QnATask

    def __init__(self, 
                 name: str, 
                 weight: int = 0.05,
                 desc: str = "", 
                 timeout: int = 12,
                 tools: List[Tool] = [],
                 messages: List[ChatMessage] = [],
                 criteria: List[Criterion] = default_criteria,
                 response_should_contain: str = None, 
                 correct_answer: str=None
                        ) -> None:
        random.seed(None)
        self.name=name
        self.weight = weight
        self.desc=desc
        self.timeout=timeout
        self.criteria=criteria
        self.messages = messages
        self.response_should_contain=response_should_contain
        self.synapse = QueryTask(messages=messages, tools=tools)
        self.correct_answer = correct_answer

    def reward(self, validator: BaseValidatorNeuron, synapse: QueryTask, response:dict) -> [float, float, List[str]]:
        total_score = 0.0
        total_possible = 0.0
        results = []
        for criterion in self.criteria:
            score, max_score, result = criterion.evaluate(self, validator, synapse, response)
            total_score += score
            total_possible += max_score
            results.append(result)
        if self.correct_answer:
            correct_answer = self.correct_answer
        else:
            correct_answer = "N/A"
        return [total_score, total_possible, results, correct_answer]

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)
    
    def toJSON(self):
        return {
            "weight": self.weight,
            "name": self.name,
            "desc": self.desc,
            "messages": messages_to_list(self.messages) if isinstance(self.messages, list) else [], 
            "tools": [tool.to_dict() for tool in self.synapse.tools],
            "timeout": self.timeout,
        }

# evaluate task
def evaluate_task(validator, task:Task, synapse:bt.Synapse, response:dict) -> [float, float, List[str]]:
    # TODO tiered weighting
    return task.reward(validator, synapse, response)

# get random task
def get_random_task(validator, task_name=None, sub_task_id_to_get=None) -> Task:
    from bitagent.tasks import ToolCallTask
    random.seed(validator.random_seed())  
    task_names = list(TASK_FREQUENCY.keys())
    task_frequencies = list(TASK_FREQUENCY.values())
    choice = random.choices(task_names, weights=task_frequencies)[0]
    # (optional) override the task with provided task id
    if task_name and task_name in task_names:
        choice = task_name
    
    for _ in range(100):
        try:
            match choice:
                case "tool_call":
                    return ToolCallTask(validator=validator, name="Responds with correct function call")

        except Exception as e:
            bt.logging.warning(f'Error getting task (name {choice}): ', e)
            bt.logging.warning(traceback.format_exc())

    raise Exception("Failed to get task after 100 attempts")
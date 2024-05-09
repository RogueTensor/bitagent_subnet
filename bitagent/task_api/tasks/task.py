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

import time
import random
import bittensor as bt
from pprint import pformat
from typing import List
from bitagent.protocol import QnATask
from bitagent.task_api.criteria import Criterion, default_criteria
from common.base.validator import BaseValidatorNeuron
from redis import Redis
from rq import Queue
from bitagent.types import Tool

queue = Queue(connection=Redis(host='localhost', port=14000))

# Task()
# combines criterion/criteria with the QnATask (prompt,data) for eval to form a task for the miner
class Task():
    criteria: List[Criterion]
    synapse: QnATask

    def __init__(self, 
                 name: str, 
                 prompt: str = "", 
                 desc: str = "", 
                 datas: List[dict] = [],
                 tools: List[Tool] = [],
                 notes: str = "No Notes",
                 urls: List[str] = [], 
                 criteria: List[Criterion] = default_criteria,
                 citation_sources_should_contain: str = None, 
                 response_should_contain: str = None, 
                 task_type: str=None, 
                 task_id: str=None,
                 correct_answer: str=None
                        ) -> None:
        random.seed(None)
        if task_id:
            self.task_id=task_id
        else:
            self.task_id=str(random.getrandbits(128))
        self.name=name
        self.task_type=task_type
        self.desc=desc
        self.timeout=10.0
        self.criteria=criteria
        self.citation_sources_should_contain=citation_sources_should_contain
        self.response_should_contain=response_should_contain
        self.synapse = QnATask(prompt=prompt, urls=urls, datas=datas, notes=notes, tools=[Tool(tool) for tool in tools])
        self.correct_answer = correct_answer

    def reward(self, validator: BaseValidatorNeuron, synapse: QnATask, response:dict) -> [float, float, List[str]]:
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
    
    @classmethod
    def fromSerialized(cls, serialized):
        task = cls(
            name=serialized["name"], 
            prompt=serialized["prompt"], 
            desc=serialized["desc"], 
            datas=serialized["datas"], 
            tools=[Tool(**tool) for tool in serialized["tools"]], 
            urls=serialized["urls"], 
            criteria=[Criterion.fromSerialized(c) for c in serialized["criteria"]], 
            citation_sources_should_contain=(serialized["citation_sources_should_contain"] if "None" != serialized["citation_sources_should_contain"] else None), 
            response_should_contain=(serialized["response_should_contain"] if "None" != serialized["response_should_contain"] else None), 
            task_type=(serialized["task_type"] if "None" != serialized["task_type"] else None), 
            task_id=(serialized["task_id"] if "None" != serialized["task_id"] else None),
            correct_answer = serialized["correct_answer"]
            )
        return task
    
    def serialize(self):
        return {
            "task_id": str(self.task_id),
            "task_type": str(self.task_type),
            "name": self.name,
            "prompt": self.synapse.prompt,
            "desc": self.desc,
            "tools": [dict(tool) for tool in self.synapse.tools],
            "notes": self.synapse.notes,
            "datas": self.synapse.datas,
            "urls": self.synapse.urls,
            "timeout": self.timeout,
            "criteria": [c.serialize() for c in self.criteria],
            "citation_sources_should_contain": str(self.citation_sources_should_contain),
            "response_should_contain": str(self.response_should_contain),
            "correct_answer": (str(self.correct_answer) if self.correct_answer else "N/A")
        }
    
    def toJSON(self):
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "name": self.name,
            "prompt": self.synapse.prompt,
            "desc": self.desc,
            "datas": self.synapse.datas,
            "tools": [dict(tool) for tool in self.synapse.tools],
            "notes": self.synapse.notes,
            "urls": self.synapse.urls,
            "timeout": self.timeout,
        }

# fetch organic tasks
# organic tasks are non-generated tasks
# in this case we are first looking to see if there are any tasks in the queue
def get_organic_task():
    jobs = [job for job in queue.jobs if job.is_queued]
    if not jobs:
        return None

    job = jobs[0]
    job.set_status('started')
    try:
        job_datas = job.args[1]
    except Exception as e:
        job_datas = []

    return Task(name="Organic Task", task_type="organic", prompt=job.args[0], datas=job_datas, task_id=job.id)

# evaluate task
def evaluate_task(validator, task:Task, synapse:bt.Synapse, response:dict) -> [float, float, List[str]]:
    return task.reward(validator, synapse, response)

# get random task
# right now the core tasks are:
# - QnA with Citations
# - Summarization
# - Logic QnA (pet tricks)
def get_random_task(validator, task_id_to_get=None, sub_task_id_to_get=None) -> Task:
    from bitagent.task_api.tasks import SummaryTask, GeneratedQnATask, GeneratedLogicQnATask, GeneratedToolSelectionTask, basic_qna_miner_tasks
    random.seed(validator.random_seed())  
    task_ids = [1,2,3,4,5,6,7,8,9,10]
    weights  = [0,0,1,1,2,4,0,3,0,4]
    choice = random.choices(task_ids, weights=weights)[0]

    # (optional) override the task with provided task id
    if task_id_to_get and task_id_to_get in task_ids:
        choice = task_id_to_get

    no_task_selected = True
    while no_task_selected:
        try:
            match choice:
                case 1:
                    return GeneratedQnATask(validator=validator, name="Responds with correct citation source and valid response", timeout=3.0)
                case 2:
                    return GeneratedQnATask(validator=validator, name="Responds with correct citation source and valid response from medium corpus", n_texts=10, timeout=4.0)
                case 3:
                    return GeneratedQnATask(validator=validator, name="Responds with correct citation source and valid response from larger corpus", n_texts=20, timeout= 6.0)
                case 4:
                    return GeneratedQnATask(validator=validator, name="Responds with correct citation source and valid response from LARGE corpus", n_texts=50, timeout=8.0)
                case 5:
                    return GeneratedQnATask(validator=validator, name="Responds with correct citation source and valid response from VERY LARGE corpus", n_texts=100, timeout=10.0)
                case 6:
                    return GeneratedLogicQnATask(validator=validator, name="Responds with correct answer for logic-based question", sub_task_id_to_get=sub_task_id_to_get)
                case 7:
                    pass
                    #return GeneratedAgentTask(validator=validator, name="Interact with simulation")
                case 8:
                    return SummaryTask(validator=validator, name="Responds with correct summary")
                case 9:
                    return random.choice(basic_qna_miner_tasks)
                case 10:
                    return GeneratedToolSelectionTask(validator=validator, name="Responds with correct tool selection")
        except Exception as e:
            print('Error: ', e)
            time.sleep(15)

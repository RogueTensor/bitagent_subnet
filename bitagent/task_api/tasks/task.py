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
from bitagent.protocol import QnATask
from bitagent.schemas.tool import Tool
from common.utils.uids import get_uid_rank
from bitagent.task_api.tasks import TASK_FREQUENCY
from common.base.validator import BaseValidatorNeuron
from bitagent.task_api.postprocess import PostProcessor
from bitagent.task_api.criteria import Criterion, default_criteria
from bitagent.schemas.chat import ChatMessage, messages_from_list, messages_to_list

# Task()
# combines criterion/criteria with the QnATask (prompt,data) for eval to form a task for the miner
class Task():
    criteria: List[Criterion]
    synapse: QnATask
    postprocess: List[PostProcessor]

    def __init__(self, 
                 name: str, 
                 prompt: str = "", 
                 weight: int = 0.05,
                 desc: str = "", 
                 timeout: int = 12,
                 datas: List[dict] = [],
                 tools: List[Tool] = [],
                 messages: List[ChatMessage] = [],
                 files: List[dict] = [],
                 notes: str = "No Notes",
                 urls: List[str] = [], 
                 criteria: List[Criterion] = default_criteria,
                 postprocess: List[PostProcessor] = [],
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
        self.weight = weight
        self.desc=desc
        self.timeout=timeout
        self.criteria=criteria
        self.postprocess=postprocess
        self.citation_sources_should_contain=citation_sources_should_contain
        self.messages = messages
        self.files = files
        self.response_should_contain=response_should_contain
        self.synapse = QnATask(prompt=prompt, urls=urls, datas=datas, notes=notes, tools=tools, messages=messages, files=files)
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
            timeout=serialized["timeout"],
            weight=serialized["weight"],
            datas=serialized["datas"], 
            notes=serialized["notes"],
            tools=[Tool(**tool) for tool in serialized["tools"]], 
            urls=serialized["urls"], 
            criteria=[Criterion.fromSerialized(c) for c in serialized["criteria"]], 
            postprocess=[PostProcessor.fromSerialized(p) for p in serialized["postprocess"]],
            citation_sources_should_contain=(serialized["citation_sources_should_contain"] if "None" != serialized["citation_sources_should_contain"] else None), 
            response_should_contain=(serialized["response_should_contain"] if "None" != serialized["response_should_contain"] else None), 
            task_type=(serialized["task_type"] if "None" != serialized["task_type"] else None), 
            task_id=(serialized["task_id"] if "None" != serialized["task_id"] else None),
            correct_answer = serialized["correct_answer"],
            messages = messages_from_list(serialized['messages']),
            files = serialized['files'],
            )
        return task

    def serialize(self):
        return {
            "task_id": str(self.task_id),
            "task_type": str(self.task_type),
            "weight": self.weight,
            "name": self.name,
            "prompt": self.synapse.prompt,
            "desc": self.desc,
            "tools": [dict(tool) for tool in self.synapse.tools],
            "notes": self.synapse.notes,
            "messages": messages_to_list(self.synapse.messages) if isinstance(self.synapse.messages, list) else [],
            "datas": self.synapse.datas,
            "urls": self.synapse.urls,
            "timeout": self.timeout,
            "criteria": [c.serialize() for c in self.criteria],
            "postprocess": [p.serialize() for p in self.postprocess],
            "citation_sources_should_contain": str(self.citation_sources_should_contain),
            "response_should_contain": str(self.response_should_contain),
            "correct_answer": (str(self.correct_answer) if self.correct_answer else "N/A"),
            "files": self.synapse.files,
        }
    
    def toJSON(self):
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "weight": self.weight,
            "name": self.name,
            "prompt": self.synapse.prompt,
            "desc": self.desc,
            "messages": messages_to_list(self.messages) if isinstance(self.messages, list) else [], 
            "datas": self.synapse.datas,
            "tools": [tool.to_dict() for tool in self.synapse.tools],
            "notes": self.synapse.notes,
            "urls": self.synapse.urls,
            "timeout": self.timeout,
            "files": self.synapse.files,
        }

# evaluate task
def evaluate_task(validator, task:Task, synapse:bt.Synapse, response:dict) -> [float, float, List[str]]:
    total_score, total_possible, results, correct_answer = task.reward(validator, synapse, response)

    # if the total score is above a threshold, it's a top MINER and we have post processes to run
    # then use this data to build a dataset for future queries
    # only used for "tool_call" and "tool_gen" tasks
    try:
        if (total_score / total_possible) > 0.25 and task.postprocess and get_uid_rank(validator, validator.metagraph.hotkeys.index(response['axon_hotkey'])) < 10:
            for postprocessor in task.postprocess:
                postprocessor(task, validator, synapse, response)
    except:
        pass
    return [total_score, total_possible, results, correct_answer]

# get random task
# right now the core tasks are:
# - QnA with Citations
# - Summarization
# - Logic QnA (pet tricks)
def get_random_task(validator, task_name=None, sub_task_id_to_get=None) -> Task:
    from bitagent.task_api.tasks import SummaryTask, GeneratedQnATask, GeneratedLogicQnATask, ToolCallTask, ToolGenTask, ConversationTask, GeneratedPlotQnATask
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
                case "generated_qna":
                    sub_choice = random.choices([1,2,3], weights=[1,1,2])[0]
                    match sub_choice:
                        case 1:
                            return GeneratedQnATask(validator=validator, name="Responds with correct citation source and valid response from small corpus", n_texts=2, timeout=6.0)
                        case 2:
                            return GeneratedQnATask(validator=validator, name="Responds with correct citation source and valid response from LARGE corpus", n_texts=3, timeout=8.0)
                        case 3:
                            return GeneratedQnATask(validator=validator, name="Responds with correct citation source and valid response from VERY LARGE corpus", n_texts=5, timeout=10.0)
                case "generated_logic_qna":
                    return GeneratedLogicQnATask(validator=validator, name="Responds with correct answer for logic-based question", sub_task_id_to_get=sub_task_id_to_get)
                case "summary":
                    return SummaryTask(validator=validator, name="Responds with correct summary")
                case "tool_call":
                    return ToolCallTask(validator=validator, name="Responds with correct function call")
                case "tool_gen":
                    return ToolGenTask(validator=validator, name="Responds with correct function generation") 
                case "conversation":
                    return ConversationTask(validator=validator, name="Responds with correct assistant response") 
                case "generated_plot_qna":
                    return GeneratedPlotQnATask(validator=validator, name="Responds with correct answer to plot-based question") 

        except Exception as e:
            bt.logging.warning(f'Error getting task (name {choice}): ', e)
            bt.logging.warning(traceback.format_exc())

            # time.sleep(15)
    raise Exception("Failed to get task after 100 attempts")

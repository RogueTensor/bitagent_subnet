import random
import bittensor as bt
from pprint import pformat
from typing import Callable, List
from bitqna.protocol import QnAProtocol
from template.base.validator import BaseValidatorNeuron
from bitqna.validator.criterion import default_criteria, basic_no_citations, basic_citations, Criterion

class Task():
    criteria: List[Criterion]
    synapse: QnAProtocol

    def __init__(self, name: str, desc: str, prompt: str, 
                 urls: List[str] = [], criteria: List[Criterion] = default_criteria) -> None:
        # TODO may be something other than QnAProtocol for the task synapse, so handle that
        self.name=name
        self.criteria=criteria
        self.desc=desc
        self.synapse=QnAProtocol(prompt=prompt, urls=urls)

    def reward(self, response: str) -> float:
        total_score = 0.0
        for criterion in self.criteria:
            total_score += criterion.evaluate(response)
        return total_score

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)


miner_tasks = [
    Task(name="Responds without URL(s)", desc="", 
         criteria=default_criteria+[basic_no_citations],
         prompt='who is the most famous ghost buster'),
    Task(name="Responds with a single URL", desc="", 
         urls=["https://en.wikipedia.org/wiki/Ghostbusters"],
         criteria=default_criteria+[basic_citations],
         prompt='who is the most famous ghost buster'),
    Task(name="Responds with at least one citation", desc="", 
         urls=["https://en.wikipedia.org/wiki/Ghostbusters"],
         criteria=default_criteria+[basic_citations],
         prompt='who is the most famous ghost buster'),
]

def get_random_task() -> Task:
    return random.choice(miner_tasks)

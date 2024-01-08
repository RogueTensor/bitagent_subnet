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
from typing import Callable, List
from bitqna.protocol import QnAProtocol
from bitqna.validator.criteria import Criterion, default_criteria
from template.base.validator import BaseValidatorNeuron

# combines criterion/criteria for eval to form a task for the miner
class Task():
    criteria: List[Criterion]
    synapse: QnAProtocol

    def __init__(self, name: str, prompt: str, desc: str = "", datas: List[dict] = [],
                 urls: List[str] = [], criteria: List[Criterion] = default_criteria,
                 citation_sources_should_contain: str=None, response_should_contain: str=None) -> None:
        self.name=name
        self.desc=desc
        self.criteria=criteria
        self.citation_sources_should_contain=citation_sources_should_contain
        self.response_should_contain=response_should_contain
        self.synapse=QnAProtocol(prompt=prompt, urls=urls, datas=datas)

    def reward(self, validator: BaseValidatorNeuron, response: str) -> [float, float, List[str]]:
        total_score = 0.0
        total_possible = 0.0
        results = []
        for criterion in self.criteria:
            score, max_score, result = criterion.evaluate(self, validator, response)
            total_score += score
            total_possible += max_score
            results.append(result)
        return [total_score, total_possible, results]

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)


def get_random_task(validator: BaseValidatorNeuron) -> Task:
    from bitqna.validator.tasks import GeneratedDataTask, basic_qna_miner_tasks
    # for now just looking at validating responses and citations for 0+ data
    return random.choices([
        GeneratedDataTask(validator=validator, name="Responds with correct citation source and valid response"),
        GeneratedDataTask(validator=validator, name="Responds with correct citation source and valid response from medium corpus", n_texts=8),
        GeneratedDataTask(validator=validator, name="Responds with correct citation source and valid response from larger corpus", n_texts=20),
        GeneratedDataTask(validator=validator, name="Responds with correct citation source and valid response from LARGE corpus", n_texts=50),
        random.choice(basic_qna_miner_tasks),
        ], weights=[50,15,10,5,20])[0]

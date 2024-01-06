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
from template.base.validator import BaseValidatorNeuron
from bitqna.validator.criterion import default_criteria, basic_no_citations, basic_citations, Criterion, gen_data_task_criteria

# combines criterion/criteria for eval to form a task for the miner
class Task():
    criteria: List[Criterion]
    synapse: QnAProtocol

    def __init__(self, name: str, prompt: str, desc: str = "", datas: List[dict] = [],
                 urls: List[str] = [], criteria: List[Criterion] = default_criteria) -> None:
        self.name=name
        self.desc=desc
        self.criteria=criteria
        self.synapse=QnAProtocol(prompt=prompt, urls=urls, datas=datas)

    def reward(self, validator: BaseValidatorNeuron, response: str) -> [float, List[str]]:
        total_score = 0.0
        results = []
        for criterion in self.criteria:
            score, result = criterion.evaluate(validator, response)
            total_score += score
            results.append(result)
        return [total_score, results]

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

# generated task for n_texts of data looking for n_expected_citations of relevant citations (sources and contexts)
class GeneratedDataTask(Task):
    def __init__(self, validator: BaseValidatorNeuron, name: str, 
                 desc: str = "", n_texts:int = 3, n_expected_citations:int = 1):

        self.name=name + f" for {n_texts} texts and {n_expected_citations} expected citations"
        self.desc=desc
        self.validator=validator

        datas = self.generate_random_texts(n_texts=n_texts)
        texts = [d['context'] for d in datas]
        sources = [d['source'] for d in datas]
        selected_num=random.randrange(len(texts))
        selected_text = texts[selected_num]
        selected_source = sources[selected_num]
        question = self.get_question_for_text(text=selected_text)
    
        self.criteria=gen_data_task_criteria(selected_datas=[datas[selected_num]], n_expected_citations=n_expected_citations)
        self.synapse=QnAProtocol(prompt=question, urls=[], datas=datas)

    def generate_random_texts(self, n_texts: int = 3) -> [List[str], List[str]]:
        # get n random data
        output = []
        for _ in range(n_texts):
            text = next(self.validator.dataset)["text"]
            text = text[:500]
            # arbitrary and random source ids
            h = random.getrandbits(128)
            source = f'{time.strftime("%Y%m%d")}.bitqna.source.{h}'
            output.append({'source':source,'context':text})
        return output

    def get_question_for_text(self, text: str) -> str:
        # for the simple model, use less text, truncate
        # current model (T5) takes 512 tokens, roughly 4 chars per token (per google PaLM 2), 1900 leaves us room for the rest of the prompt
        truc_text = text[:1900]
        input_text = f"TEXT: {truc_text}\n\nProvide a 1-sentence question for the provided TEXT making sure to leverage key words from the TEXT in the question.  Please do not ask questions similar to the following:\n- What is the topic of the article?\n- What's the passage about?\n- What's the game about?\n\nRespond with an insightful question leveraging keywords from the provided TEXT.\nResponse: "
        question = self.validator.validator_llm(input_text)
        return question

basic_miner_tasks = [
    Task(name="Responds without URL(s) or Datas",
         criteria=default_criteria+[basic_no_citations],
         prompt='who is the most famous ghost buster'),
    Task(name="Responds with at least one citation",
         datas=[{'source': "simple test", "context":"foobar"}],
         criteria=default_criteria+basic_citations,
         prompt='who is the most famous ghost buster'),
]

def get_random_task(validator: BaseValidatorNeuron) -> Task:
    # for now just looking at validating responses and citations for 0+ data
    return GeneratedDataTask(validator=validator, name="Responds with correct citation source and valid response")
    #return random.choice(basic_miner_tasks)

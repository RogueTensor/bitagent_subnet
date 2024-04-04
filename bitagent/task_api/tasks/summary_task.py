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
from bitagent.protocol import QnATask
from bitagent.task_api.tasks import Task
from common.base.validator import BaseValidatorNeuron
from bitagent.task_api.criteria import default_criteria, summary_task_criteria

# Summarization task
# these tasks look to test summarization implementations
# random selections of source data is selected from the task api dataset and then rewritten
# or a random story is generated 
# from there the miner is asked to summarize the text
# the summary is checked for alignment with the text as well as ground truth
class SummaryTask(Task):
    def __init__(self, validator: BaseValidatorNeuron, name: str, desc: str = ""):
        super().__init__(name=name, desc=desc)
        self.validator=validator
        self.timeout=6.0

        prompt, summary, summary_gen = self.get_random_task()
        self.criteria=default_criteria+summary_task_criteria(summary=summary, summary_gen=summary_gen)
        self.synapse=QnATask(prompt=prompt, urls=[], datas=[])

    def get_random_texts(self) -> [str, str]:
        data = next(self.validator.summary_dataset)
        return data["text"], data["summary"]
        
    def get_random_task(self) -> [str, str, str]:
        if random.random() < 0.5:
            text, summary = self.get_random_texts()
            llm_text = self.validator.validator_llm(f"Do not summarize, do keep the same length and reword this text: {text}", temperature=0.5, max_new_tokens=3000)
            prompt = f"Summarize this and make sure to be concise: {llm_text}"
            summary_gen = self.validator.validator_llm(prompt)
            return prompt, summary, summary_gen
        else:
            text = self.validator.validator_llm(f"Please generate a fairly long story about {self.validator.fake.name()} who has a job in {self.validator.fake.job()} and make it funny with twists and turns.", temperature=0.5, max_new_tokens=3000)
            prompt = f"Summarize this and make sure to be concise: {text}"
            summary_gen = self.validator.validator_llm(prompt)
            return prompt, summary_gen, summary_gen

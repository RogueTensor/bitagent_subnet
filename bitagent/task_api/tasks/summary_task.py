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
from bitagent.task_api.tasks import TASK_WEIGHTS
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
        self.weight = TASK_WEIGHTS['summary']
        prompt, summary, summary_gen = self.get_random_task()
        self.criteria=default_criteria+summary_task_criteria(summary=summary, summary_gen=summary_gen)
        notes = """The task is built from a prompt that includes the text to be summarized.
The task is to provide a reasonable summary of the provided text using an LLM."""
        self.synapse=QnATask(prompt=prompt, urls=[], datas=[], notes=notes)

    def get_random_texts(self) -> [str, str]:
        data = next(self.validator.summary_dataset)
        return data["text"], data["summary"]
        
    def get_random_task(self) -> [str, str, str]:
        if random.random() < 0.5:
            long_enough = False
            num_tries = 0
            while not long_enough and num_tries < 5:
                text, summary = self.get_random_texts()
                num_words = len(text.split())
                rewrite_prompt = f"""Please rewrite the following text, ensuring to maintain the original meaning and nuances but altering the sentence structures, vocabulary, and overall presentation. 
                The goal is to produce a version of the text that conveys the same information and sentiments as the original, but in a fresh and distinct manner. 
                Avoid summarizing or omitting any details; instead, focus on presenting the same concepts and messages in a new light.
                
                Rewrite this text in at least {num_words} words: {text}"""
                num_tries += 1
                llm_text = self.validator.validator_llm(rewrite_prompt, temperature=0.5, max_new_tokens=4096)
                if len(llm_text.split()) > 0.9 * num_words: 
                    long_enough = True
            prompt = f"Summarize this and make sure to be concise: {llm_text}"
            summary_gen = self.validator.validator_llm(prompt)
            return prompt, summary, summary_gen
        else:
            text = self.validator.validator_llm(f"Please generate a fairly long story about {self.validator.fake.name()} who has a job in {self.validator.fake.job()} and make it funny with twists and turns.", temperature=0.5, max_new_tokens=3000)
            prompt = f"Summarize this and make sure to be concise: {text}"
            summary_gen = self.validator.validator_llm(prompt)
            return prompt, summary_gen, summary_gen
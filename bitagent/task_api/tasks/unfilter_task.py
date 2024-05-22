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

from bitagent.protocol import QnATask
from bitagent.task_api.tasks import Task
from common.base.validator import BaseValidatorNeuron
from bitagent.task_api.tasks import TASK_WEIGHTS
from bitagent.task_api.criteria import default_criteria, unfilter_task_criteria

REJECTIONS = ["I apologize I cannot do that"] 



class FilterTask(Task):
    def __init__(self, validator: BaseValidatorNeuron, name: str, desc: str = ""):
        super().__init__(name=name, desc=desc)
        self.validator=validator
        self.timeout=12.0

        self.weight = TASK_WEIGHTS['unfilter'] 
        
        prompt, ds_response, ds_response_gen = self.get_random_task()
        self.criteria=default_criteria+unfilter_task_criteria(ds_response=ds_response, ds_response_gen=ds_response_gen)
        notes = """Unfiltered"""
        self.synapse=QnATask(prompt=prompt, urls=[], datas=[], notes=notes)

    def get_random_texts(self) -> [str, str]:
        data = next(self.validator.filter_dataset)
        return data["prompt"], data["response"]
    
    def gen_rewrite_prompt(self, query):
        num_words = len(query.split())
        rewrite_prompt = f"""Please rewrite the following text, ensuring to maintain the original meaning and nuances but altering the sentence structures, vocabulary, and overall presentation. 
        The goal is to produce a version of the text that conveys the same information and sentiments as the original, but in a fresh and distinct manner. 
        Avoid summarizing or omitting any details; instead, focus on presenting the same concepts and messages in a new light.
        
        Rewrite this text in at least {num_words} words: {query}
        
        Rewritten text: """
        return rewrite_prompt
    
    def get_random_task(self) -> [str, str, str]:
        for _ in range(10):
            prompt, ds_response = self.get_random_texts()
            prompt_gen = self.validator.validator_llm(self.gen_rewrite_prompt(prompt), temperature=0.5, max_new_tokens=4096)
            if not self.validator.measure_relevance_of_texts(prompt, prompt_gen) > 0.5:
                continue
            
            ds_response_gen = self.validator.validator_llm(self.gen_rewrite_prompt(ds_response), temperature=0.5, max_new_tokens=4096)
            if not self.validator.measure_relevance_of_texts(ds_response, ds_response_gen) > 0.7:
                continue
            if any([self.validator.measure_relevance_of_texts(rej,ds_response_gen) > 0.8 for rej in REJECTIONS]):
                continue
        
            return prompt_gen, ds_response, ds_response_gen
        raise Exception("Failed to generate a valid task")
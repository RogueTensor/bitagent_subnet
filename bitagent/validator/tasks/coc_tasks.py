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
from typing import List
from bitagent.protocol import QnATask
from bitagent.validator.tasks import Task
from common.base.validator import BaseValidatorNeuron
from bitagent.validator.criteria import default_criteria, coc_task_criteria
from bitagent.validator.prompts import coc_prompts

# Chain of Code tasking with popular CoC challenges using faker data
class CoCSarcasmTask(Task):
    pass

class CoCPlacesTask(Task):
    def __init__(self, validator: BaseValidatorNeuron, name: str, desc: str = "",
                 num_countries: int = 5)

        self.name=name
        self.desc=desc
        self.validator=validator

        prompt = coc_prompts.random_places_prompt(num_countries)
    
        self.criteria=default_criteria+coc_task_criteria(prompt, num_countries)
        self.synapse=QnATask(prompt=prompt, urls=[], datas=[])

        return prompt

# TODO get_tasks():
#        task_type = random.choice(["sarcasm", "countries"])
#if task_type == "countries":

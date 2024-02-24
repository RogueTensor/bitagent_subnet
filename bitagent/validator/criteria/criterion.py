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

import bittensor as bt
from pprint import pformat
from typing import Callable,List
from common.base.validator import BaseValidatorNeuron
from bitagent.validator.criteria.utils import good_message, bad_message, received_reward_template
from bitagent.validator.criteria.default_criteria import *
from bitagent.validator.criteria.qna_criteria import *
from bitagent.validator.criteria.summary_criteria import *
from bitagent.validator.criteria.qna_logic_criteria import *

# building block for the criteria used to evaluate the miner's response
class Criterion():
    name: str
    desc: str
    eval_fx: Callable

    def __init__(self, name: str, desc: str, eval_fx: Callable, eval_args=[]) -> None:
        self.name = name
        self.desc = desc
        self.eval_fx = eval_fx
        self.eval_args = eval_args

    def evaluate(self, task, validator: BaseValidatorNeuron, response: bt.Synapse) -> [float, float, str]:
        try:
            reward, max_reward, feedback = self.eval_fx(task, validator, response, *self.eval_args)
        except Exception as e:
            bt.logging.info(f"Exception was raised during criteria evaluation: {e}")
            reward = -0.5
            max_reward = 1.0
            feedback = bad_message("Exception while processing your response, please check format per protocol")
        feedback = f"[bold blue]{self.name}[/bold blue]\n" + feedback
        return reward, max_reward, feedback

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

# the core set of tests that form a set of criteria 
def gen_data_task_criteria(selected_datas: List[dict], n_expected_citations:int) -> List[Criterion]:
    return [
        Criterion(name=f"Returns expected citation source(s)", desc="", eval_fx=contains_correct_number_of_citation_sources, eval_args=[selected_datas]),
        Criterion(name=f"Returns valid response", desc="", eval_fx=correct_response_provided, eval_args=[selected_datas]),
    ]

def gen_numerical_logic_task_criteria(expected_answer:int) -> List[Criterion]:
    return [
        Criterion(name=f"Returns expected value", desc="", eval_fx=contains_correct_numerical_logic_answer, eval_args=[expected_answer]),
    ]

def summary_task_criteria(summary: str) -> List[Criterion]:
    return [
        Criterion(name="Return valid summary", desc="", eval_fx=correct_summary_provided, eval_args=[summary]),
    ]

# simple, defaults
default_criteria = [
    Criterion(name="Does not error", desc="", eval_fx=does_not_error), 
    Criterion(name="Does not take a long time", desc="", eval_fx=does_not_take_a_long_time),
]

# basic CITATION checks
basic_citations = [
    Criterion(name="Must have one citation", desc="", eval_fx=contains_number_citations, eval_args=[1, 5]),
    Criterion(name="Must have correct citation format", desc="", eval_fx=correct_citation_format),
    Criterion(name="Must have correct citation source", desc="", eval_fx=contains_correct_citation_source),
]
basic_no_citations = [
    Criterion(name="Must not return any citations", desc="", eval_fx=contains_number_citations, eval_args=[0, 0]),
    Criterion(name="Must contain some text", desc="", eval_fx=contains_some_text),
]
simple_context_aware = Criterion(name="Must return a valid response based on context", desc="", eval_fx=correct_response_provided_simple)

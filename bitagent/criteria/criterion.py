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

import ast
import bittensor as bt
from pprint import pformat
from typing import Callable, List
from common.base.validator import BaseValidatorNeuron
from bitagent.criteria.utils import bad_message
from bitagent.criteria.default_criteria import *
from bitagent.criteria.tool_call_criteria import *

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

    def evaluate(self, task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict={}) -> [float, float, str]:
        try:
            # make sure the tool response converts nicely to an ast
            try:
                ast.parse(synapse.response)
            except:
                reward = -0.5
                max_reward = 1.0
                feedback = bad_message(f"Your response: {synapse.response} was not parsable")
                return reward, max_reward, feedback
            # make sure the response is not empty
            if synapse.response.strip() is not None and synapse.response.strip() != "":
                # actually do the evaluation 
                reward, max_reward, feedback = self.eval_fx(task, validator, synapse, response, *self.eval_args)
            else:
                reward = -0.5
                max_reward = 1.0
                feedback = bad_message("Your response was empty, please check format per protocol")
        except Exception as e:
            #bt.logging.error(f"Exception was raised during criteria evaluation: {e}")
            reward = -0.5
            max_reward = 1.0
            feedback = bad_message("Exception while processing your response, please check format per protocol")
        feedback = f"[bold blue]{self.name}[/bold blue]\n" + feedback
        return reward, max_reward, feedback

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

# Function Call
def tool_call_criteria(expected_response: dict) -> List[Criterion]:
    return [
        Criterion(name="Return correct function format", desc="", eval_fx=correct_tool_call_function_format),
        Criterion(name="Return correct function name", desc="", eval_fx=correct_tool_call_function_name, eval_args=[expected_response]),
        Criterion(name="Return function with correct argument names", desc="", eval_fx=correct_tool_argument_names, eval_args=[expected_response]),
        Criterion(name="Return function with correct argument values", desc="", eval_fx=correct_tool_argument_values, eval_args=[expected_response]),
    ]

def irrelevant_tool_call_criteria() -> List[Criterion]:
    return [
        Criterion(name="Return valid function call for irrelevant tool", desc="", eval_fx=correct_irrelevant_tool_call),
    ]

# simple, defaults
default_criteria = [
    Criterion(name="Does not error", desc="", eval_fx=does_not_error), 
    Criterion(name="Does not take a long time", desc="", eval_fx=does_not_take_a_long_time),
]
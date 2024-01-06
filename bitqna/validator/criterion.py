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
from template.base.validator import BaseValidatorNeuron

received_reward_template="\nYou received {} of {} reward."

def bad_message(text: str) -> str:
    return ":cross_mark: [red]" + text + "[/red]"

def good_message(text: str) -> str:
    return ":white_heavy_check_mark: [green]" + text + "[/green]"

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

    def evaluate(self, validator: BaseValidatorNeuron, response: bt.Synapse) -> [float, str]:
        reward, feedback = self.eval_fx(validator, response, *self.eval_args)
        feedback = f"[bold blue]{self.name}[/bold blue]\n" + feedback
        return reward, feedback

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

# CRITERION: successful call to miner
def does_not_error(validator: BaseValidatorNeuron, response: bt.Synapse) -> [float, str]:
    max_reward = 0.5
    a_status_code = response.axon.status_code
    d_status_code = response.dendrite.status_code
    reward = -0.5
    feedback = bad_message("You failed to respond correctly to the request.")
    if a_status_code == "200" and d_status_code == "200":
        reward = 0.5
        feedback = good_message("You successfully responded to the request.")
    return reward, feedback + received_reward_template(reward, max_reward)

# CRITERION: reward speedy response
def does_not_take_a_long_time(validator: BaseValidatorNeuron, response: bt.Synapse) -> [float, str]:
    max_reward = 0.5
    process_time = response.dendrite.process_time
    feedback = f"You responded to the request in {process_time}."
    reward = 0.0
    if process_time <= 2.0: 
        reward = 0.50
        return reward, good_message(feedback) + received_reward_template.format(reward,max_reward)
    if process_time <= 5.0:
        reward = 0.25
        return reward, good_message(feedback) + received_reward_template.format(reward,max_reward)
    if process_time <= 10.0:
        reward = 0.10
        return reward, bad_message(feedback) + received_reward_template.format(reward,max_reward)
    return reward, bad_message(feedback) + received_reward_template.format(reward,max_reward)

# CRITERION: reward proper citation dict format
def correct_citation_format(validator: BaseValidatorNeuron, response: bt.Synapse) -> [float, str]:
    max_reward = 1.0
    try:
        citations = response.response['citations']
        sources = [c['source'] for c in citations]
        contexts = [c['context'] for c in citations]
    except KeyError:
        reward = -0.5
        feedback = bad_message("You failed to provide the correct citation format, see details in the protocol.")
        return reward, feedback+received_reward_template.format(reward, max_reward)

    if len(sources) > 0 and len(contexts) == len(sources):
        reward = max_reward
        feedback = good_message("You successfully provided the correct citation format.")
        return reward, feedback+received_reward_template.format(reward, max_reward)

    reward = 0.0
    feedback = bad_message("You failed to provide citations or there is a mismatch count between sources and contexts.")
    return reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward proper number (or range) of citations returned
def contains_number_citations(validator: BaseValidatorNeuron, response: bt.Synapse, min_citations=1, max_citations=None) -> [float, str]:
    max_reward = 1.0
    try:
        citations = response.response['citations']
    except KeyError:
        # no citations provided and no placeholder available
        reward = -0.5
        feedback = bad_message(f"Failed to provide citations or the correct number of citations in the range of {min_citations} to {max_citations}.")
        return reward, feedback+received_reward_template.format(reward, max_reward)

    if (not min_citations or len(citations) >= min_citations) and (not max_citations or len(citations) <= max_citations):
        reward = max_reward
        feedback = good_message(f"You provided the correct number of citations in the range of {min_citations} to {max_citations}.")
        return reward, feedback+received_reward_template.format(reward, max_reward)
    # placeholder for citations, but none
    reward = -0.25
    feedback = bad_message(f"Failed to provide the correct number of citations in the range of {min_citations} to {max_citations}.")
    return reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward inclusion of correct source data in citations
def contains_correct_number_of_citation_sources(validator: BaseValidatorNeuron, response: bt.Synapse, selected_datas: List[dict]=[], selected_urls: List[str]=[]) -> [float, str]:
    max_reward = 1.5
    try:
        citations = response.response['citations']
        sources = set([c['source'] for c in citations])
    except KeyError:
        # no citations provided and no placeholder available
        # but not being evaluated by this criterion (see contains_number_citationS)
        reward = -0.5
        feedback = bad_message(f"Failed to provide the correct response formatting, see protocol definition.")
        return reward, feedback+received_reward_template.format(reward, max_reward)

    # must cite correct citation urls
    score = 0.0

    selected_sources = []
    for data in selected_datas:
        selected_sources.append(data['source'])

    for url in selected_urls:
        selected_sources.append(url)

    identified_sources = 0
    for source in sources:
        if source in selected_sources:
            identified_sources += 1
            score += 1.5/len(selected_sources)

    reward = score
    if score > 0.0:
        feedback = good_message(f"You correctly identified some or all of the correct citation sources ({identified_sources}/{len(selected_sources)} identified).")
    else:
        feedback = bad_message(f"You failed to correctly identify any of the correct citation sources.")
    return reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward seemingly valid response (per a simple LLM (hosted by the validator))
def correct_response_provided(validator: BaseValidatorNeuron, response: bt.Synapse, selected_datas: List[dict]) -> [float,str]:
    max_reward = 1.0
    try:
        citations = response.response['citations']
        cited_sources = [c['source'] for c in citations]
        cited_texts = [c['context'] for c in citations]
        prompt = response.prompt
        sources = [d['source'] for d in response.datas]
        texts = [d['context'] for d in response.datas]
        completion = response.response['response']
    except KeyError:
        # no citations provided and no placeholder available or maybe something wrong with the data sources
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct citation formatting - see protocal details.")
        return reward, feedback+received_reward_template.format(reward, max_reward)

    context = selected_datas[0]['context']
    input_text = f"Question: {prompt}\n\nAnswer: {completion}\n\nContext: {context}\n\n Is the answer to the question correct given the provided context, yes or no? Response:"
    yes_or_no = validator.validator_llm(input_text)

    if yes_or_no.strip().lower() == "yes":
        reward = 1.0
        feedback = good_message(f"You responded with a valid response from the provided context.")
        return reward, feedback+received_reward_template.format(reward, max_reward)

    reward = 0.0
    feedback = bad_message(f"You failed to respond with a valid response from the provided context.")
    return reward, feedback+received_reward_template.format(reward, max_reward)

# the core set of tests that form a set of criteria 
def gen_data_task_criteria(selected_datas: List[dict], n_expected_citations:int) -> List[Criterion]:
    return [
        Criterion(name=f"Returns expected citation source(s)", desc="", eval_fx=contains_correct_number_of_citation_sources, eval_args=[selected_datas]),
        Criterion(name=f"Returns valid response", desc="", eval_fx=correct_response_provided, eval_args=[selected_datas]),
    ]

# not really used by can be
default_criteria = [
    Criterion(name="Does not error", desc="", eval_fx=does_not_error), 
    Criterion(name="Does not take a long time", desc="", eval_fx=does_not_take_a_long_time),
]

# not really used by can be
# basic CITATION checks
basic_citations = [
    Criterion(name="Must have at least one citation", desc="", eval_fx=contains_number_citations, eval_args=[1, None]),
    Criterion(name="Must have correct citation format", desc="", eval_fx=correct_citation_format),
]
# not really used by can be
basic_no_citations = Criterion(name="Must not return any citations", desc="", eval_fx=contains_number_citations, eval_args=[0, 0])

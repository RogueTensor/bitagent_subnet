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
from typing import List
from common.base.validator import BaseValidatorNeuron
from bitagent.validator.criteria.utils import good_message, bad_message, received_reward_template

# CRITERION: reward proper citation dict format
def correct_citation_format(task, validator: BaseValidatorNeuron, response: bt.Synapse) -> [float, float, str]:
    max_reward = 1.0
    try:
        citations = response.response['citations']
        sources = [c['source'] for c in citations]
        contexts = [c['context'] for c in citations]
    except KeyError:
        reward = -0.5
        feedback = bad_message("You failed to provide the correct citation format, see details in the protocol.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if len(sources) > 0 and len(contexts) == len(sources):
        reward = max_reward
        feedback = good_message("You successfully provided the correct citation format.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    reward = 0.0
    feedback = bad_message("You failed to provide citations or there is a mismatch count between sources and contexts.")
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward proper number (or range) of citations returned
def contains_number_citations(task, validator: BaseValidatorNeuron, response: bt.Synapse, min_citations=1, max_citations=None) -> [float, float, str]:
    max_reward = 1.0
    try:
        citations = response.response['citations']
    except KeyError:
        # no citations provided and no placeholder available
        reward = -0.5
        feedback = bad_message(f"Failed to provide citations or the correct number of citations in the range of {min_citations} to {max_citations}.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if (not min_citations or len(citations) >= min_citations) and (not max_citations or len(citations) <= max_citations):
        reward = max_reward
        feedback = good_message(f"You provided the correct number of citations in the range of {min_citations} to {max_citations}.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    # placeholder for citations, but none
    reward = -0.25
    feedback = bad_message(f"Failed to provide the correct number of citations in the range of {min_citations} to {max_citations}.")
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward proper number (or range) of citations returned
def contains_some_text(task, validator: BaseValidatorNeuron, response: bt.Synapse) -> [float, float, str]:
    max_reward = 0.5
    try:
        response = response.response['response']
    except KeyError:
        # no citations provided and no placeholder available
        reward = -0.5
        feedback = bad_message(f"Failed to provide a valid response per protocol.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if len(response) > 6:
        reward = max_reward
        feedback = good_message(f"You provided some text, that's all we wanted.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    # placeholder for citations, but none
    reward = -0.25
    feedback = bad_message(f"Failed to provide some text.")
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward inclusion of correct source data in citations
def contains_correct_citation_source(task, validator: BaseValidatorNeuron, response: bt.Synapse) -> [float, float, str]:
    max_reward = 1.5
    try:
        citations = response.response['citations']
        sources = set([c['source'] for c in citations])
    except KeyError:
        # no citations provided and no placeholder available
        # but not being evaluated by this criterion (see contains_number_citationS)
        reward = -0.5
        feedback = bad_message(f"Failed to provide the correct response formatting, see protocol definition.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    score = 0.0
    for source in sources:
        if source == task.citation_sources_should_contain:
            score=max_reward
            break

    reward = score
    if score > 0.0:
        feedback = good_message(f"You correctly identified the source.")
    else:
        feedback = bad_message(f"You failed to correctly identify the correct citation source.") + f"Expected: {task.citation_sources_should_contain}, Received: {sources}"
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward inclusion of correct source data in citations
def contains_correct_number_of_citation_sources(task, validator: BaseValidatorNeuron, response: bt.Synapse, selected_datas: List[dict]=[], selected_urls: List[str]=[]) -> [float, float, str]:
    max_reward = 1.5
    try:
        citations = response.response['citations']
        sources = set([c['source'] for c in citations])
    except KeyError:
        # no citations provided and no placeholder available
        # but not being evaluated by this criterion (see contains_number_citationS)
        reward = -0.5
        feedback = bad_message(f"Failed to provide the correct response formatting, see protocol definition.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

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
        feedback += f"\nYou should have found the following sources: {selected_sources}."
        feedback += f"\nYou provided the following sources: {sources}."
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward valid response - super simple check
def correct_response_provided_simple(task, validator: BaseValidatorNeuron, response: bt.Synapse) -> [float, float, str]:
    max_reward = 1.0
    try:
        response = response.response['response']
    except KeyError:
        # no citations provided and no placeholder available or maybe something wrong with the data sources
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct citation formatting - see protocal details.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if task.response_should_contain in response:
        reward = max_reward
        feedback = good_message(f"You responded with a valid response from the provided context.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    reward = 0.0
    feedback = bad_message(f"You failed to respond with a valid response from the provided context.")
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward seemingly valid response (per a simple LLM (hosted by the validator))
def correct_response_provided(task, validator: BaseValidatorNeuron, response: bt.Synapse, selected_datas: List[dict]) -> [float, float, str]:
    max_reward = 1.0
    try:
        citations = response.response['citations']
        cited_sources = [c['source'] for c in citations]
        cited_texts = [c['context'] for c in citations]
        prompt = task.synapse.prompt
        sources = [d['source'] for d in response.datas]
        texts = [d['context'] for d in response.datas]
        completion = response.response['response']
    except KeyError:
        # no citations provided and no placeholder available or maybe something wrong with the data sources
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct citation formatting - see protocal details.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    context = selected_datas[0]['context']
    input_text = f"Question: {prompt}\n\nAnswer: {completion}\n\nContext: {context}\n\nIs the answer to the question a correct respsonse given the provided context, yes or no? Response:"
    yes_or_no = validator.validator_llm(input_text)

    # miner is trying something fishy
    if validator.validator_llm(completion).strip().lower() == "yes":
        reward = -1.0
        feedback = bad_message(f"You failed to respond with a valid response from the provided context.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if yes_or_no.strip().lower() == "yes":
        reward = max_reward
        feedback = good_message(f"You responded with a valid response from the provided context.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    elif yes_or_no.strip().lower() == "no":
        reward = 0.0
        feedback = bad_message(f"You failed to respond with a valid response from the provided context.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    else:
        reward = 0.5
        feedback = bad_message(f"You failed to respond with a comparable response.", color="yellow")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

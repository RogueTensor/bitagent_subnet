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
from sentence_transformers import SentenceTransformer, util
import torch
import bittensor as bt
from typing import List
from difflib import SequenceMatcher 
from common.base.validator import BaseValidatorNeuron
from langchain_text_splitters import CharacterTextSplitter
from bitagent.task_api.criteria.utils import good_message, bad_message, received_reward_template

# CRITERION: reward proper citation dict format
def correct_citation_format(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict={}) -> [float, float, str]:
    max_reward = 0.5
    try:
        citations = synapse.response['citations']
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
def contains_number_citations(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict={}, min_citations=1, max_citations=None) -> [float, float, str]:
    max_reward = 0.75
    try:
        citations = synapse.response['citations']
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
def contains_some_text(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response:dict = {}) -> [float, float, str]:
    max_reward = 0.25
    try:
        resp = synapse.response['response']
    except KeyError:
        # no citations provided and no placeholder available
        reward = -0.5
        feedback = bad_message(f"Failed to provide a valid response per protocol.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if len(resp) > 6:
        reward = max_reward
        feedback = good_message(f"You provided some text, that's all we wanted.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    # placeholder for citations, but none
    reward = -0.25
    feedback = bad_message(f"Failed to provide some text.")
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward inclusion of correct source data in citations
def contains_correct_citation_source(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict = {}) -> [float, float, str]:
    max_reward = 1.5
    try:
        citations = synapse.response['citations']
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
        feedback = bad_message(f"You failed to correctly identify the correct citation source.")
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward inclusion of correct source data in citations
def contains_correct_number_of_citation_sources(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict={}, selected_datas: List[dict]=[], selected_urls: List[str]=[]) -> [float, float, str]:
    max_reward = 1.5
    try:
        citations = synapse.response['citations']
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
            score += 1.5 / len(selected_sources)
        else:
            # 50% penalty for wrong source
            score -= 0.75 / len(selected_sources)

    reward = max(-max_reward, score)
    if score > 0.0:
        feedback = good_message(f"You submitted {len(sources)} sources and correctly identified some or all of the correct citation sources ({identified_sources}/{len(selected_sources)} identified).")
    else:
        feedback = bad_message(f"You failed to correctly identify any of the correct citation sources.")
        feedback += f"\nYou provided the following sources: {sources}."
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward valid response - super simple check
def correct_response_provided_simple(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict={}) -> [float, float, str]:
    max_reward = 1.0
    try:
        resp = synapse.response['response']
    except KeyError:
        # no citations provided and no placeholder available or maybe something wrong with the data sources
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct citation formatting - see protocal details.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if task.response_should_contain in resp:
        reward = max_reward
        feedback = good_message(f"You responded with a valid response from the provided context.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    reward = 0.0
    feedback = bad_message(f"You failed to respond with a valid response from the provided context.")
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward for being different than the prompt and context
def ensure_unique_response(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict, selected_datas: List[dict], response_gen: str) -> [float, float, str]:
    max_reward = 1.0
    try:
        prompt = task.synapse.messages[0].content
        completion = synapse.response['response']
    except KeyError:
        # no citations provided and no placeholder available or maybe something wrong with the data sources
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - see protocal details.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    context = selected_datas[0]['context']
    
    
    def _sliding_window_score(completion, context) -> List[float]:
        length = torch.tensor([len(c) for c in completion.split("\n")]).max()
        text_splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=length,
            chunk_overlap=length-1, 
            length_function=len,
            is_separator_regex=False,
        )
        context_windows = [split.page_content for split in text_splitter.create_documents([context])]
        split_completion = completion.split("\n")
        context_and_completions_embeddings = validator.sentence_transformer.encode(context_windows+ split_completion, show_progress_bar=False)
        context_embeddings = context_and_completions_embeddings[:len(context_windows)]
        completions_embeddings = context_and_completions_embeddings[len(context_windows):]
        
        all_scores = []
        for c_embedding in completions_embeddings:
            all_scores.append(
                util.pytorch_cos_sim(c_embedding, context_embeddings)[0].tolist()
            )
        return all_scores
    
    max_cos_score = torch.tensor(_sliding_window_score(completion, context)).max()

    match_prompt = SequenceMatcher(None, completion, prompt).ratio()
    match_context = SequenceMatcher(None, completion, context).ratio()
    match_prompt_gen = SequenceMatcher(None, response_gen, prompt).ratio()
    match_context_gen = SequenceMatcher(None, response_gen, context).ratio()

    max_response_cos_score = torch.tensor(_sliding_window_score(response_gen, context)).max()
    
    if max_response_cos_score > 0.90 or match_prompt_gen > 0.90 or match_context_gen > 0.90:
        reward = max_reward = 10**-6
        feedback = good_message(f"We couldn't generate a unique enough response from the context. No points awarded or lost.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if (match_context > 0.8 and match_context < 1.05*match_context_gen) or (match_prompt > 0.9 and match_prompt < 1.05*match_prompt_gen):
        reward = max_reward = 10**-6
        feedback = good_message(f"We couldn't generate a unique enough response from the context. No points awarded or lost.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
        
    # check relevance of response to prompt and context
    if completion in context or max_cos_score > 0.90:
        reward = -1.0
        feedback = bad_message(f"You failed to respond with a distinct response from the provided context.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if match_context > 0.80 or match_prompt > 0.90:
        reward = -1.0
        feedback = bad_message(f"You failed to respond with a comparably distinct response from the provided context.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    else:
        reward = max_reward
        feedback = good_message(f"You responded with a novel response compared to the context.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward for being relevant to the question and context
def relevant_to_provided_content(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict, selected_datas: List[dict]) -> [float, float, str]:
    max_reward = 1.0
    try:
        prompt = task.synapse.messages[0].content
        completion = synapse.response['response']
    except KeyError:
        # no citations provided and no placeholder available or maybe something wrong with the data sources
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - see protocal details.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    context = selected_datas[0]['context']
    
    rel_prompt = validator.measure_relevance_of_texts(completion, prompt)
    rel_context = validator.measure_relevance_of_texts(completion, context)

    # check relevance of response to prompt and context
    if rel_context < 0.55 or rel_prompt < 0.55:
        reward = -1.0
        feedback = bad_message(f"You failed to respond with a relevant response given the provided context.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    else:
        reward = max_reward
        feedback = good_message(f"You responded with a relevant response compared to the context.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward valid response
def correct_response_provided(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict, selected_datas: List[dict], response_gen: str) -> [float, float, str]:
    max_reward = 1.0
    try:
        prompt = task.synapse.messages[0].content
        completion = synapse.response['response']
    except KeyError:
        # no citations provided and no placeholder available or maybe something wrong with the data sources
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct citation formatting - see protocal details.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    context = selected_datas[0]['context']

    score_gen = validator.measure_relevance_of_texts(completion, response_gen)

    if score_gen < 0.60:
        reward = 0.0
        feedback = bad_message(f"You failed to provide a valid response given the provided question and context.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    else:
        reward = max_reward
        feedback = good_message(f"You responded with a valid response.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

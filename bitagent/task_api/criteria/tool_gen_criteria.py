# The MIT License (MIT)
# Copyright © 2024 RogueTensor
# Copyright © 2024 TheIntern

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

import json
import difflib
import bittensor as bt
from typing import List
from common.base.validator import BaseValidatorNeuron
from bitagent.task_api.criteria.utils import good_message, bad_message, received_reward_template



def correct_tool_gen(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response:dict, expected_tool: dict) -> [float, float, str]:
    max_reward = 2.0
    try:
        resp = synapse.response['response']
        try:
            miner_tool = json.loads(resp)
        except Exception as e:
            reward = -0.5
            feedback = bad_message(f"You failed to respond with valid json. Please format the response like so: {{\"name\": \"tool_name\", \"description\": \"tool_description\", \"arguments\": {{\"arg1\": {{\"required\": true, \"type\": \"str\", \"description\": \"arg1_description\"}}, \"arg2\": {{\"required\": false, \"type\": \"int\", \"description\": \"arg2_description\"}}}}}}")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    except KeyError:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for a list of messages.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    
    # need to check if the name is similar, if there is similar tool arguments, if the tool argument descriptions are similar, if the types are the same
    score = 0

    # check if the tool names are similar
    score += difflib.SequenceMatcher(None, expected_tool['name'], miner_tool['name']).ratio() * 30
    
    description_similarity = validator.measure_relevance_of_texts(expected_tool['description'], miner_tool['description'])
    score += description_similarity * 20  # Assign 20% for description similarity

    # Score for argument matching
    expected_args = expected_tool['arguments']
    miner_args = miner_tool['arguments']
    total_arg_comparisons = 0
    match_count = 0

    for exp_arg, exp_details in expected_args.items():
        # Find the closest match in miner_tool arguments
        possible_matches = difflib.get_close_matches(exp_arg, miner_args.keys(), n=1, cutoff=0.6)
        if possible_matches:
            gen_arg = possible_matches[0]
            gen_details = miner_args[gen_arg]
            total_arg_comparisons += 3  # For required, type, and description comparisons

            # Check for required match
            if gen_details.get('required', False) == exp_details['required']:
                match_count += 1
            # Check type similarity
            if gen_details.get('type', '') == exp_details['type']:
                match_count += 1
            # Check description similarity
            arg_description_similarity = validator.measure_relevance_of_texts(exp_details['description'], gen_details.get('description', ''))
            if arg_description_similarity > 0.5:  # Threshold for considering the descriptions similar
                match_count += 1

    if total_arg_comparisons > 0:
        score += (match_count / total_arg_comparisons) * 50  # Assign 50% for argument matching

    reward = max_reward * (score / 100)
    
    
    
    
    
    if reward == max_reward:
        feedback = good_message(f"You responded with the correct answer.")
    elif reward > max_reward*0.75:
        feedback = good_message(f"You responded with the correct answer but with a few errors.", color="yellow")
    elif reward > max_reward*0.5:
        feedback = bad_message(f"You responded with the correct answer but with some errors.", color="yellow")
    elif reward > max_reward*0.25:
        feedback = bad_message(f"You responded with the correct answer but with many errors.")
    else:
        feedback = bad_message(f"You failed to respond with the correct answer.")
            
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)



def find_first_tool_call(dicts):
    for d in dicts:
        if d.get('role') == 'tool call':
            return d
    return None  # Return None if no dictionary matches

def find_assistant_after_tool_call(dicts):
    found_tool_call = False
    for d in dicts:
        if not found_tool_call:
            if d.get('role') == 'tool call':
                found_tool_call = True
        elif d.get('role') == 'assistant':
            return d
    return None  # If no matching dictionary is found after the 'tool call'


def correct_dataset_tool_gen(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response:dict) -> [float, float, str]:
    max_reward = 3
    reward = 3
    try:
        resp = synapse.response['response']
        try:
            miner_tool = json.loads(resp)
            if not isinstance(miner_tool, dict):
                raise Exception("Miner did not return a dictionary.")
        except Exception as e:
            reward = -0.5
            feedback = bad_message(f"You failed to respond with valid json. Please format the response like so: {{\"name\": \"tool_name\", \"description\": \"tool_description\", \"arguments\": {{\"arg1\": {{\"required\": true, \"type\": \"str\", \"description\": \"arg1_description\"}}, \"arg2\": {{\"required\": false, \"type\": \"int\", \"description\": \"arg2_description\"}}}}}}")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    except KeyError:
        reward = -0.5
        feedback = bad_message(f"You failed to provide any response.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    feedback = good_message(f"You responded with the correct answer.")
    if not 'name' in miner_tool.keys():
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for a function with a 'name' key.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    if not 'description' in miner_tool.keys():
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for a function with a 'description' key.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    if not 'arguments' in miner_tool.keys():
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for a function with a 'arguments' key.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    if not isinstance(miner_tool['arguments'], dict):
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for a function with a 'arguments' key that is a dictionary.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
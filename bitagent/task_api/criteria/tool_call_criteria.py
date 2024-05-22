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
import re
import ast
import json
import difflib
import bittensor as bt
from typing import List
from common.base.validator import BaseValidatorNeuron
from bitagent.task_api.criteria.utils import good_message, bad_message, received_reward_template
from bitagent.schemas.conversation import Conversation
from bitagent.task_api.helpers.convo_parsing import find_assistant_after_tool_call, find_first_tool_call

def json_quote_fix(s):
    p = re.compile('(?<!\\\\)\'')
    s = p.sub('\"', s)
    return s

def correct_tool_use_and_response(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response:dict, expected_convo: Conversation) -> [float, float, str]:
    max_reward = 1.0
    try:
        resp = synapse.response['response']
        try:
            miner_convo = Conversation.from_list(json.loads(resp))
        except Exception as e:
            reward = -0.5
            feedback = bad_message(f"You failed to provide the correct response formatting - looking for a list of messages. Like so [{{\"role\": \"user\", \"content\": \"message\"}}, {{\"role\": \"assistant\", \"content\": \"message\"}}, {{\"role\": \"tool call\", \"content\": \"message\"}}]")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    except KeyError:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for a dict w/ a response key")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    
    if not any([msg.role == 'tool call' for msg in expected_convo.messages]):
        if any([msg.role == 'tool call' for msg in miner_convo.messages]):
            reward = -0.5
            feedback = bad_message(f"You failed to provide the correct response formatting. Was not looking for any tool calls")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
   
    try:
        expected_tool_call = ast.literal_eval(find_first_tool_call(expected_convo).content)
    except Exception as e:
        raise Exception(f"Couldnt load expected tool call. {e}")
    
    try: 
        miner_tool_call = [msg for msg in miner_convo.messages if msg.role == 'tool call'][0].content
    except:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for a tool call that is JSON formatted.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    # Compare arguments
    num_expected_keys = len(expected_tool_call['arguments'].keys()) + 1 # extra 1 is for the name
    num_gotten_keys = 0
    num_gotten_values = 0
    for miner_key, miner_value in miner_tool_call['arguments'].items():
        if miner_key in expected_tool_call['arguments']:
            num_gotten_keys += 1
            
            if difflib.SequenceMatcher(None, str(miner_value), str(expected_tool_call['arguments'][miner_key])).ratio() > 0.75:
                num_gotten_values += 1
    if difflib.SequenceMatcher(None, str(miner_tool_call['name']), str(expected_tool_call['name'])).ratio() > 0.75:
        num_gotten_values += 1
    correct_tool_percentage = (num_gotten_values+num_gotten_keys)/(2*num_expected_keys)
    try:
        expected_assistant = find_assistant_after_tool_call(expected_convo)['content']
    except Exception as e:
        bt.logging.error(f"Failed to find assistant response in expected response. {e}")
    correct_assistant_percentage = 0
    try:
        miner_assistant = find_assistant_after_tool_call(miner_convo)['content']
        sim = validator.measure_relevance_of_texts(expected_assistant, miner_assistant)
        if sim>0.90:
            correct_assistant_percentage = 1
        elif sim>0.75:
            correct_assistant_percentage = 0.75
        elif sim>0.50:
            correct_assistant_percentage = 0.25
    except Exception as e:
        bt.logging.error(f"Failed to find assistant response in miner messages. {e}")
    
    reward = 0.5 * max_reward * correct_tool_percentage + 0.5 * max_reward * correct_assistant_percentage
    
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

def correct_dataset_tool_call_response(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response:dict) -> [float, float, str]:
    max_reward = 3
    reward = 3
    try:
        resp = synapse.response['response']
        try:
            miner_convo = Conversation.from_list(json.loads(resp))
        except Exception as e:
            reward = -0.5
            feedback = bad_message(f"You failed to provide the correct response formatting - looking for a list of messages. Like so [{{\"role\": \"user\", \"content\": \"message\"}}, {{\"role\": \"assistant\", \"content\": \"message\"}}, {{\"role\": \"tool call\", \"content\": \"message\"}}]")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    except KeyError:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for a list of messages.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    try:
        tool_call = find_first_tool_call(miner_convo).content
        if isinstance(tool_call, str):
            tool_call = json.loads(tool_call)
        
        if not 'name' in tool_call or not 'arguments' in tool_call:
            raise Exception("Tool call is not formatted correctly")
    except Exception as e:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for a tool call that is JSON formatted. Error: {e}")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    
    feedback = good_message(f"You responded with the correct answer.")
    
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)


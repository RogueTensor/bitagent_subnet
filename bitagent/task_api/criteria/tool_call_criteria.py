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

from bitagent.task_api.helpers.tool_parsing import validate_tool_call
from bitagent.task_api.criteria.utils import good_message, bad_message, received_reward_template
from bitagent.task_api.helpers.convo_parsing import find_assistant_after_tool_call, find_first_tool_call, find_last_assistant
from bitagent.schemas.chat import messages_from_list

def json_quote_fix(s):
    p = re.compile('(?<!\\\\)\'')
    s = p.sub('\"', s)
    return s

def correct_irrelevant_tool_call(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response:dict, expected_convo: List[dict]) -> [float, float, str]:
    max_reward = 3.0
    reward = 3.0
    expected_convo = messages_from_list(expected_convo)
    
    try:
        resp = synapse.response['response']
        try:
            miner_convo = messages_from_list(json.loads(resp))
        except Exception as e:
            reward = -0.5
            if any([msg.role == 'tool call' for msg in expected_convo]):
                feedback = bad_message(f"You failed to provide the correct response formatting - looking for a list of messages. Like so [{{\"role\": \"tool call\", \"content\": \"message\"}}, {{\"role\": \"assistant\", \"content\": \"message\"}}]")
            else:
                feedback = bad_message(f"You failed to provide the correct response formatting - looking for a list of messages. Like so [{{\"role\": \"assistant\", \"content\": \"message\"}}]")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    except KeyError:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for a dict w/ a response key")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    
    tool_call = find_first_tool_call(miner_convo)
    tool_call_content = tool_call.content
    if isinstance(tool_call_content, str):
        tool_call_content = json.loads(tool_call_content)
    
    if tool_call_content:
        reward = -0.5
        feedback = bad_message(f"You provided a tool call, but one was not expected.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    feedback = good_message(f"You responded with the correct answer.")
    
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

def correct_tool_use_and_response(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response:dict, expected_convo: List[dict]) -> [float, float, str]:
    max_reward = 3.0
    expected_convo = messages_from_list(expected_convo)
    
    if not synapse.response['response']:
        reward = -0.5
        feedback = bad_message(f"Your response was empty, please return something.")
        return reward, max_reward, feedback + received_reward_template.format(reward, max_reward)
    
    try:
        resp = synapse.response['response']
        try:
            miner_convo = messages_from_list(json.loads(resp))
        except Exception as e:
            reward = -0.5
            if any([msg.role == 'tool call' for msg in expected_convo]):
                feedback = bad_message(f"You failed to provide the correct response formatting - looking for a list of messages. Like so [{{\"role\": \"tool call\", \"content\": \"message\"}}, {{\"role\": \"assistant\", \"content\": \"message\"}}]")
            else:
                feedback = bad_message(f"You failed to provide the correct response formatting - looking for a list of messages. Like so [{{\"role\": \"assistant\", \"content\": \"message\"}}]")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    except KeyError:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for a dict w/ a response key")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    expect_tool_call = True

    if not any([msg.role == 'tool call' for msg in expected_convo]):
        expect_tool_call = False
        if any([msg.role == 'tool call' for msg in miner_convo]):
            reward   = -0.5
            feedback = bad_message(f"You failed to provide the correct response formatting. Was not looking for any tool calls")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    
    # Check to ensure that it goes `tool call` then `assistant`
    if any([msg.role == 'tool call' for msg in expected_convo]):
        if len(miner_convo) != 2:
            reward = -0.5
            feedback = bad_message(f"You failed to provide the correct response formatting - looking for ONLY an assistant response and a tool call")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
        if miner_convo[0].role != 'tool call' or miner_convo[1].role != 'assistant':
            reward = -0.5
            feedback = bad_message(f"You failed to provide the correct response formatting - looking for an assistant response before a tool call")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    else:
        if len(miner_convo) != 1:
            reward = -0.5
            feedback = bad_message(f"You failed to provide the correct response formatting - looking for ONLY an assistant response")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    

    try:
        if isinstance(find_first_tool_call(expected_convo).content, str):
            expected_tool_call = ast.literal_eval(find_first_tool_call(expected_convo).content)
        else:
            expected_tool_call = find_first_tool_call(expected_convo).content
    except Exception as e:
        feedback = good_message(f"We failed to load the expected tool, so you get full points!")
        reward = max_reward
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    
    try: 
        miner_tool_call = [msg for msg in miner_convo if msg.role == 'tool call'][0].content
        if isinstance(miner_tool_call, str):
            # miner_tool_call = ast.literal_eval(miner_tool_call)
            miner_tool_call = json.loads(miner_tool_call)
    except:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for a tool call that can be converted into a dictionary") 
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    try:
        for tool in task.synapse.tools:
            if tool.name == miner_tool_call['name']:
                if not validate_tool_call(tool, miner_tool_call):
                    raise Exception("Miner failed to return a valid tool call.")    
    except:
        reward = -0.5
        feedback = bad_message(f"You failed to provide a valid tool call.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    
    # Compare arguments
    num_expected_keys = 2 * len(expected_tool_call['arguments'].keys())
    num_gotten_keys = 0
    num_gotten_values = 0
    for miner_key, miner_value in miner_tool_call['arguments'].items():
        if miner_key in expected_tool_call['arguments']:
            num_gotten_keys += 1
            
            if difflib.SequenceMatcher(None, str(miner_value), str(expected_tool_call['arguments'][miner_key])).ratio() > 0.75:
                num_gotten_values += 1
    if str(miner_tool_call['name']) != str(expected_tool_call['name']):
        correct_tool_percentage = 0
    else:
        correct_tool_percentage = (num_gotten_values+num_gotten_keys)/(num_expected_keys)
    
    
    try:
        if expect_tool_call:
            expected_assistant = find_assistant_after_tool_call(expected_convo)['content']
        else:
            expected_assistant = find_last_assistant(expected_convo)['content']
    except Exception as e:
        bt.logging.error(f"Failed to find assistant response in expected response.")
        reward = max_reward
        feedback = good_message(f"We experienced an error and do not want to punish you.") 
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    correct_assistant_percentage = 0
    
    try:
        if expect_tool_call:
            miner_assistant = find_assistant_after_tool_call(miner_convo)['content']
        else:
            miner_assistant = find_last_assistant(miner_convo)['content']
        sim = validator.measure_relevance_of_texts(expected_assistant, miner_assistant)
        if sim>0.80:
            correct_assistant_percentage = 1
        elif sim>0.50:
            correct_assistant_percentage = 0.75
        elif sim>0.35:
            correct_assistant_percentage = 0.25
    except Exception as e:
        bt.logging.error(f"Failed to find assistant response in miner messages. {e}")
        feedback = bad_message(f"Your response errored out the tool call criteria: {e}\n This is likely due to an incorrect assistant response given.")
        reward = -0.5
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    
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
            miner_convo = messages_from_list(json.loads(resp))
        except Exception as e:
            reward = -0.5
            feedback = bad_message(f"Your response was not json loadable. Error: {e}")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    except KeyError:
        reward = -0.5
        feedback = bad_message(f"You failed to provide a response that could be correctly grabbed.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    
    if len(miner_convo) != 2:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for ONLY an assistant response and a tool call")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    if miner_convo[0].role != 'tool call' or miner_convo[1].role != 'assistant':
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for an assistant response before a tool call")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    
    try:
        tool_call = find_first_tool_call(miner_convo).content
        if isinstance(tool_call, str):
            try: 
                tool_call = json.loads(tool_call)
            except:
                raise Exception("Tool call was a string, but not json loadable. You likely didn't return a dictionary as the tool calls content.")
        if not 'name' in tool_call or not 'arguments' in tool_call:
            raise Exception("Tool call is not formatted correctly")
    except Exception as e:
        reward = -0.5
        feedback = bad_message(f"You failed to correctly response - error: {e}")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    
    feedback = good_message(f"You responded with the correct answer.")
    
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)


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
import ast
import bittensor as bt
from typing import Tuple
from common.base.validator import BaseValidatorNeuron
from bitagent.criteria.utils import good_message, bad_message, received_reward_template

# just checking if the function can be parsed by ast
def correct_tool_call_function_format(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict) -> Tuple[float, float, str]:
    max_reward = 1.0
    reward = 1.0

    try:
        ast.parse(synapse.response)
    except Exception as e:
        reward = -1.0
        feedback = bad_message(f"Your response was not in the correct format - {e}")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    
    feedback = good_message(f"Your response was in the correct format.")
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

def extract_function_name_and_params(response: str):
    if response == "":
        return "", [], {}

    node = ast.parse(response , mode="eval")
    # handle situation where the function name has a dot in it
    try:    
        # handle a dot in the function name
        function_name = node.body.func.value.id + "." + node.body.func.attr
    except:
        # no dot in teh function name
        function_name = node.body.func.id   
    param_names = [kw.arg for kw in node.body.keywords]
    if param_names: 
        param_values = [ast.literal_eval(kw.value) for kw in node.body.keywords]
    else:
        param_values = []

    param_values_dict = {}
    for i,param_name in enumerate(param_names):
        param_values_dict[param_name] = param_values[i]

    return function_name, param_names, param_values_dict

# just checking if the function name is correct
def correct_tool_call_function_name(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict, expected_response: dict) -> Tuple[float, float, str]:
    max_reward = 3.0
    reward = 3.0    

    function_name, _, _ = extract_function_name_and_params(synapse.response)
    expected_function_name = expected_response['name']

    if function_name.strip() == expected_function_name.strip():
        feedback = good_message(f"Your function name matches the expected function name.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    else:
        reward = -0.5
        feedback = bad_message(f"Your function name does not match the expected function name.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

# comparing just the argument names
# looking for required arguments and that they are present
def correct_tool_argument_names(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict, expected_response: dict) -> Tuple[float, float, str]:
    max_reward = 3.0
    reward = 0.0        

    _, function_args, _ = extract_function_name_and_params(synapse.response)
    expected_args = expected_response['arguments'].keys()

    if len(expected_args) == 0 and len(function_args) == 0:
        reward = max_reward
        feedback = good_message("Function has no arguments, good job")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if "is_ground_truth" in expected_response.keys():
        required_args = [arg for arg in expected_args if expected_response['arguments'][arg] != [""]]
    else:
        expected_tool = [tool for tool in synapse.tools if tool.name == expected_response['name']][0]
        required_args = [k for k in expected_tool.arguments.keys() if 'required' in expected_tool.arguments[k].keys() and expected_tool.arguments[k]['required']]

    feedback = "" 

    for arg in required_args:
        if arg in function_args:
            # excessive args will be penalized
            reward += max_reward/max(len(function_args),len(expected_args))
            feedback += good_message(f"Your function has the required argument: {arg}") + "\n"
        else:
            reward -= max_reward/len(required_args)
            feedback += bad_message(f"Your function is missing the required argument: {arg}") + "\n"

    return reward, max_reward, feedback[:-1]+received_reward_template.format(reward, max_reward)

def correct_tool_argument_values(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict, expected_response: dict) -> Tuple[float, float, str]:
    max_reward = 3.0
    reward = 0.0        

    feedback = "" 

    # MINER response
    _, function_args, function_values = extract_function_name_and_params(synapse.response)
    expected_args = expected_response['arguments'].keys()

    # no args
    if len(expected_args) == 0 and len(function_args) == 0:
        reward = max_reward
        feedback = good_message("Function has no arguments, good job")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if "is_ground_truth" in expected_response.keys():
        required_args = [arg for arg in expected_args if expected_response['arguments'][arg] != [""]]
    else:
        expected_tool = [tool for tool in synapse.tools if tool.name == expected_response['name']][0]
        required_args = [k for k in expected_tool.arguments.keys() if 'required' in expected_tool.arguments[k].keys() and expected_tool.arguments[k]['required']]

    for arg in required_args:
        if arg in function_args:
            correct_values = expected_response['arguments'][arg]
            if "is_ground_truth" in expected_response.keys() and function_values[arg] in correct_values:
                reward += max_reward/max(len(function_args),len(expected_args))
                feedback += good_message(f"Your function has the required value for argument: {arg}") + "\n"
            elif function_values[arg] == correct_values:
                # TODO need to compare dict with dict and list with list - need some json dumps of non primitive values
                reward += max_reward/max(len(function_args),len(expected_args))
                feedback += good_message(f"Your function has the required value for argument: {arg}") + "\n"
            else:
                reward -= max_reward/len(required_args)
                feedback += bad_message(f"Your function has the incorrect value for argument: {arg}") + "\n"
        else:
            reward -= max_reward/len(required_args)
            feedback += bad_message(f"Your function is missing the required argument: {arg}") + "\n"

    return reward, max_reward, feedback[:-1]+received_reward_template.format(reward, max_reward)

def correct_irrelevant_tool_call(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict) -> Tuple[float, float, str]:
    max_reward = 3.0
    reward = 3.0
    
    if synapse.response.strip() != "":
        reward = -0.5
        feedback = bad_message(f"Your response (`{synapse.response}`) was not empty, expected an empty response to be returned.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    feedback = good_message(f"You responded with the expected response.")
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
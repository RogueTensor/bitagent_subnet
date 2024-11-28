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
from bitagent.criteria.utils import good_message, bad_message, received_reward_template


# just checking if the function can be parsed by ast
def correct_tool_call_function_format(task, validator, synapse: bt.Synapse) -> Tuple[float, float, str]:
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

    # Walk through the AST to extract the function name
    class FunctionNameExtractor(ast.NodeVisitor):
        def __init__(self):
            self.function_name = None

        def visit_Call(self, node):
            # Check if the node is a function call
            if isinstance(node.func, ast.Attribute):  # Handles dot notation (e.g., module.function)
                parts = []
                current = node.func
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                # Join the parts in reverse to get the full function name
                self.function_name = '.'.join(reversed(parts))
            elif isinstance(node.func, ast.Name):  # Handles simple function names (e.g., functionName)
                self.function_name = node.func.id
            # No need to visit further
            self.generic_visit(node)

    extractor = FunctionNameExtractor()
    extractor.visit(node)
    function_name = extractor.function_name

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
def correct_tool_call_function_name(task, validator, synapse: bt.Synapse, expected_response: dict) -> Tuple[float, float, str]:
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
def correct_tool_argument_names(task, validator, synapse: bt.Synapse, expected_response: dict) -> Tuple[float, float, str]:
    max_reward = 3.0
    reward = 0.0        

    function_name, function_args, _ = extract_function_name_and_params(synapse.response)
    expected_args = expected_response['arguments'].keys()

    if len(expected_args) == 0 and len(function_args) == 0 and function_name != "":
        reward = max_reward
        feedback = good_message("Function has no arguments, good job")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    # they returned arguments but the function expects none
    if len(expected_args) == 0:
        reward = 0
        feedback = bad_message(f"Function expects no arguments, but you returned arguments: {function_args}")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if "is_ground_truth" in expected_response.keys():
        required_args = [arg for arg in expected_args if expected_response['arguments'][arg] != [""]]
    else:
        expected_tool = [tool for tool in task.synapse.tools if tool.name == expected_response['name']][0]
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

def correct_tool_argument_values(task, validator, synapse: bt.Synapse, expected_response: dict) -> Tuple[float, float, str]:
    max_reward = 3.0
    reward = 0.0        

    feedback = "" 

    # MINER response
    function_name, function_args, function_values = extract_function_name_and_params(synapse.response)
    expected_args = expected_response['arguments'].keys()

    # no args
    if len(expected_args) == 0 and len(function_args) == 0 and function_name != "":
        reward = max_reward
        feedback = good_message("Function has no arguments, good job")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if "is_ground_truth" in expected_response.keys():
        required_args = [arg for arg in expected_args if expected_response['arguments'][arg] != [""]]
    else:
        expected_tool = [tool for tool in task.synapse.tools if tool.name == expected_response['name']][0]
        required_args = [k for k in expected_tool.arguments.keys() if 'required' in expected_tool.arguments[k].keys() and expected_tool.arguments[k]['required']]

    for arg in required_args:
        if arg in function_args:
            correct_values = expected_response['arguments'][arg]
            if "is_ground_truth" in expected_response.keys() and function_values[arg] in correct_values:
                reward += max_reward/max(len(function_args),len(expected_args))
                feedback += good_message(f"Your function has the required value for argument: {arg}") + "\n"
            elif function_values[arg] == correct_values:
                reward += max_reward/max(len(function_args),len(expected_args))
                feedback += good_message(f"Your function has the required value for argument: {arg}") + "\n"
            else:
                reward -= 0 # max_reward/len(required_args)
                feedback += bad_message(f"Your function has the incorrect value for argument: {arg}") + "\n"
        else:
            reward -= max_reward/len(required_args)
            feedback += bad_message(f"Your function is missing the required argument: {arg}") + "\n"

    return reward, max_reward, feedback[:-1]+received_reward_template.format(reward, max_reward)

def correct_irrelevant_tool_call(task, validator, synapse: bt.Synapse) -> Tuple[float, float, str]:
    max_reward = 3.0
    reward = 3.0
    
    if synapse.response.strip() != "":
        reward = -0.5
        feedback = bad_message(f"Your response (`{synapse.response}`) was not empty, expected an empty response to be returned.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    feedback = good_message(f"You responded with the expected response.")
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)


# Examples:
synapse_response1 = 'calculate_gpa(grades=["A", "B", "A", "C"], credit_hours=[3, 4, 3, 2])'
synapse_response2 = 'calculate_gpa(credit_hours=[3, 4, 3, 2], grades=["A", "B", "A", "C"])'
expected_response = {'name': 'calculate_gpa', 'arguments': {'grades': ['A', 'B', 'A', 'C'], 'credit_hours': [3, 4, 3, 2]}}

import unittest
from typing import List

class MockSynapse:
    from bitagent.schemas.tool import Tool
    response: str
    tools: List[Tool] = [Tool(name="calculate_gpa", description="Calculate the GPA of a student", arguments={"grades": {"type": "list", "required": True}, "credit_hours": {"type": "list", "required": True}})]

    def __init__(self, response: str):
        self.response = response

class MockTask:
    synapse: MockSynapse

    def __init__(self, synapse: MockSynapse):
        self.synapse = synapse

class TestToolCallCriteria(unittest.TestCase):

    def setUp(self):
        self.validator = ""

    def test_correct_tool_call_function_format(self):
        # Test valid function format
        synapse = MockSynapse(response="calculate_gpa(grades=['A'], credit_hours=[3])")
        task = MockTask(synapse=synapse)
        reward, max_reward, feedback = correct_tool_call_function_format(task, self.validator, synapse)
        self.assertEqual(reward, 1.0)
        self.assertEqual(max_reward, 1.0)
        self.assertTrue("was in the correct format" in feedback.lower())

        # Test invalid function format
        synapse = MockSynapse(response="invalid(function syntax")
        task = MockTask(synapse=synapse)
        reward, max_reward, feedback = correct_tool_call_function_format(task, self.validator, synapse)
        self.assertEqual(reward, -1.0)
        self.assertEqual(max_reward, 1.0)
        self.assertTrue("not in the correct format" in feedback.lower())

        # Test json response
        synapse = MockSynapse(response='{"name": "calculate_gpa", "arguments": {"grades": ["A"], "credit_hours": [3]}}')
        task = MockTask(synapse=synapse)
        reward, max_reward, feedback = correct_tool_call_function_format(task, self.validator, synapse)
        self.assertEqual(reward, 1.0)
        self.assertEqual(max_reward, 1.0)
        self.assertTrue("was in the correct format" in feedback.lower())

    def test_extract_function_name_and_params(self):
        # Test basic function extraction
        response = "calculate_gpa(grades=['A'], credit_hours=[3])"
        name, params, values = extract_function_name_and_params(response)
        self.assertEqual(name, "calculate_gpa")
        self.assertEqual(params, ["grades", "credit_hours"])
        self.assertEqual(values, {"grades": ["A"], "credit_hours": [3]})

        # Test empty response
        name, params, values = extract_function_name_and_params("")
        self.assertEqual(name, "")
        self.assertEqual(params, [])
        self.assertEqual(values, {})

        # Test function with dot notation
        response = "math.sqrt(value=16)"
        name, params, values = extract_function_name_and_params(response)
        self.assertEqual(name, "math.sqrt")
        self.assertEqual(params, ["value"])
        self.assertEqual(values, {"value": 16})

        # Test function with dot notation and no value
        response = "math.sqrt()"
        name, params, values = extract_function_name_and_params(response)
        self.assertEqual(name, "math.sqrt")
        self.assertEqual(params, [])
        self.assertEqual(values, {})

    def test_correct_irrelevant_tool_call(self):
        # Test empty response (correct)
        synapse = MockSynapse(response="")
        task = MockTask(synapse=synapse)
        reward, max_reward, feedback = correct_irrelevant_tool_call(task, self.validator, synapse)
        self.assertEqual(reward, 3.0)
        self.assertEqual(max_reward, 3.0)
        self.assertTrue("expected response" in feedback.lower())

        # Test non-empty response (incorrect)
        synapse = MockSynapse(response="some_function()")
        task = MockTask(synapse=synapse)
        reward, max_reward, feedback = correct_irrelevant_tool_call(task, self.validator, synapse)
        self.assertEqual(reward, -0.5)
        self.assertEqual(max_reward, 3.0)
        self.assertTrue("not empty" in feedback.lower())

    def test_correct_tool_call_function_name(self):
        # Test correct function name
        synapse = MockSynapse(response="calculate_gpa(grades=['A'])")
        task = MockTask(synapse=synapse)
        expected = {"name": "calculate_gpa", "arguments": {"grades": ["A"]}}
        reward, max_reward, feedback = correct_tool_call_function_name(task, self.validator, synapse, expected)
        self.assertEqual(reward, 3.0)
        self.assertEqual(max_reward, 3.0)
        self.assertTrue("matches the expected function name" in feedback.lower())

        # Test incorrect function name
        synapse = MockSynapse(response="wrong_function(grades=['A'])")
        task = MockTask(synapse=synapse)
        reward, max_reward, feedback = correct_tool_call_function_name(task, self.validator, synapse, expected)
        self.assertEqual(reward, -0.5)
        self.assertEqual(max_reward, 3.0)
        self.assertTrue("not match" in feedback.lower())

    def test_correct_tool_argument_names(self):
        # Test no expected arguments
        synapse = MockSynapse(response="calculate_gpa()")
        task = MockTask(synapse=synapse)
        expected = {"name": "calculate_gpa", "arguments": {}}
        reward, max_reward, feedback = correct_tool_argument_names(task, self.validator, synapse, expected)
        self.assertEqual(reward, 3.0)
        self.assertEqual(max_reward, 3.0)
        self.assertTrue("no arguments, good job" in feedback.lower())

        # Test no expected arguments, but pass in arguments anyway
        synapse = MockSynapse(response="calculate_gpa(grades=['A'])")
        task = MockTask(synapse=synapse)
        expected = {"name": "calculate_gpa", "arguments": {}}
        reward, max_reward, feedback = correct_tool_argument_names(task, self.validator, synapse, expected)
        self.assertEqual(reward, 0.0)
        self.assertEqual(max_reward, 3.0)
        self.assertTrue("expects no arguments" in feedback.lower())

        # Test correct argument names
        synapse = MockSynapse(response="calculate_gpa(grades=['A'], credit_hours=[3])")
        task = MockTask(synapse=synapse)
        expected = {"name": "calculate_gpa", "arguments": {"grades": ["A"], "credit_hours": [3]}}
        reward, max_reward, feedback = correct_tool_argument_names(task, self.validator, synapse, expected)
        self.assertEqual(reward, 3.0)
        self.assertEqual(max_reward, 3.0)
        self.assertEqual(feedback.lower().count("has the required argument"), 2)

        # Test correct argument names plus an incorrect argument
        synapse = MockSynapse(response="calculate_gpa(grades=['A'], credit_hours=[3], extra_arg=1)")
        task = MockTask(synapse=synapse)
        expected = {"name": "calculate_gpa", "arguments": {"grades": ["A"], "credit_hours": [3]}}
        reward, max_reward, feedback = correct_tool_argument_names(task, self.validator, synapse, expected)
        self.assertEqual(reward, 2.0)
        self.assertEqual(max_reward, 3.0)
        self.assertEqual(feedback.lower().count("has the required argument"), 2)

        # Test correct argument names out of order
        synapse = MockSynapse(response="calculate_gpa(credit_hours=[3], grades=['A'])")
        task = MockTask(synapse=synapse)
        expected = {"name": "calculate_gpa", "arguments": {"grades": ["A"], "credit_hours": [3]}}
        reward, max_reward, feedback = correct_tool_argument_names(task, self.validator, synapse, expected)
        self.assertEqual(reward, 3.0)
        self.assertEqual(max_reward, 3.0)
        self.assertEqual(feedback.lower().count("has the required argument"), 2)

        # Test missing argument
        synapse = MockSynapse(response="calculate_gpa(grades=['A'])")
        task = MockTask(synapse=synapse)
        reward, max_reward, feedback = correct_tool_argument_names(task, self.validator, synapse, expected)
        self.assertEqual(reward, 0.0)
        self.assertEqual(max_reward, 3.0)
        self.assertEqual(feedback.lower().count("has the required argument"), 1)
        self.assertEqual(feedback.lower().count("missing the required argument"), 1)

    def test_correct_tool_argument_values(self):
        # Test correct argument values
        synapse = MockSynapse(response="calculate_gpa(grades=['A'], credit_hours=[3])")
        task = MockTask(synapse=synapse)
        expected = {"name": "calculate_gpa", "arguments": {"grades": ["A"], "credit_hours": [3]}}
        reward, max_reward, feedback = correct_tool_argument_values(task, self.validator, synapse, expected)
        self.assertEqual(reward, 3.0)
        self.assertEqual(max_reward, 3.0)
        self.assertEqual(feedback.lower().count("has the required value for argument"), 2)

        # Test correct argument values out of order
        synapse = MockSynapse(response="calculate_gpa(credit_hours=[3], grades=['A'])")
        task = MockTask(synapse=synapse)
        expected = {"name": "calculate_gpa", "arguments": {"grades": ["A"], "credit_hours": [3]}}
        reward, max_reward, feedback = correct_tool_argument_values(task, self.validator, synapse, expected)
        self.assertEqual(reward, 3.0)
        self.assertEqual(max_reward, 3.0)
        self.assertEqual(feedback.lower().count("has the required value for argument"), 2)

        # Test incorrect argument values
        synapse = MockSynapse(response="calculate_gpa(grades=['B'], credit_hours=[4])")
        task = MockTask(synapse=synapse)
        reward, max_reward, feedback = correct_tool_argument_values(task, self.validator, synapse, expected)
        self.assertEqual(reward, -3.0)
        self.assertEqual(max_reward, 3.0)
        self.assertEqual(feedback.lower().count("has the incorrect value for argument"), 2)

        # Test incorrect argument values out of order
        synapse = MockSynapse(response="calculate_gpa(credit_hours=[4], grades=['B'])")
        task = MockTask(synapse=synapse)
        reward, max_reward, feedback = correct_tool_argument_values(task, self.validator, synapse, expected)
        self.assertEqual(reward, -3.0)
        self.assertEqual(max_reward, 3.0)
        self.assertEqual(feedback.lower().count("has the incorrect value for argument"), 2)

        # Test incorrect value types
        synapse = MockSynapse(response="calculate_gpa(grades='A', credit_hours=3)")
        task = MockTask(synapse=synapse)
        reward, max_reward, feedback = correct_tool_argument_values(task, self.validator, synapse, expected)
        self.assertEqual(reward, -3.0)
        self.assertEqual(max_reward, 3.0)
        self.assertEqual(feedback.lower().count("has the incorrect value for argument"), 2)

if __name__ == '__main__':
    # Run all tests in this file
    # You can run this file directly with: python -m bitagent.criteria.tool_call_criteria
    # Or run all tests with: python -m pytest bitagent/criteria/tool_call_criteria.py
    unittest.main(verbosity=2)

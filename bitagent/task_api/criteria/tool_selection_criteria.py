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
import bittensor as bt
from typing import List
from common.base.validator import BaseValidatorNeuron
from bitagent.task_api.criteria.utils import good_message, bad_message, received_reward_template

# CRITERION: reward valid answer to question
def contains_correct_tool_selection_answer(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response:dict, expected_answer: List[str]) -> [float, float, str]:
    max_reward = 2.0
    try:
        resp = synapse.response['response']
        try:
            resp_json = json.loads(resp)
        except Exception as e:
            reward = -0.5
            feedback = bad_message(f"You failed to provide the correct response formatting - looking for a list of tool names that is JSON formatted.")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    except KeyError:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - looking for a list of tool names.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    lcs_perc = grade_lcs(expected_answer, resp_json)
    lcs_reward = lcs_perc * max_reward * 0.50
    
    edit_dist = edit_distance(expected_answer, resp_json)
    max_distance = max(len(expected_answer), len(resp_json))
    if max_distance == 0:
        edit_dist_reward = max_reward*0.5
    else:
        similarity_score = 1 - (edit_dist / max_distance)
        if similarity_score > 0.85:
            edit_dist_reward = max_reward*0.5
        elif similarity_score > 0.5:
            edit_dist_reward = max_reward*0.25
        elif similarity_score > 0.30:
            edit_dist_reward = max_reward*0.5*0.25
        else:
            edit_dist_reward = 0
            
    reward = lcs_reward + edit_dist_reward
    
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

def edit_distance(gt, user):
    # Create a matrix to store distances between sub-lists of gt and user lists
    # The size of the matrix is (len(gt) + 1) x (len(user) + 1)
    m, n = len(gt), len(user)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base case: converting an empty gt list to a user list of size 'j' (0 to n)
    for j in range(n + 1):
        dp[0][j] = j

    # Base case: converting a gt list of size 'i' to an empty user list (0 to m)
    for i in range(m + 1):
        dp[i][0] = i

    # Fill dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i - 1] == user[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation needed if items are the same
            else:
                # Minimum of deleting from gt, inserting into gt, or replacing in gt
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # The value at dp[m][n] is the edit distance between the entire gt and user lists
    return dp[m][n]

def grade_lcs(correct_sequence, user_response):
    """Function to grade the user response based on the longest common subsequence."""
    lcs_length = lcs(user_response, correct_sequence)
    total_possible = len(correct_sequence)
    score_percentage = lcs_length / total_possible
    return score_percentage

def lcs(a, b):
    """Helper function to calculate the longest common subsequence of two lists."""
    lengths = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    # build the lengths matrix
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the substring out from the matrix
    result = []
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            result.append(a[x - 1])
            x -= 1
            y -= 1
    return len(result)
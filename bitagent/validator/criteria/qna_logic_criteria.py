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

# CRITERION: reward valid answer to question
def contains_correct_numerical_logic_answer(task, validator: BaseValidatorNeuron, response: bt.Synapse, expected_answer: int) -> [float, float, str]:
    max_reward = 1.0
    try:
        response = response.response['response']
    except KeyError:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - see protocal details.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    numbers = [int(i) for i in response.split() if i.isdigit()]
    if str(expected_answer) in response and len(numbers) == 1:
        reward = max_reward
        feedback = good_message(f"You responded with a valid answer.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    elif str(expected_answer) in response:
        reward = max_reward
        reward = 0.0
        feedback = bad_message(f"You failed to only provide the correct answer.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    # curbing attempts at prompt injection that won't work anyway
    if len(response.split(" ")) > 15:
        reward = -1.0
        feedback = bad_message(f"You failed to respond with a valid answer type, too long.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    
    reward = 0.0
    feedback = bad_message(f"You failed to respond with the correct answer: {expected_answer}.")
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

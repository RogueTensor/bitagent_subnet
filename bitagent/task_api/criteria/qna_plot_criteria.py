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

import re
import bittensor as bt
from common.base.validator import BaseValidatorNeuron
from bitagent.task_api.criteria.utils import good_message, bad_message, received_reward_template
from datetime import datetime

# CRITERION: reward valid answer to question
def contains_correct_numerical_plot_answer(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response:dict, expected_answer: str) -> [float, float, str]:
    max_reward = 5.0
    try:
        resp = synapse.response['response']
    except KeyError:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct response formatting - see protocal details.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    #def extract_numbers_from_string(s):
    #    # Regular expression to match both integers and floats
    #    pattern = r'-?\d+\.?\d*'
    #    matches = re.findall(pattern, s)

    #    # Convert matched strings to float or int and store in a list
    #    numbers = [float(match) if '.' in match else int(match) for match in matches]
    #    return numbers

    def calculate_float_points(diff):
        if diff == 0:
            return 5
        elif diff <= 2:
            return 4.5
        elif diff <= 3:
            return 4.0
        elif diff <= 4:
            return 3.5
        elif diff <= 5:
            return 3.0
        elif diff <= 6:
            return 2.5
        elif diff <= 7:
            return 2.0
        elif diff <= 8:
            return 1.5
        elif diff <= 9:
            return 1.0
        else:
            return 0.0

    def calculate_date_points(diff):
        if diff == 0:
            return 5
        elif diff <= 2:
            return 4.5
        elif diff <= 5:
            return 4.0
        elif diff <= 10:
            return 3.5
        elif diff <= 15:
            return 3.0
        elif diff <= 20:
            return 2.5
        elif diff <= 25:
            return 2.0
        elif diff <= 30:
            return 1.5
        else:
            return 1.0

    # Determine if the answer is a number or a date
    is_date = False
    try:
        expected_answer = float(expected_answer)
    except ValueError:
        expected_answer = datetime.strptime(expected_answer, '%m/%d/%Y')
        is_date = True

    #if the expected answer is a date
    if is_date:
        try:
            #convert miner response to datetime
            response_date = datetime.strptime(resp, '%m/%d/%Y')
            reward = calculate_date_points(abs((response_date - expected_answer).days))
            feedback = good_message(f"You responded with a valid answer.")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
        except:
            #failed to convert to datetime
            reward = 0.0
            feedback = bad_message(f"You failed to respond with the correct answer. We were expecting a date in the format MM/DD/YYYY.")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    #if the expected answer is a float
    elif isinstance(expected_answer, float):
        try:
            reward = (calculate_float_points(abs(float(resp) - expected_answer)) / 5) * max_reward
            feedback = good_message(f"You responded with a valid answer.")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
        except:
            reward = 0.0
            feedback = bad_message(f"You failed to respond with the correct answer. We were expecting a stringified float (e.g. '3.14').")
            return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    #if we got here, they failed to get the correct answer
    reward = 0.0
    feedback = bad_message(f"You failed to respond with the correct answer.")
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)


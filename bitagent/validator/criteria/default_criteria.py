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
from common.base.validator import BaseValidatorNeuron
from bitagent.validator.criteria.utils import good_message, bad_message, received_reward_template

# CRITERION: successful call to miner
def does_not_error(task, validator: BaseValidatorNeuron, response: bt.Synapse) -> [float, float, str]:
    max_reward = 0.5
    a_status_code = response.axon.status_code
    d_status_code = response.dendrite.status_code
    reward = 0.0
    if a_status_code == 200 and d_status_code == 200:
        reward = 0.5
        feedback = good_message("You successfully responded to the request.")
    else:
        feedback = bad_message("You failed to respond correctly to the request.")
        if d_status_code == 408:
            feedback += "You timed out and will fail the remainder of the criteria."
        feedback += f" Status Code: {a_status_code}/{d_status_code}"
        
    return reward, max_reward, feedback + received_reward_template.format(reward, max_reward)

# CRITERION: reward speedy response
def does_not_take_a_long_time(task, validator: BaseValidatorNeuron, response: bt.Synapse) -> [float, float, str]:
    max_reward = 0.5
    process_time = response.dendrite.process_time
    if not process_time:
        feedback = f"You likely ran into an error processing this task and failed to respond appropriately."
        reward = 0
        return reward, max_reward, bad_message(feedback) + received_reward_template.format(reward,max_reward)

    feedback = f"You responded to the request in {process_time}."
    reward = 0.0
    if process_time <= task.timeout/3:
        reward = 0.50
        return reward, max_reward, good_message(feedback) + received_reward_template.format(reward,max_reward)
    if process_time <= task.timeout/2:
        reward = 0.25
        return reward, max_reward, good_message(feedback, color="yellow") + received_reward_template.format(reward,max_reward)
    if process_time <= task.timeout:
        reward = 0.10
        return reward, max_reward, bad_message(feedback, color="yellow") + received_reward_template.format(reward,max_reward)
    return reward, max_reward, bad_message(feedback) + received_reward_template.format(reward,max_reward)


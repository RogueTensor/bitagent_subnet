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
from bitagent.task_api.criteria.utils import good_message, bad_message, received_reward_template

# CRITERION: reward seemingly valid response based on length
def correct_assistant_response(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict, correct_response: str) -> [float, float, str]:
    max_reward = 3
    try:
        miner_response = synapse.response['response']
        if not miner_response or not isinstance(miner_response, str):
            reward = -0.5
            feedback = bad_message(f"You responded with {miner_response}. You must respond with a string.")
            return reward, max_reward, feedback + received_reward_template.format(reward, max_reward)
    except KeyError:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct data - see protocol details.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    response_similarity = validator.measure_relevance_of_texts(correct_response,miner_response)
    
    if response_similarity < 0.45:
        reward = 0
        feedback = bad_message(f"Your response was not at all similar to what it should have been.", color="red")
    elif response_similarity < 0.75:
        reward = 0.25*max_reward
        feedback = bad_message(f"Your response was barely similar to what it should have been.", color="yellow")
    elif response_similarity < 0.8:
        reward = 0.5*max_reward
        feedback = good_message(f"Your response was almost perfect.", color="green")
    elif response_similarity >= 0.8:
        reward = 1*max_reward
        feedback = good_message(f"Your response was perfect.", color="green")
    return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
        

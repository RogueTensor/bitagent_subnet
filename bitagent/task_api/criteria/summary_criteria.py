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
def shorter_summary_length(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict, summary: str, summary_gen: str) -> [float, float, str]:
    max_reward = 0.5
    try:
        prompt = task.synapse.prompt
        completion = synapse.response['response']
    except KeyError:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct data - see protocal details.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if len(completion) >= 0.8 * len(prompt):
        reward = 0.0
        feedback = bad_message(f"You failed to provide a short summarization from the provided text.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    else:
        reward = max_reward
        feedback = good_message(f"You responded with a valid summary length.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

# CRITERION: reward seemingly valid response
def correct_summary_provided(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict, summary: str, summary_gen: str) -> [float, float, str]:
    max_reward = 1.0
    try:
        prompt = task.synapse.prompt
        completion = synapse.response['response']
    except KeyError:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct data - see protocal details.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    score_gen = validator.measure_relevance_of_texts(completion, summary_gen)
    score_grd = validator.measure_relevance_of_texts(completion, summary)
    score_prmt = validator.measure_relevance_of_texts(completion, prompt)
    score_prmt_wo = validator.measure_relevance_of_texts(completion, prompt.replace("Summarize this and make sure to be concise: ", ""))

    if summary == summary_gen and score_gen < 0.5:
        reward = -0.5
        feedback = bad_message(f"You failed to provide a valid summary from the provided text.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    elif score_prmt >= 0.99 or score_prmt_wo >= 0.99:
        reward = -0.25
        feedback = bad_message(f"You failed to provide a valid summary from the provided text.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    elif score_gen > 0.95:
        reward = max_reward
        feedback = good_message(f"You responded with a valid summary.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    elif score_grd < 0.7 or score_grd > 0.95 or score_gen < 0.7 or score_prmt > 0.95 or score_prmt_wo > 0.95:
        reward = 0.0
        feedback = bad_message(f"You failed to provide a valid summary from the provided text.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    else:
        reward = max_reward * score_gen/0.75 * score_grd/0.75
        if reward > max_reward:
            reward = max_reward
        feedback = good_message(f"You responded with a valid summary.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

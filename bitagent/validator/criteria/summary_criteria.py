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

# CRITERION: reward seemingly valid response (per a simple LLM (hosted by the validator))
def correct_summary_provided(task, validator: BaseValidatorNeuron, response: bt.Synapse, summary: str) -> [float, float, str]:
    max_reward = 1.0
    try:
        prompt = task.synapse.prompt
        completion = response.response['response']
    except KeyError:
        reward = -0.5
        feedback = bad_message(f"You failed to provide the correct data - see protocal details.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    input_text = f"SummaryA: {summary}\n\nSummaryB: {completion}\n\n\nIs SummaryA similar to SummaryB? Only respond with yes or no, no other words:"
    yes_or_no = validator.validator_llm(input_text)

    # miner trying something fishy
    if validator.validator_llm(completion).strip().lower() == "yes":
        reward = -1.0
        feedback = bad_message(f"You failed to respond with a valid summary from the provided context.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

    if yes_or_no.strip().lower() == "yes":
        reward = max_reward
        feedback = good_message(f"You responded with a valid summary.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    elif yes_or_no.strip().lower() == "no":
        reward = 0.0
        feedback = bad_message(f"You failed to respond with a valid summary.")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)
    else:
        reward = 0.0
        feedback = bad_message(f"You failed to respond with a comparable summary.", color="yellow")
        return reward, max_reward, feedback+received_reward_template.format(reward, max_reward)

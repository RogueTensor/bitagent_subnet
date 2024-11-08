# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
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

import asyncio
import bittensor as bt
from bitagent.protocol import QueryTask
from common.utils.uids import get_alive_uids
from common.utils.uids import get_random_uids
from bitagent.tasks.task import get_random_task
from bitagent.validator.offline_task import offline_task
from bitagent.validator.reward import process_rewards_update_scores_and_send_feedback

async def forward(self, synapse: QueryTask=None) -> QueryTask:
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # complete this first so it's cached for both ONLINE and OFFLINE
    get_alive_uids(self)

    # ###########################################################
    # OFFLINE TASKING
    # ###########################################################
    # if we're close to the block check point for offline HF model check, then check all miners
    if not self.running_offline_mode and self.block % self.config.neuron.block_number_check_interval_for_offline_hf_model_check <= 30:
        self.running_offline_mode = True
        asyncio.create_task(offline_task(self))

    # ###########################################################
    # ONLINE TASKING
    # ###########################################################
    try:
        # check a random sample of miners in online mode
        miner_uids = get_random_uids(self, min(self.config.neuron.sample_size, self.metagraph.n.item()))
        task = get_random_task(self)
        task.mode = "online"

        # send the task to the miners
        responses = self.dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=task.synapse,
            deserialize=False,
            timeout=task.timeout,
        )

        await asyncio.create_task(process_rewards_update_scores_and_send_feedback(self, task=task, responses=responses, miner_uids=miner_uids))

    except Exception as e:
        bt.logging.debug(f"Error in forward: {e}")
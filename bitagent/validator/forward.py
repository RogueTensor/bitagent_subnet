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
from datetime import datetime
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
    # TODO Track competition end date in config
    if not self.running_offline_mode: # TODO and datetime.now() > datetime.strptime(COMPETITION_END_DATE, "%Y-%m-%d"):
        bt.logging.debug(f"OFFLINE: Starting offline mode")
        self.last_offline_run_block = self.block
        self.running_offline_mode = True
        asyncio.create_task(offline_task(self))
    elif self.running_offline_mode:
        bt.logging.debug(f"OFFLINE: Already running offline mode")
    # TODO else:
    #    bt.logging.debug(f"OFFLINE: Not running offline mode. Block number: {self.block}, block number check interval: {interval}")
    #    remainder = self.block % interval
    #    next_block = self.block + interval - remainder
    #    next_block_hours = (interval - remainder) * 12.0/60.0/60.0 # 12 seconds per block, converted to hours
    #    bt.logging.debug(f"OFFLINE: Blocks until next offline run is {interval - remainder}: {next_block}; roughly {next_block_hours} hours")

    # ###########################################################
    # ONLINE TASKING
    # ###########################################################
    try:
        bt.logging.debug(f"ONLINE: Starting online run")
        # check a random sample of miners in online mode
        bt.logging.debug(f"ONLINE: Getting random miner uids")
        miner_uids = get_random_uids(self, min(self.config.neuron.sample_size, self.metagraph.n.item()))
        bt.logging.debug(f"ONLINE: Getting random task")
        task = get_random_task(self)
        task.mode = "online"

        # send the task to the miners
        bt.logging.debug(f"ONLINE: Sending task to miners")
        responses = self.dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=task.synapse,
            deserialize=False,
            timeout=task.timeout,
        )

        bt.logging.debug(f"ONLINE: Evaluating responses")
        await asyncio.create_task(process_rewards_update_scores_and_send_feedback(self, task=task, responses=responses, miner_uids=miner_uids))
        bt.logging.debug(f"ONLINE: Evaluation complete")

    except Exception as e:
        bt.logging.debug(f"Error in forward: {e}")

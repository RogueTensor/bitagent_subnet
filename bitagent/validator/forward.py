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

import bittensor as bt

from bitagent.validator.reward import get_rewards
from common.utils.uids import get_random_uids

from bitagent.validator.tasks import get_random_task
from bitagent.protocol import QnAResult

async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    try:
        miner_uids = get_random_uids(self, k=min(self.config.neuron.sample_size, self.metagraph.n.item()))
    except Exception as e:
        bt.logging.warning(f"Trouble setting miner_uids: {e}")
        bt.logging.warning("Defaulting to 1 uid, k=1")
        miner_uids = get_random_uids(self, k=1)

    task = get_random_task(self)

    bt.logging.debug(f"Task prompt: {task.synapse.prompt[:2000]}")

    # The dendrite client queries the network.
    responses = self.dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        # Construct a query. 
        synapse=task.synapse,
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        deserialize=False,
        timeout=task.timeout,
    )

    # Log the results for monitoring purposes.
    #bt.logging.debug(f"Received responses: {responses}")

    # Adjust the scores based on responses from miners.
    # also gets results for feedback to the miners
    rewards, results = get_rewards(self, task=task, responses=responses, miner_uids=miner_uids)

    #bt.logging.info(f"Scored responses: {rewards}")

    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, miner_uids)
    
    # The dendrite client queries the network to send feedback to the miner
    for i,uid in enumerate(miner_uids):
        _ = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid]],
            # Construct a query. 
            synapse=QnAResult(results=results[i]),
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=False,
        )

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

from bitagent.validator.reward import process_rewards_update_scores_and_send_feedback
from bitagent.tasks.task import get_random_task
from common.utils.uids import get_random_uids

from bitagent.tasks.task import get_random_task
from bitagent.protocol import QueryTask

async def forward(self, synapse: QueryTask=None) -> QueryTask:
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """

    # TODO get all miner UIDs to compare against the TOP HF model
    miner_uids = get_random_uids(self, k=min(self.config.neuron.sample_size, self.metagraph.n.item()))
    task = get_random_task(self)

    #bt.logging.debug(f"Task prompt: {task.synapse.prompt[:2000]}")

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

    # Generated Request
    # When the task is generated, we need to evaluate the responses.
    # Adjust the scores based on responses from miners.
    # also gets results for feedback to the miners
    await asyncio.create_task(process_rewards_update_scores_and_send_feedback(self, task=task, responses=responses, miner_uids=miner_uids))
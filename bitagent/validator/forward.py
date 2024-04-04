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

import torch
import requests
import bittensor as bt

from bitagent.validator.reward import get_rewards
from common.utils.uids import get_random_uids

from bitagent.validator.tasks import get_random_task
from bitagent.protocol import QnATask, QnAResult

async def forward(self, synapse: QnATask=None) -> QnATask:
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Handles 3 cases:
     - Organic task coming in through the axon
     - Organic task coming in through the Task API
     - Generated task coming in through the Task API

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    try:
        # don't grab miner ids if they are passed in from the organic axon request
        if not synapse or len(synapse.miner_uids) == 0:
            miner_uids = get_random_uids(self, k=min(self.config.neuron.sample_size, self.metagraph.n.item()))
    except Exception as e:
        bt.logging.warning(f"Trouble setting miner_uids: {e}")
        bt.logging.warning("Defaulting to 1 uid, k=1")
        miner_uids = get_random_uids(self, k=1)

    # Organic Validator Axon Request
    # if a request comes into the validator through the axon handle it 
    # in this case, timeout can be provided to work with your application's needs
    # and miner_uids can be provided (optional) to specify which miners to send the task to if you want to be specific
    # any tasks that come in through the validator axon are organic tasks and not evaluated
    if synapse:
        task = None
        bt.logging.debug(f"Received organic task over axon: {synapse.prompt[:200]} ...")
        task_synapse = synapse
        # check for miner uids to use, otherwise will use random
        # only one miner response will be returned
        if len(synapse.miner_uids) > 0:
            bt.logging.debug("setting miner uids: ", synapse.miner_uids)
            miner_uids = torch.tensor(synapse.miner_uids)
        else:
            bt.logging.debug("using randomly selected miner uids")
        # check for timeout to use, otherwise will use default of 10.0 seconds
        if synapse.timeout:
            bt.logging.debug('setting timeout: ', synapse.timeout)
            task_timeout = synapse.timeout
        else:
            bt.logging.debug('setting timeout to default: 10.0') 
            task_timeout = 10.0

    # Task API Request (Organic or Generated)
    # otherwise we are looking to the Task API to grab organic tasks off the queue
    # or generated tasks if no organic tasks exist and evaluate it
    else: 
        organic_miner_uids, task = get_random_task()
        if task.task_type == "organic" and len(organic_miner_uids) > 0:
            bt.logging.debug('Received organic task with miner uids: ', organic_miner_uids)
            miner_uids = torch.tensor(organic_miner_uids)
        elif task.task_type == "organic":
            bt.logging.debug('Received organic task without miner uids')
        else:
            bt.logging.debug('Received generated task that will require evaluation')
        task_synapse = task.synapse
        task_timeout = task.timeout

    #bt.logging.debug(f"Task prompt: {task.synapse.prompt[:2000]}")

    # The dendrite client queries the network.
    # this is used for all 3 cases:
    # - Organic task coming in through the axon
    # - Organic task coming in through the Task API
    # - Generated task coming in through the Task API
    responses = self.dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        # Construct a query. 
        synapse=task_synapse,
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        deserialize=False,
        timeout=task_timeout,
    )

    # Organic Task API Request
    # if organic then return the response to the task API for the downstream application
    if task and task.task_type == "organic":
        for response in responses:
            organic_return_url = f"https://roguetensor.com/api/task_api/organic_response"
            headers = {"Content-Type": "application/json"}
            data = {
                "task_id": task.task_id,
                "response": response.response,
                "miner_uids": miner_uids.tolist()
            }
            requests.post(organic_return_url, headers=headers, json=data) #, verify=False)

    # Log the results for monitoring purposes.
    #bt.logging.debug(f"Received responses: {responses}")

    # Generated Request
    # When the task is generated, we need to evaluate the responses.
    # Adjust the scores based on responses from miners.
    # also gets results for feedback to the miners
    if not synapse and task and task.task_type != "organic":
        rewards, results = get_rewards(self, task=task, responses=responses, miner_uids=miner_uids)
        # we may need to skip some of the results b/c the API goes down
        # if we skip a result, the returned data is None
        # so if we see None, we'll want to remove the corresponding miner_uid
        # this way we don't score them unfairly
        offset=0
        temp_miner_uids = miner_uids.tolist() # work in list b/c easier
        for i in range(len(results)):
            result = results[i-offset]
            if not result:
                rewards.pop(i-offset)
                temp_miner_uids.pop(i-offset)
                results.pop(i-offset)
                offset += 1

        #bt.logging.info(f"Scored responses: {rewards}")

        miner_uids = torch.tensor(temp_miner_uids)
        # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
        self.update_scores(torch.FloatTensor(rewards).to(self.device), miner_uids)

        # For generated/evaluated tasks, we send the results back to the miner so they know how they did and why
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
                timeout=1 # quick b/c we are not awaiting a response
            )

    # Organic Validator Axon Request
    # If the request came in through the axon, return the response to the validator
    if synapse:
        # check for a valid response from the collection
        # only sending one back
        for response in responses:
            if response.response:
                if "response" in response.response.keys():
                    return response
        return responses[0]
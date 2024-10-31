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

import time
import asyncio
import numpy as np
import bittensor as bt

from bitagent.helpers.llms import llm, get_openai_llm
from common.utils.uids import get_random_uids
from bitagent.tasks.task import get_random_task
from bitagent.helpers.dockers import create_container, wait_for_container
from bitagent.protocol import QueryTask, GetHFModelName, GetHFRunModelName
from bitagent.validator.reward import process_rewards_update_scores_and_send_feedback

class QuickObj:
    pass

async def forward(self, synapse: QueryTask=None) -> QueryTask:
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """

    # if we're close to the block check point for offline HF model check, then check all miners
    if False: #self.block % self.config.neuron.block_number_check_interval_for_offline_hf_model_check <= 30:
        try:
            # get all miner UIDs to compare against the TOP HF model
            miner_uids = range(self.metagraph.n.item())
            # load and call model directly on validator hardware per miner
            for miner_uid in miner_uids:
                # Request the miner's HF model name to evaluate offline
                responses = self.dendrite.query(
                    axons=[self.metagraph.axons[miner_uid]],
                    synapse=GetHFModelName(),
                    deserialize=False,
                    timeout=5.0,
                )
                top_hf_model_name = responses[0].hf_model_name
                if top_hf_model_name is None or top_hf_model_name == "":
                    bt.logging.warning(f"Miner {miner_uid} returned empty HF model name")
                    self.offline_scores[miner_uid] = 0.0
                    continue

                # start the docker container named "offline-hf-model-check" (config)
                # load the miner's model directly on validator hardware
                bt.logging.debug(f"Starting container for miner {miner_uid} with HF model name: ", top_hf_model_name)
                container = create_container(self.config.validator_hf_container_name, top_hf_model_name, self.config.validator_hf_vllm_port)
                wait_for_container(get_openai_llm(self), top_hf_model_name)

                # get the result for 8000 random tasks
                all_scores = []
                for i in range(8000):
                    task = get_random_task(self)
                    task.mode = "offline"

                    # call miners LLM
                    t0 = time.time()
                    llm_response = llm(self, task.synapse.messages, task.synapse.tools, top_hf_model_name)
                    t1 = time.time()

                    response = QuickObj()
                    response.response = llm_response.strip()
                    response.dendrite = QuickObj()
                    response.dendrite.process_time = t1 - t0
                    response.dendrite.status_code = 200
                    response.axon = QuickObj()
                    response.axon.status_code = 200
                    response.axon.hotkey = self.metagraph.axons[miner_uid].hotkey

                    # evaluate, track score and add to wandb
                    scores = await asyncio.create_task(process_rewards_update_scores_and_send_feedback(self, task=task, responses=[response], miner_uids=[miner_uid], run_models=[top_hf_model_name]))
                    all_scores.append(scores[0])

                # After we're dong with that miner's runs, we can update the validator's state
                self.offline_scores[miner_uid] = np.mean(all_scores)
                self.offline_model_names[miner_uid] = top_hf_model_name

                # kill the docker container named "offline-hf-model-check"
                container.remove(force=True)

        except Exception as e:
            bt.logging.debug(f"Error in forward: {e}")
            #raise e
        finally:
            # kill the docker container named "offline-hf-model-check"
            if container:
                container.remove(force=True)

        # TODO send out the TOP HF model name to all miners after we've checked all of them
        # and have the miners store it in their state

    else:
        # otherwise, just check a random sample of miners in online mode
        try:
            miner_uids = get_random_uids(self, min(self.config.neuron.sample_size, self.metagraph.n.item()))
            task = get_random_task(self)
            task.mode = "online"

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

            # The dendrite client queries the network.
            run_models = self.dendrite.query(
                # Send the query to selected miner axons in the network.
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                # Construct a query. 
                synapse=GetHFRunModelName(),
                # All responses have the deserialize function called on them before returning.
                # You are encouraged to define your own deserialization function.
                deserialize=False,
                timeout=task.timeout,
            )

            await asyncio.create_task(process_rewards_update_scores_and_send_feedback(self, task=task, responses=responses, miner_uids=miner_uids, run_models=run_models))
        except Exception as e:
            bt.logging.debug(f"Error in forward: {e}")
            #raise e

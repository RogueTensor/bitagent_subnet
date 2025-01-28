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
import numpy as np
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
    # if all miners have been processed for this competition, then don't run offline mode
    self.update_competition_numbers()

    wandb_data = {
        "task_name": "offline_model_check",
        "task_mode": "offline",
        "validator_uid": self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address),
        "val_spec_version": self.spec_version,
        "highest_score_for_miners_with_this_validator": self.scores.max(),
        "median_score_for_miners_with_this_validator": np.median(self.scores),
        "highest_offline_score_for_miners_with_this_validator": self.offline_scores[self.competition_version].max(),
        "median_offline_score_for_miners_with_this_validator": np.median(self.offline_scores[self.competition_version]),
        "average_offline_score_for_miners_with_this_validator": np.mean(self.offline_scores[self.competition_version]),
        # "prior_highest_offline_score_for_miners_with_this_validator": self.offline_scores[self.previous_competition_version].max(),
        # "prior_median_offline_score_for_miners_with_this_validator": np.median(self.offline_scores[self.previous_competition_version]),
        # "prior_average_offline_score_for_miners_with_this_validator": np.mean(self.offline_scores[self.previous_competition_version]),
        "competition_version": self.competition_version,
    }

    if self.config.subtensor.network != "test":
        if len(self.miners_left_to_score) == 0:
            if self.offline_status != "complete":
                self.offline_status = "complete"
                wandb_data['offline_status'] = self.offline_status
                wandb_data['num_miners_left_to_score'] = len(self.miners_left_to_score)
                self.log_event(wandb_data)
                wandb_data.pop('offline_status')
                wandb_data.pop('num_miners_left_to_score')
            self.running_offline_mode = False
            #bt.logging.debug(f"OFFLINE: No miners left to score for competition {self.competition_version}")
            pass
        elif not self.running_offline_mode:
            bt.logging.debug(f"OFFLINE: Starting offline mode for competition {self.competition_version}")
            #bt.logging.debug(f"OFFLINE: Miners left to score: {self.miners_left_to_score}")
            self.running_offline_mode = True
            self.offline_status = "starting"
            wandb_data['offline_status'] = self.offline_status
            wandb_data['num_miners_left_to_score'] = len(self.miners_left_to_score)
            wandb_data['miners_left_to_score'] = self.miners_left_to_score
            self.log_event(wandb_data)
            wandb_data.pop('num_miners_left_to_score')
            wandb_data.pop('miners_left_to_score')
            wandb_data.pop('offline_status')
            asyncio.create_task(offline_task(self, wandb_data))
            self.running_offline_mode = False
        elif self.running_offline_mode:
            #bt.logging.debug(f"OFFLINE: Already running offline mode for competition {self.competition_version}")
            #bt.logging.debug(f"OFFLINE: Miners left to score: {self.miners_left_to_score}")
            if self.offline_status != "running":
                self.offline_status = "running"
                wandb_data['offline_status'] = self.offline_status
                wandb_data['num_miners_left_to_score'] = len(self.miners_left_to_score)
                self.log_event(wandb_data)
                wandb_data.pop('num_miners_left_to_score')
                wandb_data.pop('offline_status')
            pass
    else:
        bt.logging.debug("OFFLINE: Skipping offline for testnet")

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

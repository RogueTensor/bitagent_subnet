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

import os
import copy
import time
import asyncio
import threading
import numpy as np
import bittensor as bt
from datetime import datetime, timezone, date
from huggingface_hub import dataset_info
from scoring_utils import score_spreading
from common.utils.uids import get_alive_uids
from bitagent.datasources.tools import ToolDataset
from bitagent.validator.constants import DEPLOYED_DATE, COMPETITION_LENGTH_DAYS, TESTNET_COMPETITION_LENGTH_DAYS, COMPETITION_PREFIX, COMPETITION_PREVIOUS_PREFIX
from common.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
)
from typing import List
from traceback import print_exception

from common.base.neuron import BaseNeuron
from common.utils.uids import check_uid_availability

class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    def __init__(self, config=None):
        super().__init__(config=config)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")

        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.offline_scores = {}        
        self.offline_miners_scored = {}
        self.offline_model_names = {}
        self.running_offline_mode = False
        self.offline_status = None
        self.regrade_version = dataset_info("BitAgent/tool_shuffle_small").last_modified.strftime("%Y%m%d%H")
        self.update_competition_numbers()
        self.max_div = 0.0006
        self.min_div = 0.00015
        self.state_file_name = "ft_state.npz"
        # set random seed to be encoded regrade version based later
        self.seed = int(datetime.strptime(self.regrade_version, "%Y%m%d%H").timestamp())
        np.random.seed(self.seed)
        bt.logging.info(f"Startup regrade_version: {self.regrade_version}")

        # Init sync with the network. Updates the metagraph.
        if os.path.exists(self.config.neuron.full_path + f"/{self.state_file_name}"):
            # if we are booting up and have this file, then we'll want to load it
            # otherwise, if we save state, it will overwrite from the sync
            self.sync(save_state=False)
        else:
            # if no state file then we'll create one on init
            self.hotkeys=[]
            self.sync()
        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            #self.serve_axon()
            pass
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            self.axon.attach(
                forward_fn=self.forward_fn,
                blacklist_fn=self.blacklist_fn,
                priority_fn=self.priority_fn,
            )

            try:
                self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
                self.axon.start()
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines,return_exceptions=True)

    def tool_dataset_regen(self):

        mod_date = dataset_info("BitAgent/tool_shuffle_small").last_modified.strftime("%Y%m%d%H")
        
        if mod_date != self.regrade_version:
            bt.logging.info(f"Dataset Regeneration: Regrade version{self.regrade_version} has changed, updating to {mod_date}")
            self.tool_dataset = ToolDataset(False, self.seed)
            self.task_dataset = ToolDataset(True, self.seed)
            self.regrade_version = mod_date
            self.update_competition_numbers()
            bt.logging.debug("Data regenerated.")
        else:
            bt.logging.info(f"Dataset Regeneration: {self.regrade_version} is the same as check_date: {mod_date}, passing.")
            return
        
    def _thread_entrypoint(self):
        """This is the function that the background thread will run.
           It sets up an event loop and runs the async 'run' method."""
        asyncio.run(self.run())

    async def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()
        bt.logging.info(
            f"Running validator on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.block}")
        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                try:
                    bt.logging.info(f"step({self.step}) block({self.block})")
                except Exception as e:
                    bt.logging.error(f"Error logging step and block, likely socket issue, will update next round: {e}")
                    #if "Broken pipe" in str(e):
                    #    print("======= Exiting due to a broken pipe ========")
                    #    self.axon.stop()
                    #    self.should_exit = True
                    #    exit()

                if self.step % 500 == 0:
                    self.tool_dataset_regen()
                    

                # Run multiple forwards concurrently.
                await self.concurrent_forward()
                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                try:
                    self.sync()
                except Exception as e:
                    bt.logging.error(f"Error syncing metagraph during run loop: {e}")
                
                bt.logging.info(f"Validator still running... step: {self.step}, no new offline scoring activity")
                await asyncio.sleep(30)
                self.step += 1
        except Exception as e:
            bt.logging.error(f"Unexpected error during run: {e}")

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(
                print_exception(type(err), err, err.__traceback__)
            )

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            #self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread = threading.Thread(target=self._thread_entrypoint, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """
        
        if self.config.subtensor.network == "test":
            return # Don't set weights on testnet.
        # with temporary_logging_state(state):
        self.divisions = int(np.floor(self.block / 1000))
        current_odds = self.offline_scores[self.competition_version]
        current_odds[current_odds < 0] = 0

        bt.logging.info(f"Offline Scores: {current_odds}")
        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )
        # correct validator scores to be 0 
        for uid, hotkey in enumerate(self.hotkeys):
            if not check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit): 
                # if validator, set validators scores to 0
                self.scores[uid] = 0 
                #self.offline_scores[self.previous_competition_version][uid] = 0
                self.offline_scores[self.competition_version][uid] = 0
                self.offline_miners_scored[self.competition_version][self.regrade_version].append(uid)
                self.offline_model_names[self.competition_version][uid] = ""

        # always fit scores to weighted curve
        # to change random seed to be encoded regrade version based later
        np.random.seed(self.seed)
        weighted_scores = score_spreading(current_odds,self.divisions,self.min_div,self.max_div, kurtosis_factor=0.5, divisions=np.random.randint(2,7))
        bt.logging.debug(f"weighted_scores: {weighted_scores}")

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        norm = np.linalg.norm(weighted_scores, ord=1, axis=0, keepdims=True)
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)
        raw_weights = weighted_scores/norm

        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )

        # Set the weights on chain via our subtensor connection.

        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            bt.logging.info(f"set_weights on chain for version: {self.spec_version} successfully!")
        else:
            bt.logging.error(f"set_weights failed: {msg}")

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        try:
            self.metagraph.sync(subtensor=self.subtensor)
            self.last_block_sync = self.block

            # Check if the metagraph axon info has changed.
            if self.hotkeys == self.metagraph.hotkeys:
                bt.logging.debug("Metagraph hotkeys are the same, skipping resync")
                return

            bt.logging.info("Metagraph updated, resyncing hotkeys, dendrite pool and moving averages")  
            # Normalize all hotkeys that have been replaced, and zero out all hotkeys that are no longer available
            for uid, hotkey in enumerate(self.hotkeys):
                if hotkey != self.metagraph.hotkeys[uid]:
                    bt.logging.debug(f"RESYNC: hotkey changed for uid: {uid}")
                    self.scores[uid] = np.median(self.scores)
                    #self.offline_scores[self.previous_competition_version][uid] = 0
                    self.offline_scores[self.competition_version][uid] = 0
                    if uid in self.offline_miners_scored[self.competition_version][self.regrade_version]:
                        self.offline_miners_scored[self.competition_version][self.regrade_version].remove(uid)
                    self.offline_model_names[self.competition_version][uid] = ""
                    #self.offline_model_names[self.previous_competition_version][uid] = ""

            # Check to see if the metagraph has changed size.
            # If so, we need to add new hotkeys and moving averages.
            if len(self.hotkeys) < len(self.metagraph.hotkeys):
                # Update the size of the moving average scores.
                new_moving_average = np.zeros((self.metagraph.n))
                min_len = min(len(self.hotkeys), len(self.scores))
                new_moving_average[:min_len] = self.scores[:min_len]
                self.scores = new_moving_average

                # previous offline scores
                # new_moving_average = np.zeros((self.metagraph.n))
                # min_len = min(len(self.hotkeys), len(self.offline_scores[self.previous_competition_version]))
                # new_moving_average[:min_len] = self.offline_scores[self.previous_competition_version][:min_len]
                # self.offline_scores[self.previous_competition_version] = new_moving_average

                # current offline scores
                new_moving_average = np.zeros((self.metagraph.n))
                min_len = min(len(self.hotkeys), len(self.offline_scores[self.competition_version]))
                new_moving_average[:min_len] = self.offline_scores[self.competition_version][:min_len]
                self.offline_scores[self.competition_version] = new_moving_average

            # Update the hotkeys.
            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        except Exception as e:
            bt.logging.error(f"Could not resync with metagraph right now, will try later. Error: {e}")

    def update_offline_scores(self, rewards: np.ndarray, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""
        if np.isnan(rewards).any():
            #bt.logging.debug(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        scattered_rewards: np.ndarray = self.offline_scores[self.competition_version].copy()
        scattered_rewards[uids_array] = rewards

        bt.logging.debug(f"OFFLINE Scattered rewards: {rewards}")

        self.offline_scores[self.competition_version]: np.ndarray = scattered_rewards # type: ignore
        self.offline_miners_scored[self.competition_version][self.regrade_version].extend([int(x) for x in uids_array])
        bt.logging.debug(f"Updated moving avg OFFLINE scores for Competition {self.competition_version}: {self.offline_scores[self.competition_version]}")
        self.save_state()

    def update_scores(self, rewards: np.ndarray, uids: List[int], alpha=None):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""
        if np.isnan(rewards).any():
            #bt.logging.debug(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        scattered_rewards: np.ndarray = self.scores.copy()
        scattered_rewards[uids_array] = rewards

        bt.logging.debug(f"ONLINE Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        if not alpha:
            alpha: float = self.config.neuron.moving_average_alpha
        self.scores: np.ndarray = alpha * scattered_rewards + ( 1 - alpha) * self.scores
        bt.logging.debug(f"Updated moving avg ONLINE scores: {self.scores}")
    
    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.debug(f"Saving validator state - {self.state_file_name}.")

        # Save the state of the validator to file.
        try:
            np.savez(
                self.config.neuron.full_path + f"/{self.state_file_name}",
                step=self.step,
                scores=self.scores,
                offline_scores=self.offline_scores,
                regrade_version=self.regrade_version,
                offline_miners_scored=np.array(list(self.offline_miners_scored.items()), dtype=object),
                offline_model_names=self.offline_model_names,
                hotkeys=self.hotkeys,
                allow_pickle=True,
            )
        except Exception as e:
            bt.logging.error(f"OFFLINE: Error saving validator state: {e}")
        
    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")
        state = np.load(self.config.neuron.full_path + f"/{self.state_file_name}",allow_pickle=True)
        bt.logging.debug(f"OFFLINE: LOADING STATE: {state}")

        self.step = state["step"]
        if 'hotkeys' in state:
            self.hotkeys = list(state["hotkeys"])
        else:
            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        if 'scores' in state:
            loaded_scores = state["scores"]
            self.scores[:len(loaded_scores)] = loaded_scores

        if 'offline_scores' in state:
            loaded_offline_scores = state["offline_scores"]
            if isinstance(loaded_offline_scores, dict):
                self.offline_scores = loaded_offline_scores
            elif isinstance(loaded_offline_scores, np.ndarray):
                self.offline_scores = loaded_offline_scores.item()
            else:
                bt.logging.error(f"OFFLINE: loaded_offline_scores is not a dict or array, type: {type(loaded_offline_scores)}")

            # if self.offline_scores.get(self.previous_competition_version) is None:
            #     self.offline_scores[self.previous_competition_version] = np.zeros(self.metagraph.n, dtype=np.float32)
            #for uid in self.metagraph.uids:
            #    if uid not in self.offline_scores[self.previous_competition_version]:
            #        self.offline_scores[self.previous_competition_version][uid] = 0
        if 'offline_miners_scored' in state:
            loaded_offline_miners_scored = state["offline_miners_scored"]
            self.offline_miners_scored = dict(loaded_offline_miners_scored)

        if 'offline_model_names' in state:
            loaded_offline_model_names = state["offline_model_names"]
            if isinstance(loaded_offline_model_names, dict):
                self.offline_model_names = loaded_offline_model_names
            elif isinstance(loaded_offline_model_names, np.ndarray):
                self.offline_model_names = loaded_offline_model_names.item()
            else:
                bt.logging.error(f"OFFLINE: loaded_offline_model_names is not a dict or array, type: {type(loaded_offline_model_names)}")

    def update_competition_numbers(self):
        try:

            self.competition_version = f"{COMPETITION_PREFIX}-0"

            if self.offline_scores.get(self.competition_version) is None:
                self.offline_scores[self.competition_version] = np.zeros(self.metagraph.n, dtype=np.float32)

            # SETUP OFFLINE MINERS SCORED
            if self.offline_miners_scored.get(self.competition_version) is None:
                self.offline_miners_scored[self.competition_version] = {}

            if not isinstance(self.offline_miners_scored[self.competition_version], dict):
                self.offline_miners_scored[self.competition_version] = {}

            if self.offline_miners_scored[self.competition_version].get(self.regrade_version) is None:
                self.offline_miners_scored[self.competition_version][self.regrade_version] = []


            # SETUP OFFLINE MODEL NAMES
            if self.offline_model_names.get(self.competition_version) is None:
                self.offline_model_names[self.competition_version] = {}

            self.miners_left_to_score = []

            # if an offline_score is 0 (we should try again), we need to add the miner to the list of miners left to score
            # so clear out the offline_miners_scored for this competition, for those miners
            to_remove = []
            for uid in self.offline_miners_scored[self.competition_version][self.regrade_version]:
                if self.offline_scores[self.competition_version][uid] <= 0.01:
                    to_remove.append(uid)

            for uid in to_remove:
                self.offline_miners_scored[self.competition_version][self.regrade_version].remove(uid)


            # add all miners that are alive and not already scored to the list of miners left to score
            for uid in get_alive_uids(self):
                if uid not in [int(x) for x in self.offline_miners_scored[self.competition_version][self.regrade_version]]:
                    self.miners_left_to_score.append(int(uid))

            # if a regrade has been set for the comp, then reset the scores for the miners
            #bt.logging.debug(f"OFFLINE: regrade version: {self.regrade_version}")
            #bt.logging.debug(f"OFFLINE: regrade check - offline miners scored: {self.offline_miners_scored[self.competition_version][self.regrade_version]}")
            #bt.logging.debug(f"OFFLINE: regrade check - offline scores: {self.offline_scores[self.competition_version]}")
            for uid,score in enumerate(self.offline_scores[self.competition_version]):
                #bt.logging.debug(f"OFFLINE: regrade check for uid: {uid}")
                if score > 0.0 and uid not in [int(x) for x in self.offline_miners_scored[self.competition_version][self.regrade_version]]:
                    #bt.logging.debug(f"OFFLINE: resetting miner {uid}'s score for competition {self.competition_version} for regrade")
                    self.offline_scores[self.competition_version][uid] = 0.0
                #bt.logging.debug(f"OFFLINE: regrade check for uid done: {uid}")

        except Exception as e:
            bt.logging.error(f"Error updating competition numbers: {e}")
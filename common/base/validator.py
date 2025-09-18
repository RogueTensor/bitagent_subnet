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
from pathlib import Path
from datetime import datetime, timezone, date
from typing import List
from traceback import print_exception

from common.base.neuron import BaseNeuron
from common.utils.uids import check_uid_availability
from common.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
)

# === Emission hotkeys:
EMISSION_HOTKEYS: List[str] = [
    "5HgA4szVBVRXLuUGMbrDgewUe51iMS8ngCtDzp14Jaa6adPD",
    "5H6CrN8Bnmx9fHtZpfEdBSRjf8ig7naj1GyQ6BndnJ3sGhzw",
    "5CJzrhAWPaQa9FMxaR5dLNAJiNSrs1ECcb2p6J5trTmkGLKN",
    "5Cqz2G6EoYUB567NcUekPEA9BX5EdQHp6JRgUT2wosw52gYp",
    "5GGbcsZqYuJZRYxATCvECkcKs1pSJUhJg7oE2VUfJkKsi7G8",
    "5G1hemwjyDVsYxSz43UyP5ZiHwAnehMxeyAy4KYjpEGVtxMB"
]

class BaseValidatorNeuron(BaseNeuron):
    """
    Minimal validator focused on setting weights every ~100 seconds.
    All state-file and BFCL dataset plumbing removed.
    """

    neuron_type: str = "ValidatorNeuron"

    def __init__(self, config=None):
        super().__init__(config=config)

        self.state_file_name      = "ft_state.npz"  
        self.dataset_history_len  = 0           
        self.dataset_scores       = {}       
        self.competition_version  = "noop-0"        
        self.regrade_version      = datetime.now(timezone.utc).strftime("%Y%m%d%H") 

        # Networking
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Runtime buffers
        self.scores                = np.zeros(self.metagraph.n, dtype=np.float32)
        self.offline_scores        = {self.competition_version: np.zeros(self.metagraph.n, dtype=np.float32)} 
        self.offline_miners_scored = {self.competition_version: {self.regrade_version: []}}                  
        self.offline_model_names   = {self.competition_version: {}}                            
        self.hotkeys               = []
        self.running_offline_mode  = False
        self.offline_status        = None

        # Deterministic RNG (not critical, but harmless to keep)
        self.seed = 11123421
        np.random.seed(self.seed)
        bt.logging.info(f"Startup regrade_version: {self.regrade_version}")


        # Axon serve 
        if not self.config.neuron.axon_off:
            pass
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Threading helpers
        self.should_exit = False
        self.is_running  = False
        self.thread      = None
        self.lock        = asyncio.Lock()



    def serve_axon(self):
        """Serve axon to enable external connections (unchanged, but likely unused)."""
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
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass


    def find_bounty_uids(self) -> List[int]:
        """Return UIDs for all emission hotkeys currently in the metagraph."""
        uids: List[int] = []
        for uid, hotkey in enumerate(self.metagraph.hotkeys):
            if hotkey in EMISSION_HOTKEYS:
                uids.append(uid)
        return uids

    async def run(self):
        """
        Minimal loop: sync → set_weights → sleep(100).
        """
        # Initial sync to fetch metagraph etc.
        self.sync()
        bt.logging.info(
            f"Running validator on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        bt.logging.info(f"Validator starting at block: {self.block}")

        try:
            while True:
                try:
                    bt.logging.info(f"step({self.step}) block({self.block})")
                except Exception as e:
                    bt.logging.error(f"Error logging step/block: {e}")

                # Try to set weights
                try:
                    self.set_weights()
                except Exception as e:
                    bt.logging.error(f"Error during set_weights: {e}")

                # Resync metagraph 
                try:
                    self.sync()
                except Exception as e:
                    bt.logging.error(f"Error syncing metagraph during run loop: {e}")

                if self.should_exit:
                    break

                bt.logging.info(f"Sleeping 100s before next weight update...")
                await asyncio.sleep(100)
                self.step += 1

        except KeyboardInterrupt:
            try:
                self.axon.stop()
            except Exception:
                pass
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()
        except Exception as err:
            bt.logging.error("Error during validation: %s", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))

    def _thread_entrypoint(self):
        asyncio.run(self.run())

    def run_in_background_thread(self):
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self._thread_entrypoint, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
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
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    # --- Core: evenly split all emissions across EMISSION_HOTKEYS ---
    def set_weights(self):
        """
        Even-split emissions across EMISSION_HOTKEYS present in the metagraph.
        Attempts to set every call; no testnet writes.
        """
        if self.config.subtensor.network == "test":
            bt.logging.info("Testnet detected; skipping set_weights.")
            return

        # Refresh bounty/emission UIDs from current metagraph
        bounty_uids = self.find_bounty_uids()
        if not bounty_uids:
            bt.logging.error("No EMISSION_HOTKEYS found in metagraph; cannot set weights.")
            return

        # Build weight vector (zeros except our target UIDs)
        n = self.metagraph.n
        weighted_scores = np.zeros(n, dtype=np.float32)
        share = 1.0 / float(len(bounty_uids))
        for uid in bounty_uids:
            # Safety: ignore out-of-range or unavailable UIDs
            if 0 <= uid < n and check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit):
                weighted_scores[uid] = share

        if weighted_scores.sum() <= 0:
            bt.logging.error("All target UIDs filtered out or unavailable; aborting set_weights.")
            return

        # Normalize (should already sum to 1.0, but guard anyway)
        norm = np.linalg.norm(weighted_scores, ord=1)
        if norm == 0 or np.isnan(norm):
            bt.logging.error("Normalization failed; aborting set_weights.")
            return
        raw_weights = weighted_scores / norm

        # Pipe through chain limits
        processed_uids, processed_weights = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        uint_uids, uint_weights = convert_weights_and_uids_for_emit(
            uids=processed_uids, weights=processed_weights
        )

        # Emit
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
            bt.logging.info(
                f"set_weights success (version {self.spec_version}); "
                f"even split across {len(bounty_uids)} hotkeys: {bounty_uids}"
            )
        else:
            bt.logging.error(f"set_weights failed: {msg}")

 
    def resync_metagraph(self):
        """Kept for compatibility; now only updates hotkeys & sizes without touching offline state."""
        bt.logging.info("resync_metagraph()")
        try:
            previous_metagraph = copy.deepcopy(self.metagraph)
            self.metagraph.sync(subtensor=self.subtensor)
            self.last_block_sync = self.block

            if self.hotkeys == self.metagraph.hotkeys:
                bt.logging.debug("Metagraph hotkeys are the same, skipping resync")
                return

            bt.logging.info("Metagraph updated; resizing score arrays.")
            # Resize runtime arrays to match new n
            if len(self.scores) != self.metagraph.n:
                new_scores = np.zeros((self.metagraph.n), dtype=np.float32)
                min_len = min(len(self.scores), len(new_scores))
                new_scores[:min_len] = self.scores[:min_len]
                self.scores = new_scores

            if self.competition_version not in self.offline_scores:
                self.offline_scores[self.competition_version] = np.zeros(self.metagraph.n, dtype=np.float32)
            elif len(self.offline_scores[self.competition_version]) != self.metagraph.n:
                new_os = np.zeros(self.metagraph.n, dtype=np.float32)
                min_len = min(len(self.offline_scores[self.competition_version]), len(new_os))
                new_os[:min_len] = self.offline_scores[self.competition_version][:min_len]
                self.offline_scores[self.competition_version] = new_os

            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        except Exception as e:
            bt.logging.error(f"Could not resync with metagraph; will try later. Error: {e}")

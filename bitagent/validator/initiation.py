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

import os
import glob
import wandb
import shutil
import bittensor as bt
from datetime import datetime

# setup validator with wandb
# clear out the old wandb dirs if possible
def initiate_validator(self):
    # clear out wandb runs that may be left over
    def clear_wandb():
        try:
            bt.logging.debug("Clearing out stale wandb runs")
            if os.path.exists("wandb"):
                bt.logging.debug("Found the wandb dir...")
                bt.logging.debug("Contents before delete: ", os.listdir("wandb"))
                for f in glob.glob("wandb/run-*"):
                    shutil.rmtree(f)
                bt.logging.debug("Contents after delete: ", os.listdir("wandb"))
            else:
                bt.logging.debug('Could not find wandb dir, moving on')

        except Exception as e:
            bt.logging.debug(f"Error while trying to remove stale wandb runs: {e}")
    clear_wandb()
    self.clear_wandb = clear_wandb

    # wandb setup
    def init_wandb(miner_uid=None, validator_uid=None):
        try:
            run_name = f"{miner_uid}_{validator_uid}"
            tags = ["validator_miner_runs", 
                    f"validator_uid_{self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)}",
                    f"net_uid{self.config.netuid}",
                    datetime.now().strftime('%Y_%m_%d'),
            ]
            if self.config.subtensor.network == "test" or self.config.netuid == 76: # testnet wandb
                # bt.logging.debug("Initializing wandb for testnet")
                return wandb.init(name=run_name, anonymous="allow", entity="bitagenttao", project="bitagent-testnet-logging", config=self.config, tags=tags)
            elif self.config.subtensor.network == "finney" or self.config.netuid == 20: # mainnet wandb
                # bt.logging.debug("Initializing wandb for mainnet")
                return wandb.init(name=run_name, anonymous="allow", entity="bitagenttao", project="bitagent-logging", config=self.config, tags=tags)
            else: # unknown network, not initializing wandb
                # bt.logging.debug("Not initializing wandb, unknown network")
                return None
        except Exception as e:
            #bt.logging.error("Wandb could not be initialized: ", e)
            return None
    self.init_wandb = init_wandb

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


import copy
import wandb
import bittensor as bt
from datetime import datetime
from bitagent.task_api.initiation import initiate_validator as initiate_validator_local

# setup validator with wandb
# clear out the old wandb dirs if possible
def initiate_validator(self):
    
    def should_reinit_wandb(self):
        # Check if wandb run needs to be rolled over.
        return (
            not self.config.wandb.off
            and self.step
            and self.step % self.config.wandb.run_step_length == 0
        )

    def init_wandb(self, reinit=False):
        uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        spec_version = self.spec_version

        """Starts a new wandb run."""
        tags = [
            self.wallet.hotkey.ss58_address,
            str(spec_version),
            f"netuid_{self.config.netuid}",
        ]

        wandb_config = {
            key: copy.deepcopy(self.config.get(key, None))
            for key in ("neuron", "reward", "netuid", "wandb")
        }
        wandb_config["neuron"].pop("full_path", None)
        wandb_config["validator_uid"] = uid

        project_name = "testnet" # for TN76
        if self.config.netuid == 20:
            project_name = "mainnet"


        self.wandb = wandb.init(
            anonymous="allow",
            reinit=reinit,
            entity='bitagentsn20',
            project=project_name,
            config=wandb_config,
            dir=self.config.neuron.full_path,
            tags=tags,
            resume='allow',
            name=f"{uid}-{spec_version}-{datetime.today().strftime('%Y-%m-%d')}",
        )
        bt.logging.success(f"Started a new wandb run <blue> {self.wandb.name} </blue>")


    def reinit_wandb(self):
        """Reinitializes wandb, rolling over the run."""
        self.wandb.finish()
        init_wandb(self, reinit=True)


    def log_event(event):
        #bt.logging.debug("Writing to WandB ....")
        #if self.config.netuid != 20 and self.config.netuid != 76:
        #    return

        if not self.config.wandb.on:
            return

        if not getattr(self, "wandb", None):
            init_wandb(self)

        # Log the event to wandb.
        self.wandb.log(event)
        #bt.logging.debug("Logged event to WandB ....")

    self.log_event = log_event

    if self.config.run_local:
        def random_seed():
            None
        self.random_seed = random_seed
        initiate_validator_local(self)

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
import time
import bitagent
from typing import Tuple

# Bittensor
import bittensor as bt

# Bittensor Validator Template:
from bitagent.validator import forward, initiate_validator

# import base validator class which takes care of most of the boilerplate
from common.base.validator import BaseValidatorNeuron

class Validator(BaseValidatorNeuron):
    """
    BitAgent validator neuron class.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        self.first_forward_pass_completed = False
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        bt.logging.info("initiate_validator()")
        initiate_validator(self)
        bt.logging.debug(f"spec_version: {self.spec_version}")
        if self.config.neuron.visible_devices:
            print(f"Setting CUDA_VISIBLE_DEVICES to: {self.config.neuron.visible_devices}")
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.neuron.visible_devices
        else:
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                del os.environ["CUDA_VISIBLE_DEVICES"]
                
        # check if the sglang python executable exists
        python_path = f"{os.getcwd()}/.venvsglang/bin/python"
        if not os.path.exists(python_path):
            raise FileNotFoundError(f"The required sglang python executable does not exist at {python_path}")
        bt.logging.info(f"sglang python executable found at {python_path}")


    async def forward(self, synapse: bitagent.protocol.QueryTask=None):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        fwd = await forward(self, synapse)
        self.first_forward_pass_completed = True
        return fwd

    async def forward_fn(self, synapse: bitagent.protocol.QueryTask=None) -> bitagent.protocol.QueryTask:
        return await self.forward(synapse)

    async def blacklist_fn(self, synapse: bitagent.protocol.QueryTask) -> Tuple[bool, str]:
        # Add hotkeys to blacklist here as needed
        # blacklist the hotkeys mining on the subnet to prevent any potential issues
        #hotkeys_to_blacklist = [h for i,h in enumerate(self.hotkeys) if self.metagraph.S[i] < 20000 and h != self.wallet.hotkey.ss58_address]
        #if synapse.dendrite.hotkey in hotkeys_to_blacklist:
        #    return True, "Blacklisted hotkey - miners can't connect, use a diff hotkey."
        return False, ""

    async def priority_fn(self, synapse: bitagent.protocol.QueryTask) -> float:
        # high priority for organic traffic
        return 1000000.0

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(15)
            if validator.should_exit:
                bt.logging.warning("Ending validator...")
                break

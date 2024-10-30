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

import bitagent
import numpy as np
from bitagent.helpers.llms import llm
from collections import Counter

def miner_init(self, config=None):
    self.top_model_name = get_top_miner_HF_model_name(self)
    self.llm = llm

def miner_process(self, synapse: bitagent.protocol.QueryTask) -> bitagent.protocol.QueryTask:
    llm_response = self.llm(self, synapse.messages, synapse.tools, self.top_model_name)
    synapse.response = llm_response

    return synapse

# each validator sends their top miner's HF model name to each miner
def get_top_miner_HF_model_name(self):
    # miner can specify a HF model name to run
    if self.config.hf_model_name_to_run != "none":
        return self.config.hf_model_name_to_run

    # if no specific model name is specified, miner will use the top model name from the validators' votes
    if not self.hf_top_model_names or len(self.hf_top_model_names) == 0:
        return self.config.hf_model_name_to_run
    else:
        # get the most common model name
        most_common_model_name = Counter(self.hf_top_model_names).most_common(1)[0][0]
        return most_common_model_name

def save_top_model_from_validator(self, top_hf_model_name, validator_uid):
    # save off the top model from this validator
    bt.logging.debug(f"Saving top HF model name from validator {validator_uid} state - {self.config.neuron.full_path}/miner_state.npz.")

    self.hf_top_model_names[validator_uid] = top_hf_model_name

    # Save the state of the miner to file.
    np.savez(
        self.config.neuron.full_path + "/miner_state.npz",
        hf_top_model_names=self.hf_top_model_names,
    )
    
def load_state(self):
    """Loads the state of the miner from a file."""
    bt.logging.info("Loading miner state.")
    state = np.load(self.config.neuron.full_path + "/miner_state.npz")
    if 'hf_top_model_names' in state:
        loaded_hf_top_model_names = state["hf_top_model_names"]
        self.hf_top_model_names = loaded_hf_top_model_names
    else:
        self.hf_top_model_names = {}
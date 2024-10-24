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
import bittensor as bt

def miner_init(self, config=None):

    def get_top_miner_HF_model_name():
        if config.openai_model_name != "none":
            return config.openai_model_name

        top_miner_uid = self.metagraph.I.argmax()

        top_miner_HF_model_name_response = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[top_miner_uid]],
            # Construct a query. 
            synapse=bitagent.protocol.GetHFModelName,
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=False,
            timeout=5.0,
        )

        return top_miner_HF_model_name_response[0].hf_model_name

    # TODO handle tools along with the messages
    def llm(messages, max_new_tokens = 160, temperature=0.7):
        if isinstance(messages, str):
            messages = [{"role":"user","content":messages}]

        hf_model_name = get_top_miner_HF_model_name()
        llm = ChatOpenAI(
            openai_api_key=self.config.openai_api_key,
            openai_api_base=self.config.openai_api_base,
            model_name=hf_model_name,
            max_tokens = max_new_tokens,
            temperature = temperature,
        )
        return llm.invoke(messages).content.strip()

    self.llm = llm

def miner_process(self, synapse: bitagent.protocol.QueryTask) -> bitagent.protocol.QueryTask:

    llm_response = self.llm(synapse.messages)
    synapse.response["response"] = llm_response

    return synapse

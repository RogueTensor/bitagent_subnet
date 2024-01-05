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
from typing import List, Tuple
import bittensor as bt

# Bittensor Miner Template:
import bitqna
from bitqna.miner.context_util import get_relevant_context_and_citations_from_synapse

# import base miner class which takes care of most of the boilerplate
from template.base.miner import BaseMinerNeuron
from transformers import T5Tokenizer, T5ForConditionalGeneration

class Miner(BaseMinerNeuron):
    """
    BitQnA miner neuron class. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. Modify, if you need to define custom capability.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map=self.device)

        def llm(input_text):
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            outputs = self.model.generate(input_ids, max_length=60)
            result = self.tokenizer.decode(outputs[0])
            # response is typically: <pad> text</s>
            result = result.replace("<pad>","").replace("</s>","").strip()
            return result

        self.llm = llm

    async def forward(
        self, synapse: bitqna.protocol.QnAProtocol
    ) -> bitqna.protocol.QnAProtocol:
        """
        Processes the incoming BitQnA synapse and returns response.

        Args:
            synapse (bitqna.protocol.QnAProtocol): The synapse object containing the urls and prompt.

        Returns:
            bitqna.protocol.QnAProtocol: The synapse object with the 'response' field set to the generated response and citations

        """
        if not synapse.urls and not synapse.datas:
            context = ""
            citations = []
        else:
            context, citations = get_relevant_context_and_citations_from_synapse(synapse)

        query_text = f"Given the following CONTEXT:\n\n{context}\n\nPlease provide the user with an answer to their question: {synapse.prompt}.\n\n Response: "
        llm_response = self.llm(query_text)

        synapse.response["response"] = llm_response
        synapse.response["citations"] = citations

        return synapse

    async def blacklist(
        self, synapse: bitqna.protocol.QnAProtocol
    ) -> Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (bitqna.protocol.QnAProtocol): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        # TODO(developer): Define how miners should blacklist requests.
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: bitqna.protocol.QnAProtocol) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (bitqna.protocol.QnAProtocol): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)

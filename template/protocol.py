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

from typing import Optional, List
import bittensor as bt

# TODO we should add another protocol for IsAlive

class QnAProtocol(bt.Synapse):
    """
    A simple BitQnA protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling validator request and miner response communication

    Attributes:
    - urls: list of urls for data context (urls can be empty, urls can contain wildcards)
    - prompt: user prompt
    - repsonse: a dict containing the response along with citations from the provided data context (urls)
        - {response: str, citations: List[str]}
    """

    # Required request input, filled by sending dendrite caller.
    urls: List[str]
    prompt: str

    # Optional request output, filled by recieving axon.
    response: Optional[dict] = {}

    def deserialize(self) -> dict:
        """
        Deserialize the miner response. 

        Returns:
        - dict: The deserialized response, which in this case is the miner's response.

        """
        return self.response

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

class QnATask(bt.Synapse):
    """
    A simple BitAgent protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling validator request and miner response communication

    Attributes:
    #- urls: list of urls for data context (urls can be empty, urls can contain wildcards)
    - datas: list of data {source & context} in a List of dicts
    - prompt: user prompt
    - repsonse: a dict containing the response along with citations from the provided data context (urls or datas)
        - {response: str, citations: List[dict]}
        - the citations are a list of dicts {source & content: relevant content chunk from the source}
        - there are multiple citations, multiple returned dicts can contains the same source of reference
    - timeout: time in seconds to wait for the response (ONLY used for tasks coming in through validator axon)
    - miner_uids: list of miner uids to send the task to (ONLY used for tasks coming in through validator axon)
    """

    # Required request input, filled by sending dendrite caller.
    urls: Optional[List[str]] = [] # not used at the moment
    datas: List[dict] = []
    prompt: str = ""

    # Optional request output, filled by recieving axon.
    response: Optional[dict] = {}

    # used only for requests coming in through validator axon
    timeout: Optional[float] = None
    miner_uids: Optional[List[int]] = []

class QnAResult(bt.Synapse):
    """
    Provide feedback on last task request from validator to inform Miner of performance.
    This is a one-way request does not require a response.
    Attributes:
    - results: string of results to be printed to the logs
    """
    results: str

class IsAlive(bt.Synapse):
    response: bool

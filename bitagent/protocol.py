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
from bitagent.schemas.chat import ChatMessage
from bitagent.schemas.tool import Tool

class QueryTask(bt.Synapse):
    """
    A simple BitAgent protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling validator request and miner response communication

    Attributes:
    - messages: a list of ChatMessage (see bitagent/schemas) - will be used for every task except Tool Gen
    - tools: list of tools {name, description, arguments } in a List of dicts
    - repsonse: the tool calling response messages
    # TODO can we reduce to just response now?
        - {response: <messages>}
    - timeout: time in seconds to wait for the response (ONLY used for tasks coming in through validator axon)
    - miner_uids: list of miner uids to send the task to (ONLY used for tasks coming in through validator axon)
    """

    # Required request input, filled by sending dendrite caller.
    tools: List[Tool] = []
    messages: List[ChatMessage] = []

    # Optional request output, filled by recieving axon.
    response: str = ""

    # used only for requests coming in through validator axon
    timeout: Optional[float] = None
    miner_uids: Optional[List[int]] = []

class QueryResult(bt.Synapse):
    """
    Provide feedback on last task request from validator to inform Miner of performance.
    This is a one-way request does not require a response.
    Attributes:
    - results: string of results to be printed to the logs
    """
    results: str

class IsAlive(bt.Synapse):
    response: bool

# Validator calls this to get the HF model name that this miner hosts on HF
class GetHFModelName(bt.Synapse):
    hf_model_name: Optional[str] = None

# Validator calls this to have the miner set the TOP HF model for this miner to run
class SetHFModelName(bt.Synapse):
    hf_model_name: str
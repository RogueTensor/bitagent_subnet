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
from bitagent.schemas.tool import Tool
from bitagent.schemas.conversation import Conversation
import bittensor as bt 
import bitagent
import httpx
import os

HUST_ENDPOINT = os.environ.get("HUST_ENDPOINT", "http://localhost:10001/api/bitagent")

def miner_init(self, config=None):
    pass

async def miner_process(self, synapse: bitagent.protocol.QnATask) -> bitagent.protocol.QnATask:
    # print(synapse.messages)
    try:
        # if "conversation history" in synapse.notes or "Tool Calling" in synapse.notes:
        message_history=[]
        for item in synapse.messages:
            role=str(item.role)
            json_item={
                "role":role,
                "content":item.content
            }
            message_history.append(json_item)
        tools = [t.to_dict() for t in synapse.tools]
        data = {
            "prompt": synapse.prompt,
            "datas": synapse.datas,
            "timeout": synapse.timeout,
            "notes": synapse.notes,
            "message_history": message_history,
            "tools" : tools
        }
    except:
        bt.logging.info('BBBBB')
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=synapse.timeout)) as client:
            hust_response = await client.post(HUST_ENDPOINT, json=data)
            hust_response = hust_response.json()
    except Exception as e:
        print(e)
        hust_response = {
            "response": synapse.prompt * 10,
            "citations": [],
        }
    try:
        if "Tool Calling" in synapse.notes:
            bt.logging.info(message_history)
    except:
        bt.logging.info("logging fail for tool calling message history")
    synapse.response = hust_response

    return synapse

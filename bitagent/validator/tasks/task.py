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

import time
import requests
import bittensor as bt
from pprint import pformat
from typing import List
from bitagent.protocol import QnATask
from bitagent.schemas.tool import Tool
from bitagent.schemas.conversation import Conversation

# combines criterion/criteria plus QnATask synapse for eval to form a task for the miner
class Task:

    synapse: QnATask

    def __init__(
        self,
        task_id: str,
        name: str,
        weight: int,
        prompt: str,
        desc: str = "",
        task_type: str = None,
        datas: List[dict] = [],
        tools: List[Tool] = [],
        notes: str = "No Notes",
        message_history: Conversation = [],
        urls: List[str] = [],
        timeout: float = 12.0,
    ) -> None:
        self.task_id = task_id
        self.task_type = task_type
        self.weight = weight
        self.name = name
        self.desc = desc
        self.tools = tools
        self.notes = notes
        self.message_history = message_history
        self.timeout = timeout
        self.synapse = QnATask(
            prompt=prompt, urls=urls, datas=datas, tools=tools, notes=notes, message_history=message_history
        )

    @classmethod
    def create_from_json(cls, data):
        return Task(
            data["task_id"],
            data["name"],
            data["weight"],
            data["prompt"],
            data["desc"],
            data["task_type"],
            data["datas"],
            [Tool(**tool) for tool in data["tools"]],
            data["notes"],
            Conversation.from_list(data["message_history"]),
            data["urls"],
            data["timeout"],
        )

    def reward(self, validator, response):
        num_retries = 0
        while num_retries < 3:
            try:
                reward_url = (
                    f"{validator.config.task_api_host}/task_api/evaluate_task_response"
                )
                headers = {"Content-Type": "application/json"}
                data = {
                    "task_id": self.task_id,
                    "response": {
                        "response": response.response,
                        # "prompt": response.prompt,
                        # "urls": response.urls,
                        # "datas": response.datas,
                        # "tools": response.tools,
                        # "notes": response.notes,
                        "axon_hotkey": response.axon.hotkey,
                        "dendrite_process_time": response.dendrite.process_time,
                        "dendrite_status_code": response.dendrite.status_code,
                        "axon_status_code": response.axon.status_code,
                    },
                }

                result = requests.post(
                    reward_url, headers=headers, json=data
                )  # , verify=False)
                result.raise_for_status()
                return result.json()["result"]
            except KeyError as e:
                bt.logging.error("Key Error with API: ", e)
                num_retries += 1
                if "task_id" in e:
                    return False, "Task API restarted, task_id not found"
            except Exception as e:
                bt.logging.error("Error with Task API: ", e)
                time.sleep(25)
                num_retries += 1
        return False, "Waiting for Task API to come back online ..."

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

def get_random_task(validator) -> [List[int], Task]:
    if validator.config.run_local:
        pass
    else:
        while True:
            try:
                task_url = f"{validator.config.task_api_host}/task_api/get_new_task"
                headers = {"Content-Type": "application/json"}
                data = {}

                response = requests.post(
                    task_url, headers=headers, json=data
                )  # , verify=False)
                if response.status_code == 502:
                    # bt.logging.debug("Task API is down, should be back up shortly: ", response)
                    time.sleep(25)
                elif response.status_code != 200:
                    bt.logging.error(
                        "Error connecting to Task API, might be access restriction: ",
                        response,
                    )
                    time.sleep(25)
                else:
                    jdata = response.json()

                    # if miner_uids are empty, then leave empty and they will be randomly selected
                    if "miner_uids" in jdata.keys():
                        # get miner uids to call for organic traffic
                        miner_uids = jdata["miner_uids"]
                        # bt.logging.debug("Received miner uids: ", miner_uids)
                    else:
                        miner_uids = []
                    task = Task.create_from_json(jdata["task"])
                    return miner_uids, task

            except Exception as e:
                bt.logging.debug(
                    "Likely waiting for Task API to come back online ... ", e
                )
                time.sleep(25)

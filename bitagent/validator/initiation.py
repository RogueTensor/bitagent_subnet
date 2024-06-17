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


import os
import glob
import shutil
import bittensor as bt
from bitagent.task_api.initiation import initiate_validator as initiate_validator_local

import comet_llm
import os
import json
import requests
import multiprocessing
import time
import signal

def write_to_comet_ml_thread(workspace, project_name, step_log):
    comet_llm.log_prompt(
        prompt=step_log["prompt"],
        output=step_log["completion_response"],
        project=project_name,
        workspace=workspace,
        metadata=step_log
    )

def comet_ml_logger(directory, workspace, project_name):
    """Function to simulate logging process."""
    try:
        while True:
            for filename in os.listdir(directory):
                if filename.endswith(".txt"):
                    filepath = os.path.join(directory, filename)
                    with open(filepath, 'r') as file:
                        log_data = file.readlines()
                    for log in log_data:
                        write_to_comet_ml_thread(workspace, project_name, json.loads(log))
                    os.remove(filepath)
            time.sleep(10)  # Check every 10 seconds
    except KeyboardInterrupt:
        bt.logging.warning("Logger process terminated.")

# setup validator with wandb
# clear out the old wandb dirs if possible
def initiate_validator(self):

    #task_api_cml = f"{self.config.task_api_host}/task_api/get_cml_api_data"
    #headers = {"Content-Type": "application/json"}
    #data = {}
    #if self.config.netuid == 76: # testnet
    #    data = {
    #        "network": "testnet",
    #    }
    #cml_results = requests.get(task_api_cml, headers=headers, json=data)
    #cml_data = cml_results.json()
    #comet_llm.init(api_key=cml_data['api_key'])

    #if not self.config.log_dir.endswith("/"):
    #    self.config.log_dir = self.config.log_dir + "/"

    #if self.config.netuid == 76: # testnet
    #    self.log_directory = self.config.log_dir +  ".comet_testnet-llm-logs"
    #    workspace = cml_data['workspace']
    #    project_name = cml_data['project_name'] 
    #elif self.config.netuid == 20: # mainnet
    #    self.log_directory = self.config.log_dir + ".comet_mainnet-llm-logs"
    #    workspace = cml_data['workspace'] 
    #    project_name = cml_data['project_name']
    #else: # unknown, maybe local
    #    self.log_directory = None
    #    workspace = None
    #    project_name = None

    if False: #self.log_directory:
        os.makedirs(self.log_directory, exist_ok=True)
        logger_process = multiprocessing.Process(target=comet_ml_logger, args=(self.log_directory, workspace, project_name,))
        logger_process.start()
        def signal_handler(signal_received, frame):
            """Handles cleanup when receiving a signal."""
            print('Signal received, cleaning up.')
            logger_process.terminate()
            logger_process.join()
            print("Logger process terminated cleanly.")
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    if self.config.run_local:
        def random_seed():
            None
        self.random_seed = random_seed
        initiate_validator_local(self)

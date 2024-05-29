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

from comet_ml import Experiment
import os
import json
import multiprocessing
import time
import signal

def write_to_comet_ml_thread(project_name, step_log):
    experiment = Experiment(
        api_key="x6TeIvmRgto7KhgAeMVJqkZRQ",
        project_name=project_name,
        workspace="roguetensor",
        display_summary_level=0,
    )
    experiment.log_parameters(step_log)
    experiment.end()

def comet_ml_logger(directory, project_name):
    """Function to simulate logging process."""
    try:
        while True:
            for filename in os.listdir(directory):
                if filename.endswith(".txt"):
                    filepath = os.path.join(directory, filename)
                    with open(filepath, 'r') as file:
                        log_data = file.readlines()
                    for log in log_data:
                        write_to_comet_ml_thread(project_name, json.loads(log))
                    os.remove(filepath)
            time.sleep(10)  # Check every 10 seconds
    except KeyboardInterrupt:
        bt.logging.warning("Logger process terminated.")

# setup validator with wandb
# clear out the old wandb dirs if possible
def initiate_validator(self):
    if self.config.netuid == 76: # testnet
        self.log_directory = self.config.log_dir +  ".testnet-logs"
        project_name = "bitagent-testnet"
    elif self.config.netuid == 20: # mainnet
        self.log_directory = self.config.log_dir + ".mainnet-logs"
        project_name = "bitagent-mainnet"
    else: # unknown, maybe local
        self.log_directory = None
        project_name = None

    if self.log_directory:
        os.makedirs(self.log_directory, exist_ok=True)
        logger_process = multiprocessing.Process(target=comet_ml_logger, args=(self.log_directory, project_name,))
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

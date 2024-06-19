# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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
import time
import json
import pickle
import random
import bittensor as bt
from collections.abc import Iterator
from distutils.util import strtobool
from datasets import load_dataset, load_from_disk
from bitagent.task_api.datasources.loaders import huggingface_loader

root_data_dir = "bitagent.data"

class QnADataset(Iterator):
    def __init__(self):
        super().__init__()
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        seed = random.randint(0, 10000)
        wiki = huggingface_loader("wikipedia", name="20220301.en")
        
        self.datasets = {"wiki": iter(wiki.shuffle(seed=seed))}
        

    def __next__(self):
        bt.logging.debug("Retrieving Q&A data from dataset...")
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        while True:
            try:
                dname, ds = random.choice(list(self.datasets.items()))
                text = next(ds)["text"]

                # Check if the text is not empty or does not consist only of newline characters
                if text.strip():
                    return {"text": text}

            except Exception as e:
                bt.logging.debug(f"HuggingFace issue ... {e}")
                time.sleep(15)


class SummaryDataset(Iterator):
    def __init__(self):
        super().__init__()
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        seed = random.randint(0, 1000)
        bt.logging.debug("Loading Samsum from HuggingFace")
        ss_data_dir = f"{root_data_dir}/samsum"
        if os.path.exists(f"{ss_data_dir}/state.json"):
            bt.logging.debug(f"Loading from disk ({ss_data_dir}) ...")
            ss_ds = load_from_disk(ss_data_dir)
        else:
            bt.logging.debug("Loading from web ...")
            ss_ds = load_dataset("samsum", split="train")
            ss_ds.save_to_disk(ss_data_dir)
        bt.logging.debug("Loaded.")

        bt.logging.debug("Loading CNN Daily from HuggingFace")
        cnn_data_dir = f"{root_data_dir}/cnn_daily"
        if os.path.exists(f"{cnn_data_dir}/state.json"):
            bt.logging.debug(f"Loading from disk ({cnn_data_dir}) ...")
            cnn_ds = load_from_disk(cnn_data_dir)
        else:
            bt.logging.debug("Loading from web ...")
            cnn_ds = load_dataset("cnn_dailymail", "3.0.0", split="train")
            cnn_ds.save_to_disk(cnn_data_dir)
        bt.logging.debug("Loaded.")

        self.keys = {"cnn_dailymail": {"text": "article", "summary": "highlights"},
                     "samsum": {"text": "dialogue", "summary":"summary"}}

        self.datasets = { 
            "samsum": iter(ss_ds.shuffle(seed=seed)),
            "cnn_dailymail": iter(cnn_ds.shuffle(seed=seed))
        }

    def __next__(self):
        bt.logging.debug("Retrieving summarization data from dataset...")
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        while True:
            try:
                dname, ds = random.choice(list(self.datasets.items()))
                data = next(ds)
                text = data[self.keys[dname]["text"]]
                summary = data[self.keys[dname]["summary"]]

                # Check if the text is not empty or does not consist only of newline characters
                if text.strip():
                    return {"text": text, "summary": summary}
            except Exception as e:
                bt.logging.debug(f"HuggingFace issue ... {e}")
                time.sleep(15)

from bitagent.task_api.helpers.ansible import AnsibleRepoAnalyzer
class AnsibleDataset(Iterator):
    def __init__(self):
        super().__init__()
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        self.dataset = pickle.load(open(f"{root_data_dir}/ansible-repos.pkl", "rb"))

    def __next__(self):
        bt.logging.debug("Retrieving ansible data from dataset...")
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        happy = False
        task_set = None
        while not happy:
            try:
                repo = random.choice(list(self.dataset))
                with AnsibleRepoAnalyzer(repo, os.environ["GITHUB_ACCESS_TOKEN"]) as repo_analyzer:
                    all_task_sets = repo_analyzer.get_ansible_tasks()
                    if len(all_task_sets.keys()) > 0:
                        num_tries = 0
                        while num_tries < 5 and not happy:
                            random_task_set_key = random.choice(list(all_task_sets.keys()))
                            task_set = all_task_sets[random_task_set_key]
                            if len(task_set) > 3 and len(task_set) < 10:
                                happy = True
                            num_tries += 1
                return task_set
            except Exception as e:
                bt.logging.debug(f"Github issue ... {e}")
        return task_set
 
from bitagent.task_api.datasources.api_constants import apis
class APIDataset(Iterator):
    def __init__(self):
        super().__init__()
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        # self.dataset = json.load(open(f"{root_data_dir}/apis.json", "r"))
        self.dataset = apis

    def __next__(self):
        bt.logging.debug("Retrieving api data from dataset...")
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        while True:
            try:
                api = random.choice(list(self.dataset))
                return api
            except Exception as e:
                bt.logging.debug(f"Github issue ... {e}")

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
import random
import bittensor as bt
from datasets import load_dataset, load_from_disk
from collections.abc import Iterator

root_data_dir = "bitagent.data"

class QnADataset(Iterator):
    def __init__(self):
        super().__init__()
        seed = random.randint(0, 1000)
        bt.logging.debug("Loading OpenWebText from HuggingFace")
        owt_data_dir = f"{root_data_dir}/openwebtext"
        if os.path.exists(f"{owt_data_dir}/state.json"):
            bt.logging.debug(f"Loading from disk ({owt_data_dir}) ...")
            owt_ds = load_from_disk(owt_data_dir)
        else:
            bt.logging.debug("Loading from web ...")
            owt_ds = load_dataset("openwebtext", split="train", trust_remote_code=True)
            owt_ds.save_to_disk(owt_data_dir)

        bt.logging.debug("Loaded.")
        self.datasets = [ 
            iter(owt_ds.shuffle(seed=seed)),
        ]

    def __next__(self):
        bt.logging.debug("Retrieving Q&A data from dataset...")
        while True:
            try:
                ds = random.choice(self.datasets)
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

import os
import pandas as pd
import bittensor as bt
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download

class ShuffledJSONDatasetIterator:
    def __init__(self):
        dataframes = []

        # TODO - other BFCL task types:
        # irrelevance and live_irrelevance - answer is NOTHING
        # exec_* (simple, multiple, parallel, parallel_multiple) - answer in the file itself
        # multi_turn_* - answer in the file itself
        # parallel* - answer in the file itself
        # rest - maybe later - calls to API that the validator would need to setup

        for filename in ["java", "javascript", "simple", "multiple", "sql", "live_simple", "live_multiple"]:
            bfcl_path = "bitagent.data/bfcl/BFCL_v3_{filename}.json"
            bfcl_answer_path = "bitagent.data/bfcl/possible_answer/BFCL_v3_{filename}.json"
            file_path = bfcl_path.format(filename=filename)
            answer_path = bfcl_answer_path.format(filename=filename)
            df_data = pd.read_json(file_path, lines=True)
            df_answer = pd.read_json(answer_path, lines=True)
            df_data['ground_truth'] = df_answer['ground_truth']
            dataframes.append(df_data[['id','question','function','ground_truth']])
        self.all_data = pd.concat(dataframes)
        self._shuffle_data()

    def _shuffle_data(self):
        self.shuffled_data = self.all_data.sample(frac=1, random_state=572343).reset_index(drop=True)
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.shuffled_data):
            row = self.shuffled_data.iloc[self.index]
            self.index += 1
            return row
        else:
            self._shuffle_data()  # Shuffle and reset index if end is reached
            return self.__next__()

def huggingface_loader(dataset_name, root_data_dir="bitagent.data", split="train", name=None):
    bt.logging.debug(f"Loading {dataset_name}")
    dataset_dir = f"{root_data_dir}/{dataset_name.replace('/','_')}"
    if os.path.exists(f"{dataset_dir}/state.json") and dataset_name != "BitAgent/tool_calling_shuffle":
        bt.logging.debug(f"Loading from disk ({dataset_dir}) ...")
        ds = load_from_disk(dataset_dir)
    else:
        bt.logging.debug("Loading from web ...") 
        ds = load_dataset(dataset_name, split=split, name=name, token=os.getenv("HF_TOKEN", None))
        ds.save_to_disk(dataset_dir)
    bt.logging.debug("Loaded.")
    return ds

def load_bfcl_dataset(dataset_name, root_data_dir="bitagent.data", split="train", name=None):
    snapshot_download(repo_id=dataset_name, allow_patterns="*.json", repo_type="dataset", local_dir="bitagent.data/bfcl/")

    return ShuffledJSONDatasetIterator()
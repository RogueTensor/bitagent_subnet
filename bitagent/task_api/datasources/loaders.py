import os
import bittensor as bt
from datasets import load_dataset, load_from_disk


def huggingface_loader(dataset_name, root_data_dir="bitagent.data", split="train", name=None):
    bt.logging.debug(f"Loading {dataset_name}")
    dataset_dir = f"{root_data_dir}/{dataset_name.replace('/','_')}"
    if os.path.exists(f"{dataset_dir}/state.json"):
        bt.logging.debug(f"Loading from disk ({dataset_dir}) ...")
        ds = load_from_disk(dataset_dir)
    else:
        bt.logging.debug("Loading from web ...") 
        ds = load_dataset(dataset_name, split=split, name=name, token=os.getenv("HF_TOKEN", None))
        ds.save_to_disk(dataset_dir)
    bt.logging.debug("Loaded.")
    return ds
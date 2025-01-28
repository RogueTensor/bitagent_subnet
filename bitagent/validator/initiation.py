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
import copy
import wandb
import shutil
import bittensor as bt
from datetime import datetime
from bitagent.datasources import ToolDataset
from langchain_openai import ChatOpenAI
from sentence_transformers import util
from bitagent.helpers.sbert import CachedSentenceTransformer

# setup validator with wandb
# clear out the old wandb dirs if possible
def initiate_validator(self):
    
    def init_wandb(self, reinit=False):
        uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        spec_version = self.spec_version

        """Starts a new wandb run."""
        tags = [
            self.wallet.hotkey.ss58_address,
            str(spec_version),
            f"netuid_{self.config.netuid}",
        ]

        wandb_config = {
            key: copy.deepcopy(self.config.get(key, None))
            for key in ("neuron", "reward", "netuid", "wandb")
        }
        wandb_config["neuron"].pop("full_path", None)
        wandb_config["validator_uid"] = uid

        if self.config.netuid == 20:
            project_name = "mainnet"
        elif self.config.netuid == 76:
            project_name = "testnet" # for TN76
        else:
            self.wandb = "errored"
            return # must be using a local netuid, no need to log to wandb

        try:
            self.wandb = wandb.init(
                anonymous="allow",
                reinit=reinit,
                entity='bitagentsn20',
                project=project_name,
                config=wandb_config,
                dir=self.config.neuron.full_path,
                tags=tags,
                resume='allow',
                name=f"{uid}-{spec_version}-{datetime.today().strftime('%Y-%m-%d')}",
            )
            bt.logging.success(f"Started a new wandb run <blue> {self.wandb.name} </blue>")
        except Exception as e:
            self.wandb = "errored"
            bt.logging.error("Could not connect to wandb ... continuing without it ...")
            bt.logging.error(f"WANDB Error: {e}")

    def log_event(event):
        #bt.logging.debug("Writing to WandB ....")

        if not self.config.wandb.on:
            return

        if not getattr(self, "wandb", None):
            clear_wandb_dir(self)
            init_wandb(self)

        if self.wandb == "errored":
            return

        # Log the event to wandb.
        self.wandb.log(event)
        #bt.logging.debug("Logged event to WandB ....")

    self.log_event = log_event

    initiate_validator_local(self)

def clear_wandb_dir(self):
    wandb_path = os.path.join(self.config.neuron.full_path, "wandb")
    if os.path.exists(wandb_path):
        bt.logging.info(f"Clearing WandB directory of old runs")
        for item in os.listdir(wandb_path):
            item_path = os.path.join(wandb_path, item)
            try:
                if os.path.islink(item_path):  # If it's a symbolic link
                    os.unlink(item_path)  # Remove the symlink
                elif os.path.isfile(item_path):  # If it's a regular file
                    os.remove(item_path)
                elif os.path.isdir(item_path):  # If it's a directory
                    shutil.rmtree(item_path)
            except Exception as e:
                bt.logging.warning(f"Failed to remove {item_path}: {e}")
        bt.logging.info(f"Cleared WandB directory of old runs")

# provide some capabilities to the task API (LLM, cossim)
def initiate_validator_local(self):
    #bt.logging.info("Initializing Validator - this may take a while (downloading data and models).")
    self.tool_dataset = ToolDataset()
    self.check_date = ""
    #bt.logging.debug("Initializing Validator - this may take a while (downloading data and models) - loading model ...")
    self.sentence_transformer = CachedSentenceTransformer('BAAI/bge-small-en-v1.5')

    def llm(messages, max_new_tokens = 160, temperature=0.7):
        if isinstance(messages, str):
            messages = [{"role":"user","content":messages}]
        llm = ChatOpenAI(
            openai_api_key=self.config.openai_api_key,
            openai_api_base=self.config.openai_api_base,
            model_name=self.config.validator_model_name,
            max_tokens = max_new_tokens,
            temperature = temperature,
        )
        return llm.invoke(messages).content.strip()
    
    self.llm = llm
    
    #bt.logging.debug("Initializing Validator - this may take a while (downloading data and models) - finished loading model")

    # code to measure the relevance of the response to the question
    def measure_relevance_of_texts(text1, text2): 
        # Encode the texts to get the embeddings
        if type(text2) == list:
            embeddings = self.sentence_transformer.encode([text1,*text2], convert_to_tensor=True, show_progress_bar=False)
        else:
            embeddings = self.sentence_transformer.encode([text1,text2], convert_to_tensor=True, show_progress_bar=False)
        # Compute the cosine similarity between the embeddings
        if type(text2) == list:
            return util.pytorch_cos_sim(embeddings[0], embeddings[1:])[0]
        else:
            return float(util.pytorch_cos_sim(embeddings[0], embeddings[1:])[0][0])

    self.measure_relevance_of_texts = measure_relevance_of_texts
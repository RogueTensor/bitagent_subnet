import json
import random
import bittensor as bt
from pydantic import BaseModel
from collections.abc import Iterator
from datasets import load_dataset, load_from_disk
from bitagent.task_api.datasources.loaders import huggingface_loader
from bitagent.task_api.helpers.string_parse import parse_multiple_space_sep_json
from bitagent.schemas.conversation import Conversation
from bitagent.schemas.chat import ChatMessage




class ChatDataset(Iterator):
    def __init__(self):
        super().__init__()
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        seed = random.randint(0, 1000)
        #TODO add API key if wanna use wildchat
        # wildchat = huggingface_loader("allenai/WildChat-1M")
        lmsys = huggingface_loader('lmsys/lmsys-chat-1m')
        wizardlm = huggingface_loader('cognitivecomputations/WizardLM_alpaca_evol_instruct_70k_unfiltered')
        wizard_vic = huggingface_loader('cognitivecomputations/wizard_vicuna_70k_unfiltered')
        self.datasets = { 
            #TODO Decide if to use wildchat
            # "wildchat": iter(wildchat.shuffle(seed=seed)),
            "lmsys": iter(lmsys.shuffle(seed=seed)),
            "wizardlm": iter(wizardlm.shuffle(seed=seed)),
            "wizard_vic": iter(wizard_vic.shuffle(seed=seed))
            # https://huggingface.co/datasets/H-D-T/Buzz
            # https://huggingface.co/datasets/xzuyn/open-instruct-uncensored-alpaca
            
        }
    
    def wizardlm_formatter(self,row):
        conversation = [{'role': 'user', 'content': row['instruction']}, {'role': 'assistant', 'content': row['output']}]
        return Conversation.from_list(conversation)
    
    def wizard_vic_formatter(self,row):
        convos = row['conversations']
        for convo in convos:
            convo['role'] = convo.pop('from')
            convo['content'] = convo.pop('value')
            if convo['role'] == 'human':
                convo['role'] = 'user'
            if convo['role'] == 'gpt':
                convo['role'] = 'assistant'
        return Conversation.from_list(convos)
    
    def __next__(self) -> Conversation:
        bt.logging.debug("Retrieving chat data from dataset...")
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        for _ in range(20):
            try:
                dname, ds = random.choice(list(self.datasets.items()))
                if dname == "wizardlm":
                    row = next(ds)
                    return self.wizardlm_formatter(row)
                if dname == "wizard_vic":
                    row = next(ds)
                    return self.wizard_vic_formatter(row)
                
                row = next(ds)
                if row['language'] != "English":
                    continue
                conversation = row["conversation"]
                conversation = [{'role': msg['role'], 'content': msg['content']} for msg in conversation]
                
                return Conversation.from_list(conversation)
            except Exception as e:
                bt.logging.debug(f"Issue getting chat history {e}")
            

        
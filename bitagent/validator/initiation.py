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

import random
import threading
import socketserver
import bittensor as bt
from faker import Faker
from datetime import datetime
from http.server import SimpleHTTPRequestHandler
from bitagent.validator.dataset import QnADataset, SummaryDataset
from common.base.validator import BaseValidatorNeuron

import transformers
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from sentence_transformers.cross_encoder import CrossEncoder

def initiate_validator(self):
    bt.logging.info("Initializing Validator - this may take a while (downloading data and models).")
    bt.logging.debug("Initializing Validator - this may take a while (downloading data and models) - loading dataset 1/2 ...")
    self.qna_dataset = QnADataset()
    bt.logging.debug("Initializing Validator - this may take a while (downloading data and models) - finished loading dataset 1/2")
    bt.logging.debug("Initializing Validator - this may take a while (downloading data and models) - loading dataset 2/2 ...")
    self.summary_dataset = SummaryDataset()
    bt.logging.debug("Initializing Validator - this may take a while (downloading data and models) - finished loading dataset 2/2")

    bt.logging.debug("Initializing Validator - this may take a while (downloading data and models) - loading model ...")
    # load a simple LLM for evals
    transformers.logging.set_verbosity_error()
    model_name_or_path = "TheBloke/Mistral-7B-OpenOrca-AWQ"
    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    self.model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                              trust_remote_code=False, safetensors=True)
    bt.logging.debug("Initializing Validator - this may take a while (downloading data and models) - finished loading model")

    bt.logging.debug("Initializing Validator - this may take a while (downloading data and models) - loading cross encoder ...")
    self.cross_encoder = CrossEncoder("cross-encoder/stsb-distilroberta-base")
    bt.logging.debug("Initializing Validator - this may take a while (downloading data and models) - finished loading cross encoder")

    def validator_llm(input_text):
        text = f'''
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
'''
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(input_ids, max_new_tokens=160, temperature=0.7, top_k=40, top_p=0.95, do_sample=True, repetition_penalty=1.1)
        result = self.tokenizer.decode(outputs[0])
        result = result.split("<|im_start|> assistant\n")[-1].replace("<|im_end|>","").strip()
        return result

    self.validator_llm = validator_llm

    # faker data
    Faker.seed(random.randint(0,2000))
    self.fake = Faker()

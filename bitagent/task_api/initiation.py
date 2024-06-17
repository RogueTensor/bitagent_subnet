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
from faker import Faker
from bitagent.task_api.dataset import QnADataset, SummaryDataset
from bitagent.task_api.datasources import ToolDataset, ChatDataset, LocalToolDataset
from langchain_community.llms import VLLMOpenAI
from langchain_openai import ChatOpenAI, OpenAI
from sentence_transformers import SentenceTransformer, util
from bitagent.task_api.helpers.sbert import CachedSentenceTransformer

# provide some capabilities to the task API (LLM, cossim and faker)
def initiate_validator(self):
    #bt.logging.info("Initializing Validator - this may take a while (downloading data and models).")
    self.qna_dataset = QnADataset()
    self.summary_dataset = SummaryDataset()
    self.tool_dataset = ToolDataset()
    self.convo_dataset = ChatDataset()
    self.local_tool_gen_dataset = LocalToolDataset(table_name='tool_gen', db_type='sqlite', db_path='bitagent.data/tools.db')
    self.local_tool_call_dataset = LocalToolDataset(table_name='tool_call', db_type='sqlite', db_path='bitagent.data/tools.db')
    #bt.logging.debug("Initializing Validator - this may take a while (downloading data and models) - loading model ...")
    self.sentence_transformer = CachedSentenceTransformer('BAAI/bge-small-en-v1.5')

    def chat_llm(messages, max_new_tokens = 160, temperature=0.7):
        if isinstance(messages, str):
            messages = [{"role":"user","content":messages}]
        llm = ChatOpenAI(
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1",
            model_name="thesven/Mistral-7B-Instruct-v0.3-GPTQ",
            max_tokens = max_new_tokens,
            temperature = temperature,
        )
        return llm.invoke(messages).content.strip()
    
    self.chat_llm = chat_llm
    self.validator_llm = chat_llm
    
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

    # countering the effect of setting seed for task orchestration from validators
    random.seed(None)
    Faker.seed(random.randint(0,2000))
    self.fake = Faker()

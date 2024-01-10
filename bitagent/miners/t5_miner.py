# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
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

import bitagent
import transformers
from common.base.miner import BaseMinerNeuron
from transformers import T5Tokenizer, T5ForConditionalGeneration
from bitagent.miners.context_util import get_relevant_context_and_citations_from_synapse

def miner_init(self, config=None):
    transformers.logging.set_verbosity_error()
    self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
    self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map=self.device)

    def llm(input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(input_ids, max_length=60)
        result = self.tokenizer.decode(outputs[0])
        # response is typically: <pad> text</s>
        result = result.replace("<pad>","").replace("</s>","").strip()
        return result

    self.llm = llm

def miner_process(self, synapse: bitagent.protocol.QnATask) -> bitagent.protocol.QnATask:
    if not synapse.urls and not synapse.datas:
        context = ""
        citations = []
    else:
        context, citations = get_relevant_context_and_citations_from_synapse(synapse)

    query_text = f"Please provide the user with an answer to their question: {synapse.prompt}.\n\n Response: "
    if context:
        query_text = f"Given the following CONTEXT:\n\n{context}\n\n{query_text}"

    llm_response = self.llm(query_text)

    synapse.response["response"] = llm_response
    synapse.response["citations"] = citations

    return synapse

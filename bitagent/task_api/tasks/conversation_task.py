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
from bitagent.protocol import QnATask
from bitagent.task_api.tasks import Task, TASK_WEIGHTS
from common.base.validator import BaseValidatorNeuron
from bitagent.task_api.criteria import default_criteria, conversation_task_criteria 
from bitagent.schemas.conversation import Conversation

REWRITE_PROMPT = """Please rewrite the following text, ensuring to maintain the original meaning and nuances but altering the sentence structures, vocabulary, and overall presentation. 
The goal is to produce a version of the text that conveys the same information and sentiments as the original, but in a fresh and distinct manner. 
Avoid summarizing or omitting any details; instead, focus on presenting the same concepts and messages in a new light.
        
Rewrite this text: {text}"""


class ConversationTask(Task):
    def __init__(self, validator: BaseValidatorNeuron, name: str, desc: str = ""):
        super().__init__(name=name, desc=desc)
        self.validator=validator
        self.timeout=12.0
        self.weight = TASK_WEIGHTS['conversation']
        # convo is of type Conversation
        self.message_history, assistant_response = self.get_convo()
        self.criteria = default_criteria + conversation_task_criteria(correct_response=assistant_response)
        notes = """The task is to correctly respond to the user based on the conversation history."""
        self.synapse=QnATask(notes=notes, message_history=self.message_history)

    def reword(self, text: str) -> str:
        return self.validator.validator_llm(REWRITE_PROMPT.format(text=text))
    
    def get_convo(self) -> [Conversation, str]:
        convo: Conversation = next(self.validator.convo_dataset)
        
        assistant_idxs = [idx for idx,msg in enumerate(convo.messages) if msg.role == 'assistant']
        rand_idx = random.choice(assistant_idxs)
        
        # truncate convo upto (not including) the last assistants response 
        convo.messages = convo.messages[0:rand_idx+1]
        for msg in convo.messages: #message is of type ChatMessage
            msg.content = self.reword(msg.content)
        assistant_response = convo.messages[rand_idx].content
        
        convo.messages = convo.messages[0:rand_idx] 
        return convo, assistant_response
            

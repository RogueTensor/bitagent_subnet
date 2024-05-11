from typing import List, Dict
from pydantic import BaseModel
from bitagent.schemas.chat import ChatMessage

class Conversation(BaseModel):
    messages: List[ChatMessage] = []
    
    @classmethod
    def from_list(cls, data_list: List[Dict[str, str]]):
        """Create a Conversation object from a list of dictionaries."""
        messages = [ChatMessage.from_dict(item) for item in data_list]
        return cls(messages=messages)
    
    def to_list(self):
        return [msg.to_dict() for msg in self.messages]
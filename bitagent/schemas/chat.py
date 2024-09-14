from strenum import StrEnum
from typing import Dict, List
from pydantic import BaseModel, Field

class ChatRole(StrEnum):
    """One of ASSISTANT|USER to identify who the message is coming from."""

    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL_CALL = "tool call"
    TOOL_RESPONSE = "tool response"


class ChatMessage(BaseModel):
    """A list of previous messages between the user and the model, meant to give the model conversational context for responding to the user's message."""

    role: ChatRole = Field(
        title="One of the ChatRole's to identify who the message is coming from.",
    )
    content: str | dict | list = Field( # TODO the dict/list was added to support json loading the function calls. this should maybe be done inside  a ToolMessage type
        title="Contents of the chat message.",
    )

    @classmethod
    def from_dict(cls, data: Dict[str, str]):
        """Create a ChatMessage object from a dictionary."""
        return cls(role=ChatRole(data['role']), content=data['content'])
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}
    

def messages_from_list(data_list: List[Dict[str, str]]):
    messages = [ChatMessage.from_dict(item) for item in data_list]
    return messages

def messages_to_list(messages: List[ChatMessage]):
    return [msg.to_dict() for msg in messages]

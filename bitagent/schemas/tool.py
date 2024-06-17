from pydantic import BaseModel
from typing import Dict, Any, List

class Tool(BaseModel):
    """
    Attributes:
    - name: str
    - description: str
    - arguments: dict where the key is the name of the argument and the value is a dict containing the keys (required:bool, type:str, description:str)
    """
    name: str
    description: str
    arguments: Dict[str, Any]
    
    def to_dict(self):
        return self.dict()
    
    
class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]
    
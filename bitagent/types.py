from pydantic import BaseModel

class Tool(BaseModel):
    """
    Attributes:
    - name: str
    - description: str
    - arguments: dict where the key is the name of the argument and the value is a dict containing the keys (optional:bool, type:str, description:str)
    """
    name: str
    description: str
    arguments: dict

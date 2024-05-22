import re
import json
import ast
import random
import bittensor as bt
from collections.abc import Iterator
from datasets import load_dataset, load_from_disk
from bitagent.task_api.datasources.loaders import huggingface_loader
from bitagent.task_api.helpers.string_parse import parse_multiple_space_sep_json
from pydantic import BaseModel
from bitagent.schemas.tool import Tool
from bitagent.schemas.conversation import Conversation
from bitagent.schemas.chat import ChatMessage


def split_dialogue(text):
    # Define a pattern to match the roles and capture messages
    pattern = r"(USER|ASSISTANT|TOOL CALL|TOOl RESPONSE): (.*?)(?=\s*(USER|ASSISTANT|TOOL CALL|TOOL RESPONSE):|$)"
    
    # Find all matches in the text using the pattern
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Create a list of dictionaries based on the matches
    dialogue_list = [{"role": role.lower(), "content": message.strip().replace('\'','')} for role, message, _ in matches]
    
    return Conversation.from_list(dialogue_list)

def clean_text(text):
    text = text.replace("<|endoftext|>","")
    text = text.replace("ASSISTANT: <functioncall>", "TOOL CALL: ")
    text = text.replace("FUNCTION RESPONSE", "TOOL RESPONSE")
    text = text.replace("  "," ")
    return text.strip()

def json_schema_to_pydantic_tool(schema: dict) -> Tool:
    tool_name = schema.get('name', '')
    tool_description = schema.get('description', '')
    
    schema_parameters = schema.get('parameters', {})
    properties = schema_parameters.get('properties', {})
    required_params = schema_parameters.get('required', [])

    parameters = {}
    for param_name, param_info in properties.items():
        parameters[param_name] = {
            'required': param_name in required_params,
            'type': param_info.get('type', ''),
            'description': param_info.get('description', '')
        }

    return Tool(name=tool_name, description=tool_description, arguments=parameters)


class ToolCallData(BaseModel):
    convo: Conversation 
    tools: list[Tool]

# TODO (intern) - more datasets
class ToolDataset(Iterator):
    def __init__(self):
        super().__init__()
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        seed = random.randint(0, 1000)
        glaive_ds = huggingface_loader("glaiveai/glaive-function-calling-v2")

        self.datasets = { 
            "glaive": iter(glaive_ds.shuffle(seed=seed)),
        }
    def __next__(self) -> ToolCallData:
        bt.logging.debug("Retrieving function call data from dataset...")
        # countering the effect of setting seed for task orchestration from validators
        count = 0
        while count < 25:
            count += 1
            try:
                random.seed(None)
                dname, ds = random.choice(list(self.datasets.items()))
                data = next(ds)
                system_prompt = data['system'].replace("SYSTEM: ", "")
                if "following functions" not in system_prompt:
                    continue 
                
                chat_history = clean_text(data['chat'])
                tools = parse_multiple_space_sep_json(system_prompt.replace("You are a helpful assistant with access to the following functions. Use them if required - ",""))
                tools = [json_schema_to_pydantic_tool(tool) for tool in tools]
                convo = split_dialogue(chat_history)
                
                return ToolCallData(convo=convo, tools=tools)
            except Exception as e:
                bt.logging.debug(f"Issue getting tool call from dataset ... {e}")

from bitagent.task_api.datasources.sql import SQLDataset 

class LocalToolDataset(SQLDataset):
    def __init__(self, table_name: str = 'tools', db_type: str = 'sqlite', **kwargs):
        
        self.columns = ['conversation', 'tools']
        self.table_name = table_name
        super().__init__(table_name=self.table_name, columns=self.columns,db_type=db_type, **kwargs)
        self._create_table()
        self.shuffle()
    
    def __next__(self) -> ToolCallData:
        bt.logging.debug("Retrieving function call data from local dataset...")
        tool_data = super().__next__()
        for key,value in tool_data.items():
            tool_data[key] = json.loads(value)
        convo = Conversation.from_list(tool_data['conversation'])
        if isinstance(tool_data['tools'], str):
            tools = [json_schema_to_pydantic_tool(tool) for tool in json.loads(tool_data['tools'])]    
        elif isinstance(tool_data['tools'], list):
            tools = [json_schema_to_pydantic_tool(tool) for tool in tool_data['tools']]     
        else:
            raise ValueError(f"Invalid format for tools: {tool_data['tools']}")
        return ToolCallData(convo=convo, tools=tools)
        

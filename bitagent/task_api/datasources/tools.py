import re
import json
import ast
import random
import bittensor as bt
from pydantic import BaseModel
from typing import List, Dict, Any
from collections.abc import Iterator
from bitagent.schemas.tool import Tool
from bitagent.task_api.datasources.sql import SQLDataset
from bitagent.schemas.chat import ChatMessage, messages_from_list
from bitagent.task_api.datasources.loaders import huggingface_loader
from bitagent.task_api.helpers.string_parse import parse_multiple_space_sep_json


def split_dialogue(text) -> List[ChatMessage]:
    # Define a pattern to match the roles and capture messages
    pattern = r"(USER|ASSISTANT|TOOL CALL|TOOl RESPONSE): (.*?)(?=\s*(USER|ASSISTANT|TOOL CALL|TOOL RESPONSE):|$)"

    # Find all matches in the text using the pattern
    matches = re.findall(pattern, text, re.DOTALL)

    # Create a list of dictionaries based on the matches
    dialogue_list = [{"role": role.lower(), "content": message.strip().replace('\'','')} for role, message, _ in matches]
    
    for message in dialogue_list:
        if not message['role']:
            raise ValueError("There is a message with no role.")
     
    return messages_from_list(dialogue_list)


def clean_text(text):
    text = text.replace("<|endoftext|>", "")
    text = text.replace("ASSISTANT: <functioncall>", "TOOL CALL: ")
    text = text.replace("FUNCTION RESPONSE", "TOOL RESPONSE")
    text = text.replace("  ", " ")
    return text.strip()

def custom_json_schema_to_pydantic_tool(schema: dict) -> Tool:
    tool_name = schema.get("name", "")
    tool_description = schema.get("description", "")

    schema_arguments = schema.get("arguments", {})
    parameters = {}
    for param_name, param_info in schema_arguments.items():
        parameters[param_name] = {
            "required": param_info.get("required", False),
            "type": param_info.get("type", ""),
            "description": param_info.get("description", ""),
        }

    return Tool(name=tool_name, description=tool_description, arguments=parameters)

def json_schema_to_pydantic_tool(schema: dict) -> Tool:
    tool_name = schema.get("name", "")
    tool_description = schema.get("description", "")

    schema_parameters = schema.get("parameters", {})
    if not schema_parameters:
        schema_parameters = schema.get("arguments", {})
    properties = schema_parameters.get("properties", {})
    required_params = schema_parameters.get("required", [])
    if isinstance(required_params, bool):
        required_params = list(properties.keys()) if required_params else []
    elif not isinstance(required_params, list):
        required_params = []
    parameters = {}
    for param_name, param_info in properties.items():
        if param_name == "required":
            continue
        parameters[param_name] = {
            "required": param_name in required_params,
            "type": param_info.get("type", ""),
            "description": param_info.get("description", ""),
        }
    return Tool(name=tool_name, description=tool_description, arguments=parameters)




class ToolCallData(BaseModel):
    messages: List[ChatMessage]
    tools: list[Tool]

TYPES = ["str", "int", "dict", "list", "float", "bool", "string", "integer", "number", "boolean", "dictionary", "object"]

def detect_type(value: Any) -> str:
    type_mapping = {
        int: 'integer',
        float: 'number',
        str: 'string',
        bool: 'boolean',
        list: 'array',
        dict: 'object'
    }
    return type_mapping.get(type(value), 'string')

def add_extra_arguments(tool_call: Dict[str, Any], tools: List[Tool]):
    # Find the tool in the list
    tool_name = tool_call['name']
    arguments = tool_call.get('arguments', {})
    
    for tool in tools:
        if tool.name == tool_name:
            for arg_name, arg_value in arguments.items():
                if arg_name not in tool.arguments:
                    # Detect the type of the argument
                    arg_type = detect_type(arg_value)
                    # Add the new argument to the tool's schema
                    tool.arguments[arg_name] = {
                        'required': False, # assume false
                        'type': arg_type,
                        'description': arg_name
                    }
            break

class ToolDataset(Iterator):
    def __init__(self):
        super().__init__()
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        seed = random.randint(0, 1000)
        glaive_ds = huggingface_loader("glaiveai/glaive-function-calling-v2")
        bitagent_ds = huggingface_loader("BitAgent/tool_calling")

        self.datasets = {
            "glaive": iter(glaive_ds.shuffle(seed=seed)),
            "bitagent": iter(bitagent_ds.shuffle(seed=seed)),
        }

    def __next__(self) -> ToolCallData:
        bt.logging.debug("Retrieving function call data from dataset...")
        # countering the effect of setting seed for task orchestration from validators
        count = 0
        while count < 25:
            count += 1
            try:
                random.seed(None)
                dname, ds = random.choices(list(self.datasets.items()), [1, 10])[0]
                data = next(ds)
                if dname == "glaive":
                    system_prompt = data["system"].replace("SYSTEM: ", "")
                    if "following functions" not in system_prompt:
                        continue

                    chat_history = clean_text(data["chat"])
                    tools = parse_multiple_space_sep_json(
                        system_prompt.replace(
                            "You are a helpful assistant with access to the following functions. Use them if required - ",
                            "",
                        )
                    )
                    tools = [json_schema_to_pydantic_tool(tool) for tool in tools]
                    messages = split_dialogue(chat_history)

                    # Add arguments that werent defined in schema to the tool
                    for msg in messages:
                        if msg.role == "tool call":
                            tool_call = None
                            if isinstance(msg.content, str):
                                tool_call = json.loads(msg.content)
                            else:
                                tool_call = msg.content
                            
                            add_extra_arguments(tool_call, tools) 

                    
                    return ToolCallData(messages=messages, tools=tools)
                else:
                    for key, value in data.items():
                        if isinstance(value, str):
                            data[key] = json.loads(value)
                    messages = messages_from_list(data["conversation"])
                    if isinstance(data["tools"], str):
                        tools = [
                            json_schema_to_pydantic_tool(tool)
                            for tool in json.loads(data["tools"])
                        ]
                    elif isinstance(data["tools"], list):
                        tools = [Tool(**tool) for tool in data["tools"]]
                    else:
                        raise ValueError(f"Invalid format for tools: {data['tools']}")
                    for tool in tools:
                        for arg_name, arg_value in tool.arguments.items():
                            if arg_value["type"] not in TYPES:
                                raise ValueError(f"Inavlid type used type: {arg_value['type']}")
                    return ToolCallData(messages=messages, tools=tools)
                    
            except Exception as e:
                bt.logging.debug(f"Issue getting tool call from dataset ... {e}")



class LocalToolDataset(SQLDataset):
    def __init__(self, table_name: str = "tools", db_type: str = "sqlite", **kwargs):

        self.columns = ["conversation", "tools"]
        self.table_name = table_name
        super().__init__(
            table_name=self.table_name, columns=self.columns, db_type=db_type, **kwargs
        )
        self._create_table()
        self.shuffle()

    def __next__(self) -> ToolCallData:
        bt.logging.debug("Retrieving function call data from local dataset...")
        tool_data = super().__next__()
        for key, value in tool_data.items():
            tool_data[key] = json.loads(value)
        messages = messages_from_list(tool_data["conversation"])
        if isinstance(tool_data["tools"], str):
            tools = [
                custom_json_schema_to_pydantic_tool(tool)
                for tool in json.loads(tool_data["tools"])
            ]
        elif isinstance(tool_data["tools"], list):
            tools = [Tool(**tool) for tool in tool_data["tools"]]
        else:
            raise ValueError(f"Invalid format for tools: {tool_data['tools']}")
        return ToolCallData(messages=messages, tools=tools)

    def load_data(self, tool_data):
        for key, value in tool_data.items():
            tool_data[key] = json.loads(value)
        messages = messages_from_list(tool_data["conversation"])
        if isinstance(tool_data["tools"], str):
            tools = [
                custom_json_schema_to_pydantic_tool(tool)
                for tool in json.loads(tool_data["tools"])
            ]
        elif isinstance(tool_data["tools"], list):
            tools = [custom_json_schema_to_pydantic_tool(tool) for tool in tool_data["tools"]]
        else:
            raise ValueError(f"Invalid format for tools: {tool_data['tools']}")
        return ToolCallData(messages=messages, tools=tools)

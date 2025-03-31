import re
import json
import random
import bittensor as bt
from pydantic import BaseModel
from typing import List, Dict, Any
from collections.abc import Iterator
from bitagent.schemas.tool import Tool
from bitagent.schemas.chat import ChatMessage, messages_from_list
from bitagent.datasources.loaders import huggingface_loader, load_bfcl_dataset
from bitagent.helpers.string_parse import parse_multiple_space_sep_json


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


def cycle_hf_dataset(dataset, seed=572343):
    while True:
        ds_shuf = dataset.shuffle(seed=seed)
        for item in ds_shuf:
            yield item

class ToolDataset(Iterator):
    def __init__(self, task_dataset_flag=False):
        super().__init__()
        # Always load the "BitAgent/tool_shuffle_small" dataset
        seed = 572343
        bitagent_ds = huggingface_loader("BitAgent/tool_shuffle_small")
        # Wrap it in an infinite-cycle generator
        if task_dataset_flag:
            self.bitagent_iter = iter(bitagent_ds)
        else:
            self.bitagent_iter = cycle_hf_dataset(bitagent_ds, seed=seed)

    def __next__(self) -> ToolCallData:
        count = 0
        while count < 25:
            count += 1
            try:
                # Always pull from "bitagent"
                data = next(self.bitagent_iter)

                # Convert any string columns with JSON content into Python objects
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

                # Validate argument types
                for tool in tools:
                    for arg_name, arg_value in tool.arguments.items():
                        if arg_value["type"] not in TYPES:
                            raise ValueError(f"Inavlid type used type: {arg_value['type']}")

                return ToolCallData(messages=messages, tools=tools)

            except Exception as e:
                bt.logging.debug(f"Issue getting tool call from dataset ... {e}")
                pass

        # If we tried 25 times and still haven't returned, raise StopIteration
        raise StopIteration("Unable to retrieve a valid ToolCallData after 25 attempts.")
import bittensor as bt
from typing import Dict, Any, List
from pydantic import ValidationError
from bitagent.schemas.tool import Tool, ToolCall
from bitagent.schemas.chat import ChatMessage, messages_to_list

# Mapping from type strings to Python types
type_mapping = {
    "str": str,
    "int": int,
    "dict": Dict,
    "list": List,
    "float": float,
    "bool": bool,
    "string": str,
    "integer": int,
    "number": (int, float),  # Allow both int and float for 'number'
    "boolean": bool,
    "array": List,
    "dictionary": Dict,
    "object": Dict,  # Handle nested objects as dictionaries
}

def validate_tool_call(tool: Tool, tool_call: Dict[str, Any]) -> bool:
    try:
        # Validate the tool call structure
        tool_call_validated = ToolCall(**tool_call)

        # Check if the tool call name matches the tool name
        if tool_call_validated.name != tool.name:
            #bt.logging.warning(f"Tool name mismatch: {tool_call_validated.name} != {tool.name}")
            return False
        
        if len(tool_call_validated.arguments.keys()) < len([argname for argname, argdict in tool.arguments.items() if argdict['required']]) or len(tool_call_validated.arguments.keys()) > len([argname for argname, argdict in tool.arguments.items()]):
            #bt.logging.warning(f"Argument length mismatch")
            return False

        # Check arguments
        for arg_name, arg_schema in tool.arguments.items():
            if arg_schema['required'] and arg_name not in tool_call_validated.arguments:
                #bt.logging.warning(f"Missing required argument: {arg_name}")
                return False
            if arg_name in tool_call_validated.arguments:
                expected_type = type_mapping.get(arg_schema['type'])
                if expected_type is None:
                    #bt.logging.warning(f"Unknown type for argument {arg_name}: {arg_schema['type']}")
                    return False
                
                # Handle nested objects
                if 'is_ground_truth' in list(tool_call.keys()) and tool_call['is_ground_truth']:
                    arg_value = tool_call_validated.arguments[arg_name][-1]
                else:
                    arg_value = tool_call_validated.arguments[arg_name]
                # convert int to float if expected type is float
                if expected_type == float and type(arg_value) == int:
                    arg_value = float(arg_value)
                # convert str to float if expected type is float
                if expected_type == float and type(arg_value) == str:
                    arg_value = float(arg_value)
                # convert str to int if expected type is int
                if expected_type == int and type(arg_value) == str:
                    arg_value = int(arg_value)

                if expected_type == dict:
                    if not isinstance(arg_value, dict):
                        #bt.logging.warning(f"""Argument {arg_name} has incorrect type. 
                        #                    Expected {expected_type}, got {type(arg_value)}""")
                        return False
                else:
                    if not isinstance(arg_value, expected_type):
                        #bt.logging.warning(f"""Argument {arg_name} has incorrect type. 
                        #                    Expected {expected_type}, got {type(arg_value)}""")
                        return False
        
        # All checks passed
        return True
    except ValidationError as e:
        #bt.logging.warning(f"Validation error: {e}")
        return False

def find_first_tool_call(messages: List[ChatMessage]):
    for msg in messages:
        if msg.role == 'tool call':
            return msg
    
def find_msgs_before_tool_call(messages: List[ChatMessage]):
    result = []
    for msg in messages:
        if msg.role == 'tool call':
            break
        result.append(msg)
    return result
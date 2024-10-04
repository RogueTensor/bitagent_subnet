import bittensor as bt
from typing import Dict, Any, List
from pydantic import ValidationError
from bitagent.schemas.tool import Tool, ToolCall

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
            bt.logging.warning(f"Tool name mismatch: {tool_call_validated.name} != {tool.name}")
            return False
        
        if not len(tool_call_validated.arguments.keys()) >= len([argname for argname, argdict in tool.arguments.items() if argdict['required']]) and not len(tool_call_validated.arguments.keys()) <= len([argname for argname, argdict in tool.arguments.items()]):
            bt.logging.warning(f"Argument length mismatch")
            return False
        
        # Check arguments
        for arg_name, arg_schema in tool.arguments.items():
            if arg_schema['required'] and arg_name not in tool_call_validated.arguments:
                bt.logging.warning(f"Missing required argument: {arg_name}")
                return False
            if arg_name in tool_call_validated.arguments:
                expected_type = type_mapping.get(arg_schema['type'])
                if expected_type is None:
                    bt.logging.warning(f"Unknown type for argument {arg_name}: {arg_schema['type']}")
                    return False
                
                # Handle nested objects
                if expected_type == dict:
                    if not isinstance(tool_call_validated.arguments[arg_name], dict):
                        bt.logging.warning(f"Argument {arg_name} has incorrect type. Expected {expected_type}, got {type(tool_call_validated.arguments[arg_name])}")
                        return False
                else:
                    if not isinstance(tool_call_validated.arguments[arg_name], expected_type):
                        bt.logging.warning(f"Argument {arg_name} has incorrect type. Expected {expected_type}, got {type(tool_call_validated.arguments[arg_name])}")
                        return False
        
        # All checks passed
        return True
    except ValidationError as e:
        bt.logging.warning(f"Validation error: {e}")
        return False
from bitagent.schemas.chat import ChatMessage, messages_to_list
from typing import List

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


# TODO this converts from type to dict, keep it typed
def find_assistant_after_tool_call(messages: List[ChatMessage]):
    found_tool_call = False
    for d in messages_to_list(messages):
        if not found_tool_call:
            if d.get('role') == 'tool call':
                found_tool_call = True
        elif d.get('role') == 'assistant':
            return d
    return None  # If no matching dictionary is found after the 'tool call'


def find_last_assistant(messages: List[ChatMessage]):
    for d in reversed(messages_to_list(messages)):
        if d.get('role') == 'assistant':
            return d
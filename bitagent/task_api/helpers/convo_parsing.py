from bitagent.schemas.conversation import Conversation


def find_first_tool_call(convo: Conversation):
    for msg in convo.messages:
        if msg.role == 'tool call':
            return msg
    
def find_msgs_before_tool_call(convo: Conversation):
    result = []
    for msg in convo.messages:
        if msg.role == 'tool call':
            break
        result.append(msg)
    return result

def find_assistant_after_tool_call(convo: Conversation):
    found_tool_call = False
    for d in convo.to_list():
        if not found_tool_call:
            if d.get('role') == 'tool call':
                found_tool_call = True
        elif d.get('role') == 'assistant':
            return d
    return None  # If no matching dictionary is found after the 'tool call'


def find_last_assistant(convo: Conversation):
    for d in reversed(convo.to_list()):
        if d.get('role') == 'assistant':
            return d
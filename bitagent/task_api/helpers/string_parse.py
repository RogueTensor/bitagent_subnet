import re
import json

def extract_text_inside_quotes(s):
    match = re.search(r'"(.*?)"', s)
    if match:
        return match.group(1)  # Returns the text inside the first pair of double quotes
    else:
        return s  # Returns the original string if no double quotes are found

def parse_multiple_space_sep_json(json_str):
    """
    Parses a string containing multiple JSON objects separated by whitespace.
    
    {} {} -> [{},{}]
    """
    results = []
    start = 0
    json_str = json_str.strip()  # Remove leading and trailing whitespace
    while start < len(json_str):
        # Find the start of a JSON object
        start = json_str.find('{', start)
        if start == -1:  # No more JSON object
            break
        try:
            obj, index = json.JSONDecoder().raw_decode(json_str[start:])
            results.append(obj)
            start += index
            while start < len(json_str) and json_str[start] in ' \t\n\r':  # Skip whitespace
                start += 1
        except json.JSONDecodeError:
            # Move start forward and try again
            start += 1
    return results

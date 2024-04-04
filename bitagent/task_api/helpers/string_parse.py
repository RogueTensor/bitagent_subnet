import re

def extract_text_inside_quotes(s):
    match = re.search(r'"(.*?)"', s)
    if match:
        return match.group(1)  # Returns the text inside the first pair of double quotes
    else:
        return s  # Returns the original string if no double quotes are found

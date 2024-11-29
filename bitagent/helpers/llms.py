# The MIT License (MIT)
# Copyright © 2024 RogueTensor

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
from openai import OpenAI

# specifically for the validator
def get_openai_llm(self, hugging_face=False):
    if "validator" in self.__class__.__name__.lower() and hugging_face and self.config.validator_hf_server_port:
        # stand up a vLLM server on this port for the OFFLINE HF model evals
        base_url = f'http://localhost:{self.config.validator_hf_server_port}/v1'
    else:
        base_url = self.config.openai_api_base

    return OpenAI(
        api_key=self.config.openai_api_key,
        base_url=base_url
    )

def system_prompt(tools):
    prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
    If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
    You should only return the function call in tools call sections.

    If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1="params_string_value1", params_name2=params_value2...), func_name2(params)]
    Notice that any values that are strings must be put in quotes like this: "params_string_value1"
    You SHOULD NOT include any other text in the response.
    Here is a list of functions in JSON format that you can invoke.\n{functions}\n
    """

    return prompt.format(functions=tools)


def llm(self, messages, tools, model_name, hugging_face=False,max_new_tokens = 160, temperature=0.7):
    prompt = system_prompt(tools)

    try:
        #try:
        #    new_messages = [{"role":"system", "content":prompt}] + messages
        #    response = get_openai_llm(self, hugging_face).chat.completions.create(
        #        messages=new_messages,
        #        max_tokens=max_new_tokens,
        #        model=model_name,
        #        temperature=temperature
        #    )
        #except Exception as e:
            # errored b/c the model does not allow system prompts
        messages[0].content = prompt + "\n\n" + messages[0].content
        response = get_openai_llm(self, hugging_face).chat.completions.create(
            messages=messages,
            max_tokens=max_new_tokens,
            model=model_name,
            temperature=temperature
        )

    except Exception as e:
        bt.logging.error(f"Error calling to LLM: {e}")
        return ""

    if hugging_face:
        return response.choices[0].message.content.strip(), response.choices[0].finish_reason
    else:
        return response.choices[0].message.content.strip()
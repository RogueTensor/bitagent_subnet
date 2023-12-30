from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

device = "cuda" # the device to load the model onto

llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(llm_model)
tokenizer = AutoTokenizer.from_pretrained(llm_model)

def prompt_llm(prompt: str, context: str) -> str:
    messages = __build_messages(prompt, context)
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]

def __build_messages(prompt: str, context: str) -> List:
    # TODO make these messages configurable
    user_entry_context = "Hi, you're super helpful and I appreciate all the support!"
    ai_assistant_context = "I am an AI assistant and my top priority is achieving user fulfillment by helping you with your requests."
    if context:
        context_lead_in = "Given the following content:\n"
        prompt_lead_in = "\n\nProvide the user with an answer to this question:\n"
    else:
        # we may not have context if either a) urls were not passed as data or b) relevant context does not match
        context_lead_in = ""
        prompt_lead_in = ""
        context = ""

    query_prompt = context_lead_in + context + prompt_lead_in + prompt

    return [
        {"role": "user", "content": user_entry_context},
        {"role": "assistant", "content": ai_assistant_context},
        {"role": "user", "content": query_prompt}
    ]


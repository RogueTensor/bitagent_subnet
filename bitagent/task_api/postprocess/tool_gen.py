import json
import bittensor as bt
from typing import List
from common.base.validator import BaseValidatorNeuron
from bitagent.task_api.criteria.utils import good_message, bad_message, received_reward_template

def store_gen_tool(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict):
    # need to store in the database the tool that was generated
    validator.local_tool_gen_dataset.add_item({"conversation": json.dumps([{"role": "user", "content": synapse.prompt}]), "tools": json.dumps([json.loads(synapse.response['response'])])})
    
    
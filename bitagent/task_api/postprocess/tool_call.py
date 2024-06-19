import json
import bittensor as bt
from typing import List
from common.base.validator import BaseValidatorNeuron
from bitagent.task_api.criteria.utils import good_message, bad_message, received_reward_template
from bitagent.schemas.conversation import Conversation

def store_tool_call(task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response:dict):
    convo = [{"role": "user", "content": task.synapse.prompt}] + json.loads(synapse.response['response'])  
    validator.local_tool_call_dataset.add_item({"conversation":json.dumps(convo),"tools":json.dumps([dict(task.synapse.tools[0])])})
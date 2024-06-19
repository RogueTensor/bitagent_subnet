from typing import Callable, List
import bittensor as bt
from common.base.validator import BaseValidatorNeuron
from bitagent.task_api.postprocess.tool_gen import *
from bitagent.task_api.postprocess.tool_call import *

class PostProcessor():
    name: str
    desc: str
    func: Callable

    def __init__(self, name: str, desc: str, func: Callable, func_args=[]) -> None:
        self.name = name
        self.desc = desc
        self.func = func
        self.func_args = func_args
        
    def __call__(self, task, validator: BaseValidatorNeuron, synapse: bt.Synapse, response: dict={}):
        return self.func(task, validator, synapse, response, *self.func_args)

    @classmethod
    def fromSerialized(cls, serialized):
        return cls(
            name=serialized["name"],
            desc=serialized["desc"],
            func=eval(serialized["func"]),
            func_args=serialized["func_args"]
        )
    
    def serialize(self):
        return {
            "name": self.name,
            "desc": self.desc,
            "func": self.func.__name__,
            "func_args": self.func_args
        }



def tool_gen_postprocess() -> List[PostProcessor]:
    return [
        PostProcessor("store_gen_tool", "store the generated tool in the database", store_gen_tool),
    ]

def tool_call_postprocess() -> List[PostProcessor]:
    return [
        PostProcessor("store_tool_call", "store the tool call in the database", store_tool_call),
    ]
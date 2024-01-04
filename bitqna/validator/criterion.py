import bittensor as bt
from pprint import pformat
from typing import Callable

class Criterion():
    name: str
    desc: str
    eval_fx: Callable

    def __init__(self, name: str, desc: str, eval_fx: Callable, eval_args=[]) -> None:
        self.name = name
        self.desc = desc
        self.eval_fx = eval_fx
        self.eval_args = eval_args

    def evaluate(self, response: bt.Synapse) -> float:
        return self.eval_fx(response, *self.eval_args)

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

# TODO we can provide log info to the miner as to what they missed
def does_not_error(response: bt.Synapse) -> float:
    a_status_code = response.axon.status_code
    d_status_code = response.dendrite.status_code
    if a_status_code == "200" and d_status_code == "200":
        return 1.0 
    return 0.0

def does_not_take_a_long_time(response: bt.Synapse) -> float:
    process_time = response.dendrite.process_time
    if process_time <= 2.0: return 1.00 
    if process_time <= 5.0: return 0.75
    if process_time <= 10.0: return 0.50
    if process_time <= 15.0: return 0.25
    return 0.00

def contains_citations(response: bt.Synapse, min_citations=1, max_citations=None) -> float:
    try:
        citations = response.response['citations']
    except KeyError:
        return 0.0

    if (not min_citations or len(citations) >= min_citations) and (not max_citations or len(citations) <= max_citations):
        return 1.0
    return 0.0
    

default_criteria = [
    Criterion(name="Does not error", desc="", eval_fx=does_not_error), 
    Criterion(name="Does not take a long time", desc="", eval_fx=does_not_take_a_long_time),
]

# basic CITATION checks
basic_citations = Criterion(name="Must have at least one citation", desc="", eval_fx=contains_citations, eval_args=[1, None])
basic_no_citations = Criterion(name="Must not return any citations", desc="", eval_fx=contains_citations, eval_args=[0, 0])

"""
{   'axon': TerminalInfo(status_code=200, status_message='Success', process_time=None, ip='..', port=.., version=650, nonce=.., uuid='..-..-..-..-..', hotkey='..', signature='0x..'),
    'computed_body_hash': '',
    'dendrite': TerminalInfo(status_code=200, status_message='Success', process_time=0.026878833770751953, ip='..', port=.., version=650, nonce=.., uuid='..', hotkey='..', signature='0x..'),
    'header_size': 0,
    'name': 'QnAProtocol',
    'prompt': ...
    'required_hash_fields': [   ],
    'response': {   'citations': [   ],
                    'response': ...  '},
    'timeout': ...,
    'total_size': ..,
    'urls': [   '..']}

"""

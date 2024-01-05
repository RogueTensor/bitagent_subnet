import bittensor as bt
from pprint import pformat
from typing import Callable,List
from template.base.validator import BaseValidatorNeuron

class Criterion():
    name: str
    desc: str
    eval_fx: Callable

    def __init__(self, name: str, desc: str, eval_fx: Callable, eval_args=[]) -> None:
        self.name = name
        self.desc = desc
        self.eval_fx = eval_fx
        self.eval_args = eval_args

    def evaluate(self, validator: BaseValidatorNeuron, response: bt.Synapse) -> float:
        return self.eval_fx(validator, response, *self.eval_args)

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

# TODO we can provide log info to the miner as to what they missed
def does_not_error(validator: BaseValidatorNeuron, response: bt.Synapse) -> float:
    a_status_code = response.axon.status_code
    d_status_code = response.dendrite.status_code
    if a_status_code == "200" and d_status_code == "200":
        return 1.0 
    return -0.1

def does_not_take_a_long_time(validator: BaseValidatorNeuron, response: bt.Synapse) -> float:
    process_time = response.dendrite.process_time
    if process_time <= 2.0: return 1.00 
    if process_time <= 5.0: return 0.75
    if process_time <= 10.0: return 0.50
    if process_time <= 15.0: return 0.25
    return -0.1

def correct_citation_format(validator: BaseValidatorNeuron, response: bt.Synapse) -> float:
    try:
        citations = response.response['citations']
        sources = [c['source'] for c in citations]
        contexts = [c['context'] for c in citations]
    except KeyError:
        # no citations provided and no placeholder available
        return -2.0

    if len(sources) > 0 and len(contexts) == len(sources):
        return 1.0

    return 0.0

def contains_number_citations(validator: BaseValidatorNeuron, response: bt.Synapse, min_citations=1, max_citations=None) -> float:
    try:
        citations = response.response['citations']
    except KeyError:
        # no citations provided and no placeholder available
        return -0.5

    if (not min_citations or len(citations) >= min_citations) and (not max_citations or len(citations) <= max_citations):
        return 1.0
    # placeholder for citations, but none
    return -0.25


def contains_correct_number_of_citation_sources(validator: BaseValidatorNeuron, response: bt.Synapse, selected_datas: List[dict]=[], selected_urls: List[str]=[]) -> float:
    try:
        citations = response.response['citations']
        # TODO may need to do unique here
        sources = set([c['source'] for c in citations])
    except KeyError:
        # no citations provided and no placeholder available
        # but not being evaluated by this criterion (see contains_number_citationS)
        return 0.0

    # must cite correct citation urls
    score = 0.0

    selected_sources = []
    for data in selected_datas:
        selected_sources.append(data['source'])

    for url in selected_urls:
        selected_sources.append(url)

    for source in sources:
        if source in selected_sources:
            score += 1.5/len(selected_sources)

    return score

# TODO more general capability for testing validity of response
def correct_response_provided(validator: BaseValidatorNeuron, response: bt.Synapse, selected_datas: List[dict]) -> float:
    try:
        citations = response.response['citations']
        cited_sources = [c['source'] for c in citations]
        cited_texts = [c['context'] for c in citations]
        prompt = response.prompt
        sources = [d['source'] for d in response.datas]
        texts = [d['context'] for d in response.datas]
        completion = response.response['response']
    except KeyError:
        # no citations provided and no placeholder available or maybe something wrong with the data sources
        # but not being evaluated by this criterion (see contains_number_citations)
        return 0.0

    # TODO handle case for more selected datas than just 1
    context = selected_datas[0]['context']
    # TODO instead of yes or no, have it score the answer 1-10
    input_text = f"Question: {prompt}\n\nAnswer: {completion}\n\nContext: {context}\n\n Is the answer to the question correct given the provided context, yes or no? Response:"
    yes_or_no = validator.validator_llm(input_text)

    if yes_or_no.strip().lower() == "yes":
        return 1.0

    return 0.0


def gen_data_task_criteria(selected_datas: List[dict], n_expected_citations:int) -> List[Criterion]:
    return [
        Criterion(name=f"Returns expected citation source(s)", desc="", eval_fx=contains_correct_number_of_citation_sources, eval_args=[selected_datas]),
        Criterion(name=f"Returns expected citation chunk/text(s)", desc="", eval_fx=correct_response_provided, eval_args=[selected_datas]),
    ]

    
## URL Stuff, handle later ##
#def url_task_criteria(selected_texts: List[str], selected_urls: List[str], n_expected_citations:int) -> List[Criterion]:
#    return [
#        Criterion(name=f"Returns expected citation url(s)", desc="", eval_fx=contains_correct_citation_urls, eval_args=[selected_urls])
#        # TODO add another criterion that confirms validity of the answer to some extent (correct_citation_provided method?)
#    ]

default_criteria = [
    Criterion(name="Does not error", desc="", eval_fx=does_not_error), 
    Criterion(name="Does not take a long time", desc="", eval_fx=does_not_take_a_long_time),
]

# basic CITATION checks
basic_citations = [
    Criterion(name="Must have at least one citation", desc="", eval_fx=contains_number_citations, eval_args=[1, None]),
    Criterion(name="Must have correct citation format", desc="", eval_fx=correct_citation_format),
]
basic_no_citations = Criterion(name="Must not return any citations", desc="", eval_fx=contains_number_citations, eval_args=[0, 0])

# NOTE this is the content/format of response
"""
{   'axon': TerminalInfo(status_code=200, status_message='Success', process_time=None, ip='..', port=.., version=650, nonce=.., uuid='..-..-..-..-..', hotkey='..', signature='0x..'),
    'computed_body_hash': '',
    'dendrite': TerminalInfo(status_code=200, status_message='Success', process_time=0.026878833770751953, ip='..', port=.., version=650, nonce=.., uuid='..', hotkey='..', signature='0x..'),
    'header_size': 0,
    'name': 'QnAProtocol',
    'prompt': ...
    'datas': ...
    'required_hash_fields': [   ],
    'response': {   'citations': [   ],
                    'response': ...  '},
    'timeout': ...,
    'total_size': ..,
    'urls': [   '..']}

"""

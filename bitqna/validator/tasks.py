import time
import random
import bittensor as bt
from pprint import pformat
from typing import Callable, List
from bitqna.protocol import QnAProtocol
from template.base.validator import BaseValidatorNeuron
from bitqna.validator.criterion import default_criteria, basic_no_citations, basic_citations, Criterion, url_task_criteria

# TODO in addition to dataset data, use faker data too for html gen 

class Task():
    criteria: List[Criterion]
    synapse: QnAProtocol

    def __init__(self, name: str, prompt: str, desc: str = "",
                 urls: List[str] = [], criteria: List[Criterion] = default_criteria) -> None:
        # TODO may be something other than QnAProtocol for the task synapse, so handle that
        self.name=name
        self.desc=desc
        self.criteria=criteria
        self.synapse=QnAProtocol(prompt=prompt, urls=urls)

    def reward(self, validator: BaseValidatorNeuron, response: str) -> float:
        total_score = 0.0
        for criterion in self.criteria:
            total_score += criterion.evaluate(validator, response)
        return total_score

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

# generated task for n_texts of URLs looking for n_expected_citations of relevant citations
class URLTask(Task):
    # TODO use n_expected_citations (i.e., 3 urls were found to be relevant to answer the question) for combined answering
    def __init__(self, validator: BaseValidatorNeuron, name: str, 
                 desc: str = "", n_texts:int = 3, n_expected_citations:int = 1):

        self.name=name + f" for {n_texts} texts and {n_expected_citations} expected citations"
        self.desc=desc
        self.validator=validator

        html_urls, texts = self.generate_random_urls_for_random_data(n_texts=n_texts)
        selected_num=random.randrange(len(texts))
        selected_text = texts[selected_num]
        selected_url = html_urls[selected_num]
        question = self.get_question_for_text(text=selected_text)
    
        # TODO should be more general with the selected texts and urls to include more than just 1
        self.criteria=url_task_criteria(selected_texts=[selected_text], selected_urls=[selected_url], n_expected_citations=n_expected_citations)
        self.synapse=QnAProtocol(prompt=question, urls=html_urls)

    def generate_random_urls_for_random_data(self, n_texts: int = 3) -> [List[str], List[str]]:
        # get n random data
        texts = []
        html_filenames = []
        for _ in range(n_texts):
            text = next(self.validator.dataset)["text"]
            texts.append(text)
            # put these texts under /tmp/<some random hash>
            h = random.getrandbits(128)
            html_filename = f'{time.strftime("%Y%m%d")}.bitqna.testfile.{h}.html'
            html_filenames.append(html_filename)
            f = open(f"/tmp/{html_filename}", "w")
            f.write(text)
            f.close()

        # TODO turn into html file urls with validator IP and port
        # TODO port should be configurable
        return [html_filenames, texts]

    def get_question_for_text(self, text: str) -> str:
        # for the simple model, use less text, truncate
        # TODO find a good size for the t5 model
        # current model takes 512 tokens, roughly 4 chars per token (per google PaLM 2), 2000 leaves us room for the rest of the prompt
        # TODO consider making random section of the text instead of always the first portion, just make sure it comes out to 2000ish
        truc_text = text[:2000]
        input_text = f"TEXT: {truc_text}\n\nProvide a 1-sentence question for the provided TEXT making sure to leverage key words from the TEXT in the question.\n\n Response: "
        response = self.validator.validator_llm(input_text)
        # response is typically: <pad> text</s>
        question = response.replace("<pad>","").replace("</s>","").strip()
        return question

basic_miner_tasks = [
    Task(name="Responds without URL(s)",
         criteria=default_criteria+[basic_no_citations],
         prompt='who is the most famous ghost buster'),
    Task(name="Responds with a single URL",
         urls=["https://en.wikipedia.org/wiki/Ghostbusters"],
         criteria=default_criteria+[basic_citations],
         prompt='who is the most famous ghost buster'),
    Task(name="Responds with at least one citation",
         urls=["https://en.wikipedia.org/wiki/Ghostbusters"],
         criteria=default_criteria+[basic_citations],
         prompt='who is the most famous ghost buster'),
]

def get_random_task(validator: BaseValidatorNeuron) -> Task:
    # TODO there should be other task types like URLTask that are more generative with some notion of ~ ground truth
    return URLTask(validator=validator, name="Responds with correct citation")
    # TODO randomly select between the basic tests and the harder ones
    return random.choice(basic_miner_tasks)

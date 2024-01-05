import threading
import socketserver
from http.server import SimpleHTTPRequestHandler
from bitqna.validator.dataset import Dataset
from template.base.validator import BaseValidatorNeuron
from transformers import T5Tokenizer, T5ForConditionalGeneration


def initiate_validator(self):
    # load a simple LLM for evals
    # TODO coullddddd make configurable
    self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map=self.device)

    def validator_llm(input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(input_ids, max_length=60)
        result = self.tokenizer.decode(outputs[0])
        # response is typically: <pad> text</s>
        result = result.replace("<pad>","").replace("</s>","").strip()
        return result

    self.validator_llm = validator_llm

    # WEB SERVER # TODO - some complications come from this, like having the validator open a port for these queries
    # TODO add another protocol to make sure http server is up and the right tmp file can be accessed during initiation
    # TODO handle any errors like port taken, etc
    # TODO we can come back to this later if we need / want to - this allows the validator to host data on their own for url testing
    # TODO in the meantime we can work with datas instead of urls in the protocol to get desired behavior
    # TODO testing that they can retrieve and handle urls for data may be a valuable eval down the line
    #def create_server():
    #    # TODO launch simple http server with randomly selected text from datasets
    #    # TODO make port configurable
    #    PORT = 8888
    #    handler = SimpleHTTPRequestHandler
    #
    #    # TODO set some sort of director to be configurable
    #    with socketserver.TCPServer(("", PORT), handler) as httpd:
    #        print("Server started at localhost:" + str(PORT))
    #        httpd.serve_forever()
    #
    #threading.Thread(target=create_server).start()
    # 

    # set our dataset for for starter text
    self.dataset = Dataset()


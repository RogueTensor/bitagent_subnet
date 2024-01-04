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
        return result

    self.validator_llm = validator_llm

    # TODO add another protocol to make sure http server is up and the right tmp file can be accessed during initiation
    # TODO handle any errors like port taken, etc
    def create_server():
        # TODO launch simple http server with randomly selected text from datasets
        # TODO make port configurable
        PORT = 8888
        handler = SimpleHTTPRequestHandler

        # TODO set some sort of director to be configurable
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            print("Server started at localhost:" + str(PORT))
            httpd.serve_forever()

    threading.Thread(target=create_server).start()

    # set our dataset for for starter text
    self.dataset = Dataset()


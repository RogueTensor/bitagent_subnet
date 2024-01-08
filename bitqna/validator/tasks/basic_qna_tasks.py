from bitqna.validator.tasks import Task
from bitqna.validator.criterion import default_criteria, basic_no_citations, basic_citations, simple_context_aware

basic_qna_miner_tasks = [
    Task(name="Q&A - Responds with no citations",
         criteria=default_criteria+[basic_no_citations],
         prompt='who is the most famous ghost buster'),
    Task(name="Q&A - Responds with at least one citation",
         datas=[{'source': "simple test", "context":"The most famous ghost buster is Bob."}],
         criteria=default_criteria+basic_citations,
         citation_sources_should_contain="simple test",
         prompt='who is the most famous ghost buster'),
    Task(name="Q&A - Responds with correct citation and data relevant to context",
         datas=[{'source': "simple test", "context":"Frogs are mammals that live in trees and eat bacon."}],
         criteria=default_criteria+basic_citations+[simple_context_aware],
         citation_sources_should_contain="simple test",
         response_should_contain="bacon",
         prompt='What do frogs eat?'),
    Task(name="Q&A - Responds with correct citation and data relevant to context",
         datas=[{'source': "simple test", "context":"Bees are mammals that live in trees and eat bacon."}],
         citation_sources_should_contain="simple test",
         criteria=default_criteria+basic_citations+[simple_context_aware],
         response_should_contain="trees",
         prompt='Where do bees live?'),
]

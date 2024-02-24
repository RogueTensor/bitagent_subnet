# The MIT License (MIT)
# Copyright © 2023 RogueTensor

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from bitagent.validator.tasks import Task
from bitagent.validator.criteria import default_criteria, basic_no_citations, basic_citations, simple_context_aware

basic_qna_miner_tasks = [
    Task(name="Q&A - Responds with no citations",
         criteria=default_criteria+basic_no_citations,
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

# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
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

import torch
from typing import List
from bitagent.validator.tasks import Task
from common.base.validator import BaseValidatorNeuron

def get_rewards(validator: BaseValidatorNeuron, task: Task, responses: List[str], 
                miner_uids: List[int]) -> [torch.FloatTensor, List[str]]:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - task (Task): The task sent to the miner.
    - responses (List[float]): A list of responses from the miner.
    - miner_uids (List[int]): A list of miner UIDs. The miner at a particular index has a response in responses at the same index.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    - results: A list of feedback for the miner
    """
    # Get all the reward results by iteratively calling your reward() function.
    scores = []
    results = []
    for i, response in enumerate(responses):
        miner_uid = miner_uids[i]
        score, max_possible_score, task_results = task.reward(validator, response)
        normalized_score = score/max_possible_score
        scores.append(normalized_score)
        results.append(f"""
[bold]Task: {task.name}[/bold]\n[bold]Results:[/bold]
=====================\n"""+
"\n".join(task_results) + f"""
[bold]Total reward:[/bold] {score}
[bold]Total possible reward:[/bold] {max_possible_score}
[bold]Normalized reward:[/bold] {normalized_score}
---
Stats with this validator:
Your Average Score: {validator.scores[miner_uid]}
Highest Score across all miners on Subnet for this Validator: {validator.scores.max()}
Median Score across all miners on Subnet for this Validator: {validator.scores.median()}""")

    return [torch.FloatTensor(scores).to(validator.device), results]

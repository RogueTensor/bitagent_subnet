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

import os
import torch
import shutil
import bittensor as bt
from typing import List
from rich.console import Console
from bitagent.validator.tasks import Task
from common.base.validator import BaseValidatorNeuron

rich_console = Console()
os.environ["WANDB_SILENT"] = "true"

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

    prompt = task.synapse.prompt
    wandb_basics = {
        "task_name": task.name,
        "prompt": prompt,
        "validator_uid": validator.metagraph.hotkeys.index(validator.wallet.hotkey.ss58_address),
    }
    
    scores = []
    results = []
    for i, response in enumerate(responses):
        miner_uid = miner_uids[i]
        vwandb = None
        # only write out to wandb if the response is successful
        if (response.axon.status_code == 200 or response.dendrite.status_code == 200):
            vwandb = validator.init_wandb(miner_uid, wandb_basics['validator_uid'])

        reward = task.reward(response)
        if len(reward) == 4:
            score, max_possible_score, task_results, correct_answer = reward
        elif len(reward) == 2: # skip it
            bt.logging.error(f"Skipping results for this task b/c Task API rebooted: {reward[1]}")
            scores.append(-10)
            results.append(None)
            continue
        else:
            bt.logging.error(f"Skipping results for this task b/c not enough information")
            scores.append(-10)
            results.append(None)
            continue

        normalized_score = score/max_possible_score
        scores.append(normalized_score)

        # extra transparent details for miners
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
Highest Score across all miners: {validator.scores.max()}
Median Score across all miners: {validator.scores.median()}""")

        # only log to wandb if the response is successful
        if (response.axon.status_code == 200 or response.dendrite.status_code == 200) and vwandb:
            step_log = {
                "completion": response.response,
                "correct_answer": correct_answer,
                "miner_uid": miner_uid,
                "score": score,
                "max_possible_score": max_possible_score,
                "normalized_score": normalized_score,
                "average_score_for_miner_with_this_validator": validator.scores[miner_uid],
                "highest_score_for_miners_with_this_validator": validator.scores.max(),
                "median_score_for_miners_with_this_validator": validator.scores.median(),
                "stake": validator.metagraph.S[miner_uid].item(),
                "trust": validator.metagraph.T[miner_uid].item(),
                "incentive": validator.metagraph.I[miner_uid].item(),
                "consensus": validator.metagraph.C[miner_uid].item(),
                "dividends": validator.metagraph.D[miner_uid].item(),
                "task_results": "\n".join(task_results),
                "dendrite_process_time": response.dendrite.process_time,
                "dendrite_status_code": response.dendrite.status_code,
                "axon_status_code": response.axon.status_code,
                **wandb_basics
            }
            try:
                vwandb.log(step_log)
            except Exception as e:
                bt.logging.error(f"Failed to log to wandb. Error: {e}. It's likely that your network is incorrect which caused validator.init_wandb() to return None.")
                
        # cleanup the wandb directory files so they don't eat up space
        if vwandb:
            try:
                bt.logging.debug("Writing to wandb and cleaning up.")
                vwandb.finish()
                wandb_dir_to_delete = vwandb.dir
                if "files" in wandb_dir_to_delete:
                    wandb_dir_to_delete = wandb_dir_to_delete.split("files")[0]
                if wandb_dir_to_delete and os.path.exists(wandb_dir_to_delete):
                    shutil.rmtree(wandb_dir_to_delete)
            except Exception as e:
                bt.logging.error(f"Failed to delete wandb directory. Error: {e}.")
    
    return [scores, results]
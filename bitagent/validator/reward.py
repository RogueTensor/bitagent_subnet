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
import time
import json
from comet_ml import Experiment
import torch
import shutil
import asyncio
import bittensor as bt
import multiprocessing
from typing import List
from datetime import datetime
from rich.console import Console
from bitagent.validator.tasks import Task
from bitagent.protocol import QnAResult
from common.base.validator import BaseValidatorNeuron

rich_console = Console()

async def send_results_to_miner(validator, result, miner_axon):
    # extra transparent details for miners

    # For generated/evaluated tasks, we send the results back to the miner so they know how they did and why
    # The dendrite client queries the network to send feedback to the miner
    _ = validator.dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=[miner_axon],
        # Construct a query. 
        synapse=QnAResult(results=result),
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        deserialize=False,
        timeout=5.0 # quick b/c we are not awaiting a response
    )

async def evaluate_task(validator, task, response):
    rewards = []
    if validator.config.run_local:
        rewards.append(task.reward(validator, response, response))
        # TOOD take out - but compare first so we can do both levels of validator hosting
        # response
        # timeout=10.0 urls=[] datas=[] prompt="Summarize this and make sure to be concise:  .." 
        # response={'response': "...", 'citations': [], 'context': ''} miner_uids=[]
    else:
        rewards.append(task.reward(validator, response))

    return rewards

async def return_results(validator, task, miner_uid, reward):
    # means we got all of the information we need to score the miner and update wandb
    if len(reward) == 4:
        score, max_possible_score, task_results, correct_answer = reward
        # make sure the score is not None
        if score and max_possible_score:
            normalized_score = score/max_possible_score

            result = f"""
[bold]Task: {task.name}[/bold]\n[bold]Results:[/bold]
=====================\n"""+"\n".join(task_results) + f"""
[bold]Total reward:[/bold] {score}
[bold]Total possible reward:[/bold] {max_possible_score}
[bold]Normalized reward:[/bold] {normalized_score}
---
Stats with this validator:
Your Average Score: {validator.scores[miner_uid]}
Highest Score across all miners: {validator.scores.max()}
Median Score across all miners: {validator.scores.median()}"""

            # send results
            await send_results_to_miner(validator, result, validator.metagraph.axons[miner_uid])

            return task_results
        return None
    elif len(reward) == 2: # skip it
        #bt.logging.debug(f"Skipping results for this task b/c Task API seems to have rebooted: {reward[1]}")
        #time.sleep(25)
        return None
    else:
        #bt.logging.debug(f"Skipping results for this task b/c not enough information")
        #time.sleep(25)
        return None

async def process_rewards_update_scores_and_send_feedback(validator: BaseValidatorNeuron, task: Task, responses: List[str], 
                miner_uids: List[int]) -> None:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - task (Task): The task sent to the miner.
    - responses (List[float]): A list of responses from the miner.
    - miner_uids (List[int]): A list of miner UIDs. The miner at a particular index has a response in responses at the same index.
    """

    # common wandb setup
    prompt = task.synapse.prompt
    log_basics = {
        "task_name": task.name,
        "prompt": prompt,
        "validator_uid": validator.metagraph.hotkeys.index(validator.wallet.hotkey.ss58_address),
        "val_spec_version": validator.spec_version,
        "highest_score_for_miners_with_this_validator": validator.scores.max().item(),
        "median_score_for_miners_with_this_validator": validator.scores.median().item(),
    }
    
    # run these in parallel but wait for the reuslts b/c we need them downstream
    rewards = await asyncio.gather(*[evaluate_task(validator, task, response) for response in responses])

    # track which miner uids are scored for updating the scores
    temp_miner_uids = [miner_uids[i] for i, reward in enumerate(rewards) if len(reward[0]) == 4 and reward[0][0] is not None and reward[0][1] is not None]
    scores = [reward[0][0]/reward[0][1] for reward in rewards if len(reward[0]) == 4 and reward[0][0] is not None and reward[0][1] is not None]

    results = await asyncio.gather(*[return_results(validator, task, miner_uids[i], reward[0]) for i, reward in enumerate(rewards)])

    for i, result in enumerate(results):
        if result is not None:
            response = responses[i]
            miner_uid = miner_uids[i]
            score,max_possible_score,_,correct_answer = rewards[i][0]
            normalized_score = score/max_possible_score
            resp = "None"
            citations = "None"
            try:
                resp = response.response["response"]
                citations = json.dumps(response.response["citations"])
            except:
                pass
            step_log = {
                "completion_response": resp,
                "completion_citations": citations,
                "correct_answer": correct_answer,
                "miner_uid": miner_uids[i].item(),
                "score": score,
                "max_possible_score": max_possible_score,
                "normalized_score": normalized_score,
                "average_score_for_miner_with_this_validator": validator.scores[miner_uid].item(),
                "stake": validator.metagraph.S[miner_uid].item(),
                "trust": validator.metagraph.T[miner_uid].item(),
                "incentive": validator.metagraph.I[miner_uid].item(),
                "consensus": validator.metagraph.C[miner_uid].item(),
                "dividends": validator.metagraph.D[miner_uid].item(),
                "task_results": "\n".join(result),
                "dendrite_process_time": response.dendrite.process_time,
                "dendrite_status_code": response.dendrite.status_code,
                "axon_status_code": response.axon.status_code,
                **log_basics
            }

            if (step_log["axon_status_code"] == 200 or step_log["dendrite_status_code"] == 200):
                validator_network = validator.config.subtensor.network
                validator_netuid = validator.config.netuid
                if validator_network == "test" or validator_netuid == 76: # testnet wandb
                    #bt.logging.debug("Initializing wandb for testnet")
                    project_name = "bitagent-testnet"
                elif validator_network == "finney" or validator_netuid == 20: # mainnet wandb
                    # bt.logging.debug("Initializing wandb for mainnet")
                    project_name = "bitagent-mainnet"
                else: # unknown network, not initializing wandb
                    # bt.logging.debug("Not initializing wandb, unknown network")
                    project_name = None

                if project_name:
                    experiment = Experiment(
                        api_key="x6TeIvmRgto7KhgAeMVJqkZRQ",
                        project_name=project_name,
                        workspace="roguetensor"
                    )
                    experiment.log_parameters(step_log)
                    experiment.end()

    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    miner_uids = torch.tensor(temp_miner_uids)
    validator.update_scores(torch.FloatTensor(scores).to(validator.device), miner_uids)

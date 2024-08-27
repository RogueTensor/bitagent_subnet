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
import json
import asyncio
import requests
import numpy as np
import bittensor as bt
from typing import List, Any
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
        resp = {
            "response": response.response,
            #"prompt": response.prompt,
            #"urls": response.urls,
            #"datas": response.datas,
            #"tools": response.tools,
            #"notes": response.notes,
            "axon_hotkey": response.axon.hotkey,
            "dendrite_process_time": response.dendrite.process_time,
            "dendrite_status_code": response.dendrite.status_code,
            "axon_status_code": response.axon.status_code,
        }
        try:
            rewards.append(task.reward(validator, response, resp))
        except Exception as e:
            bt.logging.warning(f"An exception calling task.reward: {e}")
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
Median Score across all miners: {np.median(validator.scores)}"""
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
                        
async def process_rewards_update_scores_and_send_feedback(validator: BaseValidatorNeuron, task: Task, responses: List[Any], 
                miner_uids: List[int]) -> None:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - task (Task): The task sent to the miner.
    - responses (List[float]): A list of responses from the miner.
    - miner_uids (List[int]): A list of miner UIDs. The miner at a particular index has a response in responses at the same index.
    """
    # common wandb setup
    try:
        prompt = task.synapse.prompt
        messages = task.synapse.messages
        tools = task.synapse.tools
        datas = task.synapse.datas
        files = task.synapse.files
        task_name = task.name
    except Exception as e:
        bt.logging.error("Could not setup common data - ", e)

    # run these in parallel but wait for the reuslts b/c we need them downstream
    rewards = await asyncio.gather(*[evaluate_task(validator, task, response) for response in responses])
    try:
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

                try:
                    task_id = task.task_id
                    data = {
                        "task_id": task_id,
                        "task_name": task_name,
                        "prompt": prompt,
                        "messages": [{'role': m.role, 'content': m.content} for m in messages],
                        "prompt_len": len(prompt),
                        "tools": [{'name': t.name, 'description': t.description, 'arguments': t.arguments} for t in tools],
                        "miners_count": len(miner_uids),
                        "messages_count": len(messages),
                        "tools_count": len(tools),
                        "datas_count": len(datas),
                        "files_count": len(files),
                        "datas": datas,
                        "files": files,
                        "response": resp,
                        "citations": citations,
                        "miner_uid": miner_uids[i],
                        "score": score,
                        "normalized_score": normalized_score,
                        "average_score_for_miner_with_this_validator": validator.scores[miner_uid],
                        "stake": validator.metagraph.S[miner_uid],
                        "trust": validator.metagraph.T[miner_uid],
                        "incentive": validator.metagraph.I[miner_uid],
                        "consensus": validator.metagraph.C[miner_uid],
                        "dividends": validator.metagraph.D[miner_uid],
                        "results": "\n".join(result),
                        "dendrite_process_time": response.dendrite.process_time,
                        "dendrite_status_code": response.dendrite.status_code,
                        "axon_status_code": response.axon.status_code,
                        "validator_uid": validator.metagraph.hotkeys.index(validator.wallet.hotkey.ss58_address),
                        "val_spec_version": validator.spec_version,
                        "highest_score_for_miners_with_this_validator": validator.scores.max(),
                        "median_score_for_miners_with_this_validator": np.median(validator.scores),
                        #"correct_answer": correct_answer, # TODO best way to send this without lookup attack?
                    }

                    try:
                        validator.log_event(data)
                    except Exception as e:
                        bt.logging.debug("WandB failed to log, moving on ... exception: ", e)

                except Exception as e:
                    bt.logging.warning("Exception in log_rewards: {}".format(e))

    except Exception as e:
        bt.logging.warning(f"Error logging reward data: {e}")

    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    miner_uids = temp_miner_uids
    validator.update_scores(scores, miner_uids, alpha=task.weight)

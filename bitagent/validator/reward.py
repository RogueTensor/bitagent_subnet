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
import asyncio
import numpy as np
import bittensor as bt
from typing import List, Any
from rich.console import Console
from bitagent.tasks.task import Task
from bitagent.protocol import QueryResult
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
        synapse=QueryResult(results=result),
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        deserialize=False,
        timeout=5.0 # quick b/c we are not awaiting a response
    )

async def evaluate_task(validator, task, response):
    try:
        return [task.reward(validator, response)]
    except Exception as e:
        bt.logging.warning(f"An exception calling task.reward: {e}")

async def return_results(validator, task, miner_uid, reward, response):
    # means we got all of the information we need to score the miner and update wandb
    if len(reward) == 4:
        score, max_possible_score, task_results, correct_answer = reward
        # make sure the score is not None
        if score and max_possible_score:
            normalized_score = score/max_possible_score

            result = f"""
[bold]Task: {task.name}[/bold]
[bold]Messages:[/bold] {task.synapse.messages}
[bold]Tools:[/bold] {[t.name for t in task.synapse.tools]}
[bold]Response:[/bold] `{response.response}`
\n[bold]Results:[/bold]\n
=====================\n"""+"\n".join(task_results) + f"""
[bold]Total reward:[/bold] {score}
[bold]Total possible reward:[/bold] {max_possible_score}
[bold]Normalized reward:[/bold] {normalized_score}
---
Stats with this validator:
Your Average Score: {validator.scores[miner_uid]}
Highest Score across all miners: {validator.scores.max()}
Median Score across all miners: {np.median(validator.scores)}
Your Offline Model Score for Competition {validator.previous_competition_version}: {validator.offline_scores[validator.previous_competition_version][miner_uid]}
Your Offline Model Score for Competition {validator.competition_version}: {validator.offline_scores[validator.competition_version][miner_uid]}"""
# TODO need to add BFCL scores when we do them
            # send results
            if task.mode == "online":
                await send_results_to_miner(validator, result, validator.metagraph.axons[miner_uid])
            else:
                # useful if validators want to see progress or results of offline tasks
                # rich_console.print("this is a non-online task")
                # rich_console.print(result)
                pass
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

async def write_to_wandb(validator: BaseValidatorNeuron, task: Task, responses: List[Any], miner_uids: List[int], rewards: List[List[float]], results: List[List[str]]) -> None:
    # common wandb setup
    try:
        messages = task.synapse.messages
        tools = task.synapse.tools
        task_name = task.name
        task_mode = task.mode
    except Exception as e:
        bt.logging.error("Could not setup common data - ", e)

    for i in range(len(responses)):
        response = responses[i]
        miner_uid = miner_uids[i]
        score,max_possible_score,_,correct_answer = rewards[i][0]
        normalized_score = score/max_possible_score

        resp = "None"
        try:
            resp = response.response
            run_model = response.hf_run_model_name
        except:
            pass

        try:
            data = {
                "task_name": task_name,
                "task_mode": task_mode,
                "messages": [{'role': m.role, 'content': m.content} for m in messages],
                "tools": [{'name': t.name, 'description': t.description, 'arguments': t.arguments} for t in tools],
                "miners_count": len(miner_uids),
                "messages_count": len(messages),
                "tools_count": len(tools),
                "response": resp,
                "miner_uid": miner_uids[i],
                "score": score,
                "normalized_score": normalized_score,
                "average_score_for_miner_with_this_validator": validator.scores[miner_uid],
                "stake": validator.metagraph.S[miner_uid],
                "trust": validator.metagraph.T[miner_uid],
                "incentive": validator.metagraph.I[miner_uid],
                "consensus": validator.metagraph.C[miner_uid],
                "dividends": validator.metagraph.D[miner_uid],
                "results": "\n".join(str(item) for item in results[i]) if results[i] else "None",
                "dendrite_process_time": response.dendrite.process_time,
                "dendrite_status_code": response.dendrite.status_code,
                "axon_status_code": response.axon.status_code,
                "validator_uid": validator.metagraph.hotkeys.index(validator.wallet.hotkey.ss58_address),
                "val_spec_version": validator.spec_version,
                "highest_score_for_miners_with_this_validator": validator.scores.max(),
                "median_score_for_miners_with_this_validator": np.median(validator.scores),
                "offline_score_for_miner_with_this_validator": validator.offline_scores[validator.competition_version][miner_uid],
                "highest_offline_score_for_miners_with_this_validator": validator.offline_scores[validator.competition_version].max(),
                "median_offline_score_for_miners_with_this_validator": np.median(validator.offline_scores[validator.competition_version]),
                "average_offline_score_for_miners_with_this_validator": np.mean(validator.offline_scores[validator.competition_version]),
                "prior_highest_offline_score_for_miners_with_this_validator": validator.offline_scores[validator.previous_competition_version].max(),
                "prior_median_offline_score_for_miners_with_this_validator": np.median(validator.offline_scores[validator.previous_competition_version]),
                "prior_average_offline_score_for_miners_with_this_validator": np.mean(validator.offline_scores[validator.previous_competition_version]),
                "competition_version": validator.competition_version,
                # TODO add BFCL scores
                #"correct_answer": correct_answer, # TODO best way to send this without lookup attack?
            }

            try:
                #if task.mode == "offline":
                #    bt.logging.debug(f"OFFLINE Logging to WandB")
                #else:
                #    bt.logging.debug(f"ONLINE Logging to WandB")
                validator.log_event(data)
                #if task.mode == "offline":
                #    bt.logging.debug(f"OFFLINE Logged to WandB")
                #else:
                #    bt.logging.debug(f"ONLINE Logged to WandB")
            except Exception as e:
                bt.logging.warning("WandB failed to log, moving on ... exception: {}".format(e))

        except Exception as e:
            bt.logging.warning("Exception in logging to WandB: {}".format(e))

# all of these miners are scored the same way with the same tasks b/c this is scoring offline models
async def process_rewards_update_scores_for_many_tasks_and_many_miners(
    validator: BaseValidatorNeuron, tasks: List[Task], responses: List[Any], 
    miner_uids: List[int], wandb_data: dict
) -> None:
    # Gather rewards in parallel
    rewards = await asyncio.gather(*[
        evaluate_task(validator, tasks[i], responses[i]) for i in range(len(responses))
    ])

    try:
        scores = []
        miner_tasks = []  # Collect tasks to execute in parallel for each miner
        for i, reward in enumerate(rewards):
            if len(reward[0]) == 4 and reward[0][0] is not None and reward[0][1] is not None:
                scores.append(reward[0][0] / reward[0][1])

                # Create a coroutine chain for each miner_uid
                for miner_uid in miner_uids:
                    async def process_miner_task(task_idx, miner_uid, reward, response):
                        # Get the result for this miner
                        result = await return_results(validator, tasks[task_idx], miner_uid, reward[0], response)
                        # Write the result to wandb
                        await write_to_wandb(validator, tasks[task_idx], [response], [miner_uid], rewards, result)
                    
                    # Append the task for execution
                    miner_tasks.append(process_miner_task(i, miner_uid, reward, responses[i]))
            else:
                # Bad reward, so 0 score
                scores.append(0.0)

        # Await all miner-specific tasks concurrently
        await asyncio.gather(*miner_tasks)

    except Exception as e:
        bt.logging.warning(f"OFFLINE: Error logging reward data: {e}")
        wandb_data['event_name'] = "Processing Rewards - Error"
        wandb_data['miner_uids'] = miner_uids
        wandb_data['error'] = e
        validator.log_event(wandb_data)
        wandb_data.pop('error')
        wandb_data.pop('miner_uids')

    # Compute and log the mean score
    score = np.mean(scores)
    wandb_data['event_name'] = "Processing Rewards - Score"
    wandb_data['score'] = score
    wandb_data['miner_uids'] = miner_uids
    validator.log_event(wandb_data)
    wandb_data.pop('score')
    wandb_data.pop('miner_uids')

    # Update scores
    validator.update_offline_scores([score] * len(miner_uids), miner_uids)

    return score

async def process_rewards_update_scores_and_send_feedback(validator: BaseValidatorNeuron, task: Task, responses: List[Any], 
                miner_uids: List[int]) -> None:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - task (Task): The task sent to the miner.
    - responses (List[float]): A list of responses from the miner.
    - miner_uids (List[int]): A list of miner UIDs. The miner at a particular index has a response in responses at the same index.
    """
    # run these in parallel but wait for the reuslts b/c we need them downstream
    rewards = await asyncio.gather(*[evaluate_task(validator, task, response) for response in responses])
    try:
        # track which miner uids are scored for updating the scores
        #temp_miner_uids = [miner_uids[i] for i, reward in enumerate(rewards) if len(reward[0]) == 4 and reward[0][0] is not None and reward[0][1] is not None]
        scores = []
        results = []
        for i, reward in enumerate(rewards):
            if len(reward[0]) == 4 and reward[0][0] is not None and reward[0][1] is not None:
                scores.append(reward[0][0]/reward[0][1])
                results.append(await return_results(validator, task, miner_uids[i], reward[0], responses[i]))
            else:
                # bad reward, so 0 score
                scores.append(0.0)
                results.append(None)

        await write_to_wandb(validator, task, responses, miner_uids, rewards, results)

    except Exception as e:
        bt.logging.warning(f"ONLINE: Error logging reward data: {e}")

    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    #miner_uids = temp_miner_uids
    validator.update_scores(scores, miner_uids, alpha=task.weight)

    return scores
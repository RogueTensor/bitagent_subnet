import os
import shutil
import asyncio

from sglang.utils import ( # type: ignore
    execute_shell_command,
    wait_for_server,
    terminate_process)

import bittensor as bt
from bitagent.helpers.llms import llm
from common.utils.uids import get_alive_uids
from bitagent.tasks.task import get_random_task
from bitagent.protocol import GetHFModelName
from bitagent.validator.reward import process_rewards_update_scores_for_many_tasks_and_many_miners

# Delete the model from the huggingface cache when we're done serving it so we don't run out of disk space
def delete_model_from_hf_cache(self, model_name: str):
    # Determine the cache directory
    cache_dir = os.path.expanduser(self.config.validator_hf_cache_dir)
    
    # Format the directory name based on the model name
    model_cache_dir = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    
    # Check if the directory exists and delete it
    if os.path.exists(model_cache_dir):
        try:
            shutil.rmtree(model_cache_dir)
            bt.logging.debug(f"OFFLINE: Model '{model_name}' has been removed from the cache.")
        except Exception as e:
            bt.logging.error(f"OFFLINE: Error deleting model: '{model_name}' from HF cache: {e}")
    else:
        bt.logging.debug(f"OFFLINE: Model '{model_name}' not found in the cache: {model_cache_dir}")

# ###########################################################
# OFFLINE TASKING
# ###########################################################

# TODO also run the bfcl suite on the validator - but skip the API calls, don't use those at first
# TODO store TOP score from last round and all-time in validator state
async def offline_task(self):
    bt.logging.debug("OFFLINE: Starting offline task")
    self.running_offline_mode = True
    # get all alive miner UIDs to compare against the top scores from the last round
    miner_uids = await asyncio.to_thread(get_alive_uids, self)

    # TODO potentially fetch prompt template from miner too
    # Grab all the models that the miners submitted
    responses = await self.dendrite.forward(
        axons=[self.metagraph.axons[miner_uid] for miner_uid in miner_uids],
        synapse=GetHFModelName(),
        deserialize=False,
        timeout=5.0,
    )

    # TODO check status codes of the responses and score accordingly

    # get all the HF model names from the responses
    miner_hf_model_names = [response.hf_model_name for response in responses]
    bt.logging.debug(f"OFFLINE: Miner HF model names: {miner_hf_model_names}")

    hf_model_name_to_miner_uids = {}
    for i,miner_uid in enumerate(miner_uids):
        if responses[i].hf_model_name is not None:
            if responses[i].hf_model_name not in hf_model_name_to_miner_uids:
                hf_model_name_to_miner_uids[responses[i].hf_model_name] = []
            hf_model_name_to_miner_uids[responses[i].hf_model_name].append(miner_uid)

    # Group all the models together uniquely and share the same inference server
    unique_miner_hf_model_names = [m for m in list(set(miner_hf_model_names)) if m not in [None, ""]]
    bt.logging.debug(f"OFFLINE: Unique miner HF model names: {unique_miner_hf_model_names}")

    if len(unique_miner_hf_model_names) > 0:
        bt.logging.debug(f"OFFLINE: Generating tasks")
        # Generate a set of tasks to run on all the offline models
        tasks = []
        for _ in range(1000):
            task = await asyncio.to_thread(get_random_task, self)
            task.mode = "offline"
            tasks.append(task)

    bt.logging.debug(f"OFFLINE: Generated {len(tasks)} tasks")
    for hf_model_name in unique_miner_hf_model_names:
        bt.logging.debug(f"OFFLINE: Running tasks for model {hf_model_name}")
        if hf_model_name is None or hf_model_name == "" or hf_model_name.lower() == "none":
            bt.logging.debug(f"OFFLINE: Miner returned empty HF model name ... skipping")
            for miner_uid in hf_model_name_to_miner_uids[hf_model_name]:
                self.offline_scores[miner_uid] = 0.0
            continue # skip this model

        # TODO check size of model on disk - if larger than 8B params, then don't run and set score to 0
        #try:
        #    if model.num_parameters() > 10000000000:
        #        self.offline_scores[miner_uid] = 0.0
        #        continue
        #except:
        #    try:
        #        if sum(p.numel() for p in model.parameters()) > 10000000000:
        #            self.offline_scores[miner_uid] = 0.0
        #            continue
        #    except:
        #        self.offline_scores[miner_uid] = 0.0
        #        continue
        bt.logging.debug(f"OFFLINE: Starting server for model {hf_model_name}")
        try:
            # Start the server for the model
            server_process = await asyncio.to_thread(execute_shell_command,
            f"""
            python -m sglang.launch_server --model-path {hf_model_name} \
            --port {self.config.validator_hf_server_port} --host 0.0.0.0
            --mem-fraction-static 0.45
            """
            )
            await asyncio.to_thread(wait_for_server, f"http://localhost:{self.config.validator_hf_server_port}")
        except Exception as e:
            bt.logging.error(f"OFFLINE: Error starting sglang server for model: {hf_model_name}: {e}")
            # TODO determine if this is a problem with the model or the server
            # right now assuming problem with the model
            for miner_uid in hf_model_name_to_miner_uids[hf_model_name]:
                self.offline_scores[miner_uid] = 0.0
            continue

        # get LLM responses
        bt.logging.debug(f"OFFLINE: Getting LLM responses for model {hf_model_name}")
        llm_responses = await asyncio.gather(
            *[asyncio.to_thread(llm, self, task.synapse.messages, task.synapse.tools, hf_model_name, hugging_face=True)
              for task in tasks]
        )
        bt.logging.debug(f"OFFLINE: Got {len(llm_responses)} LLM responses for model: {hf_model_name}")

        # terminate the server after getting all the responses
        bt.logging.debug(f"OFFLINE: Terminating server for model: {hf_model_name}")
        await asyncio.to_thread(terminate_process, server_process)
        bt.logging.debug(f"OFFLINE: Terminated server for model: {hf_model_name}")

        these_miner_uids = hf_model_name_to_miner_uids[hf_model_name]
        responses = []
        for i, llm_response in enumerate(llm_responses):
            task = tasks[i]
            response = task.synapse.model_copy()
            response.response = llm_response.strip()
            response.dendrite.process_time = 5.0 # TODO may be useful to test performance of the model itself
            response.dendrite.status_code = 200 
            response.axon.status_code = 200
            response.hf_run_model_name = hf_model_name
            responses.append(response)

        # evaluate, track score and add to wandb
        # TODO need to see if this SCORE is higher than the all-time top score
        # if so, update the all-time top score and model name and reward TOP miners
        # if not, then temporal decay of scores
        bt.logging.debug(f"OFFLINE: Processing rewards for model: {hf_model_name}, for miners: {these_miner_uids}")
        await process_rewards_update_scores_for_many_tasks_and_many_miners(self, tasks=tasks, responses=responses, miner_uids=these_miner_uids)
    
        # remove old files from HF cache
        bt.logging.debug(f"OFFLINE: Deleting model from HF cache: {hf_model_name}")
        await asyncio.to_thread(delete_model_from_hf_cache, self, hf_model_name)
        # TODO handle temporal decay of scores if no miners outperform all time top score
        # TODO handle temporal decay of all scores depending on a) if no new TOP score and b) if new TOP score
        # TODO if doing tier emissions - check wandb for the TOP score and what associated model name it is - could do that here

    bt.logging.debug(f"OFFLINE: Finished processing offline tasks")
    self.running_offline_mode = False
import os
import shutil
import asyncio

from sglang.utils import ( # type: ignore
    execute_shell_command,
    wait_for_server,
    terminate_process)

import bittensor as bt
from bitagent.helpers.llms import llm
from huggingface_hub import model_info
from common.utils.uids import get_alive_uids
from bitagent.tasks.task import get_random_task
from bitagent.protocol import GetHFModelName
from bitagent.validator.reward import process_rewards_update_scores_for_many_tasks_and_many_miners

# TODO overall for tracking, would be nice to track based on hotkey instead of UID
# it's currently handled for uid and new hotkeys taking over a uid, but might be cleaner

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

    # get all the HF model names from the responses
    miner_hf_model_names = [response.hf_model_name for response in responses]
    bt.logging.debug(f"OFFLINE: Miner HF model names: {miner_hf_model_names}")

    hf_model_name_to_miner_uids = {}
    for i,miner_uid in enumerate(miner_uids):
        if responses[i].hf_model_name is not None:
            if responses[i].hf_model_name not in hf_model_name_to_miner_uids:
                hf_model_name_to_miner_uids[responses[i].hf_model_name] = []
            hf_model_name_to_miner_uids[responses[i].hf_model_name].append(int(miner_uid))

    # Group all the models together uniquely and share the same inference server
    unique_miner_hf_model_names = [m for m in list(set(miner_hf_model_names)) if m not in [None, ""]]
    if len(unique_miner_hf_model_names) == 0:
        for miner_uid in hf_model_name_to_miner_uids[hf_model_name]:
            self.offline_scores[self.competition_version][miner_uid] = 0.0
            self.offline_miners_scored[self.competition_version].append(int(miner_uid))
        bt.logging.debug(f"OFFLINE: No unique miner HF model names to evaluate in OFFLINE mode")
        return

    bt.logging.debug(f"OFFLINE: Unique miner HF model names: {unique_miner_hf_model_names}")

    if len(unique_miner_hf_model_names) > 0:
        bt.logging.debug(f"OFFLINE: Generating tasks")
        # Generate a set of tasks to run on all the offline models
        num_tasks = 1000
        batch_size = 100
        generated_tasks = []
        for _ in range(0, num_tasks, batch_size):
            generated_tasks.append(await asyncio.gather(*[asyncio.to_thread(get_random_task, self) for _ in range(batch_size)]))
        tasks = []
        for task in generated_tasks:
            task.mode = "offline"
            tasks.append(task)
        bt.logging.debug(f"OFFLINE: Generated {len(tasks)} tasks of {num_tasks} total")

    for hf_model_name in unique_miner_hf_model_names:
        bt.logging.debug(f"OFFLINE: Running tasks for model {hf_model_name}")
        if hf_model_name is None or hf_model_name == "" or hf_model_name.lower() == "none":
            bt.logging.debug(f"OFFLINE: Miner returned empty HF model name ... skipping")
            for miner_uid in hf_model_name_to_miner_uids[hf_model_name]:
                self.offline_scores[self.competition_version][miner_uid] = 0.0
                self.offline_miners_scored[self.competition_version].append(int(miner_uid))
            continue # skip this model

        # Extract the model card data for the model from HF
        info = model_info(hf_model_name)
        license = info.card_data['license']
        total_size = info.safetensors.total

        # confirm model license is apache-2.0 or nc-by-nc-4.0 or mit
        # TODO eventually ONLY accept apache-2.0
        if license not in ["apache-2.0", "cc-by-nc-4.0", "mit"]:
            bt.logging.debug(f"OFFLINE: Skipping model {hf_model_name} due to license: {license}")
            for miner_uid in hf_model_name_to_miner_uids[hf_model_name]:
                self.offline_scores[self.competition_version][miner_uid] = 0.0
                self.offline_miners_scored[self.competition_version].append(int(miner_uid))
            continue

        # confirm model size is less than 10B params (want 8B or less models)
        if total_size > 10000000000:
            bt.logging.debug(f"OFFLINE: Skipping model {hf_model_name} due to size: {total_size}")
            for miner_uid in hf_model_name_to_miner_uids[hf_model_name]:
                self.offline_scores[self.competition_version][miner_uid] = 0.0
                self.offline_miners_scored[self.competition_version].append(int(miner_uid))
            continue

        bt.logging.debug(f"OFFLINE: Starting server for model {hf_model_name}")
        try:
            # Start the server for the model
            server_process = await asyncio.to_thread(execute_shell_command,
            f"""
            {os.getcwd()}/.venvsglang/bin/python -m sglang.launch_server --model-path {hf_model_name} \
            --port {self.config.validator_hf_server_port} --host 0.0.0.0 \
            --mem-fraction-static 0.5
            """
            )

            bt.logging.debug(f"OFFLINE: Started server for model {hf_model_name}, waiting for it to start on port {self.config.validator_hf_server_port} (could take several minutes)")
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(wait_for_server, f"http://localhost:{self.config.validator_hf_server_port}"), 
                    timeout=60*10 # wait up to 10 minutes
                )
                bt.logging.debug(f"OFFLINE: Server for model {hf_model_name} started")
            except asyncio.TimeoutError:
                # likely a validator error
                bt.logging.error(f"OFFLINE: Timeout waiting for server for model {hf_model_name} to start")
                # can't score this model, so skipping it for now, the miner will be tried again if this runs again
                continue
            except Exception as e:
                bt.logging.error(f"OFFLINE: Error waiting for server: {e}")

        except Exception as e:
            # likely a validator error
            bt.logging.error(f"OFFLINE: Error starting sglang server for model: {hf_model_name}: {e}")
            # can't score this model, so skipping it for now, the miner will be tried again if this runs again
            # could be an issue with model size
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
            response.competition_version = self.competition_version
            responses.append(response)

        # evaluate, track score and add to wandb
        # TODO need to see if this SCORE is higher than the all-time top score
        # TODO if so, update the all-time top score and model name and reward TOP miners
        # TODO if not, then temporal decay of scores
        bt.logging.debug(f"OFFLINE: Processing rewards for model: {hf_model_name}, for miners: {these_miner_uids}")
        await process_rewards_update_scores_for_many_tasks_and_many_miners(self, tasks=tasks, responses=responses, miner_uids=these_miner_uids)
    
        # remove old files from HF cache
        bt.logging.debug(f"OFFLINE: Deleting model from HF cache: {hf_model_name}")
        await asyncio.to_thread(delete_model_from_hf_cache, self, hf_model_name)
        # TODO handle temporal decay of scores if no miners outperform all time top score
        # TODO handle temporal decay of all scores depending on a) if no new TOP score and b) if new TOP score

    bt.logging.debug(f"OFFLINE: Finished processing offline tasks")
    self.running_offline_mode = False
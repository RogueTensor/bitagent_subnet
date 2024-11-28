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
            bt.logging.debug(f"OFFLINE: Model has been removed from the HF cache.")
        except Exception as e:
            bt.logging.error(f"OFFLINE: Error deleting model: from HF cache: {e}")
    else:
        bt.logging.debug(f"OFFLINE: Model not found in the cache, could not delete")

# ###########################################################
# OFFLINE TASKING
# ###########################################################

# TODO also run the bfcl suite on the validator - but skip the API calls, don't use those at first
# TODO store TOP score from last round and all-time in validator state

async def offline_task(self, wandb_data):
    bt.logging.debug("OFFLINE: Starting offline task")
    self.running_offline_mode = True
    wandb_data['event_name'] = "offline_task_started"
    self.log_event(wandb_data)

    # get all alive miner UIDs to compare against the top scores from the last round
    miner_uids = self.miners_left_to_score

    # TODO potentially fetch prompt template from miner too
    # Grab all the models that the miners submitted
    responses = await self.dendrite.forward(
        axons=[self.metagraph.axons[miner_uid] for miner_uid in miner_uids],
        synapse=GetHFModelName(),
        deserialize=False,
        timeout=5.0,
    )

    wandb_data['event_name'] = "GetHFModelName Responses Fetched"
    self.log_event(wandb_data)

    # get all the HF model names from the responses
    miner_hf_model_names = [response.hf_model_name for response in responses]
    bt.logging.debug(f"OFFLINE: Miner HF model names: {len(miner_hf_model_names)}")

    try:
        hf_model_name_to_miner_uids = {}
        for i,miner_uid in enumerate(miner_uids):
            if responses[i].hf_model_name is not None:
                if responses[i].hf_model_name not in hf_model_name_to_miner_uids:
                    hf_model_name_to_miner_uids[responses[i].hf_model_name] = []
                hf_model_name_to_miner_uids[responses[i].hf_model_name].append(int(miner_uid))

        # Group all the models together uniquely and share the same inference server
        unique_miner_hf_model_names = [m for m in list(set(miner_hf_model_names)) if m not in [None, ""]]
        if len(unique_miner_hf_model_names) == 0:
            bt.logging.debug(f"OFFLINE: No unique miner HF model names to evaluate in OFFLINE mode")
            for miner_uid in miner_uids:
                self.offline_scores[self.competition_version][miner_uid] = 0.0
                #self.offline_miners_scored[self.competition_version].append(int(miner_uid))
            wandb_data['event_name'] = "No Unique HF Models"
            wandb_data['miners_left_to_score'] = miner_uids
            self.log_event(wandb_data)
            wandb_data.pop('miners_left_to_score')
            self.running_offline_mode = False
            return
    except Exception as e:
        bt.logging.error(f"OFFLINE: Error getting unique miner HF model names: {e}")
        wandb_data['event_name'] = "Error Getting Unique HF Models"
        wandb_data['error'] = e
        self.log_event(wandb_data)
        wandb_data.pop('error')
        self.running_offline_mode = False
        return

    bt.logging.debug(f"OFFLINE: Unique miner HF model names: {len(unique_miner_hf_model_names)}")
    wandb_data['event_name'] = "Unique HF Model Fetched"
    wandb_data['num_unique_hf_models'] = len(unique_miner_hf_model_names)
    self.log_event(wandb_data)
    wandb_data.pop('num_unique_hf_models')

    if len(unique_miner_hf_model_names) > 0:
        bt.logging.debug(f"OFFLINE: Generating tasks")
        # Generate a set of tasks to run on all the offline models
        num_tasks = 1000
        batch_size = 100
        wandb_data['event_name'] = "Generating Tasks"
        self.log_event(wandb_data)
        tasks = []
        for i,_ in enumerate(range(0, num_tasks, batch_size)):
            #bt.logging.debug(f"OFFLINE: Generating tasks batch {i+1} of {num_tasks // batch_size}")
            tasks.extend(await asyncio.gather(*[asyncio.to_thread(get_random_task, self, offline=True) for _ in range(batch_size)]))
            #bt.logging.debug(f"OFFLINE: Generated tasks batch {i+1} of {num_tasks // batch_size}")
        bt.logging.debug(f"OFFLINE: Generated {len(tasks)} tasks of {num_tasks} total")
        wandb_data['event_name'] = "Generated Tasks"
        wandb_data['num_tasks'] = len(tasks)
        self.log_event(wandb_data)
        wandb_data.pop('num_tasks')

    for i,hf_model_name in enumerate(unique_miner_hf_model_names):
        bt.logging.debug(f"OFFLINE: Running tasks for model {i+1} of {len(unique_miner_hf_model_names)}")
        wandb_data['event_name'] = "Running HF Model"
        wandb_data['num_hf_model'] = i
        wandb_data['miner_uids'] = hf_model_name_to_miner_uids[hf_model_name]
        self.log_event(wandb_data)
        wandb_data.pop('miner_uids')

        if hf_model_name is None or hf_model_name == "" or hf_model_name.lower() == "none":
            bt.logging.debug(f"OFFLINE: Miner returned empty HF model name ... skipping")
            for miner_uid in hf_model_name_to_miner_uids[hf_model_name]:
                self.offline_scores[self.competition_version][miner_uid] = 0.0
                #self.offline_miners_scored[self.competition_version].append(int(miner_uid))
            wandb_data['event_name'] = "Skipping Empty HF Model"
            wandb_data['miner_uids'] = hf_model_name_to_miner_uids[hf_model_name]
            self.log_event(wandb_data)
            wandb_data.pop('miner_uids')
            continue # skip this model

        # Extract the model card data for the model from HF
        info = model_info(hf_model_name)
        license = info.card_data['license']
        total_size = info.safetensors.total

        # confirm model license is apache-2.0 or nc-by-nc-4.0 or mit
        # TODO eventually ONLY accept apache-2.0
        if license not in ["apache-2.0", "cc-by-nc-4.0", "mit"]:
            bt.logging.debug(f"OFFLINE: Skipping model {i+1} of {len(unique_miner_hf_model_names)} due to license: {license}")
            for miner_uid in hf_model_name_to_miner_uids[hf_model_name]:
                self.offline_scores[self.competition_version][miner_uid] = 0.0
                #self.offline_miners_scored[self.competition_version].append(int(miner_uid))
            wandb_data['event_name'] = "Skipping Model Due to License"
            wandb_data['miner_uids'] = hf_model_name_to_miner_uids[hf_model_name]
            self.log_event(wandb_data)
            wandb_data.pop('miner_uids')
            continue

        # confirm model size is less than 10B params (want 8B or less models)
        if total_size > 10000000000:
            bt.logging.debug(f"OFFLINE: Skipping model {i+1} of {len(unique_miner_hf_model_names)} due to size: {total_size}")
            for miner_uid in hf_model_name_to_miner_uids[hf_model_name]:
                self.offline_scores[self.competition_version][miner_uid] = 0.0
                #self.offline_miners_scored[self.competition_version].append(int(miner_uid))
            wandb_data['event_name'] = "Skipping Model Due to Size"
            wandb_data['miner_uids'] = hf_model_name_to_miner_uids[hf_model_name]
            self.log_event(wandb_data)
            wandb_data.pop('miner_uids')
            continue

        bt.logging.debug(f"OFFLINE: Starting server for model {i+1} of {len(unique_miner_hf_model_names)}")
        wandb_data['event_name'] = "HF Model Eval Server Starting"
        self.log_event(wandb_data)
        try:
            # Start the server for the model
            server_process = await asyncio.to_thread(execute_shell_command,
            f"""
            {os.getcwd()}/.venvsglang/bin/python -m sglang.launch_server --model-path {hf_model_name} \
            --port {self.config.validator_hf_server_port} --host 0.0.0.0 \
            --mem-fraction-static {self.config.validator_hf_server_mem_fraction_static}
            """
            )

            bt.logging.debug(f"OFFLINE: Started server for model {i+1} of {len(unique_miner_hf_model_names)}, waiting for it to start on port {self.config.validator_hf_server_port} (could take several minutes)")
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(wait_for_server, f"http://localhost:{self.config.validator_hf_server_port}"), 
                    timeout=60*10 # wait up to 10 minutes
                )
                bt.logging.debug(f"OFFLINE: Server for model {i+1} of {len(unique_miner_hf_model_names)} started")
                wandb_data['event_name'] = "HF Model Eval Server Started"
                self.log_event(wandb_data)
            except asyncio.TimeoutError:
                # likely a validator error
                bt.logging.error(f"OFFLINE: Timeout waiting for server for model {i+1} of {len(unique_miner_hf_model_names)} to start")
                wandb_data['event_name'] = "Timeout Waiting for HF Model Eval Server"
                wandb_data['miner_uids'] = hf_model_name_to_miner_uids[hf_model_name]
                self.log_event(wandb_data)
                wandb_data.pop('miner_uids')
                wandb_data.pop('num_hf_model')
                # can't score this model, so skipping it for now, the miner will be tried again if this runs again
                continue
            except Exception as e:
                bt.logging.error(f"OFFLINE: Error waiting for server: {e}")
                wandb_data['event_name'] = "Error Waiting for HF Model Eval Server"
                wandb_data['error'] = e
                wandb_data['miner_uids'] = hf_model_name_to_miner_uids[hf_model_name]
                self.log_event(wandb_data)
                wandb_data.pop('error')
                wandb_data.pop('num_hf_model')
                wandb_data.pop('miner_uids')
                continue

        except Exception as e:
            # likely a validator error
            bt.logging.error(f"OFFLINE: Error starting sglang server for model: {i+1} of {len(unique_miner_hf_model_names)}: {e}")
            wandb_data['event_name'] = "Error Starting HF Model Eval Server"
            wandb_data['error'] = e
            wandb_data['miner_uids'] = hf_model_name_to_miner_uids[hf_model_name]
            self.log_event(wandb_data)
            wandb_data.pop('error')
            wandb_data.pop('num_hf_model')
            wandb_data.pop('miner_uids')
            # can't score this model, so skipping it for now, the miner will be tried again if this runs again
            # could be an issue with model size
            continue

        # get LLM responses
        bt.logging.debug(f"OFFLINE: Getting LLM responses for model {i+1} of {len(unique_miner_hf_model_names)}")
        wandb_data['event_name'] = "Getting LLM Responses"
        self.log_event(wandb_data)
        llm_responses = await asyncio.gather(
            *[asyncio.to_thread(llm, self, task.synapse.messages, task.synapse.tools, hf_model_name, hugging_face=True)
              for task in tasks]
        )
        bt.logging.debug(f"OFFLINE: Got {len(llm_responses)} LLM responses for model: {i+1} of {len(unique_miner_hf_model_names)}")
        wandb_data['event_name'] = "Got LLM Responses"
        self.log_event(wandb_data)

        # terminate the server after getting all the responses
        bt.logging.debug(f"OFFLINE: Terminating server for model: {i+1} of {len(unique_miner_hf_model_names)}")
        wandb_data['event_name'] = "HF Model Eval Server Terminating"
        self.log_event(wandb_data)
        await asyncio.to_thread(terminate_process, server_process)
        bt.logging.debug(f"OFFLINE: Terminated server for model: {i+1} of {len(unique_miner_hf_model_names)}")
        wandb_data['event_name'] = "HF Model Eval Server Terminated"
        self.log_event(wandb_data)

        these_miner_uids = hf_model_name_to_miner_uids[hf_model_name]
        responses = []
        for j, llm_response in enumerate(llm_responses):
            task = tasks[j]
            response = task.synapse.model_copy()
            response.response = llm_response.strip()
            response.dendrite.process_time = 5.0 # TODO may be useful to test performance of the model itself
            response.dendrite.status_code = 200 
            response.axon.status_code = 200
            response.competition_version = self.competition_version
            responses.append(response)

        # evaluate, track score and add to wandb
        # TODO need to see if this SCORE is higher than the all-time top score
        # TODO if so, update the all-time top score and model name and reward TOP miners
        # TODO if not, then temporal decay of scores
        bt.logging.debug(f"OFFLINE: Processing rewards for model: {i+1} of {len(unique_miner_hf_model_names)}, for miners: {these_miner_uids}")
        wandb_data['event_name'] = "Processing Rewards"
        self.log_event(wandb_data)
        await process_rewards_update_scores_for_many_tasks_and_many_miners(self, tasks=tasks, responses=responses, miner_uids=these_miner_uids, wandb_data=wandb_data)
    
        # remove old files from HF cache
        bt.logging.debug(f"OFFLINE: Deleting model from HF cache: {i+1} of {len(unique_miner_hf_model_names)}")
        wandb_data['event_name'] = "Deleting HF Model from Cache"
        self.log_event(wandb_data)
        await asyncio.to_thread(delete_model_from_hf_cache, self, hf_model_name)
        # TODO handle temporal decay of scores if no miners outperform all time top score
        # TODO handle temporal decay of all scores depending on a) if no new TOP score and b) if new TOP score
        wandb_data['event_name'] = "Finished Processing Rewards"
        wandb_data['miner_uids'] = these_miner_uids
        self.log_event(wandb_data)
        wandb_data.pop('num_hf_model')
        wandb_data.pop('miner_uids')

    bt.logging.debug(f"OFFLINE: Finished processing offline tasks")
    self.running_offline_mode = False
    wandb_data['event_name'] = "Finished Processing Offline Tasks"
    wandb_data['miner_uids'] = miner_uids
    self.log_event(wandb_data)
    wandb_data.pop('miner_uids')

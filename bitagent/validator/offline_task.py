import os
import time
import json
import shutil
import psutil
import asyncio
import requests
import subprocess
import bittensor as bt
from sglang.utils import (
    terminate_process)
from bitagent.helpers.llms import llm
from huggingface_hub import model_info, hf_hub_download
from bitagent.protocol import GetHFModelName
from bitagent.tasks.tool_call_task import ToolCallTask
from common.utils.shell import execute_shell_command
from bitagent.helpers.logging import temporary_logging_state
from bitagent.validator.multi_turn_evaluator import BFCLEvaluator
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

# added our own wait for server to check the process itself 
# this will check to see if the sglang process crashes due to limited VRAM
def wait_for_server(base_url: str, server_process, timeout: int = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        server_process: The process to terminate if the server is ready
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.time()
    procutil = psutil.Process(int(server_process.pid))
    while True:
        try:
            if timeout and time.time() - start_time > timeout:
                bt.logging.error(f"OFFLINE: Server did not become ready within timeout period")
                raise TimeoutError("Server did not become ready within timeout period")

            # Use psutil to monitor the process
            if not procutil.is_running():  # Check if process is still running
                bt.logging.error(f"OFFLINE: Server process terminated unexpectedly, check VRAM usage")
                raise Exception("Server process terminated unexpectedly, potentially VRAM usage issue")
            if server_process.poll() is not None:
                bt.logging.error(f"OFFLINE: Server process terminated with code {server_process.poll()}")
                raise Exception(f"Server process terminated with code {server_process.poll()}")

            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                time.sleep(5)
                break

        except requests.exceptions.RequestException:
            time.sleep(1)



def has_chat_template(model_id: str) -> bool:
    """
    Return True if tokenizer_config.json (local dir or hub) contains
    either 'chat_template' or 'chat_template_name'.
    Uses the default Hugging Face cache (~/.cache/huggingface) for downloads.
    """
    cfg = None

    # 1) local directory
    if os.path.isdir(model_id):
        cfg_path = os.path.join(model_id, "tokenizer_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

    # 2) hub repo
    else:
        try:
            cfg_path = hf_hub_download(
                repo_id=model_id,
                filename="tokenizer_config.json",
                repo_type="model",          # optional, keeps it explicit
            )
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            print(f"[WARNING] Could not fetch/parse tokenizer_config.json for {model_id}")

    if not cfg:
        return False

    return bool(cfg.get("chat_template") or cfg.get("chat_template_name"))


# ###########################################################
# OFFLINE TASKING
# ###########################################################

# TODO also run the bfcl suite on the validator - but skip the API calls, don't use those at first
# TODO store TOP score from last round and all-time in validator state
async def offline_task(self, wandb_data):

    # evaluate single turn logic
    st = False


    bt.logging.debug("OFFLINE: Starting offline task")
    self.running_offline_mode = True
    wandb_data['event_name'] = "offline_task_started"
    self.log_event(wandb_data)

    # get all alive miner UIDs to compare against the top scores from the last round
    miner_uids = self.miners_left_to_score

    # -------------------------------------------------------------------------
    # First Pass: Call GetHFModelName only for miners with an empty string
    #             i.e., those who have never locked a model or who were reset.
    # -------------------------------------------------------------------------
    miners_needing_model_lookup = [
        uid for uid in miner_uids
        if not self.offline_model_names[self.competition_version].get(uid, "")
    ]

    bt.logging.debug(f"OFFLINE: Miners needing model lookup: {miners_needing_model_lookup}")
    responses = await self.dendrite.forward(
        axons=[self.metagraph.axons[uid] for uid in miners_needing_model_lookup],
        synapse=GetHFModelName(),
        deserialize=False,
        timeout=15.0,
    )
    with temporary_logging_state('Warning'):
        if miners_needing_model_lookup:
            wandb_data['event_name'] = "GetHFModelName Responses Fetched"
            self.log_event(wandb_data)

            # Assign any valid HF model to the miner, with the commit hash
            for i, uid in enumerate(miners_needing_model_lookup):
                try:
                    hf_model_name = responses[i].hf_model_name or ""
                except:
                    hf_model_name = ""

                # If the response has "/", we assume it's a valid HF model name
                if "/" in hf_model_name:
                    try:
                        info = model_info(hf_model_name)
                        time.sleep(30)
                        hf_model_name_hash = hf_model_name + "@" + info.sha
                        self.offline_model_names[self.competition_version][uid] = hf_model_name_hash
                    except Exception as e:
                        bt.logging.error(f"OFFLINE: Error getting model info for a miner: {e}")
                        # if the model is not found, set the score to 0.02 and remove the miner from the list of miners to score    
                        if "404" in str(e):
                            bt.logging.error(f"OFFLINE: Model not found, setting score to 0.02 and removing miner from list of miners to score")
                            self.offline_model_names[self.competition_version][uid] = "404"
                        
                        if self.offline_model_names[self.competition_version][uid] != "404":
                            self.offline_model_names[self.competition_version][uid] = ""
                else:
                    # No valid model => remain empty
                    self.offline_model_names[self.competition_version][uid] = ""

        else:
            bt.logging.debug("OFFLINE: No miners need model lookup.")

    # -------------------------------------------------------------------------
    # Second Pass: Gather all miners' locked model names (including newly set).
    # Build 'hf_model_name_to_miner_uids' for the scoring phase.
    # -------------------------------------------------------------------------
    hf_model_name_to_miner_uids = {}
    for uid in miner_uids:
        existing_model_name = self.offline_model_names[self.competition_version].get(uid, "")
        if "@" in existing_model_name:
            # Group these miners by their 'model@commit'
            if existing_model_name not in hf_model_name_to_miner_uids:
                hf_model_name_to_miner_uids[existing_model_name] = []
            hf_model_name_to_miner_uids[existing_model_name].append(uid)
        else:
            # This miner has no valid model => will end up with 0.0 score or skip
            pass

    # Convert to a list of unique locked models
    miner_hf_model_names = list(hf_model_name_to_miner_uids.keys())

    # If none have valid models, skip out with 0.0 for everyone
    if not miner_hf_model_names:
        bt.logging.info(f"OFFLINE: No unique miner HF model names to evaluate in OFFLINE mode")
        for miner_uid in miner_uids:
            self.offline_scores[self.competition_version][miner_uid] = 0.0
        wandb_data['event_name'] = "No Unique HF Models"
        wandb_data['miners_left_to_score'] = miner_uids
        self.log_event(wandb_data)
        wandb_data.pop('miners_left_to_score', None)
        self.running_offline_mode = False
        return

    unique_miner_hf_model_names = [m for m in miner_hf_model_names if m not in [None, ""]]

    bt.logging.debug(f"OFFLINE: Unique miner HF model names: {len(unique_miner_hf_model_names)}")
    wandb_data['event_name'] = "Unique HF Model Fetched"
    wandb_data['num_unique_hf_models'] = len(unique_miner_hf_model_names)
    self.log_event(wandb_data)
    wandb_data.pop('num_unique_hf_models', None)

    # no need to regrade if score exists for the same model
    models_to_skip = []
    
    for hfmn in unique_miner_hf_model_names:


        uids_with_same_model = []
        scores_with_same_model = []
        for k, model_name in self.offline_model_names[self.competition_version].items():
            if model_name == hfmn:
                uids_with_same_model.append(k)
                scores_with_same_model.append(self.offline_scores[self.competition_version][k])
        
        if len(uids_with_same_model) > 0:
            max_score_for_model = max(scores_with_same_model)  # Calculate max score once

            if max_score_for_model <= 0:
                # Skip adding to models_to_skip if max score is zero
                continue
            
            models_to_skip.append(hfmn)  # Add only if max_score > 0
            
            # Process the miners
            the_uids = hf_model_name_to_miner_uids[hfmn]

            bt.logging.debug(f"OFFLINE: Found miner with same model, using existing score")
            if hfmn == "404":
                max_score_for_model = 0.02

            for uid in the_uids:
                self.offline_scores[self.competition_version][uid] = max_score_for_model
            self.update_offline_scores([max_score_for_model] * len(the_uids), the_uids)

    # skip the models we already have scores for
    unique_miner_hf_model_names = [m for m in unique_miner_hf_model_names if m not in models_to_skip]

    if len(unique_miner_hf_model_names) > 0:
        bt.logging.debug("OFFLINE: Generating tasks")
        # Generate a set of tasks to run on all the offline models
        num_tasks = 100
        wandb_data['event_name'] = "Generating Tasks"
        self.log_event(wandb_data)

        tasks = [ToolCallTask(validator=self, name="Responds with correct function call", offline=True) for _ in range(num_tasks)]

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
            wandb_data['event_name'] = "Skipping Empty HF Model"
            wandb_data['miner_uids'] = hf_model_name_to_miner_uids[hf_model_name]
            self.log_event(wandb_data)
            wandb_data.pop('miner_uids')
            continue # skip this model

        # Extract the model card data for the model from HF
        # ensure logger doesn't print the model name publicly, so restrict to only HF warnings
        # Temporarily set logging to WARNING within the context manager
        with temporary_logging_state('Warning'):
            try:
                info = model_info(hf_model_name.split("@")[0])
                total_size = info.safetensors.total
                license = info.card_data['license']
            except Exception:
                bt.logging.error("OFFLINE: Issue getting model info, skipping")
                license = 'No license available'
                continue
            # ── chat‑template filter ───────────────────────────────────────
            if not has_chat_template(hf_model_name.split("@")[0]):
                wandb_data['event_name'] = "Skipping (no chat_template found)"
                wandb_data['miner_uids'] = hf_model_name_to_miner_uids[hf_model_name]
                for miner_uid in hf_model_name_to_miner_uids[hf_model_name]:
                    self.offline_scores[self.competition_version][miner_uid] = 0.02
                self.log_event(wandb_data)
                wandb_data.pop('miner_uids')
                continue


        # confirm model license is apache-2.0 or nc-by-nc-4.0 or mit
        # temporarily including the salesforce xlam license as a jumping off point for performance
        if license not in ["apache-2.0", "cc-by-nc-4.0", "mit"]:
            bt.logging.debug(f"OFFLINE: Skipping model {i+1} of {len(unique_miner_hf_model_names)} due to license: {license}")
            for miner_uid in hf_model_name_to_miner_uids[hf_model_name]:
                self.offline_scores[self.competition_version][miner_uid] = 0.02
            wandb_data['event_name'] = "Skipping Model Due to License"
            wandb_data['miner_uids'] = hf_model_name_to_miner_uids[hf_model_name]
            self.log_event(wandb_data)
            wandb_data.pop('miner_uids')
            continue

        # confirm model size is less than 10B params (want 8B or less models)
        if total_size > 10000000000:
            bt.logging.debug(f"OFFLINE: Skipping model {i+1} of {len(unique_miner_hf_model_names)} due to size: {total_size}")
            for miner_uid in hf_model_name_to_miner_uids[hf_model_name]:
                self.offline_scores[self.competition_version][miner_uid] = 0.02
            wandb_data['event_name'] = "Skipping Model Due to Size"
            wandb_data['miner_uids'] = hf_model_name_to_miner_uids[hf_model_name]
            self.log_event(wandb_data)
            wandb_data.pop('miner_uids')
            continue

        bt.logging.debug(f"OFFLINE: Starting server for model {i+1} of {len(unique_miner_hf_model_names)}")
        wandb_data['event_name'] = "HF Model Eval Server Starting"
        self.log_event(wandb_data)

        # see if we have a snapshot already in the cache
        latest_snapshot = None

        try:
            # Start the server for the model
            try:
                cache_dir = os.path.expanduser(self.config.validator_hf_cache_dir)
                snapshot_dir = f"{cache_dir}/models--{hf_model_name.replace('/', '--')}/snapshots/"
                
                # Get all snapshot directories
                snapshots = [os.path.join(snapshot_dir, d) for d in os.listdir(snapshot_dir) if os.path.isdir(os.path.join(snapshot_dir, d))]
                
                # Sort snapshots by creation time (os.path.getctime) or modification time (os.path.getmtime)
                latest_snapshot = max(snapshots, key=os.path.getctime)
                # TODO if the latest snapshot is older than a week, delete it and download a new one

            except Exception as e:
                bt.logging.debug(f"OFFLINE: Error getting latest snapshot")
                latest_snapshot = None
            
            # # either load an existing snapshot or download the model
            # if os.path.exists(snapshot_dir) and latest_snapshot:    
            #     model_path = latest_snapshot
            # else:
            #     # need to download from hugging face
            model_path = hf_model_name.split("@")[0]
            model_commit = hf_model_name.split("@")[1]
            os.environ["ACTUAL_MODEL_PATH"] = model_path  # Store the actual model path in env var for multi turn
            # log the vram status prior to sglang launch
            wandb_data['event_name'] = "Launching sglang server"
            command = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
            wandb_data['current_vram_utilization'] = subprocess.check_output(command, text=True)
            self.log_event(wandb_data)
            wandb_data.pop('current_vram_utilization')

            # launch sglang server
            server_process = await asyncio.to_thread(execute_shell_command,
                f"""
                {os.getcwd()}/.venvsglang/bin/python -m sglang.launch_server \
                --model-path {model_path} \
                --port {self.config.validator_hf_server_port} \ 
                --revision {model_commit} \
                --host 0.0.0.0 \
                --mem-fraction-static {self.config.validator_hf_server_mem_fraction_static} \
                --disable-cuda-graph \
                --context-length 25000
                """, 
                model_path
            )


            bt.logging.debug(f"OFFLINE: Started server for model {i+1} of {len(unique_miner_hf_model_names)}, waiting for it to start on port {self.config.validator_hf_server_port} (could take several minutes)")
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(wait_for_server, f"http://localhost:{self.config.validator_hf_server_port}", server_process), 
                    timeout=60*15 # wait up to 15 minutes
                )
                bt.logging.debug(f"OFFLINE: Server for model {i+1} of {len(unique_miner_hf_model_names)} started")
                wandb_data['event_name'] = "HF Model Eval Server Started"
                self.log_event(wandb_data)
            except asyncio.TimeoutError as e:
                # likely a validator error
                bt.logging.error(f"OFFLINE: Timeout waiting for server for model {i+1} of {len(unique_miner_hf_model_names)} to start, skipping")
                wandb_data['event_name'] = "Timeout Waiting for HF Model Eval Server"
                wandb_data['miner_uids'] = hf_model_name_to_miner_uids[hf_model_name]
                self.log_event(wandb_data)
                wandb_data.pop('miner_uids')
                wandb_data.pop('num_hf_model')
                # can't score this model, so skipping it for now, the miner will be tried again if this runs again
                continue
            except Exception as e:
                bt.logging.error(f"OFFLINE: Error waiting for server: {e}, skipping")
                wandb_data['event_name'] = "Error Waiting for HF Model Eval Server"
                wandb_data['error'] = f"{e}"
                wandb_data['miner_uids'] = hf_model_name_to_miner_uids[hf_model_name]
                wandb_data['current_vram_utilization'] = subprocess.check_output(command, text=True)
                self.log_event(wandb_data)
                wandb_data.pop('current_vram_utilization')
                wandb_data.pop('error')
                wandb_data.pop('num_hf_model')
                wandb_data.pop('miner_uids')
                continue

        except Exception as e:
            # likely a validator error
            bt.logging.error(f"OFFLINE: Error starting sglang server for model: {i+1} of {len(unique_miner_hf_model_names)}: {e}")
            wandb_data['event_name'] = "Error Starting HF Model Eval Server"
            wandb_data['error'] = f"{e}"
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

        # at most 5 LLM calls concurrently
        sem = asyncio.Semaphore(5) 

        async def call_llm_with_semaphore(task):
            async with sem:
                return await asyncio.to_thread(
                    llm, self, task.synapse.messages, task.synapse.tools, model_path, hugging_face=True
                )

        llm_responses_and_finishes = await asyncio.gather(
            *[call_llm_with_semaphore(task) for task in tasks]
        )
        try:    
            llm_responses = [r[0] for r in llm_responses_and_finishes]
            llm_finishes = [r[1] for r in llm_responses_and_finishes]
        except Exception as e:
            bt.logging.error(f"OFFLINE: Error getting LLM responses: {e}, have to skip this model")
            continue

        bt.logging.debug(f"OFFLINE: Got {len(llm_responses)} LLM responses for model: {i+1} of {len(unique_miner_hf_model_names)}")
        wandb_data['event_name'] = "Got LLM Responses"
        self.log_event(wandb_data)

        # -----------------------------------------
        # Run single-turn evaluation
        # -----------------------------------------
        st_score = 0.0
        these_miner_uids = hf_model_name_to_miner_uids[hf_model_name]
        if st:
            
            responses = []
            for j, llm_response in enumerate(llm_responses):
                task = tasks[j]
                response = task.synapse.model_copy()
                response.response = llm_response.strip()
                response.dendrite.process_time = 5.0
                response.dendrite.status_code = 200 
                response.axon.status_code = 200
                response.competition_version = self.competition_version
                responses.append(response)

            # Get single-turn score
            bt.logging.debug(f"OFFLINE: Processing single-turn rewards for model {i+1} of {len(unique_miner_hf_model_names)}")
            wandb_data['event_name'] = "Processing Single-Turn Rewards"
            self.log_event(wandb_data)
            
            # Get the single-turn score 
            st_score = await process_rewards_update_scores_for_many_tasks_and_many_miners(
                self, tasks=tasks, responses=responses, miner_uids=these_miner_uids, wandb_data=wandb_data
            )

            # Run multi-turn evaluation 
            bt.logging.debug(f"OFFLINE: Starting BFCL evaluation for model {i+1} of {len(unique_miner_hf_model_names)}")
            wandb_data['event_name'] = "Starting BFCL Evaluation"
            self.log_event(wandb_data)
        
        # The server is already running from the single-turn evaluation
        mt_score = 0.0
        
        # Use temporary_logging_state to silence BFCL logging
        # with temporary_logging_state('Warning'):
        try:
            # Create evaluator instance for multi-turn evaluation
            # Set verbose=False to avoid any model info leaking
            with temporary_logging_state('Warning'):
                evaluator = BFCLEvaluator(verbose=True, project_root=None)
                mt_results = evaluator.evaluate_model(
                    model_name="BitAgent",
                    endpoint="localhost",
                    port=self.config.validator_hf_server_port,
                    test_category="multi_turn_base",
                    backend="sglang",
                    cleanup_files=True
                )
                mt_score = mt_results.get("overall_score", 0.0)
            bt.logging.info(f"OFFLINE: BFCL evaluation score: {mt_score:.4f}")
            if mt_score <= 0.0:
                mt_score = 0.02
            
        except Exception as e:
            # Avoid including model info in error messages
            bt.logging.error(f"OFFLINE: Error in BFCL evaluation: {str(e).replace(model_path, 'REDACTED')}")
            mt_score = 0.0
        
        wandb_data['event_name'] = "Completed BFCL Evaluation"
        wandb_data['BFCL_score'] = mt_score
        self.log_event(wandb_data)
        wandb_data.pop('BFCL_score', None)
        
        # -----------------------------------------
        # Calculate combined score

        combined_score = mt_score
        
        bt.logging.info(f"OFFLINE: Combined score for model {i+1} of {len(unique_miner_hf_model_names)}: {combined_score:.4f}")
        
        wandb_data['event_name'] = "Final Combined Scores"
        wandb_data['single_turn_score'] = st_score
        wandb_data['BFCL_score'] = mt_score
        wandb_data['combined_score'] = combined_score
        wandb_data['miner_uids'] = these_miner_uids
        self.log_event(wandb_data)
        wandb_data.pop('single_turn_score', None)
        wandb_data.pop('BFCL_score', None)
        wandb_data.pop('combined_score', None)
        wandb_data.pop('miner_uids', None)
        
        # Update scores with the combined score instead of just single-turn
        self.update_offline_scores([combined_score] * len(these_miner_uids), these_miner_uids)

        # terminate the server after getting all the responses
        bt.logging.debug(f"OFFLINE: Terminating server for model {i+1} of {len(unique_miner_hf_model_names)}")
        wandb_data['event_name'] = "HF Model Eval Server Terminating"
        self.log_event(wandb_data)
        await asyncio.to_thread(terminate_process, server_process)
        bt.logging.debug(f"OFFLINE: Terminated server for model {i+1} of {len(unique_miner_hf_model_names)}")
        wandb_data['event_name'] = "HF Model Eval Server Terminated"
        self.log_event(wandb_data)

        # remove newly downloaded files from HF cache if were not already in cache
        # todo re-evaluate how we handle storage of models
        if not latest_snapshot:
            bt.logging.debug(f"OFFLINE: Deleting model from HF cache: {i+1} of {len(unique_miner_hf_model_names)}")
            wandb_data['event_name'] = "Deleting HF Model from Cache"
            self.log_event(wandb_data)
            await asyncio.to_thread(delete_model_from_hf_cache, self, hf_model_name.split("@")[0])
        else:
            bt.logging.debug(f"OFFLINE: NOT Deleting model from HF cache: {i+1} of {len(unique_miner_hf_model_names)}")
            wandb_data['event_name'] = "NOT Deleting HF Model from Cache - snapshot found, so no new download to revert"
            self.log_event(wandb_data)

        wandb_data['event_name'] = "Finished Processing Rewards"
        wandb_data['miner_uids'] = these_miner_uids
        self.log_event(wandb_data)
        wandb_data.pop('num_hf_model')
        wandb_data.pop('miner_uids')

        time.sleep(5)

    bt.logging.debug(f"OFFLINE: Finished processing offline tasks")
    self.running_offline_mode = False
    wandb_data['event_name'] = "Finished Processing Offline Tasks"
    wandb_data['miner_uids'] = miner_uids
    self.log_event(wandb_data)
    wandb_data.pop('miner_uids')
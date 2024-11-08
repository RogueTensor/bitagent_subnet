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

# ###########################################################
# OFFLINE TASKING
# ###########################################################

# TODO also run the bfcl suite on the validator - but skip the API calls, don't use those at first
# TODO store TOP score from last round and all-time in validator state
async def offline_task(self):
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
    miner_hf_model_names = [response.hf_model_name for response in responses]

    hf_model_name_to_miner_uids = {}
    for i,miner_uid in enumerate(miner_uids):
        if responses[i].hf_model_name is not None:
            if responses[i].hf_model_name not in hf_model_name_to_miner_uids:
                hf_model_name_to_miner_uids[responses[i].hf_model_name] = []
            hf_model_name_to_miner_uids[responses[i].hf_model_name].append(miner_uid)

    # Group all the models together uniquely and share the same inference server
    unique_miner_hf_model_names = [m for m in list(set(miner_hf_model_names)) if m not in [None, ""]]

    if len(unique_miner_hf_model_names) > 0:
        # Generate a set of tasks to run on all the offline models
        tasks = []
        for _ in range(1000):
            task = await asyncio.to_thread(get_random_task, self)
            task.mode = "offline"
            tasks.append(task)

    for hf_model_name in unique_miner_hf_model_names:
        if hf_model_name is None or hf_model_name == "" or hf_model_name.lower() == "none":
            #bt.logging.warning(f"Miner returned empty HF model name")
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
            #bt.logging.warning(f"Error starting server for model {hf_model_name}: {e}")
            # TODO determine if this is a problem with the model or the server
            # right now assuming problem with the model
            for miner_uid in hf_model_name_to_miner_uids[hf_model_name]:
                self.offline_scores[miner_uid] = 0.0
            continue

        # get LLM responses
        llm_responses = await asyncio.gather(
            *[asyncio.to_thread(llm, self, task.synapse.messages, task.synapse.tools, hf_model_name, hugging_face=True)
              for task in tasks]
        )

        # terminate the server after getting all the responses
        await asyncio.to_thread(terminate_process, server_process)

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
        await process_rewards_update_scores_for_many_tasks_and_many_miners(self, tasks=tasks, responses=responses, miner_uids=these_miner_uids)
    
        # TODO remove old files from HF cache
        # TODO handle temporal decay of scores if no miners outperform all time top score
        # TODO handle temporal decay of all scores depending on a) if no new TOP score and b) if new TOP score
        # TODO if doing tier emissions - check wandb for the TOP score and what associated model name it is - could do that here

    self.running_offline_mode = False

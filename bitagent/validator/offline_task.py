import os
import json
import shutil
import asyncio
import subprocess
import time
import csv
import bittensor as bt
from common.utils.shell import execute_shell_command
from huggingface_hub import model_info, snapshot_download
from bitagent.protocol import GetHFModelName
from typing import Dict, List, Optional




def parse_percentage(value: Optional[str]) -> Optional[float]:
    """Parse percentage string to float, handling N/A and None values."""
    if value is None or value == 'N/A' or value == '':
        return None
    try:
        return float(value.rstrip('%')) / 100.0
    except:
        return None

async def offline_task(self, wandb_data):
    """Evaluate models using BFCL."""
    bt.logging.debug("OFFLINE: Starting offline task")
    self.running_offline_mode = True
    wandb_data['event_name'] = "offline_task_started"
    self.log_event(wandb_data)
    
    # Get miner UIDs and their models
    miner_uids = self.miners_left_to_score
    
    # Get HF model names for miners who need lookup
    miners_needing_lookup = [
        uid for uid in miner_uids
        if not self.offline_model_names[self.competition_version].get(uid, "")
    ]
    
    if miners_needing_lookup:
        bt.logging.debug(f"OFFLINE: Miners needing model lookup: {miners_needing_lookup}")
        responses = await self.dendrite.forward(
            axons=[self.metagraph.axons[uid] for uid in miners_needing_lookup],
            synapse=GetHFModelName(),
            deserialize=False,
            timeout=15.0,
        )
        
        wandb_data['event_name'] = "GetHFModelName Responses Fetched"
        self.log_event(wandb_data)
        
        for i, uid in enumerate(miners_needing_lookup):
            try:
                hf_model_name = responses[i].hf_model_name or ""
            except:
                hf_model_name = ""
            
            if "/" in hf_model_name:
                try:
                    info = model_info(hf_model_name)
                    self.offline_model_names[self.competition_version][uid] = f"{hf_model_name}@{info.sha}"
                except:
                    self.offline_model_names[self.competition_version][uid] = ""
    
    # Group miners by model
    model_to_uids = {}
    for uid in miner_uids:
        model_name = self.offline_model_names[self.competition_version].get(uid, "")
        if "@" in model_name:
            if model_name not in model_to_uids:
                model_to_uids[model_name] = []
            model_to_uids[model_name].append(uid)
    
    # Convert to list of unique models
    unique_miner_hf_model_names = list(model_to_uids.keys())
    
    if not unique_miner_hf_model_names:
        bt.logging.info(f"OFFLINE: No unique miner HF model names to evaluate")
        for miner_uid in miner_uids:
            self.offline_scores[self.competition_version][miner_uid] = 0.0
        wandb_data['event_name'] = "No Unique HF Models"
        wandb_data['miners_left_to_score'] = miner_uids
        self.log_event(wandb_data)
        wandb_data.pop('miners_left_to_score', None)
        self.running_offline_mode = False
        return
    
    bt.logging.debug(f"OFFLINE: Unique miner HF model names: {len(unique_miner_hf_model_names)}")
    wandb_data['event_name'] = "Unique HF Model Fetched"
    wandb_data['num_unique_hf_models'] = len(unique_miner_hf_model_names)
    self.log_event(wandb_data)
    wandb_data.pop('num_unique_hf_models', None)
    
    # Evaluate each unique model
    for i, model_full in enumerate(unique_miner_hf_model_names):
        model_name = model_full.split("@")[0]
        commit_hash = model_full.split("@")[1] if "@" in model_full else None
        
        bt.logging.debug(f"OFFLINE: Running tasks for model {i+1} of {len(unique_miner_hf_model_names)}")
        wandb_data['event_name'] = "Running HF Model"
        wandb_data['num_hf_model'] = i
        wandb_data['miner_uids'] = model_to_uids[model_full]
        self.log_event(wandb_data)
        wandb_data.pop('miner_uids')
        
        try:
            # Check model metadata
            info = model_info(model_name)
            total_size = info.safetensors.total if hasattr(info, 'safetensors') else 0
            license_info = info.card_data.get('license', 'Unknown') if hasattr(info, 'card_data') else 'Unknown'
            
            # Skip if wrong license or too big
            if license_info not in ["apache-2.0", "cc-by-nc-4.0", "mit"] or total_size > 10_000_000_000:
                bt.logging.debug(f"OFFLINE: Skipping model {i+1} due to license: {license_info} or size: {total_size}")
                for uid in model_to_uids[model_full]:
                    self.offline_scores[self.competition_version][uid] = 0.02
                wandb_data['event_name'] = "Skipping Model Due to License or Size"
                wandb_data['miner_uids'] = model_to_uids[model_full]
                self.log_event(wandb_data)
                wandb_data.pop('miner_uids')
                continue
            
            bt.logging.info(f"OFFLINE: Evaluating model {i+1} of {len(unique_miner_hf_model_names)}")
            wandb_data['event_name'] = "HF Model Eval Starting"
            self.log_event(wandb_data)
            
            # 1. Download the model
            cache_dir = os.path.expanduser(self.config.validator_hf_cache_dir)
            bt.logging.info(f"OFFLINE: Downloading model to {cache_dir}")
            model_path = snapshot_download(
                repo_id=model_name,
                revision=commit_hash,
                cache_dir=cache_dir
            )
            bt.logging.info(f"OFFLINE: Download complete")
            
            # 2. Use a supported Salesforce model name
            base_model_name = "Salesforce/Llama-xLAM-2-8b-fc-r"
            
            # 3. Set up paths for BFCL
            original_cwd = os.getcwd()
            result_dir = os.path.join(original_cwd, "bfcl_results")
            score_dir = os.path.join(original_cwd, "bfcl_scores")
            
            os.makedirs(result_dir, exist_ok=True)
            os.makedirs(score_dir, exist_ok=True)
            
            venv_path = f"{os.getcwd()}/.venvbfcl"
            # Test the full environment
            test_cmd = f"""
/bin/bash -c '
export PATH={venv_path}/bin:$PATH && \
export PYTHONPATH={venv_path}/lib/python*/site-packages:$PYTHONPATH && \
export VIRTUAL_ENV={venv_path} && \
{venv_path}/bin/python -c "import bfcl; print('"'"'BFCL imported successfully'"'"')"
'
"""
            test_process = await asyncio.to_thread(execute_shell_command, test_cmd, model_path)
            test_returncode = await asyncio.to_thread(test_process.wait)
            bt.logging.info(f"BFCL import test returned: {test_returncode}")

            # 4. Run BFCL generate
            generate_cmd = f"""
/bin/bash -c "
export PATH={venv_path}/bin:$PATH && \
export PYTHONPATH={venv_path}/lib/python*/site-packages:$PYTHONPATH && \
export VIRTUAL_ENV={venv_path} && \
{venv_path}/bin/python -m bfcl generate \
--model {base_model_name} \
--test-category simple,parallel,multiple,parallel_multiple,java,javascript,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param \
--backend vllm \
--num-gpus 1 \
--gpu-memory-utilization {self.config.validator_hf_server_mem_fraction_static} \
--local-model-path {model_path} \
--result-dir {result_dir}
"
"""

            bt.logging.debug(f"OFFLINE: Running BFCL Generate...")
            process = await asyncio.to_thread(execute_shell_command, generate_cmd, model_path)
            returncode = await asyncio.to_thread(process.wait)
            if returncode != 0:
                bt.logging.error(f"OFFLINE: Generate failed with return code: {returncode}")
                raise Exception("Generate failed")

            # 5. Run BFCL evaluate  
            evaluate_cmd = f"""
{os.getcwd()}/.venvbfcl/bin/python -m bfcl evaluate \
--model {base_model_name} \
--test-category simple,parallel,multiple,parallel_multiple,java,javascript,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param \
--result-dir {result_dir} \
--score-dir {score_dir}
"""

            bt.logging.debug(f"OFFLINE: Running BFCL Evaluate...")
            process = await asyncio.to_thread(execute_shell_command, evaluate_cmd, model_path)
            returncode = await asyncio.to_thread(process.wait)
            if returncode != 0:
                bt.logging.error(f"OFFLINE: Evaluate failed with return code: {returncode}")
                raise Exception("Evaluate failed")
            
            time.sleep(60)
                        
            # 6. Parse the score CSV file
            overall_csv_path = os.path.join(score_dir, "data_overall.csv")
            await asyncio.sleep(2)
            
            if not os.path.exists(overall_csv_path):
                bt.logging.error(f"OFFLINE: Score file not found at {overall_csv_path}")
                raise Exception(f"Score file not found")
            
            # Parse CSV to extract all scores
            scores_data = {}
            with open(overall_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Model') == 'xLAM-2-8b-fc-r (FC)' or row.get('Model') == base_model_name:
                        overall_acc = row.get('Overall Acc', '0%')
                        scores_data['overall_score'] = float(overall_acc.rstrip('%')) / 100.0
                        
                        scores_data['categories'] = {
                            'non_live_ast_acc': parse_percentage(row.get('Non-Live AST Acc')),
                            'non_live_simple_ast': parse_percentage(row.get('Non-Live Simple AST')),
                            'non_live_multiple_ast': parse_percentage(row.get('Non-Live Multiple AST')),
                            'non_live_parallel_ast': parse_percentage(row.get('Non-Live Parallel AST')),
                            'non_live_parallel_multiple_ast': parse_percentage(row.get('Non-Live Parallel Multiple AST')),
                            'live_acc': parse_percentage(row.get('Live Acc')),
                            'live_simple_ast': parse_percentage(row.get('Live Simple AST')),
                            'live_multiple_ast': parse_percentage(row.get('Live Multiple AST')),
                            'live_parallel_ast': parse_percentage(row.get('Live Parallel AST')),
                            'live_parallel_multiple_ast': parse_percentage(row.get('Live Parallel Multiple AST')),
                            'multi_turn_acc': parse_percentage(row.get('Multi Turn Acc')),
                            'multi_turn_base': parse_percentage(row.get('Multi Turn Base')),
                            'multi_turn_miss_func': parse_percentage(row.get('Multi Turn Miss Func')),
                            'multi_turn_miss_param': parse_percentage(row.get('Multi Turn Miss Param')),
                            'multi_turn_long_context': parse_percentage(row.get('Multi Turn Long Context')),
                            'relevance_detection': parse_percentage(row.get('Relevance Detection')),
                            'irrelevance_detection': parse_percentage(row.get('Irrelevance Detection')),
                        }
                        break
            
            if not scores_data:
                bt.logging.error("OFFLINE: Could not find model scores in CSV")
                raise Exception("Model scores not found in CSV")
            
            overall_score = scores_data['overall_score']
            
            bt.logging.info(f"OFFLINE: Overall score: {overall_score:.4f}")
            bt.logging.info(f"OFFLINE: Category scores: {json.dumps(scores_data['categories'], indent=2)}")
            
            # Update scores for all UIDs with this model
            for uid in model_to_uids[model_full]:
                self.offline_scores[self.competition_version][uid] = overall_score
                
                # Log to wandb - overall score and all category scores
                if wandb_data is not None:
                    wandb_data[f"offline_uid_{uid}_overall"] = overall_score
                    
                    # Log all category scores for miner feedback
                    for category, score in scores_data['categories'].items():
                        if score is not None:
                            wandb_data[f"offline_uid_{uid}_{category}"] = score
            
            self.update_offline_scores([overall_score] * len(model_to_uids[model_full]), model_to_uids[model_full])
            
            wandb_data['event_name'] = "Completed BFCL Evaluation"
            wandb_data['BFCL_score'] = overall_score
            self.log_event(wandb_data)
            wandb_data.pop('BFCL_score', None)
            
            # Cleanup
            bt.logging.info(f"OFFLINE: Cleaning up evaluation artifacts")
            
            # Clean up BFCL artifacts
            shutil.rmtree(result_dir, ignore_errors=True)
            shutil.rmtree(score_dir, ignore_errors=True)
            
            # Clean up the downloaded model
            if os.path.exists(model_path):
                bt.logging.info(f"OFFLINE: Removing downloaded model from {model_path}")
                shutil.rmtree(model_path, ignore_errors=True)
                
                # Also clean up HuggingFace cache directories
                model_cache_name = model_name.replace("/", "--")
                possible_cache_paths = [
                    os.path.join(cache_dir, f"models--{model_cache_name}"),
                    os.path.join(cache_dir, "hub", f"models--{model_cache_name}")
                ]
                
                for cache_path in possible_cache_paths:
                    if os.path.exists(cache_path):
                        bt.logging.debug(f"OFFLINE: Removing cache directory {cache_path}")
                        shutil.rmtree(cache_path, ignore_errors=True)
            
            wandb_data['event_name'] = "Finished Processing Rewards"
            wandb_data['miner_uids'] = model_to_uids[model_full]
            self.log_event(wandb_data)
            wandb_data.pop('num_hf_model')
            wandb_data.pop('miner_uids')
            
        except Exception as e:
            bt.logging.error(f"OFFLINE: Error evaluating model {i+1}: {e}")
            for uid in model_to_uids[model_full]:
                self.offline_scores[self.competition_version][uid] = 0.0
            
            wandb_data['event_name'] = "Error Evaluating Model"
            wandb_data['error'] = f"{e}"
            wandb_data['miner_uids'] = model_to_uids[model_full]
            self.log_event(wandb_data)
            wandb_data.pop('error', None)
            wandb_data.pop('num_hf_model', None)
            wandb_data.pop('miner_uids', None)
            
            # Ensure cleanup happens even on error
            try:
                if 'result_dir' in locals() and os.path.exists(result_dir):
                    shutil.rmtree(result_dir, ignore_errors=True)
                if 'score_dir' in locals() and os.path.exists(score_dir):
                    shutil.rmtree(score_dir, ignore_errors=True)
                if 'model_path' in locals() and os.path.exists(model_path):
                    shutil.rmtree(model_path, ignore_errors=True)
            except:
                pass
    
    bt.logging.debug(f"OFFLINE: Finished processing offline tasks")
    self.running_offline_mode = False
    wandb_data['event_name'] = "Finished Processing Offline Tasks"
    wandb_data['miner_uids'] = miner_uids
    self.log_event(wandb_data)
    wandb_data.pop('miner_uids')
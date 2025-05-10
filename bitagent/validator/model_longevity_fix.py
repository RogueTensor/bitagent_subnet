# model_longevity_fix.py
import datetime
import time
import numpy as np
from huggingface_hub import HfApi
import bittensor as bt

# ---------------------------------------------------------------------
# 1. Helpers to fetch dataset commit history and map to BT blocks
# ---------------------------------------------------------------------
def get_dataset_commits(repo_id: str):
    """Get commit history for a dataset repository."""
    api = HfApi()
    commits = api.list_repo_commits(repo_id=repo_id, repo_type="dataset")
    return [
        {
            "hash": c.commit_id,
            "timestamp": c.created_at.replace(tzinfo=datetime.timezone.utc),
        }
        for c in commits
    ]


def estimate_bt_block(ts_utc: datetime.datetime, subtensor):
    """Estimate Bittensor block from a UTC timestamp."""
    current_block = subtensor.get_current_block()
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    diff_sec = (now - ts_utc).total_seconds()
    return max(1, current_block - int(diff_sec // 12))  # 12-s target blocks


# ---------------------------------------------------------------------
# 2. Helpers to get registration info for all UIDs on a subnet
# ---------------------------------------------------------------------
def get_registration_blocks(netuid: int, subtensor):
    """Get registration blocks for all UIDs in a subnet."""
    substrate = subtensor.substrate
    current_block = substrate.get_block_number(None)
    total_neurons = substrate.query("SubtensorModule", "SubnetworkN", [netuid]).value
    avg_block_time = 12
    result = {}

    bt.logging.info(f"Fetching registration blocks for {total_neurons} neurons in subnet {netuid}...")
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    
    for uid in range(total_neurons):
        try:
            reg_block = substrate.query(
                "SubtensorModule", "BlockAtRegistration", [netuid, uid]
            ).value
            
            if reg_block > 0:
                blocks_ago = current_block - reg_block
                ts = now - datetime.timedelta(seconds=blocks_ago * avg_block_time)
                result[uid] = {"block": reg_block, "timestamp": ts}
                
            if uid % 10 == 0:  # Log progress every 10 UIDs
                bt.logging.debug(f"Processed UID {uid}/{total_neurons}")
                
        except Exception as e:
            bt.logging.error(f"Error getting registration block for UID {uid}: {e}")
        
        # Small delay to avoid overwhelming the node
        time.sleep(0.05)
    
    bt.logging.info(f"Successfully fetched registration blocks for {len(result)} active neurons")
    return result


# ---------------------------------------------------------------------
# 3. Main patch function
# ---------------------------------------------------------------------
def apply_registration_cutoff(
    state_path: str,
    subtensor,
    repo_id: str = "BitAgent/tool_shuffle_small",
    netuid: int = 20,
):
    """Apply model longevity fix to correct historical dataset scores.
    
    This zeroes out scores for miners who were registered after a dataset was published.
    """
    bt.logging.info("[model_longevity_fix] Starting registration cutoff patch...")
    
    # --- load state ---------------------------------------------------
    state = np.load(state_path, allow_pickle=True)
    if "dataset_scores" not in state:
        bt.logging.warning("[model_longevity_fix] No dataset_scores found in state, skipping patch")
        return
        
    dataset_scores = state["dataset_scores"].item()  # dict{rv -> np.array}
    if not dataset_scores:
        bt.logging.warning("[model_longevity_fix] Empty dataset_scores, skipping patch")
        return

    bt.logging.info(f"[model_longevity_fix] Found {len(dataset_scores)} dataset versions in state")

    # --- get commits & align with dataset_scores ----------------------
    commits = sorted(get_dataset_commits(repo_id), key=lambda x: x["timestamp"])
    n_datasets = len(dataset_scores)
    
    bt.logging.info(f"[model_longevity_fix] Found {len(commits)} commits, using last {n_datasets}")
    if len(commits) < n_datasets:
        bt.logging.warning(f"[model_longevity_fix] Not enough commits ({len(commits)}) for dataset versions ({n_datasets})")
        commits = commits  # Use all we have
    else:
        commits = commits[-n_datasets:]  # newest N commits
        
    commits.sort(key=lambda x: x["timestamp"])  # oldest→newest
    rv_sorted = sorted(dataset_scores, key=lambda x: datetime.datetime.strptime(x, "%Y%m%d%H"))

    commit_to_block = {}
    for i, rv in enumerate(rv_sorted):
        if i < len(commits):
            commit_to_block[rv] = estimate_bt_block(commits[i]["timestamp"], subtensor)
            bt.logging.debug(f"[model_longevity_fix] Mapped {rv} to block {commit_to_block[rv]}")
        else:
            # Handle case where we have more dataset versions than commits
            bt.logging.warning(f"[model_longevity_fix] No commit found for dataset version {rv}")
            # Estimate based on the date in rv (format: YYYYMMDDhh)
            try:
                rv_date = datetime.datetime.strptime(rv, "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)
                commit_to_block[rv] = estimate_bt_block(rv_date, subtensor)
                bt.logging.debug(f"[model_longevity_fix] Estimated block {commit_to_block[rv]} for {rv} from its date")
            except:
                # If all else fails, use current block (conservative)
                commit_to_block[rv] = subtensor.get_current_block()
                bt.logging.warning(f"[model_longevity_fix] Using current block for {rv}")

    # --- registration info -------------------------------------------
    bt.logging.info("[model_longevity_fix] Fetching registration blocks for all UIDs...")
    reg_info = get_registration_blocks(netuid, subtensor)
    bt.logging.info(f"[model_longevity_fix] Retrieved registration info for {len(reg_info)} UIDs")

    # --- zero out impossible scores ----------------------------------
    change_count = 0
    for rv, vec in dataset_scores.items():
        block_cutoff = commit_to_block.get(rv, 0)
        if block_cutoff == 0:
            bt.logging.warning(f"[model_longevity_fix] No block cutoff for {rv}, skipping")
            continue
            
        bt.logging.debug(f"[model_longevity_fix] Processing dataset {rv} with block cutoff {block_cutoff}")
        
        for uid, info in reg_info.items():
            if info["block"] > block_cutoff and uid < len(vec):
                if vec[uid] != 0.0:  # Only count non-zero changes
                    bt.logging.debug(f"[model_longevity_fix] Zeroing UID {uid} for dataset {rv} (registered at {info['block']} > cutoff {block_cutoff})")
                    vec[uid] = 0.0
                    change_count += 1

    bt.logging.info(f"[model_longevity_fix] Made {change_count} corrections across all datasets")

    # --- save back ----------------------------------------------------
    # Create a copy of the state to modify
    updated_state = dict(state.item()) if isinstance(state, np.ndarray) else dict(state)
    updated_state["dataset_scores"] = dataset_scores
    updated_state["longevity_fix_applied"] = True  # Add flag to indicate fix has been applied
    
    np.savez(state_path, **updated_state)
    bt.logging.info("[model_longevity_fix] State patched & saved with 'longevity_fix_applied' flag")
    
    # Return dataset_scores for immediate use
    return dataset_scores
import random
import numpy as np
import bittensor as bt
from typing import List

def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    #if not metagraph.axons[uid].is_serving:
    #    return False
    # don't hit sn owner
    if uid == 0:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # any miner receiving incentive should be queried
    if metagraph.I[uid] > 0:
        return True
    # Available otherwise.
    return True


def get_random_uids(
    self, k: int, exclude: List[int] = None
) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []
    # countering the effect of setting seed for task orchestration from validators
    random.seed(None)

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        ## actually check that the axon is there and alive
        #result = self.dendrite.query(
        #    # Send the query to selected miner axons in the network.
        #    axons=[self.metagraph.axons[uid]],
        #    # Construct a query. 
        #    synapse=bitagent.protocol.IsAlive(response=False),
        #    # All responses have the deserialize function called on them before returning.
        #    # You are encouraged to define your own deserialization function.
        #    deserialize=False,
        #)
        #uid_is_available = result[0].response

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    while True:
        try:
            if len(candidate_uids) < k:
                available_uids += random.sample(
                    [uid for uid in avail_uids if uid not in candidate_uids],
                    k - len(candidate_uids),
                )
            uids = random.sample(available_uids, k)
            return uids
        except Exception as e:
            bt.logging.debug(f"Reduced sample size from {k} to {k-1} and trying again.")
            k -= 1

def get_uid_rank(self, uid: int) -> int:
    """Returns the rank of the uid in the metagraph.
    Args:
        uid (int): uid to get the rank of.
    Returns:
        rank (int): Rank of the uid in the metagraph.
    """
    # Get the rank of the uid in the metagraph.
    rank = (-self.metagraph.I).argsort().tolist().index(uid)
    return rank
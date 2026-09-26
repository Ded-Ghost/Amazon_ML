"""Train / validation split of Source 1 entities.

We split only the S1 entities. S2 and S3 records stay shared by both halves,
the same way the test set works: every S1 entity searches the full S2/S3 pool.
IDs are sorted before shuffling so the split does not depend on file order.
"""
import numpy as np

from .config import SEED, VAL_FRACTION


def split_s1_ids(s1_ids, val_fraction=VAL_FRACTION, seed=SEED):
    """Return (train_ids, val_ids) as lists."""
    ids = np.array(sorted(s1_ids))
    order = np.random.default_rng(seed).permutation(len(ids))
    n_val = int(round(len(ids) * val_fraction))
    return ids[order[n_val:]].tolist(), ids[order[:n_val]].tolist()

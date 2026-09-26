"""Run blocking for the validation split or the test set. Run:

    python -m src.run_blocking --split train   # 300k-entity sample, to train the matcher
    python -m src.run_blocking --split val     # + recall report
    python -m src.run_blocking --split rest    # other train entities (full-competition validation)
    python -m src.run_blocking --split test    # + output/candidate_pairs.tsv

The top-100 candidates per S1 entity are saved to cache/candidates_<split>.npz
as integer arrays (1.7M x 100 pairs as Python strings would need ~10 GB of
RAM). Later phases load them and decide how many (K) to use.
"""
import argparse
import time

import numpy as np
import pandas as pd

from .blocking import generate_candidates
from .config import CACHE_DIR, OUTPUT_DIR, SEED
from .data import load_ground_truth, load_source
from .metrics import f05_entity
from .output_writer import write_submission
from .split import split_s1_ids

TEST_CANDIDATES_K = 50  # provisional; the matching phase may change it
TRAIN_SAMPLE = 300_000  # train S1 entities used to train the matcher (plenty of pairs)


def load_split(split):
    """(S1 records, S2+S3 pool) for 'train', 'val' or 'test'.

    'train' is a seeded sample of TRAIN_SAMPLE entities from the 80% train half;
    'rest' is the remainder of the train half (never used to train the matcher).
    """
    data_split = "test" if split == "test" else "train"
    s1 = load_source(data_split, 1)
    pool = pd.concat([load_source(data_split, 2), load_source(data_split, 3)], ignore_index=True)
    if split in ("train", "val", "rest"):
        train_ids, val_ids = split_s1_ids(s1["entity_id"])
        sample = sorted(np.random.default_rng(SEED).choice(train_ids, size=TRAIN_SAMPLE, replace=False))
        ids = {"train": sample, "val": val_ids, "rest": sorted(set(train_ids) - set(sample))}[split]
        s1 = s1.set_index("entity_id").loc[ids].reset_index()
    return s1, pool


def candidates_path(split):
    return CACHE_DIR / f"candidates_{split}.npz"


def save_candidates(split, s1_ids, pool_ids, cand, score):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(candidates_path(split), s1_ids=np.asarray(s1_ids, dtype="S"),
             pool_ids=np.asarray(pool_ids, dtype="S"), cand=cand, score=score)


def load_candidates(split):
    """Returns (s1_ids, pool_ids, cand, score); cand indexes pool_ids, -1 = empty."""
    z = np.load(candidates_path(split))
    return (z["s1_ids"].astype(str), z["pool_ids"].astype(str), z["cand"], z["score"])


def candidate_lists(pool_ids, cand, k):
    """{row: [pool ids]} using the first k candidates of every row."""
    return [pool_ids[row[:k][row[:k] >= 0]].tolist() for row in cand]


def recall_report(s1_df, pool_ids, cand, gt, cutoffs=(5, 10, 20, 30, 50, 100)):
    """Pair recall, oracle F0.5 and candidates per entity for several K.

    oracle F0.5 = score of a perfect matcher that picks exactly the true
    matches among the candidates: the ceiling for the matching phase.
    """
    tables = []
    for k in cutoffs:
        per_entity = []
        for s1_id, cands in zip(s1_df["entity_id"], candidate_lists(pool_ids, cand, k)):
            true = set(gt[s1_id])
            hit = true.intersection(cands)
            per_entity.append((len(hit), len(true), len(cands), f05_entity(hit, true)))
        df = pd.DataFrame(per_entity, columns=["found", "true", "cands", "oracle"])
        df["country"] = s1_df["country"].to_numpy()
        for country, g in [*df.groupby("country"), ("ALL", df)]:
            tables.append({"K": k, "country": country, "entities": len(g),
                           "pair recall": g["found"].sum() / g["true"].sum(),
                           "oracle F0.5": g["oracle"].mean(), "cands/entity": g["cands"].mean()})
    return pd.DataFrame(tables).round(4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "rest", "test"], required=True)
    args = parser.parse_args()

    start = time.time()
    s1, pool = load_split(args.split)
    print(f"{args.split}: {len(s1):,} S1 entities, {len(pool):,} pool records")
    cand, score = generate_candidates(s1, pool)
    pool_ids = pool["entity_id"].to_numpy()
    save_candidates(args.split, s1["entity_id"], pool_ids, cand, score)
    print(f"blocking done in {time.time() - start:.0f}s -> {candidates_path(args.split)}")

    if args.split == "val":
        print("\n" + recall_report(s1, pool_ids, cand, load_ground_truth()).to_string(index=False))
    if args.split == "test":
        lists = candidate_lists(pool_ids, cand, TEST_CANDIDATES_K)
        candidates = dict(zip(s1["entity_id"], lists))
        write_submission(OUTPUT_DIR, s1["entity_id"], matches={}, candidates=candidates)
        print(f"wrote candidate_pairs.tsv (K={TEST_CANDIDATES_K}) and empty matching_results.tsv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

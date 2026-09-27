"""Re-score the S1 records of some countries at a deeper blocking depth.

    python -m src.rescore_unseen --k 200 --base cache/v11_test_scores.npz              # countries without labels
    python -m src.rescore_unseen --k 200 --countries India --base cache/v11_test_scores.npz
    python -m src.predict --reuse --unseen-adapt --unseen-strict --threshold 0.675 --out <dir>

A faster equivalent of scoring the whole test set with deeper blocking lists:
S2/S3 records never match across country labels, so the S1 entities of each country
compete only with each other and their scores can be computed separately. The S1
records of countries without training labels are re-blocked to depth K and scored
with the trained models (two passes, adaptation); the saved scores of the other
countries are kept. Label-free counts are keyed by country, so they are unchanged;
only the word self-training pool differs slightly (it holds these countries' pairs
only, which is where the unknown words are). Overwrites cache/test_scores.npz.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from .blocking import generate_candidates
from .config import CACHE_DIR
from .data import load_source
from .features import add_core_names, frequent_name_words, prepare_records
from .match_features import NameCounts
from .predict import score_pairs
from .run_blocking import load_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True, help="blocking depth for countries without training labels")
    parser.add_argument("--base", required=True, help="saved test scores whose other countries are kept")
    parser.add_argument("--countries", default=None,
                        help="comma-separated country labels to re-score (default: those without training labels)")
    args = parser.parse_args()
    start = time.time()
    settings = dict(json.loads((CACHE_DIR / "matcher.json").read_text()), K=args.k)
    s1_ids, pool_ids, _, _ = load_candidates("test")
    s1_df = load_source("test", 1).set_index("entity_id").loc[s1_ids].reset_index()
    pool = pd.concat([load_source("test", 2), load_source("test", 3)], ignore_index=True)
    assert (pool["entity_id"].to_numpy() == pool_ids).all(), "pool order differs from blocking"
    train_countries = set(load_source("train", 1, usecols=["country"])["country"])
    chosen = (s1_df["country"].isin(args.countries.split(",")) if args.countries
              else ~s1_df["country"].isin(train_countries)).to_numpy()
    unseen = np.flatnonzero(chosen)  # rows to re-score
    s1_unseen = s1_df.iloc[unseen].reset_index(drop=True)
    cand, score = generate_candidates(s1_unseen, pool, args.k)
    print(f"{len(unseen):,} S1 records re-blocked to depth {args.k} ({time.time() - start:.0f}s)", flush=True)

    pool_prep, pool_numbers = prepare_records(pool)
    frequent = frequent_name_words(pool_prep)
    add_core_names(pool_prep, frequent)
    s1_prep, s1_numbers = prepare_records(s1_unseen)
    add_core_names(s1_prep, frequent)
    del pool
    name_counts = NameCounts(s1_prep, pool_prep)
    s1_row, pool_row, prob, adapted = score_pairs(s1_prep, s1_numbers, pool_prep, pool_numbers, cand, score,
                                                  name_counts, settings, train_countries=train_countries)

    base = np.load(args.base)
    keep = ~np.isin(base["s1_row"], unseen)
    np.savez(CACHE_DIR / "test_scores.npz",
             s1_row=np.concatenate([base["s1_row"][keep], unseen[s1_row]]),
             pool_row=np.concatenate([base["pool_row"][keep], pool_row]),
             prob=np.concatenate([base["prob"][keep], prob]),
             adapted=np.concatenate([base["adapted"][keep], adapted]))
    print(f"kept {keep.sum():,} pairs of the other countries, re-scored {len(s1_row):,} pairs "
          f"-> cache/test_scores.npz ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()

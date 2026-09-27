"""Predict matches for the TEST set and write the submission files. Run:

    python -m src.predict
    python -m src.predict --reuse --threshold 0.6 --out ../../output_variants/thr_0.60

The second form re-uses the saved test probabilities (cache/test_scores.npz)
and only re-applies the decision rule, e.g. to try another threshold.

Needs cache/candidates_test.npz (run_blocking --split test) and the models
from train_matcher (cache/reranker.txt, matcher.txt, matcher.json).

For the top-50 blocking candidates of each S1 entity: compute features, let
the re-ranker keep at most KEEP_MAX of them, add the word, cluster and count
features (src/match_features.py) and run the matcher on those, in two passes:
pass 1 uses the word scores learned in training; words never seen in training
(mostly French) then get scores from confident pass-1 predictions
(self-training, no labels), and pass 2 gives the final probabilities.
Writes output/matching_results.tsv and output/candidate_pairs.tsv, where
candidate_pairs.tsv holds exactly the candidates the matcher scored.
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CACHE_DIR, OUTPUT_DIR, SEED
from .data import load_source
from .features import (add_core_names, build_pairs, compute_features, entity_chunks,
                       frequent_name_words, prepare_records, refresh_context_features)
from .match_features import (NameCounts, WordScores, assemble, cluster_features, format_features, sibling_features,
                             self_trained_word_features, word_differences)
from .normalize import address_tokens
from .output_writer import write_submission
from .rerank import keep_mask, rerank_probability
from .run_blocking import load_candidates
from .train_matcher import one_owner_per_record


def load_models():
    return (lgb.Booster(model_file=str(CACHE_DIR / "reranker.txt")),
            lgb.Booster(model_file=str(CACHE_DIR / "matcher.txt")),
            WordScores.load(CACHE_DIR / "word_scores.json"))


def kept_candidate_features(s1_prep, s1_numbers, pool_prep, pool_numbers, cand, score, name_counts,
                            settings, reranker, log_prefix=""):
    """Final candidate pairs and all their matcher inputs, built chunk by chunk.

    Blocking candidates are processed in chunks of whole S1 entities: pair
    features -> re-ranker cut -> word differences, cluster, count and
    formatting features. Returns a dict with s1_row, pool_row, X_base, diffs,
    cluster, counts, formatting (word scores are applied later).
    """
    start = time.time()
    s1_names = s1_prep["name"].to_numpy()
    pool_names = pool_prep["name"].to_numpy()
    pairs = build_pairs(cand, score, settings["K"])
    parts = []
    for lo, hi in entity_chunks(pairs["s1_row"].to_numpy()):
        chunk = pairs.iloc[lo:hi].reset_index(drop=True)
        X = compute_features(chunk, s1_prep, s1_numbers, pool_prep, pool_numbers)
        keep = keep_mask(chunk["s1_row"].to_numpy(), rerank_probability(reranker, X),
                         settings["keep_max"], settings["min_prob"])
        X = X[keep]
        s1_row = chunk["s1_row"].to_numpy()[keep]
        pool_row = chunk["pool_row"].to_numpy()[keep]
        refresh_context_features(X, s1_row)
        diffs = word_differences(s1_names, pool_names, s1_row, pool_row)
        counts = name_counts.features(s1_prep, s1_numbers, pool_prep, pool_numbers, s1_row, pool_row, diffs)
        parts.append((s1_row, pool_row, X, diffs, cluster_features(s1_row, pool_row, pool_prep), counts,
                      format_features(s1_prep, pool_prep, s1_row, pool_row)))
        print(f"{log_prefix}  features for {hi:,} / {len(pairs):,} blocking pairs "
              f"({time.time() - start:.0f}s)", flush=True)
    return {
        "s1_row": np.concatenate([p[0] for p in parts]),
        "pool_row": np.concatenate([p[1] for p in parts]),
        "X_base": np.concatenate([p[2] for p in parts]),
        "diffs": [d for p in parts for d in p[3]],
        "cluster": {k: np.concatenate([p[4][k] for p in parts]) for k in parts[0][4]},
        "counts": {k: np.concatenate([p[5][k] for p in parts]) for k in parts[0][5]},
        "formatting": {k: np.concatenate([p[6][k] for p in parts]) for k in parts[0][6]},
    }


def score_pairs(s1_prep, s1_numbers, pool_prep, pool_numbers, cand, score, name_counts, settings,
                log_prefix=""):
    """Final candidate pairs (s1_row, pool_row) and their match probabilities.

    Used for the test set and for the full-competition validation. The matcher
    runs in two passes with self-training for words never seen in training.
    """
    start = time.time()
    reranker, model, word_scores = load_models()
    k = kept_candidate_features(s1_prep, s1_numbers, pool_prep, pool_numbers, cand, score, name_counts,
                                settings, reranker, log_prefix)
    k["counts"].update(sibling_features(k["diffs"], k["X_base"], s1_prep["country"].to_numpy()[k["s1_row"]]))

    def predict(word_features):
        return model.predict(assemble(k["X_base"], word_features, k["cluster"], k["counts"], k["formatting"]))

    # pass 1 with the training word scores; then words never seen in training
    # (mostly French) get scores from confident pass-1 predictions (self-training)
    first = predict(word_scores.features(k["diffs"]))
    prob = predict(self_trained_word_features(k["diffs"], first, k["s1_row"], word_scores, SEED)).astype(np.float32)
    print(f"{log_prefix}matcher done, two passes ({time.time() - start:.0f}s)", flush=True)
    return k["s1_row"], k["pool_row"], prob


def score_test(settings):
    """Final candidate pairs of the test set and their match probabilities."""
    start = time.time()
    s1_ids, pool_ids, cand, score = load_candidates("test")
    s1_df = load_source("test", 1).set_index("entity_id").loc[s1_ids].reset_index()
    pool = pd.concat([load_source("test", 2), load_source("test", 3)], ignore_index=True)
    assert (pool["entity_id"].to_numpy() == pool_ids).all(), "pool order differs from blocking"

    # frequent name words are learned from the test pool itself (any country, incl. new ones)
    pool_prep, pool_numbers = prepare_records(pool)
    frequent = frequent_name_words(pool_prep)
    add_core_names(pool_prep, frequent)
    s1_prep, s1_numbers = prepare_records(s1_df)
    add_core_names(s1_prep, frequent)
    del pool
    print(f"records prepared in {time.time() - start:.0f}s", flush=True)

    name_counts = NameCounts(s1_prep, pool_prep)  # over ALL test S1 records and the test pool
    s1_row, pool_row, prob = score_pairs(s1_prep, s1_numbers, pool_prep, pool_numbers,
                                         cand, score, name_counts, settings)
    np.savez(CACHE_DIR / "test_scores.npz", s1_row=s1_row, pool_row=pool_row, prob=prob)
    return s1_row, pool_row, prob


def _dropped_digit(a, b):
    return any((len(x) == len(y) + 1 and any(x[:i] + x[i + 1:] == y for i in range(len(x)))) or
               (len(y) == len(x) + 1 and any(y[:i] + y[i + 1:] == x for i in range(len(y))))
               for x in a for y in b)


def unseen_rows(s1_row):
    """True for pairs whose S1 entity has a country label that never occurs in training."""
    s1_ids = load_candidates("test")[0]
    country = load_source("test", 1, usecols=["entity_id", "country"]).set_index("entity_id").loc[s1_ids, "country"]
    seen = set(load_source("train", 1, usecols=["country"])["country"])
    return ~np.isin(country.to_numpy()[s1_row], list(seen))


def unseen_number_conflicts(s1_row, pool_row, predicted):
    """Predicted pairs in countries WITHOUT training labels whose house numbers conflict.

    In the training countries a same-name record at a different house number is
    still a true copy 58% of the time (the generator scrambles addresses there),
    so the matcher learned to trust names. In a country it has never seen, a
    different house number is treated as a different business (a sibling or a
    namesake) unless it only differs by a dropped digit. Applies to any
    country label that does not occur in the training data.
    """
    s1_ids = load_candidates("test")[0]
    s1 = load_source("test", 1).set_index("entity_id").loc[s1_ids].reset_index()
    pool_addr = pd.concat([load_source("test", 2, usecols=["business_address"]),
                           load_source("test", 3, usecols=["business_address"])],
                          ignore_index=True)["business_address"].to_numpy()
    s1_addr = s1["business_address"].to_numpy()
    unseen = unseen_rows(s1_row)
    conflict = np.zeros(len(s1_row), dtype=bool)
    for i in np.flatnonzero(predicted & unseen):
        a = {t for t in address_tokens(s1_addr[s1_row[i]]) if t.isdigit()}
        b = {t for t in address_tokens(pool_addr[pool_row[i]]) if t.isdigit()}
        conflict[i] = bool(a) and bool(b) and not (a & b) and not _dropped_digit(a, b)
    print(f"unseen-country rule: {conflict.sum():,} predicted pairs with conflicting house numbers removed")
    return conflict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="re-use cache/test_scores.npz")
    parser.add_argument("--threshold", type=float, default=None, help="override the tuned threshold")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="output folder")
    parser.add_argument("--unseen-strict", action="store_true",
                        help="in countries without training labels, reject matches whose "
                             "house numbers conflict")
    parser.add_argument("--unseen-threshold", type=float, default=None,
                        help="threshold for countries without training labels")
    args = parser.parse_args()
    start = time.time()
    settings = json.loads((CACHE_DIR / "matcher.json").read_text())
    threshold = settings["threshold"] if args.threshold is None else args.threshold

    if args.reuse:
        z = np.load(CACHE_DIR / "test_scores.npz")
        s1_row, pool_row, prob = z["s1_row"], z["pool_row"], z["prob"]
    else:
        s1_row, pool_row, prob = score_test(settings)
    s1_ids, pool_ids = load_candidates("test")[:2]
    s1_country = load_source("test", 1, usecols=["entity_id", "country"]).set_index("entity_id").loc[s1_ids, "country"]
    predicted = prob >= threshold
    if args.unseen_threshold is not None:
        predicted = np.where(unseen_rows(s1_row), prob >= args.unseen_threshold, predicted)
    if args.unseen_strict:
        predicted &= ~unseen_number_conflicts(s1_row, pool_row, predicted)
    if settings["one_owner_per_record"]:
        predicted = one_owner_per_record(pool_row, prob, predicted)

    matches = {s: [] for s in s1_ids}
    candidates = {s: [] for s in s1_ids}
    for r, c in zip(s1_row, pool_row):
        candidates[s1_ids[r]].append(pool_ids[c])
    for r, c in zip(s1_row[predicted], pool_row[predicted]):
        matches[s1_ids[r]].append(pool_ids[c])
    write_submission(args.out, s1_ids, matches, candidates)

    # sanity check per country: are predictions for France in the same range as US/India?
    summary = pd.DataFrame({"country": s1_country.to_numpy(),
                            "candidates": [len(candidates[s]) for s in s1_ids],
                            "matches": [len(matches[s]) for s in s1_ids]})
    print("\nper S1 entity, by country:")
    print(summary.groupby("country").agg(
        entities=("matches", "size"), candidates_mean=("candidates", "mean"),
        matches_mean=("matches", "mean"),
        share_no_match=("matches", lambda x: (x == 0).mean())).round(3).to_string())
    print(f"total candidate pairs: {len(s1_row):,} ({len(s1_row) / len(s1_ids):.2f} per S1 entity)")
    print(f"\nthreshold {threshold}: wrote {args.out / 'matching_results.tsv'} and "
          f"candidate_pairs.tsv in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()

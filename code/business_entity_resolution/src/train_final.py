"""Final matcher, trained on ALL labelled training entities. Run:

    python -m src.run_blocking --split rest      # once (candidates of the other train entities)
    python -m src.train_final                     # after train_matcher / validate_competition

train_matcher chooses everything (re-ranker, features, word dropout, model
settings) on a 300k-entity sample and measures it on the validation split.
Each doubling of the training data added about +0.001 F0.5 on validation
(67k -> 135k -> 270k entities: 0.9754 -> 0.9765 -> 0.9775), so the final
matcher is retrained with the same settings on every labelled S1 entity
(train sample + rest + validation, ~2.2M entities). The re-ranker and the
threshold are kept. Early stopping uses 5% of the entities.

Overwrites cache/matcher.txt and cache/word_scores.json; the versions trained
by train_matcher are kept as matcher_sample.txt / word_scores_sample.json.
"""
import json
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CACHE_DIR, SEED
from .data import load_ground_truth, load_source
from .features import add_core_names, frequent_name_words, prepare_records
from .match_features import (MATCH_FEATURES, NameCounts, WordScores, assemble, cross_fitted_word_features,
                             sibling_features)
from .predict import kept_candidate_features
from .run_blocking import load_candidates
from .train_matcher import LGB_PARAMS, pair_labels


def main():
    start = time.time()
    settings = json.loads((CACHE_DIR / "matcher.json").read_text())
    reranker = lgb.Booster(model_file=str(CACHE_DIR / "reranker.txt"))
    gt = load_ground_truth()
    s1_all = load_source("train", 1)
    pool = pd.concat([load_source("train", 2), load_source("train", 3)], ignore_index=True)
    pool_prep, pool_numbers = prepare_records(pool)
    frequent = frequent_name_words(pool_prep)
    add_core_names(pool_prep, frequent)
    s1_prep_all, s1_numbers_all = prepare_records(s1_all)
    add_core_names(s1_prep_all, frequent)
    name_counts = NameCounts(s1_prep_all, pool_prep)

    parts = [load_candidates(split) for split in ("train", "rest", "val")]
    pool_ids = parts[0][1]
    assert all((p[1] == pool_ids).all() for p in parts) and (pool["entity_id"].to_numpy() == pool_ids).all()
    s1_ids = np.concatenate([p[0] for p in parts])
    rows = pd.Index(s1_all["entity_id"]).get_indexer(s1_ids)
    s1_prep, s1_numbers = s1_prep_all.iloc[rows].reset_index(drop=True), s1_numbers_all[rows]
    del pool, s1_all, s1_prep_all, s1_numbers_all
    print(f"{len(s1_ids):,} labelled entities prepared ({time.time() - start:.0f}s)", flush=True)

    k = kept_candidate_features(s1_prep, s1_numbers, pool_prep, pool_numbers,
                                np.vstack([p[2] for p in parts]), np.vstack([p[3] for p in parts]),
                                name_counts, settings, reranker)
    del pool_prep, pool_numbers, parts
    k["counts"].update(sibling_features(k["diffs"], k["X_base"], s1_prep["country"].to_numpy()[k["s1_row"]]))
    s1_row = k["s1_row"]
    y = pair_labels(pd.DataFrame({"s1_row": s1_row, "pool_row": k["pool_row"]}), s1_ids, pool_ids, gt)
    print(f"{len(y):,} kept pairs, {y.sum():,} true ({time.time() - start:.0f}s)", flush=True)

    word_scores = WordScores.learn(k["diffs"], y)
    X = assemble(k["X_base"], cross_fitted_word_features(k["diffs"], y, s1_row, SEED),
                 k["cluster"], k["counts"], k["formatting"])
    del k

    holdout = (np.random.default_rng(SEED).random(len(s1_ids)) < 0.05)[s1_row]
    dtrain = lgb.Dataset(X[~holdout], y[~holdout], feature_name=MATCH_FEATURES)
    dhold = lgb.Dataset(X[holdout], y[holdout], reference=dtrain)
    model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=5000, valid_sets=[dhold],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    print(f"final matcher: {model.best_iteration} trees ({time.time() - start:.0f}s)", flush=True)

    for name in ("matcher.txt", "word_scores.json"):
        backup = CACHE_DIR / name.replace(".", "_sample.", 1)
        if not backup.exists():
            shutil.copy(CACHE_DIR / name, backup)
    model.save_model(str(CACHE_DIR / "matcher.txt"), num_iteration=model.best_iteration)
    word_scores.save(CACHE_DIR / "word_scores.json")
    settings["final_training_entities"] = int(len(s1_ids))
    (CACHE_DIR / "matcher.json").write_text(json.dumps(settings, indent=2))
    print(f"saved final matcher trained on {len(s1_ids):,} entities to {CACHE_DIR}")


if __name__ == "__main__":
    main()

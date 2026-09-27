"""Train the re-ranker and the matching model, and tune the decision rule. Run:

    python -m src.train_matcher

Needs cache/candidates_train.npz and cache/candidates_val.npz
(python -m src.run_blocking --split train / --split val).

1. Features for the top-50 blocking candidates of every S1 entity
   (src/features.py). Label = 1 if the candidate is in the ground truth.
2. Re-ranker (src/rerank.py): a small LightGBM on the 26 pair features keeps at
   most KEEP_MAX candidates per entity. These are the final candidates.
3. Extra features on the kept candidates (src/match_features.py): extra /
   missing words with scores learned from the training pairs (cross-fitted,
   with 30% word dropout for the training rows), similarity to the other kept
   candidates, and label-free counts (how many S1 businesses share a name,
   how common each differing word is, house-number distance).
4. Matcher: LightGBM binary classifier on the kept candidates of the train
   sample. Early stopping (for both models) uses 10% of the *train* entities,
   so the validation split stays unseen.
5. Decision rule, tuned on validation for the exact macro F0.5:
   - predict a candidate when its probability >= threshold
   - one owner per record: the EDA showed every S2/S3 record belongs to at
     most one S1 entity, so if a record is predicted for several S1
     entities we keep only the most probable one.
6. Saves cache/reranker.txt, cache/word_scores.json, cache/matcher.txt and
   cache/matcher.json.
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CACHE_DIR, SEED
from .data import load_ground_truth, load_source
from .features import (add_core_names, build_pairs, compute_features,
                       frequent_name_words, prepare_records, refresh_context_features)
from .match_features import (MATCH_FEATURES, NameCounts, WordScores, assemble, cluster_features, format_features,
                             sibling_features,
                             cross_fitted_word_features, self_trained_word_features,
                             word_differences)
from .metrics import macro_f05, score_report
from .rerank import KEEP_MAX, MIN_PROB, keep_mask, rerank_probability, train_reranker
from .run_blocking import load_candidates

K = 50  # blocking candidates per S1 entity given to the re-ranker
THRESHOLDS = np.round(np.arange(0.30, 0.96, 0.025), 3)
LGB_PARAMS = {
    "objective": "binary", "learning_rate": 0.1, "num_leaves": 127,
    "min_child_samples": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 1.0, "seed": SEED, "deterministic": True,
    "force_row_wise": True, "num_threads": 16, "verbose": -1,
}


def pair_labels(pairs, s1_ids, pool_ids, gt):
    """1 if (S1, candidate) is a true match, else 0."""
    s1_true, match_true = [], []
    for row, s1_id in enumerate(s1_ids):
        for match in gt[s1_id]:
            s1_true.append(row)
            match_true.append(match)
    pool_true = pd.Index(pool_ids).get_indexer(match_true)
    n_pool = np.int64(len(pool_ids))
    true_keys = np.asarray(s1_true, dtype=np.int64) * n_pool + pool_true
    keys = pairs["s1_row"].to_numpy(np.int64) * n_pool + pairs["pool_row"].to_numpy()
    return np.isin(keys, true_keys)


def one_owner_per_record(pool_row, prob, predicted):
    """Keep, for every pool record, only its most probable predicted S1 entity."""
    idx = np.flatnonzero(predicted)
    order = idx[np.lexsort((-prob[idx], pool_row[idx]))]
    first = np.r_[True, pool_row[order][1:] != pool_row[order][:-1]]
    keep = np.zeros_like(predicted)
    keep[order[first]] = True
    return keep


def fast_macro_f05(s1_row, predicted, label, n_true):
    """Vectorised version of metrics.macro_f05 (checked against it below)."""
    n = len(n_true)
    tp = np.bincount(s1_row[predicted & label], minlength=n)
    n_pred = np.bincount(s1_row[predicted], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        p, r = tp / n_pred, tp / n_true
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where(n_true == 0, (n_pred == 0).astype(float), f)
    return f.mean()


def build_split(split, s1_index, s1_prep_all, s1_numbers_all, pool_prep, pool_numbers, gt):
    """Pairs, features and labels for the 'train' or 'val' candidates."""
    s1_ids, pool_ids, cand, score = load_candidates(split)
    rows = s1_index.get_indexer(s1_ids)
    s1_prep = s1_prep_all.iloc[rows].reset_index(drop=True)
    s1_numbers = s1_numbers_all[rows]
    pairs = build_pairs(cand, score, K)
    start = time.time()
    X = compute_features(pairs, s1_prep, s1_numbers, pool_prep, pool_numbers)
    y = pair_labels(pairs, s1_ids, pool_ids, gt)
    n_true = np.array([len(gt[s]) for s in s1_ids])
    print(f"{split}: {len(s1_ids):,} entities, {len(pairs):,} pairs, {y.sum():,} positives "
          f"({time.time() - start:.0f}s for features)", flush=True)
    return {"s1_ids": s1_ids, "pool_ids": pool_ids, "pairs": pairs, "X": X, "y": y,
            "n_true": n_true, "s1_prep": s1_prep, "s1_numbers": s1_numbers}


def keep_candidates(name, split, reranker):
    """Keep only the candidates the re-ranker lets through (the final candidate set)."""
    s1_row = split["pairs"]["s1_row"].to_numpy()
    keep = keep_mask(s1_row, rerank_probability(reranker, split["X"]))
    X = split["X"][keep]
    refresh_context_features(X, s1_row[keep])
    kept = dict(split, pairs=split["pairs"][keep].reset_index(drop=True), X=X, y=split["y"][keep])

    sizes = np.bincount(s1_row[keep], minlength=len(split["s1_ids"]))
    found = np.bincount(s1_row[keep & split["y"]], minlength=len(sizes))
    n_true = split["n_true"]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = found / n_true
        oracle = np.where(n_true == 0, 1.0, np.where(found > 0, 1.25 * r / (0.25 + r), 0.0))
    print(f"{name}: {sizes.mean():.2f} candidates per entity (median {np.median(sizes):.0f}, "
          f"{100 * (sizes == 0).mean():.1f}% empty), pair recall {found.sum() / n_true.sum():.4f}, "
          f"oracle F0.5 {oracle.mean():.4f}", flush=True)
    return kept


def main():
    start = time.time()
    gt = load_ground_truth()
    s1_all = load_source("train", 1)
    pool = pd.concat([load_source("train", 2), load_source("train", 3)], ignore_index=True)
    pool_prep, pool_numbers = prepare_records(pool)
    frequent = frequent_name_words(pool_prep)
    add_core_names(pool_prep, frequent)
    # all S1 records (both halves): name counts are taken over the whole S1 table,
    # exactly as they are over the whole test S1 table at prediction time
    s1_prep_all, s1_numbers_all = prepare_records(s1_all)
    add_core_names(s1_prep_all, frequent)
    name_counts = NameCounts(s1_prep_all, pool_prep)
    for country, words in frequent.items():
        print(f"frequent name words ({country}, {len(words)}): {sorted(words)[:25]} ...")
    print(f"records prepared in {time.time() - start:.0f}s", flush=True)

    s1_index = pd.Index(s1_all["entity_id"])
    train = build_split("train", s1_index, s1_prep_all, s1_numbers_all, pool_prep, pool_numbers, gt)
    val = build_split("val", s1_index, s1_prep_all, s1_numbers_all, pool_prep, pool_numbers, gt)
    assert (train["pool_ids"] == pool["entity_id"].to_numpy()).all()
    del pool, s1_all, s1_prep_all, s1_numbers_all

    # ---------------------------------------------------------------- re-ranker
    rng = np.random.default_rng(SEED)
    holdout_entity = rng.random(len(train["s1_ids"])) < 0.1
    holdout = holdout_entity[train["pairs"]["s1_row"].to_numpy()]
    reranker = train_reranker(train["X"], train["y"], holdout)
    print(f"re-ranker: {reranker.best_iteration} trees; keep at most {KEEP_MAX} candidates "
          f"with probability >= {MIN_PROB}", flush=True)
    train = keep_candidates("train", train, reranker)
    val = keep_candidates("val", val, reranker)

    # ---------------------------------------------------------------- extra features
    pool_names = pool_prep["name"].to_numpy()
    for name, split in (("train", train), ("val", val)):
        s1_row = split["pairs"]["s1_row"].to_numpy()
        pool_row = split["pairs"]["pool_row"].to_numpy()
        split["diffs"] = word_differences(split["s1_prep"]["name"].to_numpy(), pool_names, s1_row, pool_row)
        split["cluster"] = cluster_features(s1_row, pool_row, pool_prep)
        split["format"] = format_features(split["s1_prep"], pool_prep, s1_row, pool_row)
        split["counts"] = name_counts.features(split["s1_prep"], split["s1_numbers"], pool_prep,
                                               pool_numbers, s1_row, pool_row, split["diffs"])
        split["counts"].update(sibling_features(split["diffs"], split["X"],
                                                split["s1_prep"]["country"].to_numpy()[s1_row]))
    word_scores = WordScores.learn(train["diffs"], train["y"])
    train["X"] = assemble(train["X"], cross_fitted_word_features(
        train["diffs"], train["y"], train["pairs"]["s1_row"].to_numpy(), SEED),
        train["cluster"], train["counts"], train["format"])
    val["X_base"] = val["X"]
    val["X"] = assemble(val["X"], word_scores.features(val["diffs"]), val["cluster"], val["counts"], val["format"])
    del pool_prep, pool_numbers, name_counts
    print(f"word, cluster and count features done ({time.time() - start:.0f}s)", flush=True)

    # ---------------------------------------------------------------- matcher
    holdout = holdout_entity[train["pairs"]["s1_row"].to_numpy()]
    dtrain = lgb.Dataset(train["X"][~holdout], train["y"][~holdout], feature_name=MATCH_FEATURES)
    dhold = lgb.Dataset(train["X"][holdout], train["y"][holdout], reference=dtrain)
    model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=2000, valid_sets=[dhold],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    print(f"trained {model.best_iteration} trees in {time.time() - start:.0f}s total", flush=True)

    # ---------------------------------------------------------------- tune on val
    prob = model.predict(val["X"], num_iteration=model.best_iteration)
    s1_row = val["pairs"]["s1_row"].to_numpy()
    pool_row = val["pairs"]["pool_row"].to_numpy()
    rows = []
    for t in THRESHOLDS:
        predicted = prob >= t
        rows.append({"threshold": t,
                     "F0.5": fast_macro_f05(s1_row, predicted, val["y"], val["n_true"]),
                     "F0.5 one-owner": fast_macro_f05(
                         s1_row, one_owner_per_record(pool_row, prob, predicted), val["y"], val["n_true"])})
    tuning = pd.DataFrame(rows)
    print("\n" + tuning.round(4).to_string(index=False))

    best = tuning.loc[tuning[["F0.5", "F0.5 one-owner"]].max(axis=1).idxmax()]
    use_one_owner = bool(best["F0.5 one-owner"] >= best["F0.5"])
    threshold = float(best["threshold"])
    predicted = prob >= threshold
    if use_one_owner:
        predicted = one_owner_per_record(pool_row, prob, predicted)

    # exact check with the Phase 1 scorer
    matches = {s: [] for s in val["s1_ids"]}
    for r, c in zip(s1_row[predicted], pool_row[predicted]):
        matches[val["s1_ids"][r]].append(val["pool_ids"][c])
    truth = {s: gt[s] for s in val["s1_ids"]}
    print(f"\nchosen: threshold={threshold}, one owner per record={use_one_owner}")
    print("validation (exact scorer):", score_report(matches, truth))
    countries = val["s1_prep"]["country"].to_numpy()
    for country in sorted(set(countries)):
        ids = val["s1_ids"][countries == country]
        print(f"  {country}: macro F0.5 = {macro_f05(matches, {s: gt[s] for s in ids}):.4f}")

    # robustness report: self-training (used at prediction time) and a "new language"
    # simulation where no word score is known, as for France
    def f05_at_threshold(p):
        chosen = p >= threshold
        if use_one_owner:
            chosen = one_owner_per_record(pool_row, p, chosen)
        return fast_macro_f05(s1_row, chosen, val["y"], val["n_true"])

    def predict_with(word_features):
        return model.predict(assemble(val["X_base"], word_features, val["cluster"], val["counts"], val["format"]))

    no_words = WordScores({"extra": {}, "missing": {}})
    no_words.prior = word_scores.prior
    p_unknown = predict_with(no_words.features(val["diffs"]))
    print("robustness (validation F0.5 at the chosen threshold):")
    print(f"  words known + self-training:       "
          f"{f05_at_threshold(predict_with(self_trained_word_features(val['diffs'], prob, s1_row, word_scores, SEED))):.4f}")
    print(f"  all words unknown (new language):  {f05_at_threshold(p_unknown):.4f}")
    print(f"  all words unknown + self-training: "
          f"{f05_at_threshold(predict_with(self_trained_word_features(val['diffs'], p_unknown, s1_row, no_words, SEED))):.4f}")

    importance = pd.Series(model.feature_importance("gain"), index=MATCH_FEATURES)
    print("\nfeature importance (share of total gain):")
    print((importance / importance.sum()).sort_values(ascending=False).round(4).to_string())

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_DIR / "val_scores.npz", s1_row=s1_row, pool_row=pool_row, prob=prob,
             y=val["y"], n_true=val["n_true"])  # for error analysis
    reranker.save_model(str(CACHE_DIR / "reranker.txt"), num_iteration=reranker.best_iteration)
    word_scores.save(CACHE_DIR / "word_scores.json")
    model.save_model(str(CACHE_DIR / "matcher.txt"), num_iteration=model.best_iteration)
    (CACHE_DIR / "matcher.json").write_text(json.dumps(
        {"K": K, "keep_max": KEEP_MAX, "min_prob": MIN_PROB, "threshold": threshold,
         "one_owner_per_record": use_one_owner, "features": MATCH_FEATURES}, indent=2))
    print(f"saved model and settings to {CACHE_DIR}")


if __name__ == "__main__":
    main()

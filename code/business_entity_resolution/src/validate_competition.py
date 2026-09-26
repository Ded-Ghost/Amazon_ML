"""Full-competition validation: tune the threshold the way the test set behaves. Run:

    python -m src.run_blocking --split rest     # once
    python -m src.validate_competition           # after train_matcher

On the test set every S1 business is present, so a record claimed by several
S1 entities goes to its most probable claimant (one owner per record). The
validation split holds only 20% of the S1 businesses: the true owner of a
wrongly matched record is usually absent (58% of validation false positives
belong to a training-half business), which makes the validation threshold too
cautious. Here the validation entities are scored together with the ~1.47M
other training-half entities that the matcher never saw ("rest"); the decision
rule is applied to all of them at once and macro F0.5 is measured on the
validation entities only. The best threshold is written to matcher.json.
"""
import json
import time

import numpy as np
import pandas as pd

from .config import CACHE_DIR
from .data import load_ground_truth, load_source
from .features import add_core_names, frequent_name_words, prepare_records
from .match_features import NameCounts
from .metrics import macro_f05
from .predict import score_pairs
from .run_blocking import load_candidates
from .train_matcher import THRESHOLDS, fast_macro_f05, one_owner_per_record, pair_labels


def main():
    start = time.time()
    settings = json.loads((CACHE_DIR / "matcher.json").read_text())
    gt = load_ground_truth()
    s1_all = load_source("train", 1)
    pool = pd.concat([load_source("train", 2), load_source("train", 3)], ignore_index=True)
    pool_prep, pool_numbers = prepare_records(pool)
    frequent = frequent_name_words(pool_prep)
    add_core_names(pool_prep, frequent)
    s1_prep_all, s1_numbers_all = prepare_records(s1_all)
    add_core_names(s1_prep_all, frequent)
    name_counts = NameCounts(s1_prep_all, pool_prep)

    val_ids, pool_ids, val_cand, val_score = load_candidates("val")
    rest_ids, rest_pool_ids, rest_cand, rest_score = load_candidates("rest")
    assert (pool_ids == rest_pool_ids).all() and (pool["entity_id"].to_numpy() == pool_ids).all()
    ids = np.concatenate([val_ids, rest_ids])
    rows = pd.Index(s1_all["entity_id"]).get_indexer(ids)
    s1_prep = s1_prep_all.iloc[rows].reset_index(drop=True)
    s1_numbers = s1_numbers_all[rows]
    del pool, s1_all, s1_prep_all, s1_numbers_all
    print(f"prepared {len(val_ids):,} validation + {len(rest_ids):,} other entities "
          f"({time.time() - start:.0f}s)", flush=True)

    s1_row, pool_row, prob = score_pairs(s1_prep, s1_numbers, pool_prep, pool_numbers,
                                         np.vstack([val_cand, rest_cand]), np.vstack([val_score, rest_score]),
                                         name_counts, settings)
    np.savez(CACHE_DIR / "competition_scores.npz", s1_row=s1_row, pool_row=pool_row, prob=prob)

    is_val = s1_row < len(val_ids)
    pairs = pd.DataFrame({"s1_row": s1_row[is_val], "pool_row": pool_row[is_val]})
    y = pair_labels(pairs, val_ids, pool_ids, gt)
    n_true = np.array([len(gt[s]) for s in val_ids])
    rows = []
    for t in THRESHOLDS:
        alone = one_owner_per_record(pool_row[is_val], prob[is_val], prob[is_val] >= t)
        together = one_owner_per_record(pool_row, prob, prob >= t)[is_val]
        rows.append({"threshold": t,
                     "val only (as before)": fast_macro_f05(s1_row[is_val], alone, y, n_true),
                     "with full competition": fast_macro_f05(s1_row[is_val], together, y, n_true)})
    table = pd.DataFrame(rows)
    print("\n" + table.round(4).to_string(index=False))

    best = table.loc[table["with full competition"].idxmax()]
    threshold = float(best["threshold"])
    chosen = one_owner_per_record(pool_row, prob, prob >= threshold)[is_val]
    matches = {s: [] for s in val_ids}
    for r, c in zip(s1_row[is_val][chosen], pool_row[is_val][chosen]):
        matches[val_ids[r]].append(pool_ids[c])
    countries = s1_prep["country"].to_numpy()[: len(val_ids)]
    print(f"\nbest threshold with full competition: {threshold} "
          f"(was {settings['threshold']}); validation F0.5 {best['with full competition']:.4f}")
    for country in sorted(set(countries)):
        c_ids = val_ids[countries == country]
        print(f"  {country}: macro F0.5 = {macro_f05(matches, {s: gt[s] for s in c_ids}):.4f}")

    settings.setdefault("threshold_val_only", settings["threshold"])
    settings["threshold"] = threshold
    (CACHE_DIR / "matcher.json").write_text(json.dumps(settings, indent=2))
    print(f"threshold {threshold} written to matcher.json ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()

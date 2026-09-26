"""Sanity checks for the scorer and the validation split. Run:

    python -m src.check_scorer

1. Hand-made cases, including the worked example from the problem statement.
2. On the real 20% validation split:
   - perfect predictions must score exactly 1.0
   - all-empty predictions must score exactly the singleton fraction
"""
import math

import pandas as pd

from .data import load_ground_truth, load_source
from .metrics import f05_entity, macro_f05, score_report
from .split import split_s1_ids


def check_unit_cases():
    # worked example from the PDF: P = 2/3, R = 1 -> 0.714
    pdf_example = f05_entity(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
    assert math.isclose(pdf_example, 1.25 * (2 / 3) / (0.25 * (2 / 3) + 1)), pdf_example
    assert round(pdf_example, 3) == 0.714

    assert f05_entity([], []) == 1.0              # singleton, predicted empty
    assert f05_entity(["S2-1"], []) == 0.0        # singleton, predicted something
    assert f05_entity([], ["S2-1"]) == 0.0        # missed every match
    assert f05_entity(["S3-9"], ["S2-1"]) == 0.0  # wrong match only
    assert f05_entity(["S2-1", "S2-2"], ["S2-2", "S2-1"]) == 1.0
    assert f05_entity(["S2-1", "S2-1"], ["S2-1"]) == 1.0  # duplicates don't count twice
    # precision matters more: half recall beats half precision
    assert f05_entity(["S2-1"], ["S2-1", "S2-2"]) > f05_entity(["S2-1", "S2-2"], ["S2-1"])

    truth = {"a": [], "b": ["S2-1"], "c": ["S2-2", "S3-3"]}
    assert macro_f05({}, truth) == 1 / 3
    assert macro_f05(truth, truth) == 1.0
    print(f"unit cases OK (PDF example = {pdf_example:.3f})")


def main():
    check_unit_cases()

    gt = load_ground_truth()
    train_ids, val_ids = split_s1_ids(gt.keys())
    assert not set(train_ids) & set(val_ids)
    assert len(train_ids) + len(val_ids) == len(gt)

    # describe the two halves
    countries = load_source("train", 1, usecols=["entity_id", "country"]).set_index("entity_id")["country"]
    rows = []
    for name, ids in (("train (80%)", train_ids), ("val (20%)", val_ids)):
        rows.append({
            "split": name,
            "entities": len(ids),
            "% singletons": 100 * sum(not gt[i] for i in ids) / len(ids),
            **countries.loc[ids].value_counts(normalize=True).mul(100).add_prefix("% ").to_dict(),
        })
    print("\n" + pd.DataFrame(rows).round(2).to_string(index=False))

    val_truth = {s1: gt[s1] for s1 in val_ids}
    singleton_frac = sum(not t for t in val_truth.values()) / len(val_truth)

    perfect = macro_f05(val_truth, val_truth)
    empty = macro_f05({}, val_truth)
    print(f"\nperfect predictions : {perfect!r}")
    print(f"all-empty predictions: {empty!r}")
    print(f"singleton fraction   : {singleton_frac!r}")
    assert perfect == 1.0
    assert empty == singleton_frac
    print("\nempty baseline breakdown:", score_report({}, val_truth))
    print("scorer checks OK")


if __name__ == "__main__":
    main()

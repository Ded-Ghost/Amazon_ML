"""Trivial baseline: predict "no match" for every Source 1 entity. Run:

    python -m src.baseline_empty

It scores the baseline on the validation split, then writes both submission
files for the TEST set (all lists empty) into OUTPUT_DIR. The point is to
test the plumbing end to end and to get the score floor: every real model
has to beat this number.
"""
from .config import OUTPUT_DIR
from .data import load_ground_truth, load_source
from .metrics import score_report
from .output_writer import write_submission
from .split import split_s1_ids


def main():
    gt = load_ground_truth()
    _, val_ids = split_s1_ids(gt.keys())
    val_truth = {s1: gt[s1] for s1 in val_ids}
    print("validation score of the empty baseline:", score_report({}, val_truth))

    test_s1_ids = load_source("test", 1, usecols=["entity_id"])["entity_id"].tolist()
    write_submission(OUTPUT_DIR, test_s1_ids, matches={}, candidates={})
    print(f"wrote {len(test_s1_ids):,} rows to {OUTPUT_DIR}/matching_results.tsv and candidate_pairs.tsv")


if __name__ == "__main__":
    main()

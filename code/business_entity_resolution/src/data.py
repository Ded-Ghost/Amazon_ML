"""Loading the TSV files.

Every file is TAB-separated. We read all columns as plain strings and switch
off pandas' NA detection, so a business literally called "NA" or an empty
address stays a string ("NA" / "") instead of turning into NaN. QUOTE_NONE
makes pandas treat quote characters inside names as ordinary text, which is
how the official validator reads the files too (a plain split on TAB).
"""
import csv

import pandas as pd

from .config import DATA_DIR

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path, usecols=None):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        usecols=usecols,
        encoding="utf-8",
    )


def load_source(split, source_num, usecols=None, data_dir=DATA_DIR):
    """Load e.g. load_source("train", 2) -> dataset/train/train_source2.tsv."""
    return read_tsv(data_dir / split / f"{split}_source{source_num}.tsv", usecols)


def load_ground_truth(data_dir=DATA_DIR):
    """Return {source1_entity_id: [matched S2/S3 ids]} (empty list = singleton)."""
    gt = read_tsv(data_dir / "train" / "train_ground_truth.tsv")
    return {
        s1: (matches.split(",") if matches else [])
        for s1, matches in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
    }

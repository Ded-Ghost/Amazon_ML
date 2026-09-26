"""Pair features for the matching model.

Each candidate pair (S1 record, S2/S3 record) gets a row of numbers that
describe how similar the two records are. All features are string
similarities or counts, never the country itself, so they mean the same
thing for US, India and France.

Per record we first prepare (prepare_records):
  name        cleaned name words             "blue producer private limited"
  name_glued  the same without spaces        "blueproducerprivatelimited"
  name_skel   phonetic skeleton of each word "pl prtsr prvt lmt"
  name_core   name without very frequent words of that country
              (learned from the data: "private", "limited", "llc", "sarl"...)
  address     cleaned address words
  numbers     up to 4 numbers from the address (house / plot numbers ...)

Then for every pair (compute_features):
  blocking    cosine score, rank, score relative to the entity's best one
  name        rapidfuzz ratio, token-set, token-sort, partial, Jaro-Winkler,
              ratio without spaces ("willshore" vs "will shore"),
              the same on skeletons and on core names
  address     ratio, token-set, partial; shared numbers
  context     how this candidate's name/address similarity compares with
              the best candidate of the same S1 entity
  other       name lengths, empty address flag, S2 or S3
Missing values (e.g. empty address) are NaN; LightGBM handles them natively.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .normalize import address_tokens, name_tokens, skeleton

MAX_NUMBERS = 4
FREQUENT_WORD_SHARE = 0.002  # a name word used by > 0.2% of a country's records is "frequent"

FEATURES = [
    "block_score", "block_rank", "block_score_rel",
    "name_ratio", "name_token_set", "name_token_sort", "name_partial", "name_jw", "name_nospace_ratio",
    "skel_ratio", "skel_token_set", "core_ratio", "core_token_set",
    "addr_ratio", "addr_token_set", "addr_partial",
    "num_shared", "num_s1", "num_pool", "num_pool_share",
    "name_words_s1", "name_words_pool", "pool_addr_empty", "is_s3",
    "name_token_set_vs_best", "addr_token_set_vs_best",
]


# ---------------------------------------------------------------- records

def _number_code(token):
    return int(token[:18])  # fits in int64


def _prepare_chunk(names, addresses):
    rows = []
    for name, address in zip(names, addresses):
        words = name_tokens(name)
        addr = address_tokens(address)
        numbers = sorted({t for t in addr if t.isdigit()})[:MAX_NUMBERS]
        codes = [_number_code(t) for t in numbers] + [-1] * (MAX_NUMBERS - len(numbers))
        rows.append((" ".join(words), " ".join(skeleton(t) or t for t in words), " ".join(addr), codes))
    return rows


def prepare_records(df, n_jobs=-1, chunk_size=100_000):
    """Cleaned fields for every record of df (same row order)."""
    names, addresses = df["business_name"].tolist(), df["business_address"].tolist()
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_prepare_chunk)(names[i:i + chunk_size], addresses[i:i + chunk_size])
        for i in range(0, len(names), chunk_size)
    )
    rows = [r for part in parts for r in part]
    prep = pd.DataFrame({
        "name": [r[0] for r in rows],
        "name_skel": [r[1] for r in rows],
        "address": [r[2] for r in rows],
        "name_glued": [r[0].replace(" ", "") for r in rows],
        "country": df["country"].to_numpy(),
        "is_s3": df["entity_id"].str.startswith("S3-").to_numpy(),
        "raw_name": df["business_name"].to_numpy(),        # for the formatting features
        "raw_address": df["business_address"].to_numpy(),
    })
    numbers = np.array([r[3] for r in rows], dtype=np.int64).reshape(len(rows), MAX_NUMBERS)
    return prep, numbers


def frequent_name_words(pool_prep):
    """{country: set of name words used by more than FREQUENT_WORD_SHARE of its records}."""
    frequent = {}
    for country, g in pool_prep.groupby("country"):
        counts = g["name"].str.split().explode().value_counts()
        frequent[country] = set(counts.index[counts > FREQUENT_WORD_SHARE * len(g)])
    return frequent


def add_core_names(prep, frequent):
    """prep["name_core"] = name without the frequent words of its country."""
    core = prep["name"].copy()
    for country, words in frequent.items():
        mask = (prep["country"] == country).to_numpy()
        core[mask] = [" ".join(w for w in n.split() if w not in words) for n in prep["name"][mask]]
    prep["name_core"] = core


# ---------------------------------------------------------------- pairs

def build_pairs(cand, score, k):
    """Flatten the top-k candidates into pair arrays (grouped by S1 row, best first)."""
    top = cand[:, :k]
    s1_row, rank = np.nonzero(top >= 0)
    pairs = pd.DataFrame({
        "s1_row": s1_row.astype(np.int32),
        "pool_row": top[s1_row, rank].astype(np.int32),
        "block_score": score[s1_row, rank],
        "block_rank": rank.astype(np.float32),
    })
    best = score[:, 0]
    pairs["block_score_rel"] = pairs["block_score"] / np.maximum(best[s1_row], 1e-6)
    return pairs


def _similarity(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _vs_entity_best(values, s1_row):
    """values minus the maximum of values among candidates of the same S1 row."""
    filled = np.nan_to_num(values, nan=-1.0)
    starts = np.flatnonzero(np.r_[True, s1_row[1:] != s1_row[:-1]])
    best = np.maximum.reduceat(filled, starts)
    return values - np.repeat(best, np.diff(np.r_[starts, len(s1_row)]))


def refresh_context_features(X, s1_row):
    """Recompute the *_vs_best columns after some candidates were removed.

    The re-ranker drops candidates, so "best candidate of this entity" must be
    taken over the kept candidates only. X is updated in place.
    """
    for column, base in (("name_token_set_vs_best", "name_token_set"),
                         ("addr_token_set_vs_best", "addr_token_set")):
        X[:, FEATURES.index(column)] = _vs_entity_best(X[:, FEATURES.index(base)], s1_row)


def entity_chunks(s1_row, max_pairs=5_000_000):
    """(start, end) slices of about max_pairs pairs that never split an S1 entity."""
    start = 0
    while start < len(s1_row):
        end = min(start + max_pairs, len(s1_row))
        while end < len(s1_row) and s1_row[end] == s1_row[end - 1]:
            end += 1
        yield start, end
        start = end


def compute_features(pairs, s1_prep, s1_numbers, pool_prep, pool_numbers):
    """Feature matrix (len(pairs) x len(FEATURES)), float32, built chunk by chunk.

    Chunks always contain whole S1 entities (needed for the *_vs_best features).
    """
    X = np.empty((len(pairs), len(FEATURES)), dtype=np.float32)
    for start, end in entity_chunks(pairs["s1_row"].to_numpy()):
        X[start:end] = _chunk_features(pairs.iloc[start:end], s1_prep, s1_numbers, pool_prep, pool_numbers)
    return X


def _chunk_features(pairs, s1_prep, s1_numbers, pool_prep, pool_numbers):
    i, j = pairs["s1_row"].to_numpy(), pairs["pool_row"].to_numpy()
    f = {c: pairs[c].to_numpy(np.float32) for c in ("block_score", "block_rank", "block_score_rel")}

    def column(prep, name, rows):
        return prep[name].to_numpy()[rows].tolist()

    a, b = column(s1_prep, "name", i), column(pool_prep, "name", j)
    f["name_ratio"] = _similarity(a, b, fuzz.ratio)
    f["name_token_set"] = _similarity(a, b, fuzz.token_set_ratio)
    f["name_token_sort"] = _similarity(a, b, fuzz.token_sort_ratio)
    f["name_partial"] = _similarity(a, b, fuzz.partial_ratio)
    f["name_jw"] = _similarity(a, b, JaroWinkler.normalized_similarity)
    f["name_nospace_ratio"] = _similarity(column(s1_prep, "name_glued", i), column(pool_prep, "name_glued", j), fuzz.ratio)
    f["name_words_s1"] = np.array([s.count(" ") + 1 if s else 0 for s in a], dtype=np.float32)
    f["name_words_pool"] = np.array([s.count(" ") + 1 if s else 0 for s in b], dtype=np.float32)

    a, b = column(s1_prep, "name_skel", i), column(pool_prep, "name_skel", j)
    f["skel_ratio"] = _similarity(a, b, fuzz.ratio)
    f["skel_token_set"] = _similarity(a, b, fuzz.token_set_ratio)

    a, b = column(s1_prep, "name_core", i), column(pool_prep, "name_core", j)
    core_missing = np.array([not x or not y for x, y in zip(a, b)])
    f["core_ratio"] = np.where(core_missing, np.nan, _similarity(a, b, fuzz.ratio))
    f["core_token_set"] = np.where(core_missing, np.nan, _similarity(a, b, fuzz.token_set_ratio))

    a, b = column(s1_prep, "address", i), column(pool_prep, "address", j)
    addr_missing = np.array([not x or not y for x, y in zip(a, b)])
    f["addr_ratio"] = np.where(addr_missing, np.nan, _similarity(a, b, fuzz.ratio))
    f["addr_token_set"] = np.where(addr_missing, np.nan, _similarity(a, b, fuzz.token_set_ratio))
    f["addr_partial"] = np.where(addr_missing, np.nan, _similarity(a, b, fuzz.partial_ratio))
    f["pool_addr_empty"] = np.array([not y for y in b], dtype=np.float32)

    na, nb = s1_numbers[i], pool_numbers[j]
    shared = ((na[:, :, None] == nb[:, None, :]) & (na[:, :, None] >= 0)).any(axis=2).sum(axis=1)
    f["num_shared"] = shared.astype(np.float32)
    f["num_s1"] = (na >= 0).sum(axis=1).astype(np.float32)
    f["num_pool"] = (nb >= 0).sum(axis=1).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["num_pool_share"] = np.where(f["num_pool"] > 0, f["num_shared"] / f["num_pool"], np.nan)

    f["is_s3"] = pool_prep["is_s3"].to_numpy()[j].astype(np.float32)
    f["name_token_set_vs_best"] = _vs_entity_best(f["name_token_set"], i)
    f["addr_token_set_vs_best"] = _vs_entity_best(f["addr_token_set"], i)
    return np.column_stack([np.asarray(f[c], dtype=np.float32) for c in FEATURES])

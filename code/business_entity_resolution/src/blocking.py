"""Candidate generation (blocking).

Comparing every S1 record with every S2/S3 record is impossible (millions x
millions), so we only look at pairs that share *rare keys*.

1. Every record becomes a bag of keys (see record_keys):
     n:<word>      cleaned name words
     k:<skeleton>  phonetic skeleton of each name word (typos, transliteration)
     p:<s1>_<s2>   unordered pair of name skeletons. Common words
                   ("vision", "first", "care") are useless alone but their
                   combination is rare; unordered = robust to word order.
     c:<w1w2..>    first 1-3 name words glued together, so "Will Shore"
                   meets "willshore.com" and "@SCOTTJOHNSON"
     a:<word>      cleaned address words
     b:<w1>_<w2>   adjacent address words ("12_pomeroy", "pomeroy_street")
2. Keys are hashed into columns of a sparse 0/1 matrix (HashingVectorizer,
   so there is no vocabulary to fit and train/test are encoded identically).
3. Inside one country, each key gets weight idf = log(N / df), where df is
   the number of S2/S3 records containing it. Keys with df > max_df are
   dropped: they are too common to identify anything ("ltd", "street")
   and would make the search slow.
4. score(s1, r) = TF-IDF cosine between the key vectors (pool side divided
   by its length, so long records do not win just by having many keys),
   computed for all pairs at once as a sparse matrix product.
   Each S1 record keeps its top_k pool records.

Blocking is done separately per country label (matches never cross
countries), and works for any label, including ones unseen in training.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import vstack
from sklearn.feature_extraction.text import HashingVectorizer

from .normalize import address_tokens, name_tokens, skeleton

N_HASH_FEATURES = 2 ** 24
MAX_NAME_TOKENS = 6  # limits the number of pair keys for very long names
MAX_DF = 5000
TOP_K = 100

_hasher = HashingVectorizer(
    n_features=N_HASH_FEATURES, analyzer=str.split, alternate_sign=False,
    norm=None, binary=True, dtype=np.float32,
)


def record_keys(name, address):
    """All blocking keys of one record, as one space-separated string."""
    tokens = name_tokens(name)
    skeletons = sorted({s for s in map(skeleton, tokens[:MAX_NAME_TOKENS]) if s})
    keys = [f"n:{t}" for t in tokens]
    keys += [f"k:{s}" for s in skeletons]
    keys += [f"p:{a}_{b}" for i, a in enumerate(skeletons) for b in skeletons[i + 1:]]
    keys += [f"c:{''.join(tokens[:i])}" for i in range(1, min(3, len(tokens)) + 1)]

    tokens = address_tokens(address)
    keys += [f"a:{t}" for t in tokens]
    keys += [f"b:{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return " ".join(keys)


def _hash_chunk(names, addresses):
    return _hasher.transform([record_keys(n, a) for n, a in zip(names, addresses)])


def key_matrix(df, n_jobs=-1, chunk_size=100_000):
    """Sparse (records x hashed keys) 0/1 matrix, built in parallel."""
    names, addresses = df["business_name"].tolist(), df["business_address"].tolist()
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_hash_chunk)(names[i:i + chunk_size], addresses[i:i + chunk_size])
        for i in range(0, len(names), chunk_size)
    )
    return vstack(parts, format="csr")


def _search_chunk(queries, pool_t, top_k):
    """Top-k columns of (queries @ pool_t) per row, padded with -1."""
    scores = (queries @ pool_t).tocsr()
    cand = np.full((queries.shape[0], top_k), -1, dtype=np.int32)
    best = np.zeros((queries.shape[0], top_k), dtype=np.float32)
    for i in range(scores.shape[0]):
        start, end = scores.indptr[i], scores.indptr[i + 1]
        cols, vals = scores.indices[start:end], scores.data[start:end]
        if len(vals) > top_k:
            keep = np.argpartition(-vals, top_k)[:top_k]
            cols, vals = cols[keep], vals[keep]
        order = np.argsort(-vals, kind="stable")
        cand[i, :len(order)] = cols[order]
        best[i, :len(order)] = vals[order]
    return cand, best


def block_country(s1_df, pool_df, top_k=TOP_K, max_df=MAX_DF, n_jobs=8, chunk_size=1000):
    """Candidates for the S1 records of one country.

    Returns (cand, score): int32 / float32 arrays of shape (len(s1_df), top_k).
    cand holds row positions in pool_df, best first, -1 = no candidate.
    """
    pool = key_matrix(pool_df)
    df = np.bincount(pool.indices, minlength=N_HASH_FEATURES)
    usable = (df > 0) & (df <= max_df)
    idf = np.zeros(N_HASH_FEATURES, dtype=np.float32)
    idf[usable] = np.log(pool.shape[0] / df[usable])

    pool.data = idf[pool.indices]
    pool.eliminate_zeros()  # drops the unusable keys
    norms = np.sqrt(np.asarray(pool.multiply(pool).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    pool.data /= np.repeat(norms, np.diff(pool.indptr)).astype(np.float32)
    pool_t = pool.T.tocsr()  # transpose once for fast queries @ pool.T
    del pool

    queries = key_matrix(s1_df)
    queries.data = idf[queries.indices]
    queries.eliminate_zeros()

    parts = Parallel(n_jobs=n_jobs, max_nbytes="1M")(
        delayed(_search_chunk)(queries[i:i + chunk_size], pool_t, top_k)
        for i in range(0, queries.shape[0], chunk_size)
    )
    return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])


def generate_candidates(s1_df, pool_df, top_k=TOP_K, max_df=MAX_DF):
    """Run blocking country by country (any country label works).

    Returns (cand, score) aligned with the rows of s1_df; cand holds row
    positions in pool_df (-1 = empty slot).
    """
    cand = np.full((len(s1_df), top_k), -1, dtype=np.int32)
    score = np.zeros((len(s1_df), top_k), dtype=np.float32)
    s1_country = s1_df["country"].to_numpy()
    pool_country = pool_df["country"].to_numpy()
    for country in pd.unique(s1_country):
        s1_rows = np.flatnonzero(s1_country == country)
        pool_rows = np.flatnonzero(pool_country == country)
        if len(pool_rows) == 0:
            continue
        c, s = block_country(s1_df.iloc[s1_rows], pool_df.iloc[pool_rows], top_k, max_df)
        cand[s1_rows] = np.where(c >= 0, pool_rows[np.maximum(c, 0)], -1)
        score[s1_rows] = s
        print(f"  blocked {country}: {len(s1_rows):,} S1 vs {len(pool_rows):,} pool records", flush=True)
    return cand, score

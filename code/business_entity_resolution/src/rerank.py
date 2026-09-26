"""Second blocking stage: cheap re-ranking of the blocking candidates.

The TF-IDF search (blocking.py) returns the top 50 S2/S3 records per S1 entity.
Many true matches are in that list but not near the top, so a plain top-K cut
would lose them. This stage re-orders the 50 with a small model (LightGBM,
31 leaves, a few hundred trees) on the 26 pair features of src/features.py.
With all 26 features it keeps more true matches with fewer candidates than
with an 11-feature version (validation: 97.1% of true pairs with 6.1
candidates per entity, vs 95.4% with 4.7). It keeps

    at most KEEP_MAX candidates per S1 entity, and only those whose
    re-rank probability is >= MIN_PROB

The kept candidates are the final candidate set (candidate_pairs.tsv): the
matching model (train_matcher.py) runs only on them.
"""
import lightgbm as lgb
import numpy as np

from .config import SEED
from .features import FEATURES

RERANK_FEATURES = list(FEATURES)
RERANK_COLUMNS = [FEATURES.index(f) for f in RERANK_FEATURES]
KEEP_MAX = 10
MIN_PROB = 0.005
RERANK_PARAMS = {
    "objective": "binary", "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 500,
    "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": SEED,
    "deterministic": True, "force_row_wise": True, "num_threads": 16, "verbose": -1,
}


def train_reranker(X, y, holdout):
    """Fit the re-ranker on the full feature matrix (it only uses RERANK_COLUMNS)."""
    dtrain = lgb.Dataset(X[~holdout][:, RERANK_COLUMNS], y[~holdout], feature_name=RERANK_FEATURES)
    dhold = lgb.Dataset(X[holdout][:, RERANK_COLUMNS], y[holdout], reference=dtrain)
    return lgb.train(RERANK_PARAMS, dtrain, num_boost_round=500, valid_sets=[dhold],
                     callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)])


def rerank_probability(model, X):
    return model.predict(X[:, RERANK_COLUMNS]).astype(np.float32)


def keep_mask(s1_row, prob, keep_max=KEEP_MAX, min_prob=MIN_PROB):
    """True for the candidates that survive: top keep_max per S1 row and prob >= min_prob.

    s1_row must be grouped (all candidates of one S1 entity next to each other).
    """
    order = np.lexsort((-prob, s1_row))
    grouped = s1_row[order]
    starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
    rank_in_entity = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    keep = np.zeros(len(prob), dtype=bool)
    keep[order[rank_in_entity < keep_max]] = True
    return keep & (prob >= min_prob)

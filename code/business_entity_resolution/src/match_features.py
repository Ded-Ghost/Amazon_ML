"""Extra matcher features, computed on the re-ranker's kept candidates only.

1. Word differences (the biggest single improvement, +0.008 F0.5 on validation).
   The training data contains deliberately similar *different* businesses:
   "Jimenez Distribution Midtown LLC" is not "Jimenez Distribution LLC",
   while "Jimenez Jimenez Distribution" or "The Jimenez Distribution LLC" is.
   For each pair we list
     extra words:   in the candidate's name but not in the S1 name
     missing words: in the S1 name but not in the candidate's name
   (a word counts as present when it matches exactly, by phonetic skeleton,
   or with >= 85% character similarity, so typos are not "extra").
   From the labelled training pairs we learn, per word, how often a pair with
   that extra/missing word is a true match (smoothed towards the average):
   "midtown", "south", "holdings" ~ 0.00-0.01; "services", "www", "dba" ~ 0.8-0.97.
   Features: number of extra/missing words, lowest and mean score, number of
   extra words rarely seen in training (unknown, e.g. new French words).
   For the matcher's own training rows the scores are cross-fitted (learned
   on the other half of the entities) so they are not over-optimistic.

2. Cluster similarity. The true matches of one business are copies of each
   other, so we measure how similar each kept candidate is to the *other*
   kept candidates of the same S1 entity (best name / address token-set
   similarity, and how many are near-identical by name).

3. Label-free counts over the whole dataset (NameCounts): how many S1
   businesses share the name / the address, whether the candidate's name or
   address belongs to *another* S1 business, how common each differing word
   is, and the distance between house numbers.

4. Sibling words, label-free (sibling_features): the generator places sibling
   businesses ("... Midtown", "... Holding") at a different house number,
   while true copies keep the number. So, per country and over the pairs of
   the dataset being scored (no labels), each extra word gets the share of its
   pairs whose house numbers conflict: "midtown", "westgate" ~0.8 (all true
   match rates 0.0); "dba", "formerly" ~0.00 (true match rates > 0.97). This
   is computed on the test pairs themselves, so it also covers words of a
   country never seen in training (US-only model tested on India: 0.934 ->
   0.948).

5. Formatting (format_features): whether the raw strings are identical, the
   letter case / accents / symbols of the candidate name, and whether a house
   number differs only by a dropped digit (826 -> 26 is copy noise, while
   517 -> 524 is a neighbouring business).
"""
import re
import json
from collections import Counter

import numpy as np
from joblib import Parallel, delayed
from rapidfuzz import fuzz, process

from .features import FEATURES
from .normalize import address_tokens, clean, skeleton

WORD_FEATURES = ["n_extra", "n_missing", "extra_min", "extra_mean", "extra_unknown",
                 "missing_min", "missing_mean"]
CLUSTER_FEATURES = ["clu_name_max", "clu_addr_max", "clu_n_similar"]
COUNT_FEATURES = ["s1_name_n", "s1_core_n", "cand_name_other_s1", "cand_core_other_s1", "pool_name_n",
                  "extra_df_min", "extra_df_max", "missing_df_min", "missing_df_max", "num_min_diff",
                  "s1_addr_n", "cand_addr_other_s1", "pool_addr_n", "same_addr_key"]
SIBLING_FEATURES = ["sib_max", "sib_mean", "sib_n_high"]
FORMAT_FEATURES = ["raw_equal", "raw_equal_nocase", "cand_style", "cand_accent", "cand_symbols",
                   "num_dropped_digit", "name_len_diff"]
MATCH_FEATURES = (FEATURES + WORD_FEATURES + CLUSTER_FEATURES + COUNT_FEATURES + SIBLING_FEATURES
                  + FORMAT_FEATURES)
SMOOTH = 20        # pseudo-count pulling rare words towards the average score
UNKNOWN_BELOW = 5  # a word seen fewer times than this in training counts as unknown
WORD_DROPOUT = 0.3              # share of words hidden while training (robustness to new languages)
PSEUDO_HIGH, PSEUDO_LOW = 0.9, 0.1  # confident predictions used for self-training


# ---------------------------------------------------------------- word differences

def _present(word, words, skeletons):
    if word in words:
        return True
    key = skeleton(word)
    if key and key in skeletons:
        return True
    return len(word) >= 4 and any(len(w) >= 4 and fuzz.ratio(word, w) >= 85 for w in words)


def _diff_chunk(pairs):
    out = []
    for s1_words, cand_words in pairs:
        a, b = set(s1_words), set(cand_words)
        ka = {skeleton(w) for w in a} - {""}
        kb = {skeleton(w) for w in b} - {""}
        extra = [w for w in dict.fromkeys(cand_words) if not _present(w, a, ka)]
        missing = [w for w in dict.fromkeys(s1_words) if not _present(w, b, kb)]
        out.append((extra, missing))
    return out


def word_differences(s1_names, pool_names, s1_row, pool_row, chunk_size=200_000):
    """[(extra words, missing words)] per pair; names are cleaned strings."""
    pairs = [(s1_names[i].split(), pool_names[j].split()) for i, j in zip(s1_row, pool_row)]
    parts = Parallel(n_jobs=-1)(delayed(_diff_chunk)(pairs[i:i + chunk_size])
                                for i in range(0, len(pairs), chunk_size))
    return [d for part in parts for d in part]


class WordScores:
    """Per word: P(true match | the word is extra / missing), learned from labelled pairs."""

    def __init__(self, counts):
        self.counts = counts  # {"extra": {word: [matches, total]}, "missing": {...}}
        self.prior = {kind: sum(m for m, _ in c.values()) / max(sum(t for _, t in c.values()), 1)
                      for kind, c in counts.items()}

    @classmethod
    def learn(cls, diffs, labels):
        counts = {"extra": {}, "missing": {}}
        for kind, position in (("extra", 0), ("missing", 1)):
            matches, totals = Counter(), Counter()
            for diff, label in zip(diffs, labels):
                for word in diff[position]:
                    totals[word] += 1
                    matches[word] += int(label)
            counts[kind] = {w: [matches[w], totals[w]] for w in totals}
        return cls(counts)

    def save(self, path):
        path.write_text(json.dumps(self.counts), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def _score(self, kind, word):
        m, t = self.counts[kind].get(word, (0, 0))
        return (m + SMOOTH * self.prior[kind]) / (t + SMOOTH), t

    def with_fallback(self, other):
        """Copy whose words seen < UNKNOWN_BELOW times here take their counts from `other`."""
        counts = {kind: dict(c) for kind, c in self.counts.items()}
        for kind, c in other.counts.items():
            for word, mt in c.items():
                if counts[kind].get(word, (0, 0))[1] < UNKNOWN_BELOW:
                    counts[kind][word] = mt
        merged = WordScores(counts)
        merged.prior = self.prior
        return merged

    def features(self, diffs, dropout=0.0, rng=None):
        """Word features; with dropout, each word is treated as never seen with that probability."""
        f = {k: np.full(len(diffs), np.nan, dtype=np.float32) for k in WORD_FEATURES}

        def lookup(kind, word):
            if dropout and rng.random() < dropout:
                return self.prior[kind], 0
            return self._score(kind, word)

        for i, (extra, missing) in enumerate(diffs):
            f["n_extra"][i] = len(extra)
            f["n_missing"][i] = len(missing)
            if extra:
                scored = [lookup("extra", w) for w in extra]
                f["extra_min"][i] = min(s for s, _ in scored)
                f["extra_mean"][i] = sum(s for s, _ in scored) / len(scored)
                f["extra_unknown"][i] = sum(t < UNKNOWN_BELOW for _, t in scored)
            if missing:
                scores = [lookup("missing", w)[0] for w in missing]
                f["missing_min"][i] = min(scores)
                f["missing_mean"][i] = sum(scores) / len(scores)
        return f


def cross_fitted_word_features(diffs, labels, s1_row, seed, dropout=WORD_DROPOUT):
    """Word features for training rows, each scored with words learned on the other half.

    Word dropout: each word is treated as unseen with probability `dropout`, so the
    matcher also learns what to do when a word is unknown (as for a new language).
    """
    rng = np.random.default_rng(seed)
    half = (rng.random(s1_row.max() + 1) < 0.5)[s1_row]
    f = {k: np.full(len(diffs), np.nan, dtype=np.float32) for k in WORD_FEATURES}
    for side in (False, True):
        learn_rows = np.flatnonzero(half == side)
        score_rows = np.flatnonzero(half != side)
        scores = WordScores.learn([diffs[i] for i in learn_rows], labels[learn_rows])
        part = scores.features([diffs[i] for i in score_rows], dropout, rng)
        for k in WORD_FEATURES:
            f[k][score_rows] = part[k]
    return f


def self_trained_word_features(diffs, prob, s1_row, scores, seed):
    """Word features where words unknown in training get scores from confident predictions.

    Self-training (one round, no labels): pairs predicted with prob >= PSEUDO_HIGH
    count as matches and <= PSEUDO_LOW as non-matches. Word counts are learned on
    one half of the S1 entities and used for the other half, so a pair never
    scores its own words. Words seen in training keep their training counts.
    """
    half = (np.random.default_rng(seed).random(s1_row.max() + 1) < 0.5)[s1_row]
    confident = (prob >= PSEUDO_HIGH) | (prob <= PSEUDO_LOW)
    f = {k: np.full(len(diffs), np.nan, dtype=np.float32) for k in WORD_FEATURES}
    for side in (False, True):
        learn_rows = np.flatnonzero((half == side) & confident)
        score_rows = np.flatnonzero(half != side)
        pseudo = WordScores.learn([diffs[i] for i in learn_rows], prob[learn_rows] >= PSEUDO_HIGH)
        part = scores.with_fallback(pseudo).features([diffs[i] for i in score_rows])
        for k in WORD_FEATURES:
            f[k][score_rows] = part[k]
    return f


# ---------------------------------------------------------------- cluster similarity

def cluster_features(s1_row, pool_row, pool_prep):
    """Similarity of each kept candidate to the other kept candidates of its S1 entity."""
    n = len(s1_row)
    f = {k: np.zeros(n, dtype=np.float32) for k in CLUSTER_FEATURES}
    starts = np.flatnonzero(np.r_[True, s1_row[1:] != s1_row[:-1]])
    sizes = np.diff(np.r_[starts, n])
    first, second = [], []
    for m in range(2, sizes.max(initial=1) + 1):  # all pairs inside each entity
        groups = starts[sizes == m]
        if len(groups):
            i, j = np.nonzero(np.triu(np.ones((m, m), dtype=bool), 1))
            first.append((groups[:, None] + i).ravel())
            second.append((groups[:, None] + j).ravel())
    if not first:
        return f
    first, second = np.concatenate(first), np.concatenate(second)
    a, b = pool_row[first], pool_row[second]

    def sim(column):
        values = pool_prep[column].to_numpy()
        return process.cpdist(values[a].tolist(), values[b].tolist(), scorer=fuzz.token_set_ratio,
                              workers=-1, dtype=np.float32) / 100

    name = np.maximum(sim("name"), sim("name_skel"))
    addresses = pool_prep["address"].to_numpy()
    address = np.where((addresses[a] == "") | (addresses[b] == ""), 0.0, sim("address"))
    for x in (first, second):  # both directions
        np.maximum.at(f["clu_name_max"], x, name)
        np.maximum.at(f["clu_addr_max"], x, address)
        np.add.at(f["clu_n_similar"], x, (name >= 0.85).astype(np.float32))
    return f


# ---------------------------------------------------------------- label-free counts

class NameCounts:
    """Counts over ALL S1 records and the whole S2/S3 pool of one dataset (no labels).

    Several different S1 businesses can share a name. A name-only S2/S3 record
    ("Neex Opportunities Group", no address) is a safe match when only one S1
    business has that name, and a coin toss when three have it. Likewise, a
    candidate whose name is exactly *another* S1 business's name probably
    belongs to that one. Word commonness separates one-off words (typos,
    invented trade names like "Dovasynonyx") from branch words that appear in
    thousands of names ("south", "holding"), in any language.
    """

    def __init__(self, s1_prep, pool_prep):
        self.s1_name = Counter(zip(s1_prep["country"], s1_prep["name"]))
        self.s1_core = Counter(zip(s1_prep["country"], s1_prep["name_core"]))
        self.pool_name = Counter(zip(pool_prep["country"], pool_prep["name"]))
        self.word_df = {country: Counter(w for name in g["name"] for w in set(name.split()))
                        for country, g in pool_prep.groupby("country")}
        self.address_df = {country: Counter(w for a in g["address"] for w in set(a.split()))
                           for country, g in pool_prep.groupby("country")}
        self.s1_address = Counter(k for k in self.address_keys(s1_prep) if k)
        self.pool_address = Counter(k for k in self.address_keys(pool_prep) if k)

    def address_keys(self, prep, rows=None):
        """(country, first number, rarest street word) per record, or None.

        A compact address identity that survives reordering, abbreviations
        ("Bd" / "Boulevard" are frequent words, never the rarest one) and
        missing city / state parts.
        """
        addresses = prep["address"].to_numpy()
        countries = prep["country"].to_numpy()
        if rows is not None:
            addresses, countries = addresses[rows], countries[rows]
        keys = []
        for address, country in zip(addresses, countries):
            words = address.split()
            numbers = [w for w in words if w.isdigit()]
            alpha = [w for w in words if not w.isdigit() and len(w) > 2]
            if not numbers or not alpha:
                keys.append(None)
                continue
            df = self.address_df.get(country, {})
            keys.append((country, numbers[0], min(alpha, key=lambda w: (df.get(w, 0), w))))
        return keys

    def features(self, s1_prep, s1_numbers, pool_prep, pool_numbers, s1_row, pool_row, diffs):
        n = len(s1_row)
        f = {k: np.full(n, np.nan, dtype=np.float32) for k in COUNT_FEATURES}
        country = s1_prep["country"].to_numpy()[s1_row]
        s_name, s_core = s1_prep["name"].to_numpy()[s1_row], s1_prep["name_core"].to_numpy()[s1_row]
        c_name, c_core = pool_prep["name"].to_numpy()[pool_row], pool_prep["name_core"].to_numpy()[pool_row]
        for i in range(n):
            c = country[i]
            f["s1_name_n"][i] = self.s1_name[c, s_name[i]]
            f["cand_name_other_s1"][i] = self.s1_name[c, c_name[i]] - (c_name[i] == s_name[i])
            f["pool_name_n"][i] = self.pool_name[c, c_name[i]]
            if s_core[i]:
                f["s1_core_n"][i] = self.s1_core[c, s_core[i]]
            if c_core[i]:
                f["cand_core_other_s1"][i] = self.s1_core[c, c_core[i]] - (c_core[i] == s_core[i])
            df = self.word_df.get(c, {})
            extra, missing = diffs[i]
            if extra:
                values = [np.log1p(df.get(w, 0)) for w in extra]
                f["extra_df_min"][i], f["extra_df_max"][i] = min(values), max(values)
            if missing:
                values = [np.log1p(df.get(w, 0)) for w in missing]
                f["missing_df_min"][i], f["missing_df_max"][i] = min(values), max(values)
        # smallest distance between any house/plot number of the two addresses
        # (0 = shared number; siblings sit a few doors away: 41 vs 43, 517 vs 524)
        a = s1_numbers[s1_row].astype(np.float64)
        b = pool_numbers[pool_row].astype(np.float64)
        a[a < 0], b[b < 0] = np.inf, -np.inf
        f["num_min_diff"] = np.abs(a[:, :, None] - b[:, None, :]).reshape(n, -1).min(axis=1)
        f["num_min_diff"][~np.isfinite(f["num_min_diff"])] = np.nan

        # the same for addresses: one S1 business at this address -> a record at
        # this exact address (even under an unrelated trade name) is probably ours
        s1_keys = self.address_keys(s1_prep, s1_row)
        pool_keys = self.address_keys(pool_prep, pool_row)
        for i, (a, b) in enumerate(zip(s1_keys, pool_keys)):
            if a:
                f["s1_addr_n"][i] = self.s1_address[a]
            if b:
                f["cand_addr_other_s1"][i] = self.s1_address[b] - (a == b)
                f["pool_addr_n"][i] = self.pool_address[b]
            if a and b:
                f["same_addr_key"][i] = float(a == b)
        return f


# ---------------------------------------------------------------- formatting

_NON_ASCII = re.compile(r"[^\x00-\x7F]")
_SYMBOLS = re.compile(r"[\[\]()#@*]")


def _style(name):
    letters = [ch for ch in name if ch.isalpha()]
    if not letters:
        return 0
    if all(ch.isupper() for ch in letters):
        return 1
    if all(ch.islower() for ch in letters):
        return 2
    return 3 if name.istitle() else 4


def _dropped_digit(a, b):
    """b is a with one digit removed, or the other way round (826 -> 26)."""
    if len(a) == len(b) + 1:
        return any(a[:i] + a[i + 1:] == b for i in range(len(a)))
    if len(b) == len(a) + 1:
        return any(b[:i] + b[i + 1:] == a for i in range(len(b)))
    return False


def _format_chunk(s1_names, s1_addresses, names, addresses):
    rows = []
    for a, a_addr, b, b_addr in zip(s1_names, s1_addresses, names, addresses):
        numbers_a = [t for t in address_tokens(a_addr) if t.isdigit()]
        numbers_b = [t for t in address_tokens(b_addr) if t.isdigit()]
        rows.append((float(a == b), float(a.lower() == b.lower()), _style(b),
                     float(bool(_NON_ASCII.search(b))), float(bool(_SYMBOLS.search(b))),
                     float(any(_dropped_digit(x, y) for x in numbers_a for y in numbers_b)),
                     float(len(clean(b)) - len(clean(a)))))
    return rows


def format_features(s1_prep, pool_prep, s1_row, pool_row, chunk_size=200_000):
    columns = [s1_prep["raw_name"].to_numpy()[s1_row], s1_prep["raw_address"].to_numpy()[s1_row],
               pool_prep["raw_name"].to_numpy()[pool_row], pool_prep["raw_address"].to_numpy()[pool_row]]
    parts = Parallel(n_jobs=-1)(delayed(_format_chunk)(*(c[i:i + chunk_size] for c in columns))
                                for i in range(0, len(s1_row), chunk_size))
    values = np.array([r for part in parts for r in part], dtype=np.float32).reshape(-1, len(FORMAT_FEATURES))
    return {k: values[:, j] for j, k in enumerate(FORMAT_FEATURES)}


def sibling_features(diffs, X_base, country, smooth=20):
    """Per pair: how often its extra words come with conflicting house numbers (label-free).

    Rates are computed per country over the given pairs (the dataset being
    scored), smoothed towards the country's overall conflict share.
    """
    num_s1, num_pool, num_shared = (X_base[:, FEATURES.index(c)] for c in ("num_s1", "num_pool", "num_shared"))
    conflict = (num_s1 > 0) & (num_pool > 0) & (num_shared == 0)
    agree = num_shared > 0
    f = {k: np.full(len(diffs), np.nan, dtype=np.float32) for k in SIBLING_FEATURES}
    for c in np.unique(country):
        rows = np.flatnonzero(country == c)
        n_conflict, n_agree = Counter(), Counter()
        for i in rows:
            for word in diffs[i][0]:
                n_conflict[word] += conflict[i]
                n_agree[word] += agree[i]
        prior = conflict[rows].sum() / max(conflict[rows].sum() + agree[rows].sum(), 1)
        for i in rows:
            extra = diffs[i][0]
            if extra:
                rates = [(n_conflict[w] + smooth * prior) / (n_conflict[w] + n_agree[w] + smooth) for w in extra]
                f["sib_max"][i] = max(rates)
                f["sib_mean"][i] = sum(rates) / len(rates)
                f["sib_n_high"][i] = sum(r > 0.5 for r in rates)
    return f


def assemble(X, word_features, cluster, counts, formatting):
    """Full matcher matrix in MATCH_FEATURES order (counts must include the sibling features)."""
    return np.column_stack([X] + [word_features[k] for k in WORD_FEATURES] +
                           [cluster[k] for k in CLUSTER_FEATURES] +
                           [counts[k] for k in COUNT_FEATURES] +
                           [counts[k] for k in SIBLING_FEATURES] +
                           [formatting[k] for k in FORMAT_FEATURES]).astype(np.float32)

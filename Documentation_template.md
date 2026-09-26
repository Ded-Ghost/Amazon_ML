# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Null-Pointers  
**Team Members:** Siddhanth Roy, Prithwiraj Chatterjee, Satyam Chattopadhyay, Chandan Kumar Barman  
**Submission Date:** 27/09/2026

---

## 1. Executive Summary

We use a three-step entity-resolution pipeline built to keep the candidate set small:
**(1) blocking** with a sparse TF-IDF search over hand-designed "rare keys" (name words,
phonetic skeletons, unordered skeleton pairs, glued names, address words and bigrams),
run inside each country label and returning a top-50 shortlist; **(2) a cheap re-ranker**
(a small LightGBM on 11 inexpensive signals) that cuts this shortlist to **at most 8,
on average 4.7 on validation and 5.4 on test, candidates per Source 1 entity**, the final candidate set; and **(3) a
LightGBM matcher** on 26 language-neutral similarity features, with a decision rule tuned
directly for macro F0.5. The key ideas are:
- a phonetic "skeleton" key that makes Indian-script transliterations and typos collide
  with their Latin spelling ("praaivett" / "private" → `prvt`);
- pairing common words, so that generic names become searchable;
- the learned re-ranking stage, which makes the candidate set 10.7× smaller for a
  0.0006 F0.5 cost;
- a one-owner-per-record rule derived from the training data.

On a held-out 20% of the training entities the pipeline scores **macro F0.5 = 0.9550**
(empty-prediction baseline: 0.0554).

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from exploratory analysis of the training data (`reports/eda_report.txt`):

| Observation | Value | Consequence for the design |
|---|---|---|
| Records | S1 2.21M, S2 5.03M, S3 5.29M (train); S1 1.73M, S2 4.89M, S3 5.08M (test) | All-pairs comparison impossible → blocking is mandatory |
| Singletons (S1 with no match) | 5.58% (same in US and India) | Most of the score comes from finding matches; recall matters despite β = 0.5 |
| Matches per non-singleton S1 | mean 3.67, median 4, max 11 | Predict a *set* per entity, not a single best match |
| Matches by source | S3 51.6%, S2 48.4%; 85% of matched entities have both | S2 and S3 contain internal duplicates; treat them as one pool |
| S2/S3 records matched to >1 S1 | **0** | Each pool record has at most one owner → one-owner decision rule |
| S2/S3 records never matched | ~26% | Many distractor records |
| Matched pairs with the same country label | **100.0000%** of 7.64M | Block within country label (works for any label, incl. France) |
| Indian-script names (Devanagari, Tamil, Telugu, Gujarati, Bengali, Malayalam, Gurmukhi, Odia) | ~9.4% of S2 rows, ~5.3% of S3 rows (all India) | Exact word matching fails → phonetic skeleton keys |
| Empty address | ~3% of S2/S3 rows (0% in S1) | Name-only matching path; NaN-aware features |
| Test-only country | France: 15% of test S1, never in training | No country-specific rules, no country feature |

Noise patterns seen in matched pairs: word-order transpositions ("Private Vijay Lbe Ventures
Limited"), character typos and garbling ("Sliaecvr" for "Silver"), digit-for-letter
substitutions ("Y0ga", "Ava1anche", "5equoia"), inserted accents ("ÁDVISORS"), legal-suffix
changes (LLC ↔ L.L.C., Pvt Ltd ↔ Pvt), glued names and web handles ("willshore.com",
"@SCOTTJOHNSON"), unrelated trade names ("Onyxjax"), and in addresses: reordered components,
upper-casing, abbreviations (St/Street, MH/Maharashtra), state names in native script,
missing house numbers/cities, junk prefixes ("H.no #538"), letters glued to numbers ("1604b").

### 2.2 Solution Strategy

**Approach Type:** Two-stage blocking (TF-IDF rare-key search → cheap learned re-ranker) + LightGBM classifier + F0.5-tuned decision rule
**Core Innovation:** language-neutral rare-key blocking (phonetic skeletons and unordered
skeleton pairs, TF-IDF cosine inside each country label) followed by a small re-ranking
model, which together bring 94.5% of true matches into a candidate set of only 4.7 records
per Source 1 entity (at most 8), and a decision rule that exploits the "each S2/S3 record
has at most one owner" structure of the data.

Validation protocol: S1 training entities are split 80/20 with a fixed seed (42); S2/S3
records are shared by both halves, exactly as at test time. All reported scores use an
exact re-implementation of the challenge metric (per-entity F0.5, singletons score 1 for an
empty prediction and 0 otherwise, macro average), sanity-checked to give exactly 1.0 for
perfect predictions and exactly the singleton fraction for empty predictions.

---

## 3. Candidate Generation (Blocking)

**Text normalisation** (`src/normalize.py`, identical for every country):
`unidecode` transliteration of any script to ASCII, lowercase, `&` → "and", punctuation
removed; in names, digit-for-letter typos are undone (0→o, 1→l, 3→e, 4→a, 5→s, 7→t, only
in mixed letter/digit words); in addresses, letters are split from numbers and leading
zeros removed ("012" = "12"). The **phonetic skeleton** of a word drops vowels and h/y,
merges sound-alike letters (c/k/q/g, d/t, b/p, z/x/s, w/v, ph/f) and collapses repeats.

- **Blocking keys used** (one bag of keys per record, `src/blocking.py`):
  - `n:` name words; `k:` phonetic skeleton of each name word
  - `p:` unordered pairs of name skeletons. Common words like "vision", "first" and "care"
    are useless alone but their combination is rare, and unordered pairs ignore word order
  - `c:` first 1–3 name words glued together (matches "Will Shore" ↔ "willshore.com")
  - `a:` address words; `b:` adjacent address word pairs ("12_pomeroy")
- **Search:** keys are hashed (`HashingVectorizer`, 2^24 columns) into sparse matrices.
  Within each country label, key weights are IDF computed on that country's S2+S3 pool.
  Keys that occur in more than 5,000 pool records are dropped, because they identify
  nothing and slow the search. The score is TF-IDF cosine (pool side length-normalised),
  computed as one sparse matrix product in parallel chunks. The top 50 form an internal
  shortlist for the second stage. Stage 1 on the full test set takes ~8 minutes on a
  16-thread laptop.
- **Stage 2, cheap re-ranker** (`src/rerank.py`): a small LightGBM (31 leaves, 220 trees,
  trained on the training sample) scores each of the 50 shortlisted records using 11
  inexpensive signals. These are the TF-IDF score, its rank, the score relative to the
  entity's best, name token-set, glued-name and skeleton similarity, address token-set
  similarity, and house-number overlap counts. We keep **at most 8 candidates per entity,
  and only those with re-rank probability ≥ 0.05**. The final matcher runs only on these,
  and they are exactly what `candidate_pairs.tsv` contains. The most important re-ranker
  signals are the TF-IDF score (45% of gain), name token-set similarity (30%) and the
  relative score (23%).
- **Candidate pairs generated:** 9,298,886 on the test set, **5.37 per S1
  entity** on average (median 5, maximum 8). Stage 1 alone would give 86.6M (50 per entity).
- **How we ensured true matches were not lost:** we measured pair recall and the "oracle"
  F0.5 (score of a perfect matcher restricted to the candidates) on the validation split and
  iterated on the keys. Each change was driven by inspecting missed pairs
  (`reports/blocking_experiments.md`):

| Blocking variant (20k val entities / country) | Pair recall @50 |
|---|---|
| name words + skeletons + address words, summed IDF, max_df 1000 | 0.70 (India) |
| + skeleton pairs, glued names, address bigrams (max_df 2000) | 0.942 |
| + TF-IDF cosine instead of summed IDF, max_df 5000 (final) | 0.963 |

Final blocking on the full validation split (441,364 entities):

| K | 10 | 20 | 30 | **50** | 100 |
|---|---|---|---|---|---|
| Pair recall | 0.908 | 0.943 | 0.955 | **0.965** | 0.973 |
| Oracle macro F0.5 | 0.968 | 0.980 | 0.984 | **0.987** | 0.991 |

(India 0.950 / US 0.975 pair recall at K = 50.) We also tried separate name-only and
address-only passes and their union. At the same number of candidates they were no better
than one combined pass, and they cost 3× the time.

**Shrinking the candidate set.** A plain top-K cut of the TF-IDF list loses too much: at
10 candidates, pair recall falls to 0.908 and matcher F0.5 to 0.940. A cut relative to the
best score is no better. The re-ranker fixes the *ordering* first, and only then cuts
(`reports/candidate_size_experiments.md`):

| Final candidate rule (validation) | Cands / entity | Pair recall | Oracle F0.5 | Matcher F0.5 |
|---|---|---|---|---|
| TF-IDF top 50 (stage 1 only) | 50.00 | 0.9645 | 0.9874 | 0.9556 |
| TF-IDF top 10 | 10.00 | 0.9079 | 0.9678 | 0.9399 |
| TF-IDF top 5 | 5.00 | 0.8020 | 0.9391 | 0.9145 |
| re-ranked top 5 | 5.00 | 0.9046 | 0.9774 | 0.9510 |
| **re-ranked, ≤ 8 and p ≥ 0.05 (final)** | **4.66** | **0.9449** | **0.9813** | **0.9550** |

(In the final row, the matcher F0.5 is from the matcher retrained on these candidates;
the other rows use the first matcher trained on the top-50 list.)

---

## 4. Matching Model

**Features used** (26 per candidate pair, `src/features.py`; string similarities are computed
with `rapidfuzz.process.cpdist`, in parallel):

- **Name features:** Levenshtein ratio, token-set ratio, token-sort ratio, partial ratio,
  Jaro-Winkler, ratio of the space-free names (glued names/handles); ratio and token-set
  ratio on phonetic skeletons (transliteration- and typo-robust); ratio and token-set ratio
  on "core" names with each country's frequent words removed. Frequent words are *learned*
  from the pool (words in >0.2% of that country's records, e.g. "private", "llc"; for the
  test set they are learned from the test pool, so French words such as "sarl" are handled
  without a list); number of name words on each side.
- **Address features:** ratio, token-set ratio and partial ratio (NaN if an address is
  empty); up to 4 numbers per address (house/plot/PIN), with the count shared, the count on
  each side and the share of the candidate's numbers found in the S1 address.
- **Blocking features:** cosine score, rank among the entity's candidates, score relative to
  the entity's best candidate.
- **Context features:** name and address token-set similarity minus the best value among the
  same S1 entity's candidates.
- **Other:** empty-address flag, S2-vs-S3 flag. The country label is **never** a feature.

**Model type:** LightGBM binary classifier (MIT licence, trained from scratch; no pretrained
models, no external data). 127 leaves, learning rate 0.1, feature/bagging fraction 0.8,
min 200 samples per leaf, L2 = 1, deterministic mode with seed 42. It was trained on the
re-ranker's kept candidates of a seeded sample of 300,000 training entities (about 1.4M
pairs). The two context features ("vs best") are recomputed over the kept candidates only.
Early stopping, for both the re-ranker and the matcher, used 10% of the training entities,
so the validation split stayed unseen. The matcher stopped at 1,144 trees.

**Threshold selection method:** the macro F0.5 of the validation split was computed for
every probability threshold from 0.30 to 0.95 (step 0.025), with and without a
**one-owner-per-record** rule: if a pool record is predicted for several S1 entities, it is
kept only for the most probable one (justified by the training data, where no S2/S3 record
belongs to two S1 entities). Best: threshold **0.65** with one-owner rule on. Because
precision counts double, the optimum is well above 0.5.

Most important features (share of gain): share of the candidate's numbers found in the S1
address 28%, name Jaro-Winkler 8%, name token-sort 8%, name partial ratio 8%, address
token-set vs. the entity's best candidate 5%, address token-set 5%, skeleton token-set 4%.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9550** on the validation split (441,364 S1 entities),
  exact metric.

| | macro F0.5 |
|---|---|
| Empty-prediction baseline | 0.0554 |
| Our pipeline, all entities | **0.9550** |
| — singleton entities | 0.947 |
| — entities with matches | 0.955 |
| — India / US | 0.946 / 0.961 |
| Ceiling with our final candidates (oracle) | 0.981 |
| Same matcher design on the 50-candidate list (for comparison) | 0.9556 |

| Threshold | 0.50 | 0.60 | **0.65** | 0.75 | 0.85 |
|---|---|---|---|---|---|
| F0.5, plain threshold | 0.9508 | 0.9531 | 0.9536 | 0.9535 | 0.9510 |
| F0.5, + one owner per record | 0.9535 | 0.9549 | **0.9550** | 0.9542 | 0.9513 |

- **Test-set sanity check (no labels):**

  | Test country | S1 entities | Candidates / entity | Predicted matches / entity | No match predicted |
  |---|---|---|---|---|
  | France (unseen in training) | 259,452 | 5.73 | 3.20 | 5.4% |
  | India | 809,986 | 5.47 | 3.12 | 6.6% |
  | US | 663,106 | 5.10 | 3.21 | 6.0% |

  The test set has more S2/S3 records per S1 entity than the training set (5.75 vs 4.68),
  so slightly more candidates pass the re-ranker than on validation (5.37 vs 4.66).

  In training, the true values are 3.46 matches per entity and 5.6% singletons. The unseen
  country (France) behaves like the seen ones, and the slight under-prediction is expected
  from a precision-oriented threshold.
- **Common false positives (wrong merges):** about 5% of true singletons still receive a
  wrong match (singleton F0.5 = 0.947). These are typically a different business at the same
  or a neighbouring address (same building, street and number), or a generic name that is
  very common in one city (e.g. "Shree Foods" in Pune).
- **Common false negatives (missed matches):**
  - About 3.5% of true pairs are never found by the TF-IDF search, and the re-ranker drops
    another ~2% to keep the candidate set small. These are trade names
    unrelated to the legal name ("Onyxjax", "Kordelta"), which can only be linked through
    the address, and name-only records (empty address) with heavy typos
    ("Sicbioccn" ↔ "Silicon").
  - Among the rest, the strict threshold drops weaker candidates, most often Indian-script
    names with abbreviations ("प्रा. लि.") and heavily truncated addresses. This is also
    why India (0.946) scores below the US (0.962).

---

## 6. Conclusion

A careful, data-driven blocking stage with language-neutral keys, followed by a cheap
learned re-ranker, brings 94.5% of true matches into a candidate set of only 4.7 records
per entity. A LightGBM model on string-similarity features, with a threshold tuned for F0.5
and a one-owner rule, then reaches 0.955 macro F0.5 on held-out data, and it transfers to
the unseen French data with no country-specific code. The biggest lessons:
- combinations of common words (unordered skeleton pairs) matter more than rare single
  words for blocking;
- a small re-ranking model lets the candidate set shrink about 10× with almost no loss;
- inspecting missed pairs at every step was the fastest way to improve.

A natural next step is a second-stage model that looks at all candidates of an entity
together (cluster consistency between the S2/S3 duplicates of one business).

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (Python 3.10; pinned `requirements.txt`: numpy, pandas,
scipy, scikit-learn, joblib, rapidfuzz, lightgbm, Unidecode). Entry points, run from that
folder in this order (see its `README.md`; the data folder is set with `BER_DATA_DIR`):

| Step | Command | Output | Time* |
|---|---|---|---|
| Blocking (train sample) | `python -m src.run_blocking --split train` | `cache/candidates_train.npz` | 3 min |
| Blocking (validation) | `python -m src.run_blocking --split val` | `cache/candidates_val.npz` + recall report | 4 min |
| Blocking (test) | `python -m src.run_blocking --split test` | `cache/candidates_test.npz` | 8 min |
| Train re-ranker + matcher, tune | `python -m src.train_matcher` | `cache/reranker.txt`, `cache/matcher.txt`, `cache/matcher.json` | 20 min |
| Predict test | `python -m src.predict` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` | 20 min |

\*16 logical cores, 32 GB RAM (peak use ~12 GB).

`candidate_pairs.tsv` contains exactly the candidates the re-ranker kept (at most 8 per
entity) and the matcher scored, so every matched ID is also a candidate. Both files pass `utils/validate_submission.py --check-ids`.
Supporting modules: `normalize.py` (text cleaning), `blocking.py`, `rerank.py`, `features.py`,
`metrics.py` (exact F0.5), `split.py` (seeded 80/20 split), `output_writer.py` (format
checks), plus `eda.py`, `check_scorer.py` and `baseline_empty.py` from the exploration phase.

### B. Additional Results

Detailed logs are in `code/business_entity_resolution/reports/`: `eda_report.txt`
(full EDA), `check_scorer.txt` (metric sanity checks), `blocking_experiments.md` (all
blocking variants), `candidate_size_experiments.md` (candidate-set size vs F0.5), `blocking_val.txt` (full validation recall table), `train_matcher.txt`
(training log, threshold table, feature importances), `predict_test.txt` (test statistics).

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.

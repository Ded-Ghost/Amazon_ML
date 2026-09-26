# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 business record, find all Source 2 / Source 3 records that
describe the same real-world business (zero, one or many). Scored with
macro-averaged F0.5 per Source 1 entity (singletons included).

**Result:** macro F0.5 = **0.9673** on our held-out validation split
(20% of the training S1 entities) with a final candidate set of only ~4.7
records per S1 entity (at most 8); empty-prediction baseline 0.0554.

## Layout

```
code/business_entity_resolution/
├── src/
│   ├── config.py          paths (overridable via env vars) and the random seed
│   ├── data.py            TSV loaders (sep="\t", everything as strings, no NaN)
│   ├── metrics.py         exact macro F0.5 scorer incl. singleton rules
│   ├── split.py           seeded 80/20 split of Source 1 entities
│   ├── output_writer.py   writes matching_results.tsv + candidate_pairs.tsv
│   ├── normalize.py       text cleaning + phonetic "skeleton" keys
│   ├── transliteration.py Indian-script -> Latin word dictionary learned from training pairs
│   ├── blocking.py        candidate generation (rare-key TF-IDF search per country)
│   ├── eda.py             exploratory analysis             (phase 1)
│   ├── check_scorer.py    scorer + split sanity checks     (phase 1)
│   ├── baseline_empty.py  "predict nothing" baseline       (phase 1)
│   ├── run_blocking.py    blocking for train / val / test  (phase 2)
│   ├── features.py        pair features for the re-ranker and the matcher
│   ├── rerank.py          2nd blocking stage: cheap re-ranker keeps <= 8 candidates
│   ├── match_features.py  extra/missing-word and cluster features for the matcher
│   ├── train_matcher.py   re-ranker + LightGBM matcher + F0.5 decision tuning
│   └── predict.py         test predictions -> output/*.tsv     (phase 3)
├── reports/               saved outputs of the scripts and experiment notes
├── cache/                 intermediate results + trained model (created by the scripts,
│                          not shipped: ~2.4 GB)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r code/business_entity_resolution/requirements.txt
```

By default the code expects this layout (as in the challenge repo):

```
<root>/student_resource/dataset/{train,test}/*.tsv
<root>/output/                           <- submission files are written here
<root>/code/business_entity_resolution/
```

Point it elsewhere with environment variables:
`BER_DATA_DIR=/path/to/dataset` and `BER_OUTPUT_DIR=/path/to/output`.

## Reproduce the submission (data -> blocking -> matching -> output)

Run from `code/business_entity_resolution/`, in this order. Times are for a
16-thread laptop with 32 GB RAM (peak use about 12 GB); do not let the machine
sleep during the long steps.

```bash
# 0. word dictionary for Indian-script names (learned from the 80% training half)
python -m src.transliteration              # ~3 min  -> cache/transliteration.tsv

# 1. blocking: candidate lists for a 300k training sample, the validation split and the test set
python -m src.run_blocking --split train   # ~3 min  -> cache/candidates_train.npz
python -m src.run_blocking --split val     # ~4 min  -> cache/candidates_val.npz (+ recall table)
python -m src.run_blocking --split test    # ~8 min  -> cache/candidates_test.npz

# 2. re-ranker + matching model: features, two LightGBM models, threshold tuned for F0.5
python -m src.train_matcher                # ~20 min -> cache/reranker.txt, word_scores.json,
                                           #            matcher.txt, matcher.json

# 3. test predictions and both submission files
python -m src.predict                      # ~30 min -> <root>/output/matching_results.tsv
                                           #            <root>/output/candidate_pairs.tsv
```

`run_blocking --split test` also writes a provisional `candidate_pairs.tsv` and an
empty `matching_results.tsv`; step 3 overwrites both with the final files.
Everything is seeded (`SEED = 42`), and LightGBM runs in deterministic mode. For the
previous (single-stage) version of this pipeline we re-ran `train_matcher` and got a
byte-identical model file; the re-ranker version uses the same settings.

Exploration scripts from the first phase (not needed to reproduce the output;
`baseline_empty` overwrites `<root>/output/`):

```bash
python -m src.eda              # data exploration report   -> reports/eda_report.txt
python -m src.check_scorer     # scorer sanity checks on the validation split
python -m src.baseline_empty   # "predict nothing" baseline (score floor)
```

## Pipeline

1. **Normalisation** (`normalize.py`): known Indian-script words are first replaced by
   their Latin spelling using a dictionary learned from position-aligned training
   pairs (`transliteration.py`: "प्राइवेट" -> "private", 1,312 words, covers ~93% of
   Indian-script words in the test set); then unidecode (any script -> ASCII), lowercase,
   punctuation removed, digit-for-letter typos fixed in names ("y0ga"), letters
   split from house numbers ("1604b"), and a phonetic skeleton per name word
   ("praaivett" / "private" -> "prvt").
2. **Blocking** (`blocking.py`): within each country label, records are
   described by rare keys (name words, skeletons, skeleton pairs, glued names,
   address words and bigrams). Keys are weighted by IDF, keys in more than
   5,000 records are dropped, and the top 100 S2/S3 records by TF-IDF cosine
   are kept per S1 entity. See `reports/blocking_experiments.md`.
2b. **Re-ranking, the second blocking stage** (`rerank.py`): a small LightGBM
   (31 leaves, ~220 trees, 11 cheap features: blocking score/rank, a few name
   similarities, address overlap, house-number overlap) re-orders the top 50
   and keeps at most 8 candidates with re-rank probability >= 0.05. These kept
   candidates are the final candidate set (`candidate_pairs.tsv`, ~4.7 per S1
   entity); the matcher runs only on them. See
   `reports/candidate_size_experiments.md`.
3. **Pair features** (`features.py`): 26 language-neutral numbers per
   (S1, candidate) pair: rapidfuzz name similarities (ratio, token-set,
   token-sort, partial, Jaro-Winkler, glued), the same on phonetic skeletons
   and on "core" names (frequent words of the country removed, learned from
   the data), address similarities, shared house/plot numbers, blocking
   score and rank, and comparisons with the entity's best candidate.
   The country label itself is never a feature.
3b. **Word and cluster features** (`match_features.py`, on the kept candidates):
   words the candidate adds to / drops from the S1 name, scored by how often such
   a pair is a true match in the training pairs ("midtown", "south", "holdings":
   almost never; "services", "www", "dba": usually), cross-fitted for the training
   rows; and how similar each candidate is to the *other* kept candidates of the
   same entity (true matches are copies of each other).
4. **Matcher** (`train_matcher.py`): LightGBM binary classifier (MIT licence)
   on the re-ranker's kept candidates of 300k training entities, early-stopped
   on 10% of the training entities.
5. **Decision rule**: predict pairs with probability >= threshold (tuned on
   validation for macro F0.5), then keep each S2/S3 record only for its most
   probable S1 entity (in the training data no record belongs to two S1s).

Validate the submission (from `student_resource/`):

```bash
python utils/validate_submission.py --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
```

## Validation protocol

Source 1 train entities are split 80/20 with seed 42 (`src/split.py`). The
Source 2 / Source 3 pools are shared by both halves, exactly like at test time.
Scores reported in this project are macro F0.5 on the 20% validation half.

## Rules compliance

* No external data, APIs, geocoding or lookups: only the provided TSV files are read.
* The only model is a LightGBM classifier (MIT licence) trained from scratch; there are
  no pretrained models.
* The country label is treated as an open set. Blocking runs per label found in the
  data, frequent words are learned per label, and the country is never a model feature,
  so France (test only) goes through exactly the same code as US and India.

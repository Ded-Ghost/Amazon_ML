# Business Entity Resolution — Amazon ML Challenge 2026 (team Null-Pointers)

For every Source 1 business record, find all Source 2 / Source 3 records that
describe the same real-world business (zero, one or many). Scored with
macro-averaged F0.5 per Source 1 entity (singletons included).

**Result:**
- **Validation:** macro F0.5 = **0.9797** on our held-out split (20% of the training
  S1 entities; US 0.9806, India 0.9783). With all training owners competing (measured
  for the previous K = 50 version): 0.9789.
- **Unseen country:** checked by training on one country and testing on the other, in
  both directions (see `reports/improvement_experiments.md`).
- **Candidates:** about 6.8 per S1 entity (at most 10).
- **Leaderboard:** see `reports/`.

## Layout

```
code/business_entity_resolution/
├── src/
│   ├── config.py              paths (overridable via env vars) and the random seed
│   ├── data.py                TSV loaders (sep="\t", everything as strings, no NaN)
│   ├── metrics.py             exact macro F0.5 scorer incl. singleton rules
│   ├── split.py               seeded 80/20 split of Source 1 entities
│   ├── output_writer.py       writes matching_results.tsv + candidate_pairs.tsv (format checks)
│   ├── normalize.py           text cleaning, phonetic "skeleton" keys
│   ├── transliteration.py     Indian-script -> Latin word dictionary learned from training pairs
│   ├── blocking.py            stage 1: rare-key TF-IDF search per country (top 100)
│   ├── run_blocking.py        runs blocking for train / val / rest / test
│   ├── features.py            26 pair features (names, addresses, numbers, blocking)
│   ├── rerank.py              stage 2: re-ranker keeps <= 10 candidates (the final candidate set)
│   ├── match_features.py      word-difference, cluster, count, sibling-word and formatting features
│   ├── train_matcher.py       re-ranker + matcher on a 300k sample, tuned on validation
│   ├── validate_competition.py threshold tuned with all training owners competing
│   ├── train_final.py         final matcher refitted on all 2.2M labelled entities
│   ├── predict.py             test predictions -> output/*.tsv
│   ├── eda.py, check_scorer.py, baseline_empty.py   exploration scripts (phase 1)
├── reports/                   logs of every step and experiment notes
├── cache/                     intermediate results + trained models (created by the scripts, not shipped)
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
16-thread laptop with 32 GB RAM (peak use about 20 GB); do not let the machine
sleep during the long steps.

```bash
# 0. word dictionary for Indian-script names (learned from the 80% training half)
python -m src.transliteration              # ~3 min  -> cache/transliteration.tsv

# 1. stage-1 blocking (top 100 per S1 entity)
python -m src.run_blocking --split train   # ~3 min  300k-entity sample used to develop the models
python -m src.run_blocking --split val     # ~4 min  20% validation half (+ recall table)
python -m src.run_blocking --split rest    # ~10 min other training entities (competition + final fit)
python -m src.run_blocking --split test    # ~8 min

# 2. re-ranker + matcher on the 300k sample, tuned and measured on validation
python -m src.train_matcher                # ~30 min -> cache/reranker.txt, word_scores.json, matcher.*

# 3. threshold under full owner competition (validation + all other training entities)
python -m src.validate_competition         # ~40 min -> threshold in cache/matcher.json

# 4. final matcher refitted on all labelled entities (same settings)
python -m src.train_final                  # ~60 min -> cache/matcher.txt, word_scores.json

# 5. test predictions and both submission files (as submitted: for countries without
#    training labels, the matcher is adapted to the country's own confident predictions
#    and the house-number rule is applied)
python -m src.predict --unseen-adapt --unseen-strict --threshold 0.675 --unseen-k 200
                                           # ~85 min, blocking lists of countries without
                                           # training labels deepened to 200
cp cache/test_scores.npz cache/v11_test_scores.npz
# 6. deeper blocking lists (200) for India too: re-scores only India's S1 records and keeps
#    the other countries' scores (records never match across countries), ~60 min
python -m src.rescore_unseen --k 200 --countries India --base cache/v11_test_scores.npz
python -m src.predict --reuse --unseen-adapt --unseen-strict --threshold 0.675
                                           # -> <root>/output/matching_results.tsv
                                           #    <root>/output/candidate_pairs.tsv
# (the threshold is given explicitly because step 3 was not rerun for K = 100:
#  the validation curve is flat from 0.675 to 0.775, and step 3 chose 0.65-0.675 for K = 50)
```

`run_blocking --split test` also writes a provisional `candidate_pairs.tsv` and an
empty `matching_results.tsv`; step 5 overwrites both. `python -m src.predict --reuse
--threshold T --out DIR` re-applies the decision rule to the saved test probabilities.
Everything is seeded (`SEED = 42`), and LightGBM runs in deterministic mode.

Exploration scripts from the first phase (not needed to reproduce the output;
`baseline_empty` overwrites `<root>/output/`):

```bash
python -m src.eda              # data exploration report   -> reports/eda_report.txt
python -m src.check_scorer     # scorer sanity checks on the validation split
python -m src.baseline_empty   # "predict nothing" baseline (score floor)
```

## Pipeline

1. **Normalisation** (`normalize.py`, `transliteration.py`):
   - Indian-script words are replaced by their Latin spelling, using a dictionary
     learned from position-aligned training pairs ("प्राइवेट" -> "private").
   - `unidecode`; dotted acronyms joined ("S.A.S." -> "sas"); "&"/"+" -> "and";
     punctuation removed.
   - Digit-for-letter typos fixed in names ("y0ga"); house numbers cleaned ("012" -> "12",
     "1604b" -> "1604 b").
   - A phonetic skeleton per name word ("praaivett" / "private" -> "prvt").
2. **Blocking** (`blocking.py`):
   - Inside each country label, records are described by rare keys: name words,
     skeletons, unordered skeleton pairs, glued names, and address words and bigrams.
   - Keys are IDF-weighted, keys found in more than 5,000 records are dropped, and the
     top 100 S2/S3 records by TF-IDF cosine are kept (all 100 go to the re-ranker).
3. **Re-ranking** (`rerank.py`):
   - A small LightGBM on the 26 pair features of all 100 blocking candidates keeps at
     most 10 candidates with
     probability >= 0.005.
   - These are the final candidate set (`candidate_pairs.tsv`); the matcher scores only
     them.
4. **Matcher features** (`features.py`, `match_features.py`), 60 per pair:
   - string similarities of names and addresses;
   - learned scores for the words a candidate adds to or drops from the S1 name
     ("midtown", "south": almost never a match; "services", "dba": usually);
   - similarity to the other candidates of the same entity;
   - label-free counts over the whole dataset: how many S1 businesses share the name or
     the address, how common each differing word is, and house-number distance;
   - label-free sibling-word rates: for each word a candidate adds to the S1 name, the
     share of that country's pairs (in the dataset being scored, no labels) where the
     word comes with a conflicting house number ("midtown" ~0.8: a sibling branch;
     "dba" ~0.0: a true copy). They are computed from the unlabeled pairs, so they
     also cover the words of a country never seen in training;
   - formatting (case, accents, symbols, dropped digits).
   - The country label is never a feature.
5. **Matcher** (`train_matcher.py`, `train_final.py`):
   - LightGBM (MIT licence), developed on a 300k-entity sample and refitted on all 2.2M
     labelled entities.
   - Word scores for the training rows are cross-fitted with 30% word dropout, so the
     model also learns to cope with unknown words.
6. **Prediction** (`predict.py`):
   - Two passes. Words never seen in training (e.g. French) get scores from confident
     first-pass predictions (self-training on the unlabeled test pairs; no labels used).
   - Pairs above the threshold are kept, then each S2/S3 record goes to its most
     probable S1 entity only (one owner per record, as in the training data).
   - The threshold is tuned on validation with all training owners competing
     (`validate_competition.py`).
   - For countries without training labels (`--unseen-adapt`, used for the submission),
     the matcher gets 100 more trees fitted on the country's own confident predictions
     (p ≥ 0.9 / ≤ 0.1, cross-fitted by entity halves, no labels).
   - For countries without training labels (`--unseen-strict`, used for the submission),
     a match whose house numbers conflict is rejected unless the numbers differ only by
     a dropped digit. In France, same-name records at other house numbers are mostly
     namesakes (5× more frequent per entity than in the US). `--unseen-threshold T` sets
     a separate threshold (not used).

Validate the submission (from `student_resource/`):

```bash
python utils/validate_submission.py --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
```

## Rules compliance

* **No external data, APIs, geocoding or lookups:** only the provided TSV files are read.
  The Indian-script dictionary and the word scores are learned from the provided
  training pairs. Self-training uses the model's own predictions on the provided test
  records, never labels.
* **Models:** LightGBM classifiers (MIT licence) trained from scratch; no pretrained
  models.
* **Country as an open set:** blocking runs per label found in the data, frequent words
  are learned per label, and the country is never a model feature. France (test only)
  goes through the same code as US and India. Every rule for "countries without training
  labels" applies to any such label.

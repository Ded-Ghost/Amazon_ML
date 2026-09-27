# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Null-Pointers  
**Team Members:** Siddhanth Roy, Prithwiraj Chatterjee, Satyam Chattopadhyay, Chandan Kumar Barman  
**Submission Date:** 27/09/2026

---

## 1. Executive Summary

The pipeline has three stages and is built to find every copy of a business while keeping
the candidate set small:

1. **Blocking:** a sparse TF-IDF search over hand-designed "rare keys", run inside each
   country label, returns a top-100 shortlist per Source 1 entity (200 for France and
   India in the final version). The keys are name
   words, phonetic skeletons, unordered skeleton pairs, glued names, and address words
   and bigrams.
2. **Re-ranking:** a small LightGBM cuts the shortlist to at most 10 candidates, about
   7.1 per entity on the test set. These are the final candidate set.
3. **Matching:** a LightGBM matcher scores the candidates using 60 language-neutral
   features, with a decision rule tuned for macro F0.5 under the same "every owner
   present" conditions as the test set.

The main ideas behind the result:
- **A word dictionary learned from the training pairs** for Indian-script names
  ("प्राइवेट" → "private"). India's blocking recall rose from 95.0% to 97.2%.
- **Learned "extra/missing word" scores** that separate decoy sibling businesses
  ("Jimenez Distribution *Midtown* LLC") from noisy copies. This was our largest
  single gain (+0.008).
- **Label-free counts over the whole dataset:** how many businesses share a name or an
  address, how common each differing word is, and house-number distance. These
  resolve name-only records, trade names and neighbouring businesses.
- **Robustness to the unseen country (France):** word dropout during training,
  self-training of unknown-word scores on the unlabeled test pairs, generic
  normalisation of dotted legal forms ("S.A.S." → "sas"), and **label-free
  sibling-word features** computed on the test pairs themselves (validated by training
  on one country and testing on the other, in both directions).
- **Owner competition:** each S2/S3 record goes to at most one S1 entity, and the
  threshold is tuned with all training owners competing.

Held-out validation score, measured with all owners competing, as on the test set:
**macro F0.5 = 0.9797** (US and India; final version v13 uses the same models). Public leaderboard: **0.976** (final
version v13); see the table in Section 5 for every version.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from exploratory analysis of the data (`reports/eda_report.txt`):

| Observation | Value | Consequence for the design |
|---|---|---|
| Records | S1 2.21M, S2 5.03M, S3 5.29M (train); S1 1.73M, S2 4.89M, S3 5.08M (test) | All-pairs comparison impossible → blocking is mandatory |
| Singletons (S1 with no match) | 5.58% | Most of the score comes from finding matches; recall matters despite β = 0.5 |
| Matches per S1 entity | mean 3.46 (median 4 for non-singletons, max 11) | Predict a *set* per entity |
| Matches by source | S3 51.6%, S2 48.4%; 85% of matched entities have both | S2 and S3 each contain several copies of a business |
| S2/S3 records matched to >1 S1 | **0** | One owner per record → owner-competition decision rule |
| S2/S3 records matched to nobody | 26% (train); ~40% (test, estimated) | Many decoys; the test has more decoys per business than the training data |
| Matched pairs with the same country label | **100.0000%** of 7.64M | Block within country label (works for any label, incl. France) |
| Indian-script names (8 scripts) | ~9.4% of S2, ~5.3% of S3 rows | Word dictionary learned from training pairs + phonetic keys |
| Empty address | ~3% of S2/S3 rows (0% in S1) | Name-only records are the hardest cases |
| Test-only country | France: 15% of test S1 | No country-specific model; generic rules for any unseen label |

**Test vs training distribution.** Copies per business are the same in both: 0.97
exact-name copies per unique-name US business in each. The test pool has about 23% more
S2/S3 records per S1 entity, so the extra records are **decoys**. This is why every
decision rule is tuned under full owner competition (Section 4).

**Noise patterns in true copies:**
- word-order transpositions and typos, including digit-for-letter swaps ("Y0ga");
- accents, dropped or changed legal forms (LLC ↔ L.L.C., SAS ↔ S.A.S.);
- glued names and web handles ("willshore.com", "@opticienslycee");
- unrelated trade names at the owner's address ("Dovasynonyx");
- address reordering and abbreviations, dropped digits (826 → 26), and junk prefixes.

**Decoys:**
- sibling businesses with a qualifier word and a neighbouring house number
  ("Gatto Miluna *South*, 234" vs "229");
- namesakes on other streets (frequent in France);
- near-identical typo variants at the same address;
- name-only records that drop the distinctive part of the name
  ("Johnson City Human Rights Ministries" → "Johnson City").

### 2.2 Solution Strategy

**Approach Type:** Two-stage blocking (TF-IDF rare-key search → learned re-ranker) + LightGBM classifier + F0.5 decision rule under owner competition  
**Core Innovation:** learned, language-neutral evidence about *how names differ* (extra/missing word scores, learned from labelled pairs and extended to an unseen language by word dropout and self-training), combined with label-free dataset-wide counts that tell a business's copies apart from its decoys.

**Validation protocol:**
- **Split:** Source 1 training entities are split 80/20 with seed 42, and S2/S3
  records are shared by both halves.
- **Tuning:** models are developed on a 300k-entity sample of the 80% half and scored
  on the 20% half, with the exact challenge metric.
- **Full competition:** because every owner is present on the test set, the final
  threshold is tuned by scoring the validation entities *together with* the other
  1.47M training entities and applying the one-owner rule across all of them.
- **Final fit:** the matcher is refitted on all 2.2M labelled entities.

---

## 3. Candidate Generation (Blocking)

**Normalisation** (`src/normalize.py`, identical for every country):
- **Indian-script dictionary:** known words are replaced by their Latin spelling. The
  dictionary has 1,312 words, is learned from position-aligned training pairs, and
  covers 93% of Indian-script words in the test set.
- **Cleaning:** `unidecode` to ASCII; dotted acronyms are joined
  ("E.U.R.L." → "eurl"); "&" and "+" become "and"; punctuation is removed.
- **Names:** digit-for-letter typos are undone ("y0ga").
- **Addresses:** letters are split from numbers ("1604b"), and leading zeros are dropped.
- **Phonetic skeleton:** each word keeps only its consonants, with sound-alikes merged
  ("praaivett" / "private" → `prvt`).

**Stage 1: rare-key TF-IDF search** (`src/blocking.py`):
- **Keys:** name words, skeletons, unordered skeleton pairs (common words combine into
  rare keys), the first 1–3 name words glued together, and address words and bigrams.
- **Weighting:** keys are hashed into sparse matrices and weighted by IDF within each
  country's S2+S3 pool. Keys found in more than 5,000 records are dropped.
- **Scoring:** TF-IDF cosine, computed as one sparse matrix product per chunk. The
  top 100 records are kept per S1 entity, and all 100 go to the re-ranker. In the final
  version the lists of France and India are 200 deep (Section 5): the test set is more
  crowded than the training data, so true copies sit deeper in the ranking.

**Stage 2: learned re-ranker** (`src/rerank.py`):
- **Model:** LightGBM, 31 leaves, about 360 trees, on the 26 pair features.
- **Keep rule:** at most **10** candidates per entity, and only those with re-rank
  probability ≥ **0.005**.
- **Output:** the kept pairs are `candidate_pairs.tsv`, and the matcher runs only on them.

- **Candidate pairs generated:** 12,244,224 on the test set, **7.1 per S1
  entity**, from about 280M stage-1 pairs (top 100 for the US, top 200 for France and
  India).
- **How we ensured true matches were not lost:** pair recall and the oracle F0.5 (a
  perfect matcher restricted to the candidates) were measured on validation after every
  change (`reports/blocking_experiments.md`, `reports/candidate_size_experiments.md`):

| Candidate generation (validation) | Cands/entity | Pair recall | Oracle F0.5 |
|---|---|---|---|
| words only, summed IDF (first try) | 50 | 0.704 (India) | |
| + skeleton pairs, glued names, bigrams | 50 | 0.942 | |
| + TF-IDF cosine, max_df 5000 | 50 | 0.965 | 0.987 |
| + Indian-script dictionary | 50 | 0.974 | 0.992 |
| TF-IDF top 10 only (no re-ranker) | 10 | 0.920 | 0.974 |
| + 26-feature re-ranker on the top 50, ≤10 & p≥0.005 (v4–v9) | 6.1 | 0.971 | 0.991 |
| **+ re-ranker on all 100 blocking candidates, ≤10 & p≥0.005 (final, v10)** | **6.2** | **0.977** | **0.993** |

We tried three ideas that didn't make it:
- **Separate name and address passes:** no better than one combined pass.
- **Reverse search (record → S1):** would recover about half of the missed copies, but
  at a large cost.
- **"Second-hop" search around confident matches:** rejected because only 2–3% of
  near-identical neighbours of our matches are true copies; the generator plants
  near-identical decoys.

---

## 4. Matching Model

**Features used** (60 per candidate pair, `src/features.py` and `src/match_features.py`;
rapidfuzz `cpdist`, parallel):

- **Name features (14):**
  - Levenshtein ratio, token-set, token-sort and partial ratio, Jaro-Winkler, and the
    ratio of the space-free names;
  - the same on phonetic skeletons and on "core" names (each country's frequent words
    removed; learned from the pool, so French words are handled too);
  - name lengths.
- **Address features (6):** ratio, token-set and partial ratio; shared house numbers,
  counts, and the share of the candidate's numbers found in the S1 address.
- **Blocking features (3):** cosine score, rank, and score relative to the entity's best
  candidate.
- **Word differences (7):**
  - words the candidate *adds* to or *drops* from the S1 name, where typo-tolerant
    matching means typos don't count;
  - each word scored by how often a pair with that extra/missing word is a true match
    in the training pairs ("midtown", "south", "holdings" ≈ 0.00–0.01; "services",
    "www", "dba" ≈ 0.8–0.97);
  - for the matcher's training rows the scores are cross-fitted with **30% word
    dropout**, so the model also learns what to do with unknown words.
- **Cluster features (3):** similarity of each candidate to the *other* candidates of
  the same entity (true copies are copies of each other).
- **Label-free counts (14):**
  - how many S1 businesses share this name, core name or address (house number + the
    street's rarest word);
  - whether the candidate's name or address belongs to *another* S1 business;
  - how many pool records share the name or address;
  - how common each differing word is (one-off words vs branch words);
  - the smallest house-number distance.
- **Sibling words, label-free (3):** the generator places sibling businesses
  ("... Midtown", "... Développement") at a *different* house number, while true copies
  keep it. For each country, over the pairs of the dataset being scored (no labels), each
  word a candidate adds to the S1 name gets the share of its pairs whose house numbers
  conflict ("midtown" ≈ 0.8, "dba" ≈ 0.0). The rates are turned into **percentiles within
  the country**, so they mean the same everywhere: raw rates did not transfer (conflicting
  numbers are 10% of India's pairs but 17% of the US's). Features: max, mean, and number
  of words above the 90th percentile.
- **Formatting (7):** raw strings identical, letter case, accents, symbols, a house
  number differing only by a dropped digit (826 → 26), and name-length difference.
- **Other:** empty-address flag, S2 vs S3. The country label is **never** a feature.

**Model type:** LightGBM binary classifier (MIT licence, trained from scratch, no
pretrained models, no external data):
- 127 leaves, learning rate 0.1, feature/bagging fraction 0.8, deterministic, seed 42;
- settings chosen on the 300k-entity sample;
- **final model fitted on all 2,206,821 labelled entities** (13.75M pairs, 1,979 trees;
  each doubling of the training data added ~+0.001 F0.5).

**Unseen country (France):**
- **Self-training for unknown words:** at prediction time the matcher runs twice.
  Words never seen in training get scores learned from confident first-pass
  predictions (p ≥ 0.9 / ≤ 0.1, cross-fitted by entity halves; one round).
- **Proxy test (new language):** on validation, with all word scores hidden, F0.5 is
  0.9775 without self-training and 0.9778 with it, against 0.9780 normally.
- **Cross-country simulation (new country):** we train on one training country and test
  on the other's validation entities, in both directions. This is how every
  France-targeted change was accepted or rejected:

  | Setup (F0.5 on the unseen country, threshold 0.675) | US → India | India → US |
  |---|---|---|
  | no sibling features | 0.9296 | 0.9429 |
  | raw sibling rates (rejected: fails one direction) | 0.9430 | 0.9181 |
  | **percentile sibling rates (final)** | **0.9435** | **0.9587** |
  | + full self-training with pseudo-labels (reference; ~2 h on the full data) | 0.9455 | 0.9622 |
  | **+ adaptation: 100 more trees on the country's own confident predictions (final)** | **0.9441** | **0.9621** |

- **Adaptation to the unseen country** (`predict.py --unseen-adapt`): after the two
  passes, the trained matcher gets 100 more trees fitted only on the unseen country's own
  confident predictions (p ≥ 0.9 → match, ≤ 0.1 → no match; no labels), cross-fitted by
  S1-entity halves so that no pair learns from its own pseudo-label. As good as full
  retraining with pseudo-labels in the simulation, for about one minute of compute.
  On the test set it changes 3.8% of French entities and nothing else.

- **House-number rule for countries without training labels** (`predict.py
  --unseen-strict`): a match whose house numbers conflict (no shared number, not a
  dropped digit) is rejected. In the training countries a same-name record at a
  different house number is still a true copy 58.5% of the time, but on the test set
  France has 5× more "same name, different number" candidates per entity than the US
  (0.48 vs 0.09): chains of namesakes such as "Nantes Sportive SARL" at four addresses
  in one city. The rule raised the leaderboard score from 0.968 to 0.969 (v6) and is
  part of the final submission. It applies to any country label absent from training.
- **Checked and rejected for France:** a "same street" rule. France's extra
  street-name differences among accepted matches are typos (6.4% vs 2.3% in the US),
  which are 99.9% true copies in the US/India validation data.

**Threshold selection method:**
- a probability threshold, plus the **one-owner rule**: a record predicted for several
  S1 entities goes to the most probable one;
- the threshold is tuned on validation **with all 1.9M validation + other training
  entities competing** (as on the test set): **0.65 (0.675 on the validation split alone;
  0.65 and 0.675 tie at 0.9789)**.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9797** on validation (US 0.9806, India 0.9783; singletons
  0.9792). The K = 50 version scored 0.9780 on the same split and 0.9789 under full owner
  competition. France has no labels; see Section 4 for how it was validated.

| Version | Main change | Validation F0.5 | Leaderboard |
|---|---|---|---|
| v1 | TF-IDF blocking (top 50) + LightGBM on 26 features | 0.9556 | — |
| v2 | + learned re-ranker, candidates 50 → 4.7 per entity | 0.9550 | 0.948 |
| v3 | + Indian-script dictionary, word-difference and cluster features | 0.9673 | 0.952 |
| v4 | + 26-feature re-ranker (≤10), name counts, word dropout, self-training | 0.9758 | 0.965 |
| v5 | + address counts, formatting, competition threshold, fit on all data | 0.9781 (competition 0.9789) | not submitted |
| v6 | + dotted-acronym normalisation | 0.9782 (competition 0.9791) | 0.968 |
| v6 + rule | + house-number rule for countries without training labels | same (rule only affects France) | 0.969 |
| v7 | + raw sibling-word rates | 0.9781 | not submitted (failed the India → US check) |
| v8 | + percentile sibling-word rates, with the house-number rule | 0.9780 (competition 0.9789) | not submitted |
| v9 | + adaptation to the unseen country (France only) | same (only France changes) | 0.969 |
| v10 | + all 100 blocking candidates to the re-ranker (was 50) | 0.9797 | 0.975 |
| v11 | + blocking depth 200 for countries without training labels (France) | same | 0.975 |
| **v13 (final)** | **+ blocking depth 200 for India** | **same models** | **0.976** |

- **Common false positives (wrong merges):**
  - decoys placed next to real copies: near-identical typo variants at the same address
    ("Halkos Sérvice" vs "Halcos Services");
  - sibling businesses a few doors away;
  - in France, namesakes on other streets of the same city.
- **Common false negatives (missed matches):**
  - **name-only copies:** even with a unique, exact name they are true only ~60% of the
    time, because the generator makes name-only decoys;
  - **copies never reaching the candidate set:** about 3%, mostly garbled names or
    trade names with partial addresses.
  - A perfect matcher on our candidates would score 0.993 (validation).

---

## 6. Conclusion

The biggest improvements came from studying the data generator's errors rather than
tuning models:
- **What the model couldn't know:** Indian scripts (a learned dictionary), decoy
  siblings (learned word scores), shared names and addresses (dataset-wide counts), and
  an unseen language (word dropout, self-training, generic normalisation).
- **How we measured:** every change was checked on a validation setup that mimics the
  test set, with all owners competing.

Validation rose from 0.955 to 0.979. The candidate set stays small, about 6–7 records
per business instead of 50.

On the leaderboard the France-targeted changes of v8/v9 (validated by the cross-country
simulation) left the rounded score at 0.969: their effect on the real unseen country was
below the leaderboard's 0.001 resolution. Training countries are measured precisely;
the unseen country can only be approximated.

The remaining loss is mostly:
- name-only records, which are only ~60% reliable even with a unique exact name;
- about 3% of copies that never reach the candidate set;
- the unseen country, where the matcher's learned trust in names over addresses
  doesn't carry over.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (Python 3.10; pinned `requirements.txt`: numpy,
pandas, scipy, scikit-learn, joblib, rapidfuzz, lightgbm, Unidecode). Run from that folder
in this order (full commands in its `README.md`; the data folder is set with
`BER_DATA_DIR`):

| Step | Command | Output | Time* |
|---|---|---|---|
| Indian-script dictionary | `python -m src.transliteration` | `cache/transliteration.tsv` | 3 min |
| Blocking (train sample, validation, other train, test) | `python -m src.run_blocking --split {train,val,rest,test}` | `cache/candidates_*.npz` | 25 min |
| Re-ranker + matcher on the sample, tuning | `python -m src.train_matcher` | `cache/reranker.txt`, `matcher.*`, `word_scores.json` | 30 min |
| Threshold under full owner competition | `python -m src.validate_competition` | threshold in `cache/matcher.json` | 40 min |
| Final matcher on all labelled entities | `python -m src.train_final` | `cache/matcher.txt`, `word_scores.json` | 65 min |
| Test prediction (France lists 200 deep, adaptation, house-number rule) | `python -m src.predict --unseen-adapt --unseen-strict --threshold 0.675 --unseen-k 200` | `cache/test_scores.npz` | 85 min |
| India lists 200 deep (re-scores India only) | `python -m src.rescore_unseen --k 200 --countries India --base cache/v11_test_scores.npz` (after copying the scores of the previous step to that file) | `cache/test_scores.npz` | 60 min |
| Final files | `python -m src.predict --reuse --unseen-adapt --unseen-strict --threshold 0.675` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` | 2 min |

\*16 logical cores, 32 GB RAM.

`candidate_pairs.tsv` contains exactly the candidates the re-ranker kept and the matcher
scored, so every matched ID is also a candidate. Both files pass
`utils/validate_submission.py --check-ids`.

### B. Additional Results

All logs are in `code/business_entity_resolution/reports/`:
- `eda_report.txt`: full exploratory analysis.
- `blocking_experiments.md` and `candidate_size_experiments.md`: blocking and candidate-set
  experiments.
- `improvement_experiments.md`: every feature and training experiment from v3 to v8
  (gains, the France investigation, the leaderboard audit with error buckets, and the
  hypotheses we tested and rejected: leaks, copy quotas, same-source structure,
  per-entity decision rules, noise artefacts, edit types, namesake evidence).
- `blocking_*.txt`: recall tables.
- `train_matcher*.txt`, `validate_competition*.txt`, `train_final*.txt`: training and
  tuning logs, including threshold tables and feature importances.
- `predict_test*.txt`: per-country test statistics.
- `validator_test*.txt`: validator output.

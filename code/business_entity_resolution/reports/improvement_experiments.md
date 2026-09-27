# Improvement experiments (v3 → v7)

Validation = the 20% held-out S1 entities (441,364), exact macro F0.5. The
models were developed on a 300k-entity sample of the 80% training half. Scores
are "validation only" unless marked "competition", which means the validation
entities were scored together with the other 1.47M training entities so that
every owner competes, as on the test set.

## Where the loss was (error analysis, v2 → v3)

- **v2 (0.955):** 58% of the wrong merges on validation were records owned by a
  training-half business that is absent from validation. The validation split
  therefore underestimates test precision, which led to `validate_competition.py`.
- **Missed copies outweighed wrong merges:** v3 lost 0.0327, of which about 0.023
  came from missed copies.
- **Hard negatives:** decoy siblings with a qualifier word and a neighbouring house
  number ("Jimenez Distribution *Midtown* LLC", 44 vs 39 Myrtle Ave).

## Features and training changes

| Change | Validation F0.5 | Note |
|---|---|---|
| v2 baseline (26 features, ≤8 candidates) | 0.9550 | |
| + candidate-candidate cluster similarity | 0.9562 | |
| + extra/missing word scores (learned, cross-fitted) | 0.9627 | largest single gain |
| + both | 0.9635 | |
| + Indian-script dictionary (blocking recall India 95.0 → 97.2%) = **v3** | 0.9673 | leaderboard 0.952 |
| v3 with all word scores hidden ("new language") | 0.9234 | France proxy: big drop |
| + 30% word dropout | 0.9672 / unknown 0.9606 | robustness at no cost |
| + one round of self-training for unknown words | unknown → 0.9658 | 2 or 3 rounds: worse (0.9651, 0.9640) |
| + S1 name counts (how many S1 businesses share the name) | 0.9723 / unknown 0.9669 | |
| + word commonness (log document frequency) | 0.9684 / unknown 0.9671 | |
| + all counts + house-number distance | 0.9738 / unknown 0.9728 | |
| + 26-feature re-ranker, ≤10 candidates, p ≥ 0.005 = **v4** | 0.9758 / unknown 0.9746 | leaderboard 0.965 |
| v4 under full competition | 0.9772 (competition) | best threshold 0.65 |
| + address counts (number + rarest street word) | 0.9775 | |
| + initials feature | 0.9774 | no gain, dropped |
| + formatting and dropped-digit features = **v5 (sample)** | 0.9781 | |
| v5 under full competition | 0.9789 (competition) | threshold 0.675 |
| + stage-2 neighbour support (out-of-fold probabilities) | 0.9788 | +0.0007, not used (cost) |
| bigger model (lr 0.05, 255 leaves) | 0.9774 | no gain |
| + dotted acronyms / "+" normalisation = **v6 (sample)** | 0.9782 / unknown+self-training 0.9781 | aimed at France (5.4% of French records have dotted legal forms, 0% of French S1 names) |

## Training data size (learning curve, v5 features)

| Training entities | 67.5k | 135k | 270k |
|---|---|---|---|
| Validation F0.5 | 0.9754 | 0.9765 | 0.9775 |

Each doubling added about +0.001, so the final matcher is refitted on all 2.2M
labelled entities (`train_final.py`: 13.4M pairs, 2,372 trees).

## Unseen-country (France) investigation

- **Leaderboard arithmetic:** with US/India at about 0.977 on test (competition
  validation), the leaderboard scores imply France was about 0.90 in v4.
- **Cross-country simulation:** training on US only and testing on India
  (validation):

  | Setup | India F0.5 |
  |---|---|
  | trained on US + India | 0.9769 |
  | trained on US only | 0.9287 |
  | + word self-training | 0.9338 |
  | + full self-training (pseudo-labels, p ≥ 0.9 / ≤ 0.1, 96.8% correct) | 0.9379 |
  | + full self-training (p ≥ 0.97 / ≤ 0.03) | 0.9287 |

  In an unseen country, precision tops out near 0.97 even at high thresholds: the
  errors are confident wrong merges, and the best threshold is lower (0.45–0.5).
- **Address noise differs by country:** in US/India a same-name candidate at a
  different address identity is still a true copy 58.5% of the time, so the model
  learned to trust names. In France the same pattern is mostly a namesake or
  sibling (visual inspection), which motivates the optional unseen-country
  house-number rule in `predict.py` (`--unseen-strict`, `--unseen-threshold`).
- **Test distribution:** copies per business are identical in train and test (0.97
  exact-name copies per unique US name in both). The extra 23% of test records per
  S1 entity are decoys.

## Ideas measured and rejected

| Idea | Finding |
|---|---|
| "Second-hop" search around confident matches | candidates near-identical to another candidate but rejected are true copies only 2–3% of the time (the generator plants near-identical decoys) |
| Reverse blocking (record → S1) | recovers ~50% of the copies forward blocking misses (37.7% at rank 1), but at ~1.5 h extra compute; estimated +0.002–0.003 |
| Name-only records | even a unique, exact-name, no-address record is a true copy only 60% of the time (name-only decoys); formatting helps a little (raw-identical: 81%) |

## Leaderboard audit (after v6: leaderboard 0.968, 0.969 with the unseen-country rule)

**Does local validation track the leaderboard?** The test S1 table lists only
about 82% of the businesses whose copies are in S2/S3. In training every business
is listed. We rebuilt validation under test-like conditions: a seeded 82% of the
training S1 table was kept for the label-free counts and for the owner
competition, and the copies of the other 18% were left in the pool as decoys.

| Threshold | A: all owners (as before) | only competition at 82% | only counts at 82% | B: both at 82% (like test) |
|---|---|---|---|---|
| 0.60 | 0.9789 | 0.9788 | 0.9782 | 0.9780 |
| 0.675 | 0.9791 | 0.9790 | 0.9784 | 0.9782 |
| 0.75 | 0.9788 | 0.9788 | 0.9783 | 0.9781 |
| 0.85 | 0.9777 | 0.9777 | 0.9773 | 0.9772 |

The missing owners cost only 0.001, and the best threshold does not move. So
US/India on the test set is about 0.978. The leaderboard score (0.968–0.969)
then implies that **France (15% of test entities) scores about 0.91–0.92**
(0.85 × 0.978 + 0.15 × F = 0.969). Even a perfect France would
give only about 0.981, so the remaining gap is almost entirely the unseen
country.

**Error buckets (validation, condition B, F0.5 points recovered if the bucket were fixed):**

| Bucket | Points |
|---|---|
| all missed copies (FN) | 0.0174 |
| missed copies not among the candidates | ≈ 0.0098 |
| name-only copies rejected by the matcher | 0.0047 |
| near-identical names rejected by the matcher | 0.0018 |
| all wrong merges (FP) | 0.0044 |

Of the copies missing from the candidates, 29.5k are not in the blocking top
100, 10.8k are at ranks 50–99 and 3.7k were cut by the re-ranker. Retrieving
them costs a second blocking pass (see reverse blocking below) or top-100
re-ranker input (about +0.001, memory-heavy). The name-only and near-identical
buckets are mostly generator decoys (see "Ideas measured and rejected").

## Label-free sibling-word features (v7 → v8)

The generator places sibling businesses ("... Midtown", "... Holding") at a
different house number, while true copies keep the number. For each country,
and over the pairs of the dataset being scored (no labels), each extra name
word gets the share of its pairs whose house numbers conflict, smoothed towards
the country average. Three features follow: max, mean and number of "high"
words. On validation the rate correlates with the true match rate at Spearman
−0.52 ("midtown", "westgate" ≈ 0.8, true match rate 0.0; "dba", "formerly"
≈ 0.0, true match rate > 0.97).

**Unseen-country simulation, both directions** (train on one country, test on the
other's validation entities with one round of word self-training; F0.5 at
threshold 0.675):

| Sibling features | US → India (seed 42 / 7) | India → US (seed 42 / 7) |
|---|---|---|
| none | 0.9296 / 0.9278 | 0.9429 / 0.9452 |
| raw rates (v7) | 0.9430 / 0.9414 | **0.9181 / 0.9197** |
| log(rate / country prior) | 0.9448 | 0.9498 |
| **within-country percentile (v8)** | **0.9435** | **0.9587** |

Raw rates **failed** in one direction: numbers conflict in 10% of India's pairs
but 17% of the US's (priors 0.137 vs 0.248), so a raw rate means different
things in different countries. Turning each rate into its percentile among the
country's extra-word occurrences makes the features comparable, and they gain
+0.014 and +0.016. v7 was stopped before prediction and never submitted; v8
uses percentiles. On US + India validation v8 is neutral (0.9780 vs 0.9782),
as expected. With percentile features the best unseen-country threshold is close
to the normal one (0.55–0.725; 0.85 is worse), so no separate France threshold is
used.

## Searching for the 0.99 signal (all measured on validation, condition B)

Top leaderboard teams score ≈ 0.99, which needs ≈ 0.99 on US/India, where we
have 0.978. Hypotheses tested, all with training labels only:

| Hypothesis | Finding |
|---|---|
| Row order or IDs reveal groups | No: copies are spread randomly through the files (row/ID correlation 0.000); IDs are random 9-digit numbers |
| Fixed copy quota per business | No: copies per source look random (0–5 in S2, 0–6 in S3); 86% of businesses have no name-only copy |
| Copies in one source share that source's distorted address | Partly: same-source same-address neighbours of confident matches are 94% true if already candidates, but only 3.6–4.3% true if not (recall expansion useless: +3.5k of 44k missed copies for 80–98k wrong candidates). Among uncertain pairs the shared distortion raises the true rate from 25% to 53%, not a clean split |
| Per-entity expected-F0.5 decision rule | No gain: best 0.9779 / 0.9781 vs global threshold 0.9781 / 0.9783 (two validation halves) |
| Raw noise artifacts (PO Box, &lt;NULL&gt;, brackets, "(ID:") | Some signal ("po box" 80% true, extra "floor" 4%), but the model is already calibrated on every token overall |
| Edit types between near-identical names | Legal-form swaps (LLC → Inc) and inserted "!" are decoys (≈ 0–5% true), but the model already predicts exactly those rates |
| Within-country percentiles of the base similarity features | Mixed in the simulation (+0.001 / −0.002): rejected |
| French address parsing ("COUR2" → phantom number) | Affects only 0.25% of French records |

The largest remaining errors are name-only candidates: 616k validation pairs,
only 9.4% true, 24k of our 45k wrong decisions.

More checks after v8:

| Hypothesis | Finding |
|---|---|
| Train and test share records | No: 0 of the test (name, address) strings occur in the training files; 34% of test S1 *names* do, but they are generic names ("Family Medicine") at other addresses |
| Digit patterns relate matched IDs | No: last/first digits, differences modulo 7–1009 and digit sums match random pairs exactly |
| Is the leaderboard gap a US/India train/test shift? | No: per entity, the pair types of the US/India test set match validation with 18% of owners removed, and the model accepts the same number of matches (US 3.36 vs 3.34, India 3.24 vs 3.31). The gap is France |
| France: same number, different street among accepted matches | 7.2% in France vs 3.3–3.8% elsewhere, but the excess is street *typos* (6.4% vs 2.3% in the US; 99.9% true there). Genuinely different streets are rarer in France (0.75% vs 1.0%): a street rule would lose true copies |
| France pair types on test | 5× more "same name, different house number" candidates per entity than the US (0.48 vs 0.09): chains of namesakes in one city, which is what the house-number rule targets |

## Transfer to an unseen country: data size, regularisation, self-training (v8 features)

Unseen-country simulation (train on one country, score the other; F0.5 at 0.675):

| Setting | India → US unseen | India in-domain | US → India unseen | US in-domain |
|---|---|---|---|---|
| baseline (all entities, 127 leaves, min leaf 200) | 0.9587 | 0.9763 | 0.9435 | 0.9778 |
| 10% of training entities | 0.9557 | 0.9711 | 0.9441 | 0.9737 |
| 30% of training entities | 0.9579 | 0.9744 | 0.9457 | 0.9760 |
| 31 leaves | 0.9572 | 0.9765 | 0.9402 | 0.9778 |
| min leaf 2000 | 0.9591 | 0.9763 | 0.9441 | 0.9778 |
| 15 leaves, min leaf 2000 | 0.9578 | 0.9764 | 0.9411 | 0.9777 |
| feature fraction 0.5 | 0.9596 | 0.9766 | 0.9428 | 0.9779 |

No setting is better in both directions. Less data always costs in-domain accuracy and
only helps the unseen country in one direction; smaller trees transfer worse. The model
is not overfitted to the training countries in a way that regularisation fixes.

**Adapting to the unseen country** (continue boosting the trained model on the
country's own confident predictions, p ≥ 0.9 / ≤ 0.1, cross-fitted by entity halves):

| Extra trees (learning rate) | India → US | US → India |
|---|---|---|
| none (v8) | 0.9587 | 0.9435 |
| +50 (0.05) | 0.9618 | 0.9441 |
| **+100 (0.05)** | **0.9621** | **0.9441** |
| +200 (0.05) | 0.9622 | 0.9443 |
| +100 (0.1) | 0.9622 | 0.9441 |
| full retraining with pseudo-labels (reference, ~2 h on the full data) | 0.9622 | 0.9455 |

Positive in both directions, as good as full retraining for India → US, and it only
adds 100 trees on the unseen country's pairs. Used in `predict.py --unseen-adapt`.

**Seed stability and seed ensembles** (v8 features, US + India validation, 300k-entity
sample): single matchers with seeds 42 / 7 / 123 score 0.9772 / 0.9772 / 0.9772 at the
0.675 threshold, so a validation difference of ≥ 0.0005 is not seed noise. Averaging 2 or
3 seeds gives 0.9775 (+0.0003): not used, since it would add a full final fit for about
+0.00025 on the leaderboard.

## Leaderboard result of v9

v9 (v8 + adaptation to the unseen country + house-number rule) scored **0.969**, the
same rounded score as v6 with the house-number rule. US/India are unchanged on validation
(0.9789 vs 0.9791), so the France-targeted changes, each validated in both directions of
the cross-country simulation (+0.015 and +0.002 there), moved the leaderboard by less
than its 0.001 resolution. The US ↔ India simulation predicts the direction of a change
for a new country, but not its size for France.

## v10: all 100 blocking candidates to the re-ranker

Blocking has always kept the top 100 per S1 entity, but only the top 50 went to the
re-ranker. Giving it all 100 (re-ranker and matcher retrained, same settings; log:
`train_matcher_k100.txt`):

| Validation (300k-entity sample models) | K = 50 (v8) | K = 100 (v10) |
|---|---|---|
| candidates per entity kept by the re-ranker | 6.12 | 6.24 |
| pair recall of the kept candidates | 0.9712 | 0.9774 |
| oracle F0.5 (perfect matcher on the candidates) | 0.9909 | 0.9930 |
| **macro F0.5** | 0.9780 | **0.9797** (US 0.9785 → 0.9806, India 0.9771 → 0.9783) |
| all words unknown ("new language") | 0.9775 | 0.9793 |
| all words unknown + self-training | 0.9778 | 0.9795 |

+0.0017, 17 times the seed noise, in both countries and in the unknown-language proxy.
The validation threshold curve is flat from 0.675 to 0.775 (0.9797); the competition
step was not rerun for K = 100 (it takes ~1.5 h at K = 100 and the deadline did not allow
it). For K = 50 it preferred 0.65–0.675, so the submission uses 0.675.

Also measured and rejected on the way (validation halves, condition B):
- isotonic calibration + per-entity expected-F0.5 decoding: 0.9776 / 0.9780 vs 0.9781 / 0.9783
  for the global threshold;
- per-country thresholds (US 0.675, India 0.60): +0.00005;
- S2–S3 consistency (exact-copy records given to different S1 entities): 1 group in 5.67M,
  no change.

## Leaderboard: v10 = 0.975 (+0.006) — the test set needs deeper candidate lists

v10 (all 100 blocking candidates to the re-ranker) scored **0.975**, up from 0.969,
four times what validation predicted (+0.0017 × 0.85 ≈ +0.0015 for US/India).
Explanation: the test set has 24% more S2/S3 records per S1 entity than the training
data (5.8 vs 4.67), so more decoys compete in the blocking ranking and true copies sit
deeper in the test than in validation. The K = 50 cut therefore cost much more on
the test than on validation — the "train/test shift".

Where the accepted v10 test matches come from (blocking rank, share of accepted matches):

| Country | 0–9 | 10–24 | 25–49 | 50–74 | 75–89 | 90–99 | per entity from ranks 90–99 |
|---|---|---|---|---|---|---|---|
| US | 95.06% | 3.61% | 0.84% | 0.30% | 0.12% | 0.06% | 0.002 |
| India | 95.40% | 3.06% | 0.97% | 0.36% | 0.14% | 0.07% | 0.002 |
| France | 89.15% | 5.50% | 2.97% | **1.46%** | **0.59%** | **0.32%** | **0.010** |

For US/India the tail has died out by rank 99. For France (generic names such as
"Nantes Club SAS" create long namesake chains) it is still heavy, so v11 re-blocks the
S1 records of countries without training labels to depth 200 (`predict.py --unseen-k
200`; the first 100 candidates are the same for 97.6% of French entities, the rest
differ only by ties at the boundary).

## Final leaderboard results

| Version | Change | Leaderboard |
|---|---|---|
| v10 | all 100 blocking candidates to the re-ranker | 0.975 |
| v11 | + depth 200 for countries without training labels (France) | 0.975 |
| **v13 (final)** | **+ depth 200 for India** (`rescore_unseen --countries India`) | **0.976** |

Deeper lists help on the test set mainly through owner competition: going from 50 to
100 candidates changed 4.3–4.5% of US/India entities (+32 matches and −13 matches per
1,000 entities: a copy that reaches its true owner's list stops being a false match of a
namesake). France's deeper lists added 1.5% more matches but moved the score by less
than the leaderboard's 0.001 resolution. With more time, the next step is depth 200 for
the US as well, then reverse search (each S2/S3 record looks up its best S1 entity).

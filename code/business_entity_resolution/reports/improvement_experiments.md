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

## Label-free sibling-word features (v7)

The generator places sibling businesses ("... Midtown", "... Holding") at a
different house number, while true copies keep the number. For each country,
and over the pairs of the dataset being scored (no labels), each extra name
word gets the share of its pairs whose house numbers conflict, smoothed towards
the country average. Three features follow: max, mean and number of words
above 0.5. On validation the rate correlates with the true match rate at
Spearman −0.52. "midtown" and "westgate" come out at about 0.8 (true match rate
0.0), and "dba" and "formerly" at about 0.0 (true match rate > 0.97). The rates
are computed from the test pairs themselves, so they also cover French words.

| Setup | F0.5 |
|---|---|
| US only → India (unseen), without | 0.9338 (threshold 0.45) |
| US only → India (unseen), **with** sibling features | **0.9483** (threshold 0.875) |
| with sibling features, threshold 0.675 / 0.85 / 0.90 | 0.9430 / 0.9479 / 0.9486 |
| US + India → validation, without / with | 0.9775 / 0.9772 (neutral) |

This is the largest unseen-country gain measured (+0.015 in the simulation).
France is 15% of the test S1 entities (India 46.8%, US 38.3%), so it would be
worth about +0.002 on the leaderboard if it transfers. In the unseen country the best threshold is higher
(0.85–0.9), which motivates candidate A (`--unseen-threshold 0.85`).

# Candidate-set size experiments

Goal: pass as few candidates per S1 entity as possible to the matching model
(the final candidate set, `candidate_pairs.tsv`) while keeping macro F0.5.
All numbers are on the validation split (441,364 S1 entities).
"Model F0.5" uses the first matcher (trained on the top-50 blocking
candidates), restricted to the kept candidates, with the threshold re-tuned.

## 1. Plain cut of the TF-IDF blocking list

| rule | cands/entity | pair recall | oracle F0.5 | model F0.5 |
|---|---|---|---|---|
| top 50 | 50.00 | 0.9645 | 0.9874 | 0.9556 |
| top 30 | 30.00 | 0.9547 | 0.9839 | 0.9530 |
| top 20 | 20.00 | 0.9428 | 0.9797 | 0.9497 |
| top 10 | 10.00 | 0.9079 | 0.9678 | 0.9399 |
| top 8 | 8.00 | 0.8894 | 0.9620 | 0.9349 |
| top 5 | 5.00 | 0.8020 | 0.9391 | 0.9145 |
| top 20, score >= 0.4 x best | 9.06 | 0.8967 | 0.9668 | 0.9396 |
| top 20, score >= 0.5 x best | 6.67 | 0.8425 | 0.9503 | 0.9252 |

A score-relative cut is no better than a fixed top-K at the same size: the
blocking score alone ranks too many true matches outside the top few.

## 2. Cheap re-ranker as a second blocking stage (`src/rerank.py`)

A small LightGBM (31 leaves, 220 trees, 11 cheap features) re-orders the 50
blocking candidates. Keep at most `KEEP_MAX` with re-rank probability
>= `MIN_PROB`.

| rule | cands/entity | median | p95 | empty % | pair recall | oracle F0.5 | model F0.5 |
|---|---|---|---|---|---|---|---|
| top 3 | 3.00 | 3 | 3 | 0.00 | 0.7084 | 0.9322 | 0.9141 |
| top 5 | 5.00 | 5 | 5 | 0.00 | 0.9046 | 0.9774 | 0.9510 |
| top 6 | 6.00 | 6 | 6 | 0.00 | 0.9379 | 0.9828 | 0.9544 |
| top 8 | 8.00 | 8 | 8 | 0.00 | 0.9577 | 0.9858 | 0.9556 |
| top 10 | 10.00 | 10 | 10 | 0.00 | 0.9614 | 0.9865 | 0.9556 |
| top 6, p >= 0.05 | 4.41 | 5 | 6 | 0.67 | 0.9295 | 0.9792 | 0.9538 |
| top 8, p >= 0.01 | 5.88 | 6 | 8 | 0.04 | 0.9552 | 0.9849 | 0.9555 |
| top 8, p >= 0.02 | 5.39 | 5 | 8 | 0.11 | 0.9526 | 0.9840 | 0.9554 |
| **top 8, p >= 0.05 (chosen)** | **4.66** | **5** | **8** | **0.67** | **0.9449** | **0.9813** | **0.9550** |
| top 10, p >= 0.05 | 4.72 | 5 | 8 | 0.67 | 0.9463 | 0.9814 | 0.9550 |

Re-ranker feature importance (share of gain): blocking score 0.45, name
token-set 0.30, blocking score relative to best 0.23, others < 0.02.

**Chosen:** `KEEP_MAX = 8`, `MIN_PROB = 0.05`. That is 10.7x fewer candidates
than the top-50 list (4.66 vs 50 per entity) for about 0.0006 F0.5. The matching
model is then retrained on exactly these candidates.

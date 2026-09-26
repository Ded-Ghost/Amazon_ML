# Blocking experiments (Phase 2)

All numbers: 20,000 validation S1 entities per country (India, US), searched
against the full train S2+S3 pool of the same country (4.1M India, 6.2M US).

* **pair recall**: share of true (S1, match) pairs among the candidates
* **oracle F0.5**: macro F0.5 of a perfect matcher restricted to the candidates
  (the ceiling for the matching model)

## 1. Words only (name words, name skeletons, address words), summed IDF

| max_df | K=10 | K=50 | K=100 (pair recall, India subset) |
|---|---|---|---|
| 1000 | 0.616 | 0.704 | 0.743 |
| 5000 | 0.750 | 0.811 | 0.831 |

Diagnosis: many names consist of individually common words ("Vision First
Care"); after dropping common keys only a street name is left, shared by
hundreds of records. Combinations are rare even if the words are not.

## 2. + name skeleton pairs, glued-name keys, address bigrams (max_df 2000, summed IDF)

| K | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| pair recall (ALL) | 0.888 | 0.919 | 0.942 | 0.954 | 0.964 |

## 3. Separate passes vs one combined pass, cosine vs summed IDF

Union of the name pass (K_name), address pass (K_addr) and combined pass (K_all).

| max_df | K_name | K_addr | K_all | pair recall | oracle F0.5 | cands/entity |
|---|---|---|---|---|---|---|
| 2000 | 0 | 0 | 50 | 0.9580 | 0.9845 | 50.0 |
| 2000 | 25 | 25 | 0 | 0.9387 | 0.9777 | 47.6 |
| 2000 | 20 | 20 | 40 | 0.9605 | 0.9856 | 56.1 |
| 5000 | 0 | 0 | 10 | 0.9074 | 0.9668 | 10.0 |
| **5000** | **0** | **0** | **50** | **0.9627** | **0.9861** | **50.0** |
| 5000 | 0 | 0 | 100 | 0.9720 | 0.9898 | 100.0 |
| 5000 | 20 | 20 | 40 | 0.9646 | 0.9872 | 56.2 |
| 5000 | 50 | 50 | 100 | 0.9755 | 0.9910 | 139.5 |

Cosine normalisation of the pool side (vs plain summed IDF) added about
+0.5 to +1.0 point of recall at every K. The name-only pass is weak on its
own (0.55 at K=10); the address-only pass reaches 0.77.

**Chosen:** one combined pass, cosine, max_df = 5000, keep top 100 (the
matching phase picks the final K). Multiple passes gave no better recall per
candidate and cost 3x the time.

## Remaining misses (typical)

* trade names unrelated to the legal name ("Onyxjax", "Kordelta"), matchable
  only through the address
* name-only records (empty address) with heavy typos ("Sicbioccn" vs "Silicon")
* very common names in the same city ("Shree Foods", Pune)

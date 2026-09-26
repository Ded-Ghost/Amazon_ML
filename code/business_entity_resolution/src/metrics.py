"""The competition metric: macro-averaged F0.5 over Source 1 entities.

For one S1 entity with true match set T and predicted set P:

* T empty (singleton):  score 1.0 if P is empty, else 0.0
* T non-empty:          precision = |P & T| / |P|,  recall = |P & T| / |T|
                        F0.5 = 1.25 * p * r / (0.25 * p + r)   (0.0 if no hit)

The final score is the plain average of the per-entity scores over *all*
S1 entities in the evaluation set, singletons included.
"""
import math

BETA2 = 0.5 ** 2  # beta = 0.5 -> precision counts twice as much as recall


def f05_entity(predicted, true):
    predicted, true = set(predicted), set(true)
    if not true:
        return 1.0 if not predicted else 0.0
    tp = len(predicted & true)
    if tp == 0:  # also covers "predicted nothing"
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(true)
    return (1 + BETA2) * precision * recall / (BETA2 * precision + recall)


def macro_f05(predictions, truth):
    """predictions: {s1_id: ids}, truth: {s1_id: ids}.

    The average runs over every entity in `truth`; an entity missing from
    `predictions` counts as an empty prediction.
    """
    scores = [f05_entity(predictions.get(s1, ()), true) for s1, true in truth.items()]
    # fsum keeps the sum exact, so e.g. all-empty predictions give exactly
    # (number of singletons) / (number of entities).
    return math.fsum(scores) / len(scores)


def score_report(predictions, truth):
    """Overall score plus the split between singletons and matched entities."""
    singles = {s1: t for s1, t in truth.items() if not t}
    matched = {s1: t for s1, t in truth.items() if t}
    report = {"n_entities": len(truth), "macro_f05": macro_f05(predictions, truth)}
    if singles:
        report["f05_singletons"] = macro_f05(predictions, singles)
    if matched:
        report["f05_matched"] = macro_f05(predictions, matched)
    return report

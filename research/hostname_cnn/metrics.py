"""Evaluation metrics focused on the low false-positive operating region.

Every function takes ``y`` (1 = phishing) and a score where higher means more phishing.
Bootstrap intervals are percentile intervals over resampled hostnames.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

FPRS = (0.001, 0.0025, 0.005, 0.01)


def recall_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float) -> float:
    """Recall when at most floor(fpr * negatives) negatives score strictly above the cut."""
    negatives = np.sort(score[y == 0])[::-1]
    allowed = math.floor(fpr * len(negatives))
    cut = negatives[allowed] if allowed < len(negatives) else -np.inf
    return float((score[y == 1] > cut).mean())


def partial_auc(y: np.ndarray, score: np.ndarray, max_fpr: float = 0.01) -> float:
    """McClish-standardised partial ROC-AUC over FPR in [0, max_fpr] (0.5 = chance)."""
    return float(roc_auc_score(y, score, max_fpr=max_fpr))


def ece(y: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    index = np.clip(np.digitize(probability, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(probability[mask].mean() - y[mask].mean())
    return float(total)


def ranking(y: np.ndarray, score: np.ndarray) -> dict:
    """Threshold-free metrics; empty when a class is missing."""
    if len(np.unique(y)) < 2:
        return {}
    row = {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "pauc_1pct": partial_auc(y, score),
    }
    for f in FPRS:
        row[f"recall_at_{f:g}_fpr"] = recall_at_fpr(y, score, f)
    return row


def operating(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    flagged = probability >= threshold
    row = {"n": int(len(y)), "phishing": int(y.sum()), "legitimate": int((y == 0).sum())}
    if row["phishing"]:
        row["recall_at_threshold"] = float(flagged[y == 1].mean())
    if row["legitimate"]:
        row["fpr_at_threshold"] = float(flagged[y == 0].mean())
    row["brier"] = float(np.mean((probability - y) ** 2))
    row["ece"] = ece(y, probability)
    return row


def evaluate(y, score, probability, threshold, *, bootstrap: int = 0, seed: int = 0) -> dict:
    """Full metric row; with ``bootstrap`` > 0, 95% intervals for the ranking metrics."""
    y, score, probability = np.asarray(y), np.asarray(score), np.asarray(probability)
    row = operating(y, probability, threshold) | ranking(y, score)
    if bootstrap and len(np.unique(y)) == 2:
        rng = np.random.default_rng(seed)
        draws: dict[str, list[float]] = {}
        for _ in range(bootstrap):
            i = rng.integers(0, len(y), len(y))
            for key, value in ranking(y[i], score[i]).items():
                draws.setdefault(key, []).append(value)
        for key, values in draws.items():
            row[key + "_ci95"] = [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]
    return row


def paired_difference(
    y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, *, bootstrap: int = 1000, seed: int = 0
) -> dict:
    """Bootstrap of metric(b) - metric(a) on the same hostnames (b is the candidate)."""
    y = np.asarray(y)
    base_a, base_b = ranking(y, score_a), ranking(y, score_b)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(bootstrap):
        i = rng.integers(0, len(y), len(y))
        a, b = ranking(y[i], score_a[i]), ranking(y[i], score_b[i])
        for key in a:
            draws.setdefault(key, []).append(b[key] - a[key])
    return {
        key: {
            "difference": base_b[key] - base_a[key],
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "p_candidate_not_better": float(np.mean(np.asarray(v) <= 0)),
        }
        for key, v in draws.items()
    }


def subgroup_table(y, score, probability, threshold, groups: np.ndarray, minimum: int = 50) -> dict:
    """Metrics per subgroup value (groups smaller than ``minimum`` are skipped)."""
    table = {}
    for value in np.unique(groups):
        mask = groups == value
        if mask.sum() >= minimum:
            table[str(value)] = evaluate(y[mask], score[mask], probability[mask], threshold)
    return table

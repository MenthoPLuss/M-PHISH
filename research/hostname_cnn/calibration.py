"""H10: compare calibrators and measure how stable the production threshold is.

Post-hoc and training-free: reads the saved calibration and validation logits of finished runs.
Every calibrator is fitted on the calibration split and judged on validation.

    python -m research.hostname_cnn.calibration --name c13 \
        --runs C13-v3wide-h1v2-random-s42 C13-v3wide-h1v2-random-s43 C13-v3wide-h1v2-random-s44

Scorers are each run alone and, with several runs, their mean of logits. The report goes to
``runs/research/comparisons/H10_calibration__<name>.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from backend.app.services.ml.train_url_cnn import calibrate, select_threshold

from .common import RESEARCH_RUNS, describe, load_dataset, stem_unseen_mask
from .compare import aligned, load
from .metrics import ece

EPS = 1e-6  # probabilities are clipped to [EPS, 1 - EPS] for log loss, for every calibrator
GROUP_MINIMUM = 200


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def log_losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def adaptive_ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """ECE over equal-count bins (sorted by probability), so sparse regions are not ignored."""
    order = np.argsort(p, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            total += len(chunk) / len(p) * abs(p[chunk].mean() - y[chunk].mean())
    return float(total)


def fit_calibrators(z: np.ndarray, y: np.ndarray) -> dict:
    """Fit each calibrator on (z, y); return name -> (parameters, function of logits)."""
    platt = calibrate(z, y)
    a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])

    def nll(log_t):
        return float(log_losses(y, sigmoid(z / np.exp(log_t))).mean())

    t = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(z, y)
    return {
        "none": ({}, sigmoid),
        "temperature": ({"temperature": t}, lambda v: sigmoid(v / t)),
        "platt": ({"slope": a, "intercept": b}, lambda v: sigmoid(a * v + b)),
        "isotonic": (
            {"steps": int(len(iso.X_thresholds_))},
            lambda v: iso.predict(v).astype(np.float64),
        ),
    }


def quality(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "log_loss": float(log_losses(y, p).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece(y, p),
        "adaptive_ece": adaptive_ece(y, p),
        "share_at_clip": float(np.mean((p <= EPS) | (p >= 1 - EPS))),
    }


def paired_loss_difference(y, p_base, p_other, bootstrap: int, seed: int) -> dict:
    """Bootstrap of mean loss(other) - mean loss(base) on the same hostnames (negative = better)."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, per_host in (
        ("log_loss", lambda p: log_losses(y, p)),
        ("brier", lambda p: (p - y) ** 2),
    ):
        delta = per_host(p_other) - per_host(p_base)
        draws = [delta[rng.integers(0, len(y), len(y))].mean() for _ in range(bootstrap)]
        out[name] = {
            "difference": float(delta.mean()),
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        }
    return out


def operating_point(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    flagged = p >= threshold
    return {
        "fpr": float(flagged[y == 0].mean()),
        "recall": float(flagged[y == 1].mean()),
        "false_alarms": int(flagged[y == 0].sum()),
        "legitimate": int((y == 0).sum()),
    }


def threshold_stability(y, p, threshold, y_cal, p_cal, bootstrap: int, seed: int) -> dict:
    """How far the production threshold moves with resampling, and what FPR it really gives.

    Each resample re-runs the threshold rule; the out-of-bag hostnames (about 37%) then give an
    estimate of the FPR and recall that choosing a threshold this way delivers on unseen data.
    """
    rng = np.random.default_rng(seed)
    point = operating_point(y, p, threshold)
    thresholds, oob_fpr, oob_recall, in_bag_fpr = [], [], [], []
    for _ in range(bootstrap):
        i = rng.integers(0, len(y), len(y))
        bag = np.zeros(len(y), bool)
        bag[i] = True
        t = select_threshold(y[i], p[i])
        thresholds.append(t)
        in_bag_fpr.append(float((p[i][y[i] == 0] >= t).mean()))
        out = operating_point(y[~bag], p[~bag], t)
        oob_fpr.append(out["fpr"])
        oob_recall.append(out["recall"])

    def interval(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    # Sampling uncertainty of the point threshold's FPR, with the threshold held fixed.
    fixed_fpr = []
    rng = np.random.default_rng(seed + 1)
    negatives = p[y == 0] >= threshold
    for _ in range(bootstrap):
        fixed_fpr.append(float(negatives[rng.integers(0, len(negatives), len(negatives))].mean()))
    return {
        "threshold": float(threshold),
        "validation": point,
        "fixed_threshold_fpr_ci95": interval(fixed_fpr),
        "reselected_threshold_ci95": interval(thresholds),
        "reselected_in_bag_fpr_mean": float(np.mean(in_bag_fpr)),
        "reselected_out_of_bag_fpr_mean": float(np.mean(oob_fpr)),
        "reselected_out_of_bag_fpr_ci95": interval(oob_fpr),
        "reselected_out_of_bag_recall_ci95": interval(oob_recall),
        "share_out_of_bag_fpr_above_1pct": float(np.mean(np.array(oob_fpr) > 0.01)),
        "calibration_split": operating_point(y_cal, p_cal, threshold),
    }


def site_clustering(y, p, threshold, sites: np.ndarray, bootstrap: int, seed: int) -> dict:
    """False alarms per registered domain, and an FPR interval that resamples whole sites.

    Splits are assigned by registered domain, so one site with many user subdomains lands in a
    single split and its hostnames err together. Resampling hostnames then understates the
    uncertainty; resampling registered domains does not.
    """
    negative = y == 0
    flagged = (p[negative] >= threshold).astype(float)
    domains, inverse = np.unique(sites[negative], return_inverse=True)
    count = np.bincount(inverse)
    hits = np.bincount(inverse, weights=flagged)
    rng = np.random.default_rng(seed + 2)
    draws = []
    for _ in range(bootstrap):
        k = rng.integers(0, len(domains), len(domains))
        draws.append(hits[k].sum() / count[k].sum())
    order = np.argsort(-hits, kind="stable")[:10]
    top = int(order[0])
    return {
        "legitimate_registered_domains": int(len(domains)),
        "false_alarms": int(flagged.sum()),
        "domain_bootstrap_fpr_ci95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "top_false_alarm_domains": [
            {"domain": str(domains[i]), "false_alarms": int(hits[i]), "legitimate": int(count[i])}
            for i in order
            if hits[i]
        ],
        "fpr_without_top_domain": float((flagged.sum() - hits[top]) / (count.sum() - count[top])),
    }


def group_table(y, p, threshold, groups: dict[str, np.ndarray]) -> dict:
    table = {}
    for dimension, labels in groups.items():
        rows = {}
        for value in np.unique(labels):
            mask = labels == value
            if mask.sum() < GROUP_MINIMUM:
                continue
            yy, pp = y[mask], p[mask]
            row = {
                "n": int(mask.sum()),
                "observed_phishing_rate": float(yy.mean()),
                "mean_probability": float(pp.mean()),
                "ece": ece(yy, pp),
            }
            if (yy == 0).sum():
                row["fpr_at_threshold"] = float((pp[yy == 0] >= threshold).mean())
            if (yy == 1).sum():
                row["recall_at_threshold"] = float((pp[yy == 1] >= threshold).mean())
            rows[str(value)] = row
        table[dimension] = rows
    return table


def structural_groups(hosts: list[str], record: dict) -> dict[str, np.ndarray]:
    groups = {
        dim: np.array([describe(h)[dim] for h in hosts])
        for dim in ("shape", "platform", "length", "labels", "tld")
    }
    data = load_dataset(Path(record["data"]["dataset"]), Path(record["data"]["splits"]))
    position = {h: i for i, h in enumerate(data.hosts)}
    subset = np.array([position[h] for h in hosts])
    groups["source"] = data.source[subset]
    groups["stem"] = np.where(stem_unseen_mask(data, subset), "stem unseen", "stem seen")
    return groups


def evaluate_scorer(name, z_cal, y_cal, z_val, y_val, groups, sites, bootstrap, seed) -> dict:
    report = {"n_calibration": int(len(y_cal)), "n_validation": int(len(y_val)), "methods": {}}
    probabilities = {}
    for method, (params, f) in fit_calibrators(z_cal, y_cal).items():
        p_val, p_cal = f(z_val), f(z_cal)
        probabilities[method] = (p_val, p_cal)
        threshold = select_threshold(y_val, p_val)
        report["methods"][method] = {
            "parameters": params,
            "validation": quality(y_val, p_val),
            "calibration_in_sample": quality(y_cal, p_cal),
            "threshold": float(threshold),
            "operating_point": operating_point(y_val, p_val, threshold),
        }
    base = probabilities["platt"][0]
    for method, (p_val, _) in probabilities.items():
        if method != "platt":
            report["methods"][method]["vs_platt"] = paired_loss_difference(
                y_val, base, p_val, bootstrap, seed
            )
    for method in ("platt", "isotonic"):
        p_val, p_cal = probabilities[method]
        threshold = report["methods"][method]["threshold"]
        report["methods"][method]["stability"] = threshold_stability(
            y_val, p_val, threshold, y_cal, p_cal, bootstrap, seed
        )
        report["methods"][method]["sites"] = site_clustering(
            y_val, p_val, threshold, sites, bootstrap, seed
        )
        report["methods"][method]["groups"] = group_table(y_val, p_val, threshold, groups)
    print_scorer(name, report)
    return report


def print_scorer(name: str, report: dict) -> None:
    print(f"\n{name}: calibration {report['n_calibration']}, validation {report['n_validation']}")
    print(
        f"  {'method':12s} {'log loss':>9s} {'Brier':>8s} {'ECE':>7s} {'aECE':>7s}  vs Platt log loss"
    )
    for method, row in report["methods"].items():
        q = row["validation"]
        vs = row.get("vs_platt", {}).get("log_loss")
        delta = (
            f"{vs['difference']:+.5f} [{vs['ci95'][0]:+.5f}, {vs['ci95'][1]:+.5f}]" if vs else ""
        )
        print(
            f"  {method:12s} {q['log_loss']:9.5f} {q['brier']:8.5f} {q['ece']:7.4f} "
            f"{q['adaptive_ece']:7.4f}  {delta}"
        )
    for method in ("platt", "isotonic"):
        s = report["methods"][method]["stability"]
        print(
            f"  {method} threshold {s['threshold']:.4f}: validation FPR "
            f"{s['validation']['fpr'] * 100:.2f}% (fixed-threshold 95% "
            f"[{s['fixed_threshold_fpr_ci95'][0] * 100:.2f}, {s['fixed_threshold_fpr_ci95'][1] * 100:.2f}]), "
            f"recall {s['validation']['recall'] * 100:.1f}%; calibration-split FPR "
            f"{s['calibration_split']['fpr'] * 100:.2f}%, recall {s['calibration_split']['recall'] * 100:.1f}%"
        )
        print(
            f"    re-selected threshold 95% [{s['reselected_threshold_ci95'][0]:.4f}, "
            f"{s['reselected_threshold_ci95'][1]:.4f}]; out-of-bag FPR mean "
            f"{s['reselected_out_of_bag_fpr_mean'] * 100:.2f}% "
            f"[{s['reselected_out_of_bag_fpr_ci95'][0] * 100:.2f}, "
            f"{s['reselected_out_of_bag_fpr_ci95'][1] * 100:.2f}], above 1% in "
            f"{s['share_out_of_bag_fpr_above_1pct'] * 100:.0f}% of resamples"
        )
        c = report["methods"][method]["sites"]
        top = c["top_false_alarm_domains"][0]
        print(
            f"    by site: FPR 95% [{c['domain_bootstrap_fpr_ci95'][0] * 100:.2f}, "
            f"{c['domain_bootstrap_fpr_ci95'][1] * 100:.2f}] resampling "
            f"{c['legitimate_registered_domains']} registered domains; {top['domain']} gives "
            f"{top['false_alarms']} of {c['false_alarms']} false alarms; FPR without it "
            f"{c['fpr_without_top_domain'] * 100:.2f}%"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    loaded = [load(r) for r in args.runs]
    hosts, y_val, Z = aligned(loaded)
    c_hosts, y_cal, C = aligned(loaded, "calibration")
    groups = structural_groups(hosts, loaded[0]["record"])
    sites = np.array([describe(h)["registrable"] for h in hosts])
    scorers = {run: (C[:, i], Z[:, i]) for i, run in enumerate(args.runs)}
    if len(args.runs) > 1:
        scorers[f"mean of {len(args.runs)}"] = (C.mean(1), Z.mean(1))
    report = {
        "hypothesis": "H10",
        "runs": args.runs,
        "fitted_on": "calibration split",
        "judged_on": "validation split",
        "log_loss_clip": EPS,
        "bootstrap": args.bootstrap,
        "test_split_scored": False,
        "scorers": {
            name: evaluate_scorer(
                name, zc, y_cal, zv, y_val, groups, sites, args.bootstrap, args.seed
            )
            for name, (zc, zv) in scorers.items()
        },
    }
    out = RESEARCH_RUNS / "comparisons" / f"H10_calibration__{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nSaved {out}")
    return report


if __name__ == "__main__":
    main()

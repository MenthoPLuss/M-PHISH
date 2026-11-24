"""Paired comparisons, ensembles and stacking over research runs (validation only).

    python -m research.hostname_cnn.compare diff --control P0.5-control-random-s42 --candidates E3a
    python -m research.hostname_cnn.compare combine --name ens-a --runs A B C [--stack]

Runs are aligned on the validation hostnames they share; every reported number is computed
from saved scores, never typed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .common import RESEARCH_RUNS, describe
from .metrics import evaluate, paired_difference, ranking

KEYS = ("roc_auc", "pauc_1pct", "recall_at_0.001_fpr", "recall_at_0.005_fpr", "recall_at_0.01_fpr")


def load(run: str) -> dict:
    """A run's saved scores. ``RUN@N`` uses the validation logits of epoch N instead of the
    chosen checkpoint (from ``validation_logits_by_epoch.npy``; no calibration scores)."""
    run, _, epoch = run.partition("@")
    folder = RESEARCH_RUNS / run
    record = json.loads((folder / "experiment.json").read_text())
    scores = np.load(folder / "scores.npz", allow_pickle=False)
    out = {"record": record}
    for split in ("validation", "calibration"):
        hosts = [str(h) for h in scores[f"{split}_hosts"]]
        out[split] = {
            "z": dict(zip(hosts, scores[f"{split}_logit"].tolist(), strict=True)),
            "y": dict(zip(hosts, scores[f"{split}_y"].tolist(), strict=True)),
        }
    if epoch:
        by_epoch = np.load(folder / "validation_logits_by_epoch.npy")
        hosts = [str(h) for h in scores["validation_hosts"]]  # same order as the trainer's
        out["validation"]["z"] = dict(
            zip(hosts, by_epoch[int(epoch) - 1].astype(np.float64).tolist(), strict=True)
        )
        out["calibration"] = {"z": {}, "y": {}}
    return out


GROUPS = {
    "hosting platform": lambda h: describe(h)["platform"] == "hosting platform",
    "longer than 30": lambda h: len(h) > 30,
    "4+ labels": lambda h: h.count(".") >= 3,
}


def with_extra(run: str, extra: str) -> tuple[dict, set]:
    """``load(run)`` plus its ``score --only-new`` scores on dataset ``extra``, if any."""
    data = load(run)
    path = RESEARCH_RUNS / run.partition("@")[0] / f"extra_scores__{extra}__validation.npz"
    if not path.exists():
        return data, set()
    saved = np.load(path)
    hosts = [str(h) for h in saved["hosts"]]
    data["validation"]["z"] |= dict(zip(hosts, saved["logit"].tolist(), strict=True))
    data["validation"]["y"] |= dict(zip(hosts, saved["y"].tolist(), strict=True))
    return data, set(hosts)


def fixed_recall(control: str, candidate: str, extra: str, recalls, bootstrap: int) -> dict:
    """FPR at a fixed recall, on the original validation and on hostnames dataset ``extra`` adds.

    For H1, whose dataset adds legitimate validation hostnames. A run trained without them is
    scored on them with ``score --only-new``. Each model's threshold is set to give ``recall``
    on the original validation phishing; false alarms are counted on the original and on the
    added legitimate hostnames, overall and by structural group; recall is reported by group
    too. The paired bootstrap resamples the phishing (so thresholds move) and each group.
    """
    (base, base_added), (cand, cand_added) = (
        with_extra(control, extra),
        with_extra(candidate, extra),
    )
    base_z = base["validation"]["z"]
    common = set(base_z) & set(cand["validation"]["z"])
    added = sorted(common & (base_added | cand_added))
    shared = sorted(common - set(added))
    y = {h: cand["validation"]["y"][h] for h in shared + added}
    positives = [h for h in shared if y[h] == 1]
    sets = {"shared legitimate": [h for h in shared if y[h] == 0]}
    sets["added legitimate"] = [h for h in added if y[h] == 0]
    for name, test in GROUPS.items():
        sets[f"added legitimate, {name}"] = [h for h in sets["added legitimate"] if test(h)]
        sets[f"shared legitimate, {name}"] = [h for h in sets["shared legitimate"] if test(h)]
        sets[f"shared phishing, {name}"] = [h for h in positives if test(h)]
    scores = {
        key: (
            np.array([base_z[h] for h in hosts]),
            np.array([cand["validation"]["z"][h] for h in hosts]),
        )
        for key, hosts in (sets | {"positives": positives}).items()
    }
    pa, pb = scores.pop("positives")
    rng = np.random.default_rng(0)
    report = {
        "control": control,
        "candidate": candidate,
        "added_hostnames": len(added),
        "added_phishing": int(sum(y[h] for h in added)),
        "shared_phishing": len(positives),
        "recalls": {},
    }
    print(
        f"\n{candidate} vs {control}: {len(positives)} shared phishing, "
        f"{len(sets['shared legitimate'])} shared and {len(sets['added legitimate'])} added "
        f"legitimate ({report['added_phishing']} added phishing ignored)"
    )
    for recall in recalls:
        ta, tb = np.quantile(pa, 1 - recall), np.quantile(pb, 1 - recall)
        idx_pos = rng.integers(0, len(pa), (bootstrap, len(pa)))
        boot_ta = np.quantile(pa[idx_pos], 1 - recall, axis=1)
        boot_tb = np.quantile(pb[idx_pos], 1 - recall, axis=1)
        rows = {}
        print(f"  threshold at recall {recall:.0%} on shared phishing")
        for key, (za, zb) in scores.items():
            if not len(za):
                continue
            is_phish = key.startswith("shared phishing")
            a, b = float((za > ta).mean()), float((zb > tb).mean())
            idx = rng.integers(0, len(za), (bootstrap, len(za)))
            da = (za[idx] > boot_ta[:, None]).mean(1)
            db = (zb[idx] > boot_tb[:, None]).mean(1)
            low, high = np.percentile(db - da, [2.5, 97.5])
            rows[key] = {
                "n": len(za),
                "rate": "recall" if is_phish else "false positive rate",
                "control": a,
                "candidate": b,
                "difference": b - a,
                "ci95": [float(low), float(high)],
            }
            print(
                f"    {key:42s} n={len(za):6d}  {'recall' if is_phish else 'FPR':6s} "
                f"{a * 100:6.2f}% -> {b * 100:6.2f}%  diff {(b - a) * 100:+6.2f} pts "
                f"[{low * 100:+.2f}, {high * 100:+.2f}]"
            )
        report["recalls"][str(recall)] = rows
    return report


def aligned(runs: list[dict], split: str = "validation"):
    hosts = sorted(set.intersection(*(set(r[split]["z"]) for r in runs)))
    y = np.array([runs[0][split]["y"][h] for h in hosts])
    Z = np.stack([[r[split]["z"][h] for h in hosts] for r in runs], axis=1)
    return hosts, y, Z


def fmt(value: float, key: str) -> str:
    return f"{value * 100:+.2f} pts" if key.startswith("recall") else f"{value:+.4f}"


def diff(control: str, candidates: list[str], bootstrap: int) -> dict:
    base = load(control)
    report = {"control": control, "candidates": {}}
    for name in candidates:
        cand = load(name)
        hosts, y, Z = aligned([base, cand])
        result = {"common_validation_hostnames": len(hosts)}
        result["overall"] = paired_difference(y, Z[:, 0], Z[:, 1], bootstrap=bootstrap)
        sub = {}
        for dim in ("shape", "platform", "tld", "length"):
            values = np.array([describe(h)[dim] for h in hosts])
            for v in np.unique(values):
                m = values == v
                if m.sum() >= 300 and len(np.unique(y[m])) == 2:
                    a, b = ranking(y[m], Z[m, 0]), ranking(y[m], Z[m, 1])
                    sub[f"{dim}={v}"] = {k: b[k] - a[k] for k in ("roc_auc", "recall_at_0.01_fpr")}
        result["subgroups"] = sub
        report["candidates"][name] = result
        print(f"\n{name} vs {control} on {len(hosts)} shared validation hostnames")
        for key in KEYS:
            d = result["overall"][key]
            print(
                f"  {key:22s} {fmt(d['difference'], key):>12s}  95% CI "
                f"[{fmt(d['ci95'][0], key)}, {fmt(d['ci95'][1], key)}]  "
                f"P(not better)={d['p_candidate_not_better']:.3f}"
            )
    return report


def combine(name: str, runs: list[str], stack: bool, save: bool = False) -> dict:
    """Mean of logits, or logistic-regression stacking fitted on the calibration split.

    With ``save`` the combination becomes a run ``X-<name>`` that ``diff`` can compare: its
    validation scores, and its calibration scores (in-sample for a stack, so not for
    re-stacking).
    """
    loaded = [load(r) for r in runs]
    hosts, y, Z = aligned(loaded)
    c_hosts, c_y, C = aligned(loaded, "calibration")
    if stack:
        mean, std = C.mean(0), C.std(0) + 1e-9
        model = LogisticRegression(C=1e6, max_iter=2000).fit((C - mean) / std, c_y)
        score = model.decision_function((Z - mean) / std)
        c_score = model.decision_function((C - mean) / std)
        weights = (model.coef_[0] / std).tolist()
    else:
        score, weights = Z.mean(1), [1 / len(runs)] * len(runs)
        c_score = C.mean(1)
    if save:
        folder = RESEARCH_RUNS / f"X-{name}"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "scores.npz",
            validation_hosts=np.array(hosts),
            validation_y=y,
            validation_logit=score,
            calibration_hosts=np.array(c_hosts),
            calibration_y=c_y,
            calibration_logit=c_score,
        )
        (folder / "experiment.json").write_text(
            json.dumps(
                {
                    "experiment_id": folder.name,
                    "kind": "combination",
                    "method": "stack (fitted on calibration)" if stack else "mean of logits",
                    "runs": runs,
                    "weights": weights,
                    "calibration_scores_in_sample": stack,
                    "test_split_scored": False,
                },
                indent=1,
            )
        )
    row = evaluate(y, score, 1 / (1 + np.exp(-score)), 0.5, bootstrap=200)
    singles = {r: ranking(y, Z[:, i]) for i, r in enumerate(runs)}
    print(
        f"\n{name}: {'stacked' if stack else 'mean of logits'} over {runs}, {len(hosts)} hostnames"
    )
    for key in KEYS:
        best = max(s[key] for s in singles.values())
        print(f"  {key:22s} {row[key]:.4f}  (best single {best:.4f})")
    corr = np.corrcoef(Z.T).tolist() if len(runs) > 1 else [[1.0]]
    return {
        "name": name,
        "runs": runs,
        "stack": stack,
        "weights": weights,
        "metrics": row,
        "singles": singles,
        "logit_correlation": corr,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare research runs on validation.")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("diff")
    d.add_argument("--control", required=True)
    d.add_argument("--candidates", nargs="+", required=True)
    d.add_argument("--bootstrap", type=int, default=500)
    c = sub.add_parser("combine")
    c.add_argument("--name", required=True)
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--stack", action="store_true")
    c.add_argument("--save", action="store_true", help="save as run X-<name> for diff")
    f = sub.add_parser("fixed-recall")
    f.add_argument("--control", required=True)
    f.add_argument("--candidate", required=True)
    f.add_argument("--extra", required=True, help="dataset folder name used by score.py")
    f.add_argument("--recall", type=float, nargs="+", default=[0.5, 0.59])
    f.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    out = RESEARCH_RUNS / "comparisons"
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "diff":
        report = diff(args.control, args.candidates, args.bootstrap)
        target = out / f"diff__{args.control}__{'+'.join(args.candidates)}.json"
    elif args.command == "fixed-recall":
        report = fixed_recall(args.control, args.candidate, args.extra, args.recall, args.bootstrap)
        target = out / f"fixed_recall__{args.control}__{args.candidate}.json"
    else:
        report = combine(args.name, args.runs, args.stack, args.save)
        target = out / f"combine__{args.name}.json"
    Path(target).write_text(json.dumps(report, indent=1, default=float))
    print(f"\nSaved {target}")


if __name__ == "__main__":
    main()

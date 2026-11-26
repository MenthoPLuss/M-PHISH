"""Build the final research artifact: several finished runs as one mean-of-logits ensemble.

Nothing is retrained. Each member's tensors are copied unchanged from its run's exported
artifact. Platt calibration is fitted on the calibration split, and the threshold is set on
validation by the production rule (the H10 decision). Every calibration and validation hostname
is then rescored with the NumPy runtime and checked against the members' saved PyTorch logits.
The test split and the lockbox are not touched.

    python -m research.hostname_cnn.final_build --name c13-ensemble3 \
        --runs C13-v3wide-h1v2-random-s42 C13-v3wide-h1v2-random-s43 C13-v3wide-h1v2-random-s44

Writes ``runs/research/F-<name>/``: ``url_cnn.json``, ``scores.npz`` and ``experiment.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from backend.app.services.ml.train_url_cnn import calibrate, select_threshold

from . import numpy_v4
from .calibration import site_clustering
from .common import RESEARCH_RUNS, describe, git_state, load_dataset, stem_unseen_mask
from .compare import aligned, load
from .metrics import evaluate, subgroup_table

PARITY_LIMIT = 1e-3  # the harness rule; the target is 1e-5
_model = None


def merge_artifacts(artifacts: list[dict]) -> dict:
    """One artifact whose members are the given single-member artifacts' members, in order."""
    first = artifacts[0]
    for other in artifacts[1:]:
        for key in ("format", "feature_version", "alphabet", "config"):
            if other[key] != first[key]:
                raise ValueError(f"Members disagree on {key}")
    if any(len(a["members"]) != 1 for a in artifacts):
        raise ValueError("Each source run must hold exactly one member")
    return {
        "format": first["format"],
        "feature_version": first["feature_version"],
        "alphabet": first["alphabet"],
        "config": first["config"],
        "members": [a["members"][0] for a in artifacts],
        "best_epochs": [a["best_epochs"][0] for a in artifacts],
    }


def _init(raw: bytes):
    global _model
    _model = numpy_v4.V4Numpy(json.loads(raw))


def _logit(host: str) -> float:
    return _model.logit(host)


def numpy_logits(raw: bytes, hosts: list[str], processes: int) -> np.ndarray:
    with Pool(processes, initializer=_init, initargs=(raw,)) as pool:
        return np.array(pool.map(_logit, hosts, chunksize=300))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--processes", type=int, default=6)
    args = parser.parse_args(argv)
    started = time.time()
    out = RESEARCH_RUNS / f"F-{args.name}"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists; choose another --name")

    sources = []
    for run in args.runs:
        path = RESEARCH_RUNS / run / "url_cnn.json"
        raw = path.read_bytes()
        record = json.loads((RESEARCH_RUNS / run / "experiment.json").read_text())
        if hashlib.sha256(raw).hexdigest() != record["artifact_sha256"]:
            raise SystemExit(f"{path} does not match its experiment.json")
        if record["git"]["dirty"]:
            raise SystemExit(f"{run} was trained from uncommitted code")
        sources.append((run, json.loads(raw), record))
    data_hashes = {(r["data"]["dataset_sha256"], r["data"]["splits_sha256"]) for _, _, r in sources}
    if len(data_hashes) != 1:
        raise SystemExit("The runs were trained on different datasets or splits")

    loaded = [load(run) for run in args.runs]
    hosts_v, y_v, Zv = aligned(loaded)
    hosts_c, y_c, Zc = aligned(loaded, "calibration")
    merged = merge_artifacts([a for _, a, _ in sources])

    # Score with the shipped runtime first (calibration fitted on what the app will compute).
    probe = json.dumps(merged, sort_keys=True).encode()
    zc = numpy_logits(probe, hosts_c, args.processes)
    zv = numpy_logits(probe, hosts_v, args.processes)
    parity = float(max(np.abs(zc - Zc.mean(1)).max(), np.abs(zv - Zv.mean(1)).max()))
    if parity > PARITY_LIMIT:
        raise SystemExit(f"NumPy runtime disagrees with the saved logits: {parity}")

    platt = calibrate(zc, y_c)
    slope, intercept = float(platt.coef_[0, 0]), float(platt.intercept_[0])

    def probability(z):
        return 1 / (1 + np.exp(-np.clip(slope * z + intercept, -700, 700)))

    pv = probability(zv)
    threshold = float(select_threshold(y_v, pv))
    artifact = merged | {
        "release_gate": "failed",  # research artifacts never claim a passed gate
        "model_version": f"research-F-{args.name}",
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "threshold": threshold,
    }
    raw = json.dumps(artifact, sort_keys=True, allow_nan=False).encode()
    model = numpy_v4.V4Numpy(json.loads(raw))
    sample = hosts_v[:1000]
    t0 = time.perf_counter()
    for host in sample:
        model.logit(host)
    latency_ms = (time.perf_counter() - t0) / len(sample) * 1000

    first = sources[0][2]
    data = load_dataset(Path(first["data"]["dataset"]), Path(first["data"]["splits"]))
    position = {h: i for i, h in enumerate(data.hosts)}
    subset = np.array([position[h] for h in hosts_v])
    groups = {
        dim: np.array([describe(h)[dim] for h in hosts_v])
        for dim in ("shape", "platform", "length", "labels", "tld")
    }
    groups["source"] = data.source[subset]
    groups["stem"] = np.where(stem_unseen_mask(data, subset), "stem unseen", "stem seen")
    sites = np.array([describe(h)["registrable"] for h in hosts_v])
    validation = {
        "all": evaluate(y_v, zv, pv, threshold, bootstrap=200, seed=0),
        "subgroups": {k: subgroup_table(y_v, zv, pv, threshold, g) for k, g in groups.items()},
        "sites": site_clustering(y_v, pv, threshold, sites, 1000, 0),
        "calibration_split_at_threshold": {
            "fpr": float((probability(zc)[y_c == 0] >= threshold).mean()),
            "recall": float((probability(zc)[y_c == 1] >= threshold).mean()),
        },
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "url_cnn.json").write_bytes(raw)
    np.savez_compressed(
        out / "scores.npz",
        validation_hosts=np.array(hosts_v),
        validation_y=y_v,
        validation_logit=zv,
        calibration_hosts=np.array(hosts_c),
        calibration_y=y_c,
        calibration_logit=zc,
    )
    record = {
        "experiment_id": out.name,
        "kind": "final build",
        "hypothesis": "final (C13 = H12 + H1; calibration by H10)",
        "description": "Mean-of-logits ensemble of finished runs; nothing retrained",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_state(),
        "members": [
            {
                "run": run,
                "artifact_sha256": record["artifact_sha256"],
                "git_commit": record["git"]["commit"],
                "code_sha256": record["git"].get("code_sha256"),
                "chosen_epoch": record["chosen_epoch"],
                "seed": record["training"]["seed"],
            }
            for run, _, record in sources
        ],
        "data": first["data"],
        "arch": first["arch"],
        "config": first["config"],
        "calibration": {"method": "platt", "slope": slope, "intercept": intercept},
        "threshold": threshold,
        "threshold_rule": "backend train_url_cnn.select_threshold on validation",
        "validation": validation,
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_bytes": len(raw),
        "numpy_parity_max_abs": parity,
        "numpy_parity_scope": f"all {len(hosts_c)} calibration and {len(hosts_v)} validation",
        "numpy_latency_ms": latency_ms,
        "wall_minutes": (time.time() - started) / 60,
        "test_split_scored": False,
        "lockbox_opened": False,
    }
    (out / "experiment.json").write_text(json.dumps(record, indent=1, default=float))
    v = validation["all"]
    print(
        f"DONE {out.name}: {len(sources)} members, val ROC-AUC {v['roc_auc']:.4f}, "
        f"pAUC {v['pauc_1pct']:.4f}, recall@0.5% {v['recall_at_0.005_fpr']:.4f}, "
        f"recall@1% {v['recall_at_0.01_fpr']:.4f}; threshold {threshold:.4f}: FPR "
        f"{v['fpr_at_threshold'] * 100:.2f}%, recall {v['recall_at_threshold'] * 100:.1f}%; "
        f"parity {parity:.1e}, {latency_ms:.1f} ms/hostname, "
        f"{os.path.getsize(out / 'url_cnn.json') / 1e6:.1f} MB"
    )
    return record


if __name__ == "__main__":
    main()

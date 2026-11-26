"""Phase 11-early: the final test with the material available on 2026-11-27.

The pre-registered plan, its amendment and the pass criteria are in ``RESEARCH_LOG.md``
("Plan — early final test"). Two commands, run in this order:

``build``
    The early temporal phishing holdout: hostnames on today's OpenPhish community feed and
    Phishing Army blocklist that are absent from the 2026-11-24 Phishing.Database list.
    Any hostname whose registered domain or name stem occurs anywhere in the h1-v2 dataset,
    or whose registered domain is in the lockbox bank, is excluded. The holdout's checksum is
    written to the lockbox manifest. Nothing is scored.

``score``
    Requires ``M_PHISH_LOCKBOX_UNLOCK=phase-11`` and runs once. Both the deployed model and the
    candidate are scored by the app's own loader on: the bank + holdout (the sealed set), and
    the h1-v2 test split. The pre-registered verdict is computed from the results.

    python -m research.hostname_cnn.phase11_early build
    M_PHISH_LOCKBOX_UNLOCK=phase-11 python -m research.hostname_cnn.phase11_early score
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from backend.app.services.ml import hostname_feeds as feeds
from backend.app.services.ml.offline import wilson
from backend.app.services.ml.url_cnn import UrlCnnModel

from . import lockbox
from .common import (
    BASELINE_RUN,
    FEED,
    RESEARCH_RUNS,
    ROOT,
    describe,
    git_state,
    load_dataset,
    sha256_file,
)
from .metrics import paired_difference, ranking

RAW = ROOT / "data" / "feeds" / "2026-11-27" / "raw"
FEEDS = {
    "openphish": RAW / "openphish-feed.txt",
    "phishing-army": RAW / "phishing-army-blocklist.txt",
}
H1 = ROOT / "data" / "feeds" / "research" / "h1-v2"
HOLDOUT = lockbox.LOCKBOX / "temporal_phishing_early_2026-11-27.json"
OUT = RESEARCH_RUNS / "P11-early"
TEST_DOMAINS = {"amiprotected.it"}
MODELS = {
    "deployed": ROOT / "runs" / "cnn-v3-fresh" / "url_cnn.json",
    "candidate": ROOT / "runs" / "cnn-v3wide-c13-ensemble3" / "url_cnn.json",
}
CANDIDATE_SHA256 = "6301ab4bf77bf881c117e59ba61981439890e0284bc355a8443c5e01aed10ca8"
MIN_PHISHING = 300
BOOTSTRAP = 1000
_model = None


def build() -> dict:
    if HOLDOUT.exists():
        raise SystemExit(f"{HOLDOUT} exists; the holdout is built once")
    data = load_dataset(H1 / "combined.csv", H1 / "splits.json")
    pairs = [lockbox.keys(h) for h in data.hosts]
    used_registrable = {r for r, _ in pairs}
    used_stems = {s for _, s in pairs}
    old = set(feeds.phishing_hostnames(FEED / "raw" / "phishing-domains-ACTIVE.txt"))
    bank_registrable = lockbox.reserved()["registrable"]
    found: dict[str, set[str]] = {}
    for name, path in FEEDS.items():
        for host in feeds.phishing_hostnames(path):
            found.setdefault(host, set()).add(name)
    counts = {"listed": len(found)}
    kept: dict[str, list[str]] = {}
    for host, sources in sorted(found.items()):
        registrable, stem = lockbox.keys(host)
        if registrable in TEST_DOMAINS:
            reason = "feed test domain"
        elif host in old:
            reason = "on the 2026-11-24 Phishing.Database list"
        elif registrable in used_registrable:
            reason = "registered domain in the h1-v2 dataset"
        elif stem in used_stems:
            reason = "name stem in the h1-v2 dataset"
        elif registrable in bank_registrable:
            reason = "registered domain in the lockbox bank"
        else:
            kept[host] = sorted(sources)
            continue
        counts[reason] = counts.get(reason, 0) + 1
    counts["kept"] = len(kept)
    HOLDOUT.write_text(json.dumps(kept, indent=0, sort_keys=True))
    entry = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file": HOLDOUT.name,
        "sha256": sha256_file(HOLDOUT),
        "hostnames": len(kept),
        "by_feed": {n: sum(n in s for s in kept.values()) for n in FEEDS},
        "feeds": {
            n: {"file": str(p.relative_to(ROOT)), "sha256": sha256_file(p)}
            for n, p in FEEDS.items()
        },
        "filtering": counts,
    }
    manifest = json.loads(lockbox.MANIFEST.read_text())
    manifest["early_temporal_phishing"] = entry
    lockbox.MANIFEST.write_text(json.dumps(manifest, indent=1))
    return entry


def _init(path: str):
    global _model
    raw = Path(path).read_bytes()
    _model = UrlCnnModel(path, hashlib.sha256(raw).hexdigest(), allow_ungated=True)


def _logit(host: str) -> float:
    return _model.logit(host)


def logits(path: Path, hosts: list[str], processes: int) -> np.ndarray:
    with Pool(processes, initializer=_init, initargs=(str(path),)) as pool:
        return np.array(pool.map(_logit, hosts, chunksize=300))


def operating(artifact: dict, y: np.ndarray, z: np.ndarray, groups: dict) -> dict:
    p = 1 / (
        1
        + np.exp(
            -np.clip(
                artifact["calibration_slope"] * z + artifact["calibration_intercept"], -700, 700
            )
        )
    )
    flagged = p >= artifact["threshold"]
    row = {"threshold": artifact["threshold"]}
    for name, mask in ({"all": np.ones(len(y), bool)} | groups).items():
        neg, pos = mask & (y == 0), mask & (y == 1)
        entry = {}
        if neg.any():
            entry |= {"legitimate": int(neg.sum()), "fpr": float(flagged[neg].mean()),
                      "fpr_ci95": wilson(int(flagged[neg].sum()), int(neg.sum()))}  # fmt: skip
        if pos.any():
            entry |= {"phishing": int(pos.sum()), "recall": float(flagged[pos].mean()),
                      "recall_ci95": wilson(int(flagged[pos].sum()), int(pos.sum()))}  # fmt: skip
        row[name] = entry
    return row


def summarize(artifacts, sealed_y, test_y, z, sealed_groups, test_groups, n_phishing, bootstrap):
    """Metrics for both models and the pre-registered verdict (see RESEARCH_LOG.md)."""
    results = {"sealed": {}, "test": {}}
    for name, artifact in artifacts.items():
        results["sealed"][name] = ranking(sealed_y, z[name]["sealed"]) | {
            "operating": operating(artifact, sealed_y, z[name]["sealed"], sealed_groups)
        }
        results["test"][name] = ranking(test_y, z[name]["test"]) | {
            "operating": operating(artifact, test_y, z[name]["test"], test_groups)
        }
    for split, y in (("sealed", sealed_y), ("test", test_y)):
        results[split]["candidate_minus_deployed"] = paired_difference(
            y, z["deployed"][split], z["candidate"][split], bootstrap=bootstrap, seed=11
        )
    a = results["sealed"]["candidate_minus_deployed"]["recall_at_0.01_fpr"]["ci95"][0]
    b = results["sealed"]["candidate"]["operating"]["all"]["fpr_ci95"][1]
    c = results["test"]["candidate_minus_deployed"]["pauc_1pct"]["ci95"][0]
    enough = n_phishing >= MIN_PHISHING
    verdict = {
        "A_detection": ("pass" if a > -0.005 else "fail") if enough else "inconclusive",
        "A_lower_bound": a,
        "B_false_alarms": "pass" if b <= 0.01 else "fail",
        "B_upper_bound": b,
        "C_test_split": "pass" if c > -0.005 else "fail",
        "C_lower_bound": c,
    }
    verdict["deploy"] = all(
        verdict[k] == "pass" for k in ("A_detection", "B_false_alarms", "C_test_split")
    )
    return results, verdict


def score(processes: int) -> dict:
    if (OUT / "results.json").exists():
        raise SystemExit(f"{OUT} has results; Phase 11-early is scored once")
    started = time.time()
    bank = lockbox.open_lockbox()  # refuses without the Phase 11 unlock
    manifest = json.loads(lockbox.MANIFEST.read_text())["early_temporal_phishing"]
    if sha256_file(HOLDOUT) != manifest["sha256"]:
        raise SystemExit("Holdout checksum mismatch")
    holdout = json.loads(HOLDOUT.read_text())
    raw = MODELS["candidate"].read_bytes()
    if hashlib.sha256(raw).hexdigest() != CANDIDATE_SHA256:
        raise SystemExit("The candidate is not the pre-registered artifact")
    artifacts = {k: json.loads(p.read_bytes()) for k, p in MODELS.items()}
    phishing = sorted(holdout)
    sealed_hosts = bank + phishing
    sealed_y = np.r_[np.zeros(len(bank)), np.ones(len(phishing))].astype(int)
    h1 = load_dataset(H1 / "combined.csv", H1 / "splits.json")
    test_idx = h1.indices("test")
    test_hosts, test_y = [h1.hosts[i] for i in test_idx], h1.y[test_idx].astype(int)
    deployed_data = load_dataset(FEED / "combined.csv", BASELINE_RUN / "splits.json")
    deployed_test = {deployed_data.hosts[i] for i in deployed_data.indices("test")}

    # Logits are saved before any metric is computed, so a failure later never forces a
    # rescoring; a rerun reuses them after checking they cover exactly the same hostnames.
    saved = OUT / "scores.npz"
    if saved.exists():
        npz = np.load(saved)
        if npz["sealed_hosts"].tolist() != sealed_hosts or npz["test_hosts"].tolist() != test_hosts:
            raise SystemExit("Saved scores cover different hostnames")
        z = {n: {s: npz[f"{n}_{s}_logit"] for s in ("sealed", "test")} for n in MODELS}
    else:
        z = {
            n: {
                "sealed": logits(path, sealed_hosts, processes),
                "test": logits(path, test_hosts, processes),
            }
            for n, path in MODELS.items()
        }
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            saved,
            sealed_hosts=np.array(sealed_hosts), sealed_y=sealed_y,
            test_hosts=np.array(test_hosts), test_y=test_y,
            **{f"{n}_{s}_logit": v for n, d in z.items() for s, v in d.items()},
        )  # fmt: skip
    platform = np.array([describe(h)["platform"] == "hosting platform" for h in sealed_hosts])
    feed_masks = {
        f"feed {n}": np.array([n in holdout.get(h, ()) for h in sealed_hosts]) for n in FEEDS
    }
    sealed_groups = {"hosting platform": platform, "ICANN suffix": ~platform} | feed_masks
    original = np.array([h in deployed_test for h in test_hosts])
    results, verdict = summarize(
        artifacts,
        sealed_y,
        test_y,
        z,
        sealed_groups,
        {"original v3 test": original},
        len(phishing),
        BOOTSTRAP,
    )
    record = {
        "experiment_id": OUT.name,
        "kind": "final test (early)",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_state(),
        "models": {
            k: {"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)}
            for k, p in MODELS.items()
        },
        "sealed_set": {
            "legitimate_bank": len(bank),
            "early_phishing": len(phishing),
            "holdout": manifest,
        },
        "test_split": {"hostnames": len(test_hosts), "original_v3_test": int(original.sum())},
        "bootstrap": BOOTSTRAP,
        "results": results,
        "verdict": verdict,
        "wall_minutes": (time.time() - started) / 60,
    }
    (OUT / "results.json").write_text(json.dumps(record, indent=1, default=float))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build", "score"])
    parser.add_argument("--processes", type=int, default=6)
    args = parser.parse_args(argv)
    if args.command == "build":
        print(json.dumps(build(), indent=1))
        return
    record = score(args.processes)
    for split in ("sealed", "test"):
        for name in MODELS:
            r = record["results"][split][name]
            op = r["operating"]["all"]
            print(
                f"{split:6} {name:9} ROC-AUC {r['roc_auc']:.4f} pAUC {r['pauc_1pct']:.4f} "
                f"R@0.1% {r['recall_at_0.001_fpr']:.4f} R@1% {r['recall_at_0.01_fpr']:.4f} | "
                f"thr: FPR {op.get('fpr', float('nan')) * 100:.2f}% "
                f"recall {op.get('recall', float('nan')) * 100:.1f}%"
            )
    print(json.dumps(record["verdict"], indent=1))


if __name__ == "__main__":
    main()

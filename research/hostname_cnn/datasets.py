"""Research datasets derived from the 2026-11-24 combined dataset.

``h1-v2`` (hypothesis H1): the same rows plus every unused CrUX 2026-10 hostname of the kinds
the baseline rarely sees as legitimate: hosting-platform suffixes, hostnames longer than 30
characters, and hostnames with 4 or more labels. Excluded: anything reserved by the lockbox,
any hostname on the Phishing.Database list, and any hostname whose registered domain is on
it. Earlier hostnames keep their splits (``reuse_splits``), so validation stays comparable.

    python -m research.hostname_cnn.datasets h1-v2
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

from backend.app.services.ml import hostname_feeds as feeds
from backend.app.services.ml.offline import canonical_hash
from backend.app.services.observations import _suffix_extractor

from .common import BASELINE_RUN, FEED, ROOT, load_dataset, sha256_file
from .lockbox import keys, reserved

OUT = ROOT / "data" / "feeds" / "research"


def extend_splits(hosts: list[str], previous: dict) -> tuple[dict, int]:
    """Earlier hostnames keep their split; a new hostname joins its registered domain's split,
    or a new domain is placed by a hash of its name (55/15/15/15, the baseline proportions).

    Unlike ``reuse_splits`` this accepts single-label additions (H1 adds only legitimate rows).
    """
    import hashlib

    earlier = {d: name for name, digests in previous.items() for d in digests}
    group_split: dict[str, str] = {}
    assigned: list[str] = []
    for host in hosts:
        name = earlier.get(canonical_hash(host))
        group = keys(host)[0]
        if name is not None:
            group_split.setdefault(group, name)
        assigned.append(name or "")
    kept = sum(1 for a in assigned if a)
    for i, host in enumerate(hosts):
        if assigned[i]:
            continue
        group = keys(host)[0]
        if group not in group_split:
            u = int(hashlib.sha256(group.encode()).hexdigest()[:12], 16) / 16**12
            group_split[group] = (
                "train"
                if u < 0.55
                else "calibration"
                if u < 0.70
                else "validation"
                if u < 0.85
                else "test"
            )
        assigned[i] = group_split[group]
    for host, name in zip(hosts, assigned, strict=True):
        if canonical_hash(host) in earlier and earlier[canonical_hash(host)] != name:
            raise ValueError("An earlier hostname changed split")
    names = np.array(assigned)
    return {
        name: np.flatnonzero(names == name)
        for name in ("train", "calibration", "validation", "test")
    }, kept


def build_h1_v2(seed: int = 42) -> dict:
    target = OUT / "h1-v2"
    if (target / "combined.csv").exists():
        raise SystemExit(f"{target} exists; datasets are immutable")
    base = load_dataset()
    lock = reserved()
    if not lock["hosts"]:
        raise SystemExit("Build the lockbox first so its hostnames are excluded")
    phishing = feeds.phishing_hostnames(FEED / "raw" / "phishing-domains-ACTIVE.txt")
    phishing_registrable = {keys(h)[0] for h in phishing}
    used = set(base.hosts)
    extract = _suffix_extractor()
    added: dict[str, list[str]] = {"hosting platform": [], "long (>30)": [], "deep (4+ labels)": []}
    removed = {"on phishing list": 0, "registered domain on phishing list": 0, "lockbox": 0}
    for host, raw in sorted(feeds.crux_hostnames(FEED / "raw" / "crux-202610.csv.gz").items()):
        if host in used:
            continue
        platform = extract(host).is_private
        kind = (
            "hosting platform"
            if platform
            else "long (>30)"
            if len(host) > 30
            else "deep (4+ labels)"
            if host.count(".") >= 3
            else None
        )
        if kind is None:
            continue
        registrable, stem = keys(host)
        if host in lock["hosts"] or registrable in lock["registrable"] or stem in lock["stems"]:
            removed["lockbox"] += 1
            continue
        if host in phishing:
            removed["on phishing list"] += 1
            continue
        if registrable in phishing_registrable:
            removed["registered domain on phishing list"] += 1
            continue
        added[kind].append(raw)
    target.mkdir(parents=True, exist_ok=True)
    rows = 0
    with (target / "combined.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        with base.csv_path.open(encoding="utf-8", newline="") as source:
            for row in csv.reader(source):
                writer.writerow(row)
                rows += 1
        for kind, origins in added.items():
            for origin in origins:
                writer.writerow([origin, "1", f"crux_{kind.split()[0]}"])
                rows += 1
    data_rows = rows - 1
    # Assign splits: earlier hostnames keep theirs; new registered domains are split anew.
    from backend.app.services.ml.train_url_cnn import load_csv

    hosts, y = load_csv(target / "combined.csv", "URL", "label", "1", "0")
    splits, kept = extend_splits(hosts, json.loads((BASELINE_RUN / "splits.json").read_text()))
    (target / "splits.json").write_text(
        json.dumps({k: [canonical_hash(hosts[i]) for i in v] for k, v in splits.items()})
    )
    report = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hypothesis": "H1",
        "base_dataset_sha256": sha256_file(base.csv_path),
        "lockbox_excluded": True,
        "added_legitimate": {k: len(v) for k, v in added.items()},
        "removed": removed,
        "rows": data_rows,
        "kept_split_assignments": kept,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "sha256": sha256_file(target / "combined.csv"),
        "splits_sha256": sha256_file(target / "splits.json"),
        "licence": "PhiUSIIL (CC BY 4.0); Phishing.Database (MIT); CrUX (CC BY 4.0)",
    }
    (target / "dataset.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build research datasets.")
    parser.add_argument("name", choices=["h1-v2"])
    args = parser.parse_args(argv)
    if args.name == "h1-v2":
        print(json.dumps(build_h1_v2(), indent=1))


if __name__ == "__main__":
    main()

"""Shared data access: datasets, splits, hostname descriptors and provenance helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

from backend.app.services.ml.offline import canonical_hash
from backend.app.services.ml.train_url_cnn import hostname_of, load_csv
from backend.app.services.observations import _suffix_extractor, registrable_domain

ROOT = Path(__file__).resolve().parents[2]
FEED = ROOT / "data" / "feeds" / "2026-11-24"
BASELINE_RUN = ROOT / "runs" / "cnn-v3-fresh"
RESEARCH_RUNS = ROOT / "runs" / "research"
SPLITS = ("train", "calibration", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


CODE_FILES = (
    "research/hostname_cnn",
    "backend/app/services/ml",
    "backend/app/services/observations.py",
)


def code_files() -> list[Path]:
    files: list[Path] = []
    for entry in CODE_FILES:
        path = ROOT / entry
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return files


def git_state() -> dict:
    """Commit, dirty flag and a content hash of every source file that shapes a result.

    The content hash ties a result to its exact code even before that code is committed;
    the trainer also copies those files into the run folder.
    """

    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    digest = hashlib.sha256()
    for path in code_files():
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--", *CODE_FILES)),
        "code_sha256": digest.hexdigest(),
    }


def snapshot_sha256(folder: Path) -> str:
    """The ``code_sha256`` a run's saved code snapshot had when it was taken."""
    digest = hashlib.sha256()
    for entry in CODE_FILES:
        path = folder / entry
        for file in sorted(path.rglob("*.py")) if path.is_dir() else [path]:
            digest.update(str(file.relative_to(folder)).replace("\\", "/").encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    """Repository-relative form of a path given relative to the working directory or absolute."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def snapshot_code(target: Path) -> None:
    for path in code_files():
        destination = target / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())


@dataclass
class Dataset:
    """Hostnames (model input form), labels (1 = phishing), source tag and split name."""

    hosts: list[str]
    y: np.ndarray
    source: np.ndarray
    split: np.ndarray
    csv_path: Path
    splits_path: Path
    extra: dict = field(default_factory=dict)

    def indices(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.split == name)

    def provenance(self) -> dict:
        return {
            "dataset": repo_relative(self.csv_path),
            "dataset_sha256": sha256_file(self.csv_path),
            "splits": repo_relative(self.splits_path),
            "splits_sha256": sha256_file(self.splits_path),
            "hostnames": len(self.hosts),
            "split_sizes": {name: int((self.split == name).sum()) for name in SPLITS},
        } | self.extra


def load_dataset(
    csv_path: Path = FEED / "combined.csv", splits_path: Path | None = None
) -> Dataset:
    """Load a combined CSV (URL, label, source) and its split assignment.

    ``splits_path`` must assign every hostname (a ``splits.json`` written by the trainer or
    by :func:`research.hostname_cnn.datasets.assign_splits`).
    """
    splits_path = splits_path or BASELINE_RUN / "splits.json"
    hosts, y = load_csv(csv_path, "URL", "label", "1", "0")
    source: dict[str, str] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            host = hostname_of(row["URL"])
            if host:
                source.setdefault(host, row.get("source") or "unknown")
    assigned = {
        digest: name
        for name, digests in json.loads(splits_path.read_text()).items()
        for digest in digests
    }
    split = np.array([assigned.get(canonical_hash(h), "") for h in hosts])
    if (split == "").any():
        raise ValueError(f"{int((split == '').sum())} hostnames have no split in {splits_path}")
    return Dataset(hosts, y, np.array([source[h] for h in hosts]), split, csv_path, splits_path)


@cache
def describe(host: str) -> dict:
    """Structural descriptors used for subgroup reporting (never as model input)."""
    parts = _suffix_extractor()(host)
    length = len(host)
    labels = host.count(".") + 1
    tld = host.rsplit(".", 1)[-1]
    if tld == "com":
        tld_class = "com"
    elif tld in ("net", "org"):
        tld_class = "net/org"
    elif tld.startswith("xn--"):
        tld_class = "idn tld"
    elif len(tld) == 2:
        tld_class = "ccTLD"
    else:
        tld_class = "other gTLD"
    return {
        "shape": "bare" if (registrable_domain(host) or host) == host else "subdomain",
        "platform": "hosting platform" if parts.is_private else "ICANN suffix",
        "length": next(
            name
            for name, (lo, hi) in {
                "1-10": (1, 10),
                "11-15": (11, 15),
                "16-20": (16, 20),
                "21-30": (21, 30),
                "31-50": (31, 50),
                "51+": (51, 253),
            }.items()
            if lo <= length <= hi
        ),
        "labels": "5+" if labels >= 5 else str(labels),
        "tld": tld_class,
        "punycode": "punycode" if "xn--" in host else "ascii",
        "stem": parts.domain or host,
        "suffix": parts.suffix,
        "registrable": registrable_domain(host) or host,
    }


def stem_unseen_mask(data: Dataset, subset: np.ndarray) -> np.ndarray:
    """True where the hostname's name stem never occurs in the training split."""
    train_stems = {describe(data.hosts[i])["stem"] for i in data.indices("train")}
    return np.array([describe(data.hosts[i])["stem"] not in train_stems for i in subset])


def subgroup_labels(hosts: list[str], dimension: str) -> np.ndarray:
    return np.array([describe(h)[dimension] for h in hosts])

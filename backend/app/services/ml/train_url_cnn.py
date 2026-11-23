"""Offline training for the hostname character CNN. Never imported by the API.

Examples (run from the project root):
    # Download PhiUSIIL (UCI, CC BY 4.0) and train
    python -m backend.app.services.ml.train_url_cnn --download-phiusiil data/phiusiil --output runs/cnn-1

    # Use a CSV you already have
    python -m backend.app.services.ml.train_url_cnn --csv data/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv --output runs/cnn-1

    # Fast smoke test on generated hostnames (can never pass the release gate)
    python -m backend.app.services.ml.train_url_cnn --synthetic-smoke --output runs/smoke

Outputs in --output: url_cnn.json (deployable artifact), results.json (evaluation-v2,
readable by the Research page), manifest.json, splits.json, release_targets.json,
checksums.txt. Requires PyTorch: pip install -r backend/requirements-cnn.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import random
import sys
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..observations import registrable_domain
from .offline import RELEASE_TARGETS, canonical_hash, metrics, release_gate
from .offline import split_indices as grouped_splits
from .url_cnn import (
    ALPHABET,
    ARTIFACT_FORMAT,
    FEATURE_VERSION,
    PAD,
    VOCAB_SIZE,
    UrlCnnModel,
    encode,
    model_input,
)

PHIUSIIL_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
PHIUSIIL_MANIFEST = {
    "source": "PhiUSIIL Phishing URL Dataset, UCI Machine Learning Repository (id 967), "
    "Prasad & Chandra (2024), doi:10.24432/C5Z90W",
    "license": "CC BY 4.0",
    "usage_permission": "CC BY 4.0 permits use with attribution; verify terms for your use.",
    "label_methodology": "Labels as published by the dataset authors; reduced to hostname.",
}
# Well-known legitimate sites used only to sanity-check the label direction.
LABEL_ANCHORS = {
    "google.com",
    "youtube.com",
    "wikipedia.org",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "github.com",
    "linkedin.com",
    "bbc.co.uk",
    "nytimes.com",
    "mozilla.org",
    "stackoverflow.com",
}


# ---------------------------------------------------------------- data loading


def download_phiusiil(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    existing = sorted(folder.glob("*.csv"))
    if existing:
        print(f"Using existing {existing[0]}")
        return existing[0]
    print(f"Downloading PhiUSIIL from {PHIUSIIL_URL} ...", flush=True)
    with urllib.request.urlopen(PHIUSIIL_URL, timeout=120) as response:  # noqa: S310 -- fixed https URL
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in the archive, found {names}")
        target = folder / Path(names[0]).name
        target.write_bytes(archive.read(names[0]))
    print(f"Saved {target}")
    return target


def hostname_of(url: str) -> str:
    url = url.strip()
    if "://" not in url:
        url = "http://" + url
    try:
        return model_input(urlsplit(url).hostname or "")
    except ValueError:
        return ""


def load_csv(path: Path, url_column: str, label_column: str, legit_value: str):
    """Returns (hostnames, labels with 1 = phishing) after dedupe and conflict removal."""
    csv.field_size_limit(10_000_000)
    labels_by_host: dict[str, set[int]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if url_column not in (reader.fieldnames or []) or label_column not in (
            reader.fieldnames or []
        ):
            raise ValueError(
                f"CSV needs columns {url_column!r} and {label_column!r}; found {reader.fieldnames}"
            )
        rows = 0
        for row in reader:
            rows += 1
            host = hostname_of(row[url_column])
            if not host or len(host) > 253:
                continue
            label = 0 if row[label_column].strip() == legit_value else 1
            labels_by_host.setdefault(host, set()).add(label)
    conflicts = [host for host, labels in labels_by_host.items() if len(labels) > 1]
    hosts = sorted(host for host, labels in labels_by_host.items() if len(labels) == 1)
    labels = np.array([next(iter(labels_by_host[h])) for h in hosts], dtype=np.int64)
    print(
        f"Read {rows} rows -> {len(hosts)} unique hostnames "
        f"({int(labels.sum())} phishing, {int((labels == 0).sum())} legitimate); "
        f"dropped {len(conflicts)} hostnames with conflicting labels."
    )
    check_label_direction(hosts, labels)
    return hosts, labels


def check_label_direction(hosts: list[str], labels: np.ndarray) -> None:
    """Refuse to train if famous legitimate sites look labelled as phishing."""
    found = {h: int(y) for h, y in zip(hosts, labels, strict=True) if h in LABEL_ANCHORS}
    if len(found) < 3:
        print(f"Label check: only {len(found)} anchor sites present; skipping direction check.")
        return
    phishing = [h for h, y in found.items() if y == 1]
    if len(phishing) > len(found) / 2:
        raise ValueError(
            f"Label direction looks inverted: {phishing} are labelled phishing. "
            "Re-run with the correct --legit-label value."
        )
    print(f"Label check passed: {len(found) - len(phishing)}/{len(found)} anchor sites legitimate.")


def synthetic_hostnames(n: int, seed: int):
    """Generated fixture for smoke tests only. Not realistic; cannot pass the gate."""
    rng = random.Random(seed)
    words = [
        "cloud",
        "data",
        "shop",
        "news",
        "home",
        "city",
        "green",
        "blue",
        "river",
        "north",
        "tech",
        "media",
        "book",
        "star",
    ]
    brands = ["paypal", "apple", "microsoft", "amazon", "netflix", "chase", "google", "wellsfargo"]
    bait = ["secure", "login", "verify", "account", "update", "support", "signin", "wallet"]
    hosts, labels = set(), []
    while len(hosts) < n:
        if rng.random() < 0.5:
            host = rng.choice(words) + rng.choice(["", rng.choice(words)])
            host += rng.choice([".com", ".org", ".net", ".io", ".co.uk"])
            label = 0
        else:
            parts = [rng.choice(brands), rng.choice(bait)]
            rng.shuffle(parts)
            host = "-".join(parts) + str(rng.randint(0, 999))
            host += rng.choice([".xyz", ".top", ".info", ".com", ".online"])
            label = 1
        if host not in hosts:
            hosts.add(host)
            labels.append((host, label))
    labels.sort()
    return [h for h, _ in labels], np.array([y for _, y in labels], dtype=np.int64)


# ---------------------------------------------------------------- model


def build_model(torch, embedding_dim, filters, kernel_sizes, hidden, dropout):
    nn = torch.nn

    class CharCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=PAD)
            self.convs = nn.ModuleList(nn.Conv1d(embedding_dim, filters, k) for k in kernel_sizes)
            self.dropout = nn.Dropout(dropout)
            self.dense = nn.Linear(filters * len(kernel_sizes), hidden)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x):
            e = self.embedding(x).transpose(1, 2)  # (N, E, L)
            pooled = [torch.relu(conv(e)).amax(dim=2) for conv in self.convs]
            h = torch.relu(self.dense(self.dropout(torch.cat(pooled, dim=1))))
            return self.out(self.dropout(h)).squeeze(1)

    return CharCNN()


def logits_for(torch, model, X, batch=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i : i + batch])).numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def train(torch, model, X, y, Xv, yv, epochs, batch_size, lr, seed):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    Xvt, yvt = torch.from_numpy(Xv), torch.from_numpy(yv.astype(np.float32))
    best = (float("inf"), None, 0)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(Xt), generator=generator)
        total, started = 0.0, time.perf_counter()
        for i in range(0, len(order), batch_size):
            index = order[i : i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(Xt[index]), yt[index])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xvt), yvt))
        print(
            f"epoch {epoch}/{epochs}  train_loss={total / len(Xt):.4f}  "
            f"val_loss={val_loss:.4f}  ({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
        if val_loss < best[0]:
            best = (val_loss, {k: v.clone() for k, v in model.state_dict().items()}, epoch)
    model.load_state_dict(best[1])
    print(f"Using epoch {best[2]} (lowest validation loss {best[0]:.4f})")
    return best[2]


def export_weights(model, kernel_sizes) -> dict:
    def values(tensor):
        return np.vectorize(lambda v: float(f"{v:.8g}"))(tensor.detach().numpy()).tolist()

    return {
        "embedding": values(model.embedding.weight),
        "convs": [
            {"kernel_size": k, "weight": values(conv.weight), "bias": values(conv.bias)}
            for k, conv in zip(kernel_sizes, model.convs, strict=True)
        ],
        "dense": {"weight": values(model.dense.weight), "bias": values(model.dense.bias)},
        "out": {"weight": values(model.out.weight), "bias": values(model.out.bias)},
    }


# ---------------------------------------------------------------- audits


def simple_features(texts: list[str]) -> dict[str, np.ndarray]:
    return {
        "length": np.array([len(t) for t in texts], dtype=float),
        "dots": np.array([t.count(".") for t in texts], dtype=float),
        "hyphens": np.array([t.count("-") for t in texts], dtype=float),
        "digits": np.array([sum(c.isdigit() for c in t) for t in texts], dtype=float),
        "tld_is_com": np.array([t.endswith(".com") for t in texts], dtype=float),
    }


def shortcut_audit(features: dict[str, np.ndarray], y: np.ndarray, limit=0.97) -> list[dict]:
    """Flag any single trivial feature that nearly separates the classes (leakage risk)."""
    findings = []
    for name, values in features.items():
        best = 0.0
        for cutoff in np.unique(np.quantile(values, np.linspace(0, 1, 257))):
            for predicted in (values > cutoff, values <= cutoff):
                balanced = 0.5 * (np.mean(predicted[y == 1]) + np.mean(~predicted[y == 0]))
                best = max(best, float(balanced))
        if best >= limit:
            findings.append({"feature": name, "balanced_accuracy": best})
    return findings


def select_threshold(labels, probabilities) -> float:
    """Same rule and result as offline.select_threshold, vectorized for large splits.

    Among thresholds meeting precision >= 0.95, recall >= 0.90 and FPR <= 0.01 on the
    validation split, pick the highest (recall, precision, threshold); else 0.5.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.unique(probabilities)
    order = np.sort(probabilities)
    positives = np.sort(probabilities[labels == 1])
    negatives = np.sort(probabilities[labels == 0])
    # Count of scores >= each threshold.
    tp = len(positives) - np.searchsorted(positives, thresholds, side="left")
    fp = len(negatives) - np.searchsorted(negatives, thresholds, side="left")
    predicted = len(order) - np.searchsorted(order, thresholds, side="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = tp / max(1, len(positives)) if len(positives) else np.zeros_like(thresholds)
        fpr = fp / len(negatives) if len(negatives) else np.full_like(thresholds, np.nan)
    ok = (precision >= 0.95) & (recall >= 0.90) & (fpr <= 0.01)
    if not np.any(ok):
        return 0.5
    candidates = sorted(
        zip(recall[ok].tolist(), precision[ok].tolist(), thresholds[ok].tolist(), strict=True)
    )
    return float(candidates[-1][2])


def calibrate(train_logits, labels):
    return LogisticRegression(C=1e6, max_iter=1000).fit(train_logits.reshape(-1, 1), labels)


def experiment_record(experiment_id, name, result, splits, kind, version, extra):
    return (
        result
        | {
            "experiment_id": experiment_id,
            "name": name,
            "feature_groups": ["Hostname characters"],
            "split_support": {k: int(len(v)) for k, v in splits.items()},
            "dataset_kind": kind,
            "dataset_version": version,
            "hypothesis_confirmed": "Measured comparison only; no causal claim.",
        }
        | extra
    )


# ---------------------------------------------------------------- main


def run(args) -> dict:
    try:
        import torch
    except ImportError:
        sys.exit("PyTorch is required: pip install -r backend/requirements-cnn.txt")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output folder must be empty; previous runs are preserved.")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.synthetic_smoke:
        hosts, y = synthetic_hostnames(args.max_rows or 3000, args.seed)
        manifest = {
            "dataset_version": "synthetic-smoke",
            "source": "generated by train_url_cnn.synthetic_hostnames",
            "license": "n/a",
            "usage_permission": "test fixture",
            "collected_at": datetime.now(UTC).isoformat(),
            "label_methodology": "generator rule; not real labels",
            "sha256": canonical_hash(hosts),
            "kind": "synthetic_fixture",
        }
    else:
        csv_path = (
            download_phiusiil(Path(args.download_phiusiil))
            if args.download_phiusiil
            else Path(args.csv)
        )
        hosts, y = load_csv(csv_path, args.url_column, args.label_column, args.legit_label)
        if args.max_rows and len(hosts) > args.max_rows:
            keep = np.random.RandomState(args.seed).choice(len(hosts), args.max_rows, replace=False)
            keep.sort()
            hosts, y = [hosts[i] for i in keep], y[keep]
        manifest = PHIUSIIL_MANIFEST | {
            "dataset_version": "phiusiil-" + hashlib.sha256(csv_path.read_bytes()).hexdigest()[:12],
            "collected_at": datetime.now(UTC).isoformat(),
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "kind": "real_labelled",
            "file": csv_path.name,
            "rows_used": len(hosts),
        }
        if args.csv and not args.download_phiusiil:
            manifest["source"] = "User-supplied CSV " + csv_path.name
            manifest["usage_permission"] = "Supplied by the operator; verify licence and labels."
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Subdomains of one registrable domain never straddle splits.
    groups = np.array([registrable_domain(h) or h for h in hosts])
    splits = grouped_splits(y, groups, args.seed)
    (output / "release_targets.json").write_text(json.dumps(RELEASE_TARGETS, indent=2))
    (output / "splits.json").write_text(
        json.dumps({k: [canonical_hash(hosts[i]) for i in v] for k, v in splits.items()})
    )
    X = np.stack([encode(h, args.max_length) for h in hosts])
    shortcuts = shortcut_audit(
        simple_features([hosts[i] for i in splits["train"]]), y[splits["train"]]
    )
    if shortcuts:
        print(f"WARNING shortcut findings (training split): {shortcuts}")

    model = build_model(
        torch, args.embedding_dim, args.filters, args.kernel_sizes, args.hidden, args.dropout
    )
    print(
        f"Training on {len(splits['train'])} hostnames, {sum(p.numel() for p in model.parameters())} parameters"
    )
    best_epoch = train(
        torch,
        model,
        X[splits["train"]],
        y[splits["train"]],
        X[splits["validation"]],
        y[splits["validation"]],
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
    )
    logits = {name: logits_for(torch, model, X[idx]) for name, idx in splits.items()}
    calibrator = calibrate(logits["calibration"], y[splits["calibration"]])

    def probabilities(name):
        return calibrator.predict_proba(logits[name].reshape(-1, 1))[:, 1]

    threshold = select_threshold(y[splits["validation"]], probabilities("validation"))
    test_probs = probabilities("test")
    result = metrics(y[splits["test"]], test_probs, threshold)
    gate = (
        "passed"
        if release_gate(result) and not shortcuts and manifest["kind"] == "real_labelled"
        else "failed"
    )

    artifact = {
        "format": ARTIFACT_FORMAT,
        "feature_version": FEATURE_VERSION,
        "alphabet": ALPHABET,
        "model_version": manifest["dataset_version"] + "-charcnn-v1",
        "release_gate": gate,
        "max_length": args.max_length,
        "embedding_dim": args.embedding_dim,
        "filters": args.filters,
        "kernel_sizes": args.kernel_sizes,
        "hidden": args.hidden,
        "weights": export_weights(model, args.kernel_sizes),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "threshold": float(threshold),
        "dataset_manifest": manifest,
        "release_targets": RELEASE_TARGETS,
        "best_epoch": best_epoch,
    }
    artifact_bytes = json.dumps(artifact, sort_keys=True, allow_nan=False).encode()
    artifact_path = output / "url_cnn.json"
    artifact_path.write_bytes(artifact_bytes)
    sha = hashlib.sha256(artifact_bytes).hexdigest()

    # Parity: the numpy inference used by the app must reproduce the PyTorch model.
    loaded = UrlCnnModel(str(artifact_path), sha, allow_ungated=True)
    if loaded.artifact is None:
        raise RuntimeError(loaded.reason)
    sample = splits["test"][: min(2000, len(splits["test"]))]
    started = time.perf_counter()
    numpy_logits = np.array([loaded.logit(hosts[i]) for i in sample])
    latency_ms = (time.perf_counter() - started) / max(1, len(sample)) * 1000
    parity = float(np.max(np.abs(numpy_logits - logits_for(torch, model, X[sample]))))
    if parity > 1e-3:
        raise RuntimeError(f"Numpy/PyTorch parity failed: max logit difference {parity}")

    # A trivial-feature baseline shows how much the CNN adds beyond length/dots/digits.
    feats = simple_features(hosts)
    F = np.stack(list(feats.values()), axis=1)
    mean, std = F[splits["train"]].mean(0), F[splits["train"]].std(0) + 1e-9
    base = LogisticRegression(max_iter=2000).fit(
        (F[splits["train"]] - mean) / std, y[splits["train"]]
    )
    base_cal = calibrate(
        base.decision_function((F[splits["calibration"]] - mean) / std), y[splits["calibration"]]
    )

    def base_probs(name):
        margin = base.decision_function((F[splits[name]] - mean) / std)
        return base_cal.predict_proba(margin.reshape(-1, 1))[:, 1]

    base_threshold = select_threshold(y[splits["validation"]], base_probs("validation"))
    base_result = metrics(y[splits["test"]], base_probs("test"), base_threshold)

    experiments = [
        experiment_record(
            "CNN-HOST",
            "char-CNN: hostname characters",
            result,
            splits,
            manifest["kind"],
            manifest["dataset_version"],
            {
                "num_features": args.max_length,
                "model_family": "char_cnn",
                "inference_latency_ms": latency_ms,
                "latency_scope": "numpy single-hostname inference averaged over test sample",
                "release_gate": gate,
                "numpy_torch_max_logit_difference": parity,
            },
        ),
        experiment_record(
            "BASE-HOSTFEAT",
            "logistic: length/dots/hyphens/digits/.com",
            base_result,
            splits,
            manifest["kind"],
            manifest["dataset_version"],
            {"num_features": 5, "model_family": "logistic", "release_gate": "baseline_only"},
        ),
    ]
    report = {
        "schema_version": "evaluation-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": manifest,
        "experiments": experiments,
        "shortcut_findings_train_only": shortcuts,
        "release_targets": RELEASE_TARGETS,
        "model_sha256": sha,
        "hardware": {"platform": platform.platform(), "python": platform.python_version()},
        "dependencies": {"torch": torch.__version__, "numpy": np.__version__},
        "limitations": [
            "Hostname-only model: page content, paths and behaviour are not seen.",
            "Random grouped holdout by registrable domain; temporal generalization is untested.",
            "Public URL datasets contain collection artifacts; check shortcut findings and the "
            "trivial-feature baseline before trusting headline accuracy.",
        ],
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (output / "checksums.txt").write_text(
        "\n".join(
            hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name
            for p in sorted(output.iterdir())
            if p.is_file() and p.name != "checksums.txt"
        )
    )

    def pct(v):
        return f"{v * 100:.2f}%"

    print("\n=== Test-set results (never used for training or threshold choice) ===")
    for row in experiments:
        print(
            f"{row['experiment_id']:14} precision={pct(row['precision'])} recall={pct(row['recall'])} "
            f"FPR={pct(row['false_positive_rate'])} ROC-AUC={row['roc_auc']:.4f} "
            f"(threshold {row['threshold']:.3f})"
        )
    print(
        f"Release gate: {gate.upper()}   numpy/torch parity: {parity:.2e}   "
        f"inference: {latency_ms:.2f} ms/hostname"
    )
    print(f"\nArtifact: {artifact_path.resolve()}\nSHA-256:  {sha}")
    print("To use it, set in .env (or your terminal) and restart the API/worker:")
    print(f"  ML_CNN_ARTIFACT_PATH={artifact_path.resolve()}")
    print(f"  ML_CNN_ARTIFACT_SHA256={sha}")
    if gate != "passed":
        print("  ML_CNN_ALLOW_UNGATED=true   # needed because the gate failed (local testing only)")
    print(f"  ML_RESEARCH_RESULTS={(output / 'results.json').resolve()}   # Research page")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--download-phiusiil", metavar="FOLDER", help="download PhiUSIIL into FOLDER"
    )
    source.add_argument("--csv", help="existing CSV with a URL column and a label column")
    source.add_argument("--synthetic-smoke", action="store_true", help="tiny generated dataset")
    parser.add_argument("--output", required=True, help="new empty folder for this run")
    parser.add_argument("--url-column", default="URL")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--legit-label", default="1", help="label value meaning legitimate (PhiUSIIL: 1)"
    )
    parser.add_argument("--max-rows", type=int, default=0, help="subsample for quick experiments")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()

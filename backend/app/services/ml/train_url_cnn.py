"""Offline training for the hostname character CNN. Never imported by the API.

Examples (run from the project root):
    # Download PhiUSIIL (UCI, CC BY 4.0) and train
    python -m backend.app.services.ml.train_url_cnn --download-phiusiil data/phiusiil --output runs/cnn-1

    # Use a CSV you already have
    python -m backend.app.services.ml.train_url_cnn --csv data/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv --output runs/cnn-1

    # Fast smoke test on generated hostnames (can never pass the release gate)
    python -m backend.app.services.ml.train_url_cnn --synthetic-smoke --output runs/smoke

    # v3 (default): masked gated CNN with attention, optionally an ensemble
    python -m backend.app.services.ml.train_url_cnn --arch v3 --ensemble 3 \
        --download-phiusiil data/phiusiil --output runs/cnn-v3

Outputs in --output: url_cnn.json (deployable artifact), results.json (evaluation-v2,
readable by the Research page), manifest.json, splits.json, release_targets.json,
training_config.json, history.json, checksums.txt. Requires PyTorch:
pip install -r backend/requirements-cnn.txt
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
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
from sklearn.metrics import average_precision_score

from ..observations import registrable_domain
from .offline import RELEASE_TARGETS, canonical_hash, metrics, release_gate
from .offline import split_indices as grouped_splits
from .url_cnn import (
    ALPHABET,
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_V2,
    ARTIFACT_FORMAT_V3,
    FEATURE_VERSION,
    FEATURE_VERSION_V2,
    FEATURE_VERSION_V3,
    NORM_EPS,
    PAD,
    ROLE_COUNT,
    ROLE_NAME,
    VOCAB_SIZE,
    UrlCnnModel,
    encode,
    encode_roles,
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


def load_csv(
    path: Path, url_column: str, label_column: str, legit_value: str, phishing_value: str = "0"
):
    """Returns (hostnames, labels with 1 = phishing) after dedupe and conflict removal."""
    csv.field_size_limit(10_000_000)
    if legit_value == phishing_value:
        raise ValueError("Legitimate and phishing labels must differ")
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
            raw_label = (row.get(label_column) or "").strip()
            if raw_label not in {legit_value, phishing_value}:
                raise ValueError(f"Unknown label {raw_label!r} at CSV row {rows + 1}")
            host = hostname_of(row.get(url_column) or "")
            if not host or len(host) > 253:
                continue
            label = 0 if raw_label == legit_value else 1
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


def batch_input(model, values, device):
    """Trim shared left padding only for the padding-invariant v3 architecture."""
    if getattr(model, "padding_invariant", False):
        length = max(1, int((values[:, 0] != PAD).sum(1).max()))
        values = values[:, :, -length:]
    return values.to(device)


def logits_for(torch, model, X, batch=256):
    model.eval()
    out = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for i in range(0, len(X), batch):
            values = batch_input(model, torch.from_numpy(X[i : i + batch]), device)
            out.append(model(values).cpu().numpy())
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
        return np.vectorize(lambda v: float(f"{v:.8g}"))(tensor.detach().cpu().numpy()).tolist()

    return {
        "embedding": values(model.embedding.weight),
        "convs": [
            {"kernel_size": k, "weight": values(conv.weight), "bias": values(conv.bias)}
            for k, conv in zip(kernel_sizes, model.convs, strict=True)
        ],
        "dense": {"weight": values(model.dense.weight), "bias": values(model.dense.bias)},
        "out": {"weight": values(model.out.weight), "bias": values(model.out.bias)},
    }


# ---------------------------------------------------------------- v2: deep residual CNN


def build_model_v2(torch, embedding_dim, channels, kernel_size, dilations, hidden, dropout):
    """Char + role embeddings -> stem conv -> residual dilated blocks -> max++mean pool."""
    nn = torch.nn

    class ResidualBlock(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            pad = dilation * (kernel_size - 1) // 2
            self.conv1 = nn.Conv1d(
                channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False
            )
            self.bn1 = nn.BatchNorm1d(channels)
            self.conv2 = nn.Conv1d(
                channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False
            )
            self.bn2 = nn.BatchNorm1d(channels)
            # Each residual branch starts at zero, so a deep stack starts as the identity.
            nn.init.zeros_(self.bn2.weight)

        def forward(self, h):
            branch = torch.relu(self.bn1(self.conv1(h)))
            return torch.relu(h + self.bn2(self.conv2(branch)))

    class DeepCharCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=PAD)
            self.role_embedding = nn.Embedding(ROLE_COUNT, embedding_dim, padding_idx=PAD)
            self.stem = nn.Conv1d(
                embedding_dim, channels, kernel_size, padding=(kernel_size - 1) // 2, bias=False
            )
            self.stem_bn = nn.BatchNorm1d(channels)
            self.blocks = nn.ModuleList(ResidualBlock(d) for d in dilations)
            self.dropout = nn.Dropout(dropout)
            self.dense = nn.Linear(2 * channels, hidden)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x):  # x: (N, 2, L) = character ids, role ids
            e = (self.embedding(x[:, 0]) + self.role_embedding(x[:, 1])).transpose(1, 2)
            h = torch.relu(self.stem_bn(self.stem(e)))
            for block in self.blocks:
                h = block(h)
            pooled = torch.cat([h.amax(dim=2), h.mean(dim=2)], dim=1)
            z = torch.relu(self.dense(self.dropout(pooled)))
            return self.out(self.dropout(z)).squeeze(1)

    return DeepCharCNN()


def bce(logits, labels) -> float:
    """Numerically stable binary cross-entropy on numpy logits."""
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def build_model_v3(torch, embedding_dim, channels, kernel_size, dilations, hidden, dropout):
    """Padding-invariant gated residual CNN with global and domain-aware pooling."""
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            self.depthwise = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                groups=channels,
                padding=dilation * (kernel_size - 1) // 2,
                dilation=dilation,
            )
            self.norm = nn.LayerNorm(channels, eps=NORM_EPS)
            self.gate = nn.Conv1d(channels, 2 * channels, 1)
            self.scale = nn.Parameter(torch.full((channels,), 0.1))
            self.dropout = nn.Dropout(dropout)

        def forward(self, h, mask):
            branch = self.depthwise(h).transpose(1, 2)
            branch = torch.relu(self.norm(branch)).transpose(1, 2) * mask
            branch = torch.nn.functional.glu(self.gate(branch), dim=1)
            return (h + self.dropout(branch) * self.scale[None, :, None]) * mask

    class MaskedCharCNN(nn.Module):
        padding_invariant = True

        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=PAD)
            self.role_embedding = nn.Embedding(ROLE_COUNT, embedding_dim, padding_idx=PAD)
            self.stem = nn.Conv1d(
                embedding_dim, channels, kernel_size, padding=(kernel_size - 1) // 2
            )
            self.norm = nn.LayerNorm(channels, eps=NORM_EPS)
            self.blocks = nn.ModuleList(Block(d) for d in dilations)
            self.attention = nn.Conv1d(channels, 1, 1)
            self.dropout = nn.Dropout(dropout)
            self.dense = nn.Linear(4 * channels, hidden)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x):
            mask = (x[:, 0] != PAD).unsqueeze(1)
            h = (self.embedding(x[:, 0]) + self.role_embedding(x[:, 1])).transpose(1, 2)
            h = self.stem(h * mask).transpose(1, 2)
            h = torch.relu(self.norm(h)).transpose(1, 2) * mask
            for block in self.blocks:
                h = block(h, mask)
            count = mask.sum(2).clamp_min(1)
            maximum = h.masked_fill(~mask, torch.finfo(h.dtype).min).amax(2)
            maximum = torch.where(mask.any(2), maximum, torch.zeros_like(maximum))
            scores = self.attention(h).masked_fill(~mask, torch.finfo(h.dtype).min)
            weights = torch.softmax(scores, dim=2) * mask
            name = (x[:, 1] == ROLE_NAME).unsqueeze(1) & mask
            pooled = torch.cat(
                [
                    maximum,
                    h.sum(2) / count,
                    (h * weights).sum(2),
                    (h * name).sum(2) / name.sum(2).clamp_min(1),
                ],
                dim=1,
            )
            hidden = torch.relu(self.dense(self.dropout(pooled)))
            return self.out(self.dropout(hidden)).squeeze(1)

    return MaskedCharCNN()


def export_v3(model) -> dict:
    def pack(tensor):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
        return base64.b64encode(array.tobytes()).decode("ascii")

    def pair(module):
        return {"weight": pack(module.weight), "bias": pack(module.bias)}

    return {
        "embedding": pack(model.embedding.weight),
        "role_embedding": pack(model.role_embedding.weight),
        "stem": pair(model.stem),
        "norm": pair(model.norm),
        "blocks": [
            {
                "depthwise": pair(b.depthwise),
                "norm": pair(b.norm),
                "gate": pair(b.gate),
                "scale": pack(b.scale),
            }
            for b in model.blocks
        ],
        "attention": pair(model.attention),
        "dense": pair(model.dense),
        "out": pair(model.out),
    }


def train_deep(
    torch,
    model,
    X,
    y,
    Xv,
    yv,
    *,
    epochs,
    batch_size,
    lr,
    weight_decay,
    patience,
    seed,
    class_weight="none",
):
    """AdamW + warmup/cosine LR + clipping; selection uses unweighted validation BCE."""
    generator = torch.Generator().manual_seed(seed)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0}],
        lr=lr,
    )
    steps = math.ceil(len(X) / batch_size)
    total_steps = epochs * steps
    warmup = max(1, int(0.1 * total_steps))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup - 1))
        return 0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    device = next(model.parameters()).device
    positive_weight = float((y == 0).sum() / (y == 1).sum()) if class_weight == "balanced" else 1.0
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    best, stale = (float("inf"), None, 0), 0
    model.training_history = []
    model.positive_weight = positive_weight
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(Xt), generator=generator)
        total, started = 0.0, time.perf_counter()
        for i in range(0, len(order), batch_size):
            index = order[i : i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            values = batch_input(model, Xt[index], device)
            loss = loss_fn(model(values), yt[index].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; lower the learning rate or check data")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(index)
        val_logits = logits_for(torch, model, Xv, batch=batch_size)
        val_loss = bce(val_logits, yv)
        if not math.isfinite(val_loss):
            raise ValueError("Nonfinite validation loss")
        model.training_history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(Xt),
                "validation_loss": val_loss,
                "validation_pr_auc": float(average_precision_score(yv, val_logits)),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            f"epoch {epoch}/{epochs}  train_loss={total / len(Xt):.4f}  "
            f"val_loss={val_loss:.4f}  ({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
        if val_loss < best[0] - 1e-4:
            best, stale = (
                (
                    val_loss,
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    epoch,
                ),
                0,
            )
        else:
            stale += 1
            if patience and stale >= patience:
                print(f"Early stop: no validation improvement for {patience} epochs")
                break
    model.load_state_dict(best[1])
    model.eval()
    print(f"Using epoch {best[2]} (lowest validation loss {best[0]:.4f})")
    return best[2]


def export_v2(model) -> dict:
    """Fold batch norm into the convolutions and pack every tensor as base64 float32."""

    def pack(tensor):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
        return base64.b64encode(array.tobytes()).decode("ascii")

    def fold(conv, bn):
        scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        weight = conv.weight * scale[:, None, None]
        bias = bn.bias - bn.running_mean * scale
        if conv.bias is not None:
            bias = bias + conv.bias * scale
        return {"weight": pack(weight), "bias": pack(bias)}

    return {
        "embedding": pack(model.embedding.weight),
        "role_embedding": pack(model.role_embedding.weight),
        "stem": fold(model.stem, model.stem_bn),
        "blocks": [
            {"conv1": fold(b.conv1, b.bn1), "conv2": fold(b.conv2, b.bn2)} for b in model.blocks
        ],
        "dense": {"weight": pack(model.dense.weight), "bias": pack(model.dense.bias)},
        "out": {"weight": pack(model.out.weight), "bias": pack(model.out.bias)},
    }


def model_features(hosts: list[str], arch: str, max_length: int) -> np.ndarray:
    """(N, L) character ids for v1; (N, 2, L) character + role ids for v2/v3."""
    chars = np.stack([encode(h, max_length) for h in hosts])
    if arch == "v1":
        return chars
    roles = np.stack([encode_roles(h, max_length) for h in hosts])
    return np.stack([chars, roles], axis=1)


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
    """Maximize validation recall under the release confidence bounds.

    If infeasible, retain the FPR bound where possible. Validation failure is still
    reported and prevents release; never silently fall back to an unsafe 0.5 cutoff.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.unique(np.append(probabilities, 1.0))
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

    def bounds(successes, total):
        total = np.maximum(total, 1)
        p, z = successes / total, 1.959963984540054
        center = (p + z * z / (2 * total)) / (1 + z * z / total)
        half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
        return center - half, center + half

    precision_low, _ = bounds(tp, predicted)
    recall_low, _ = bounds(tp, len(positives))
    _, fpr_high = bounds(fp, len(negatives))
    ok = (predicted > 0) & (precision_low >= 0.95) & (recall_low >= 0.90) & (fpr_high <= 0.01)
    if not np.any(ok):
        ok = (fpr_high <= 0.01) if len(negatives) else np.zeros_like(thresholds, dtype=bool)
    if not np.any(ok):
        ok = fpr <= 0.01
    if not np.any(ok):
        return 1.0
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
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
    print(f"Training {args.arch} on {device} with {args.threads} CPU threads", flush=True)

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
        hosts, y = load_csv(
            csv_path, args.url_column, args.label_column, args.legit_label, args.phishing_label
        )
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
            manifest["dataset_version"] = "csv-" + manifest["sha256"][:12]
            manifest["license"] = "Operator-supplied; not independently verified"
            manifest["label_methodology"] = "Explicit CSV labels; reduced to hostname"
            manifest["usage_permission"] = "Supplied by the operator; verify licence and labels."
    if len(hosts) < 80 or len(np.unique(y)) != 2:
        raise ValueError("At least 80 unique hostnames and both classes are required")
    manifest["label_mapping"] = {"legitimate": args.legit_label, "phishing": args.phishing_label}
    manifest["truncated_hostnames"] = sum(len(h) > args.max_length for h in hosts)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "training_config.json").write_text(
        json.dumps(vars(args) | {"resolved_device": device}, indent=2)
    )

    # Subdomains of one registrable domain never straddle splits.
    groups = np.array([registrable_domain(h) or h for h in hosts])
    splits = grouped_splits(y, groups, args.seed)
    (output / "release_targets.json").write_text(json.dumps(RELEASE_TARGETS, indent=2))
    (output / "splits.json").write_text(
        json.dumps({k: [canonical_hash(hosts[i]) for i in v] for k, v in splits.items()})
    )
    X = model_features(hosts, args.arch, args.max_length)
    shortcuts = shortcut_audit(
        simple_features([hosts[i] for i in splits["train"]]), y[splits["train"]]
    )
    if shortcuts:
        print(f"WARNING shortcut findings (training split): {shortcuts}")

    train_idx, val_idx = splits["train"], splits["validation"]
    models, best_epochs = [], []
    for member in range(args.ensemble):
        torch.manual_seed(args.seed + member)
        if args.arch in {"v2", "v3"}:
            builder = build_model_v3 if args.arch == "v3" else build_model_v2
            model = builder(
                torch,
                args.embedding_dim,
                args.channels,
                args.kernel_size,
                args.dilations,
                args.hidden,
                args.dropout,
            )
        else:
            model = build_model(
                torch,
                args.embedding_dim,
                args.filters,
                args.kernel_sizes,
                args.hidden,
                args.dropout,
            )
        model.to(device)
        parameters = sum(p.numel() for p in model.parameters())
        label = f"member {member + 1}/{args.ensemble}: "
        print(f"Training {label}{len(train_idx)} hostnames, {parameters} parameters")
        best_epochs.append(
            train_deep(
                torch,
                model,
                X[train_idx],
                y[train_idx],
                X[val_idx],
                y[val_idx],
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                seed=args.seed + member,
                class_weight=args.class_weight,
            )
        )
        # Keep completed members on CPU so ensembles do not accumulate GPU memory.
        models.append(model.cpu())
        (output / "history.json").write_text(
            json.dumps(
                [
                    {
                        "member": i,
                        "seed": args.seed + i,
                        "best_epoch": best_epochs[i],
                        "positive_weight": m.positive_weight,
                        "epochs": m.training_history,
                    }
                    for i, m in enumerate(models)
                ],
                indent=2,
                allow_nan=False,
            )
        )

    def ensemble_logits(features):
        values = []
        for m in models:
            m.to(device)
            values.append(logits_for(torch, m, features, batch=args.batch_size))
            m.cpu()
        return np.mean(values, axis=0)

    logits = {
        name: ensemble_logits(X[splits[name]]) for name in ("calibration", "validation", "test")
    }
    calibrator = calibrate(logits["calibration"], y[splits["calibration"]])

    def probabilities(name):
        return calibrator.predict_proba(logits[name].reshape(-1, 1))[:, 1]

    threshold = select_threshold(y[splits["validation"]], probabilities("validation"))
    validation_result = metrics(y[splits["validation"]], probabilities("validation"), threshold)
    test_probs = probabilities("test")
    result = metrics(y[splits["test"]], test_probs, threshold)
    gate = (
        "passed"
        if release_gate(result)
        and release_gate(validation_result)
        and not shortcuts
        and manifest["kind"] == "real_labelled"
        else "failed"
    )

    artifact = {
        "alphabet": ALPHABET,
        "release_gate": "failed",  # only promote after export parity succeeds
        "max_length": args.max_length,
        "embedding_dim": args.embedding_dim,
        "hidden": args.hidden,
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "threshold": float(threshold),
        "dataset_manifest": manifest,
        "release_targets": RELEASE_TARGETS,
    }
    if args.arch in {"v2", "v3"}:
        import tldextract

        artifact |= {
            "format": ARTIFACT_FORMAT_V3 if args.arch == "v3" else ARTIFACT_FORMAT_V2,
            "feature_version": FEATURE_VERSION_V3 if args.arch == "v3" else FEATURE_VERSION_V2,
            "model_version": manifest["dataset_version"] + "-charcnn-" + args.arch,
            "channels": args.channels,
            "kernel_size": args.kernel_size,
            "dilations": args.dilations,
            "members": [(export_v3(m) if args.arch == "v3" else export_v2(m)) for m in models],
            "best_epochs": best_epochs,
            "tldextract_version": tldextract.__version__,
        }
    else:
        artifact |= {
            "format": ARTIFACT_FORMAT,
            "feature_version": FEATURE_VERSION,
            "model_version": manifest["dataset_version"] + "-charcnn-v1",
            "filters": args.filters,
            "kernel_sizes": args.kernel_sizes,
            "weights": export_weights(models[0], args.kernel_sizes),
            "best_epoch": best_epochs[0],
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
    parity = float(np.max(np.abs(numpy_logits - ensemble_logits(X[sample]))))
    if parity > 1e-3:
        raise RuntimeError(f"Numpy/PyTorch parity failed: max logit difference {parity}")
    artifact["release_gate"] = gate
    artifact_bytes = json.dumps(artifact, sort_keys=True, allow_nan=False).encode()
    artifact_path.write_bytes(artifact_bytes)
    sha = hashlib.sha256(artifact_bytes).hexdigest()

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
            {"v1": "CNN-HOST", "v2": "CNN2-HOST", "v3": "CNN3-HOST"}[args.arch],
            "char-CNN: hostname characters"
            if args.arch == "v1"
            else f"{args.arch} residual char-CNN x{len(models)}: hostname characters + domain roles",
            result,
            splits,
            manifest["kind"],
            manifest["dataset_version"],
            {
                "num_features": args.max_length,
                "model_family": {
                    "v1": "char_cnn",
                    "v2": "deep_residual_char_cnn",
                    "v3": "masked_gated_char_cnn",
                }[args.arch],
                "parameters": int(sum(p.numel() for m in models for p in m.parameters())),
                "ensemble_members": len(models),
                "best_epochs": best_epochs,
                "inference_latency_ms": latency_ms,
                "latency_scope": "numpy single-hostname inference averaged over test sample",
                "release_gate": gate,
                "numpy_torch_max_logit_difference": parity,
                "validation_metrics": validation_result,
                "validation_gate": "passed" if release_gate(validation_result) else "failed",
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
        "training_config": vars(args) | {"resolved_device": device},
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
    parser.add_argument(
        "--phishing-label", default="0", help="phishing label; all other values fail"
    )
    parser.add_argument("--max-rows", type=int, default=0, help="subsample for quick experiments")
    parser.add_argument(
        "--arch",
        choices=["v1", "v2", "v3"],
        default="v3",
        help="v1: shallow; v2: batch-norm residual; v3 (default): masked gated CNN + attention",
    )
    parser.add_argument("--epochs", type=int, help="maximum epochs (v1: 6; v2/v3: 30)")
    parser.add_argument("--batch-size", type=int, help="v1/v2: 256; v3: 128")
    parser.add_argument("--lr", type=float, help="peak learning rate (v1/v3: 2e-3; v2: 3e-3)")
    parser.add_argument("--max-length", type=int, help="v1/v2: 64; v3: 253 (full hostname)")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, help="(v1: 64, v2: 128)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="optional training-only negative/positive class ratio",
    )
    v1 = parser.add_argument_group("v1 only")
    v1.add_argument("--filters", type=int, default=64)
    v1.add_argument("--kernel-sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    v2 = parser.add_argument_group("v2/v3 architecture and shared optimizer")
    v2.add_argument("--channels", type=int, help="v2: 128; v3: 64")
    v2.add_argument("--kernel-size", type=int, choices=[3, 5, 7])
    v2.add_argument("--dilations", type=int, nargs="+", help="one per residual block")
    v2.add_argument("--ensemble", type=int, default=1, help="independently seeded members (1-8)")
    v2.add_argument("--weight-decay", type=float, default=1e-4)
    v2.add_argument("--patience", type=int, default=5, help="early-stop epochs; 0 disables")
    args = parser.parse_args(argv)
    defaults = {
        "v1": {"epochs": 6, "batch_size": 256, "lr": 2e-3, "hidden": 64, "max_length": 64},
        "v2": {
            "epochs": 30,
            "batch_size": 256,
            "lr": 3e-3,
            "hidden": 128,
            "max_length": 64,
            "channels": 128,
            "kernel_size": 3,
            "dilations": [1, 2, 4, 8, 1, 2, 4, 8],
        },
        "v3": {
            "epochs": 30,
            "batch_size": 128,
            "lr": 2e-3,
            "hidden": 128,
            "max_length": 253,
            "channels": 64,
            "kernel_size": 5,
            "dilations": [1, 2, 4, 8, 16],
        },
    }[args.arch]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if not 1 <= args.ensemble <= 8:
        parser.error("--ensemble must be between 1 and 8")
    if args.arch == "v1" and args.ensemble != 1:
        parser.error("v1 supports only --ensemble 1")
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or args.patience < 0:
        parser.error(
            "epochs, batch size and threads must be positive; patience must be nonnegative"
        )
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be nonnegative and finite")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if (
        not 8 <= args.max_length <= 253
        or not 1 <= args.embedding_dim <= 256
        or not 1 <= args.hidden <= 1024
    ):
        parser.error("unsupported length, embedding or hidden dimensions")
    if args.max_rows < 0 or (args.max_rows and args.max_rows < 80):
        parser.error("--max-rows must be 0 (all) or at least 80")
    if not 0 <= args.seed < 2**32 - 8:
        parser.error("--seed must be between 0 and 2**32 - 9")
    if args.legit_label == args.phishing_label:
        parser.error("--legit-label and --phishing-label must differ")
    if args.arch == "v1":
        if not 1 <= args.filters <= 512 or not all(
            1 <= k <= args.max_length for k in args.kernel_sizes
        ):
            parser.error("unsupported v1 filters or kernel sizes")
    elif not (
        2 <= args.channels <= 512
        and 1 <= len(args.dilations) <= 16
        and all(1 <= d <= 64 for d in args.dilations)
    ):
        parser.error("unsupported channels or dilations")
    return run(args)


if __name__ == "__main__":
    main()

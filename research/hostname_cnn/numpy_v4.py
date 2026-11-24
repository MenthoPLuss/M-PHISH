"""NumPy inference for v4 research artifacts (no PyTorch), mirroring ``models.build_v4``.

Research artifacts use the format ``m-phish-charcnn-v4-research``, which the deployed app
refuses by design. A candidate that wins Phase 10 gets a production format and loader.
"""

from __future__ import annotations

import base64
import math

import numpy as np

from backend.app.services.ml.url_cnn import (
    NORM_EPS,
    PAD,
    ROLE_NAME,
    ROLE_SUBDOMAIN,
    ROLE_SUFFIX,
    conv_same,
)

from .models import ngram_buckets, v4_input

FORMAT = "m-phish-charcnn-v4-research"
FEATURE_VERSION = "hostname-chars-roles-structure-v1"


def _decode(entry) -> np.ndarray:
    shape, packed = entry
    array = np.frombuffer(base64.b64decode(packed, validate=True), dtype="<f4").astype(np.float64)
    if array.size != math.prod(shape) or not np.all(np.isfinite(array)):
        raise ValueError("Invalid tensor in v4 research artifact")
    return array.reshape(shape)


def _norm(x, weight, bias):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(
        x.var(-1, keepdims=True) + NORM_EPS
    ) * weight + bias


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


class V4Numpy:
    def __init__(self, artifact: dict):
        if artifact.get("format") != FORMAT or artifact.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Not a v4 research artifact")
        self.artifact = artifact
        self.config = artifact["config"]
        self.members = [{k: _decode(v) for k, v in m.items()} for m in artifact["members"]]

    def logit(self, text: str) -> float:
        return float(np.mean([self.member_logit(i, text) for i in range(len(self.members))]))

    def member_logit(self, index: int, text: str) -> float:
        c, w = self.config, self.members[index]
        length = max(1, min(c["max_length"], len(text)))
        x = v4_input(text, length)
        chars, roles = x[0], x[1]
        mask = (chars != PAD)[:, None]
        e = w["embedding.weight"][chars] + w["role_embedding.weight"][roles]
        if c["structure_embeddings"]:
            e = e + w["label_embedding.weight"][x[2]] + w["position_embedding.weight"][x[3]]
        h = conv_same(e * mask, w["stem.weight"], w["stem.bias"], 1)
        h = np.maximum(_norm(h, w["norm.weight"], w["norm.bias"]), 0) * mask
        for b, dilation in enumerate(c["dilations"]):
            p = f"blocks.{b}."
            weight, bias = w[p + "depthwise.weight"], w[p + "depthwise.bias"]
            k = weight.shape[-1]
            pad = dilation * (k - 1) // 2
            padded = np.pad(h, ((pad, pad), (0, 0)))
            branch = (
                sum(
                    padded[j * dilation : j * dilation + length] * weight[:, 0, j] for j in range(k)
                )
                + bias
            )
            branch = _norm(branch, w[p + "norm.weight"], w[p + "norm.bias"])
            if c["block"] == "glu":
                branch = np.maximum(branch, 0) * mask
                gated = branch @ w[p + "gate.weight"][:, :, 0].T + w[p + "gate.bias"]
                value, gate = np.split(gated, 2, axis=1)
                branch = value / (1 + np.exp(-np.clip(gate, -700, 700)))
            else:
                up = _gelu(branch @ w[p + "up.weight"].T + w[p + "up.bias"])
                branch = (up @ w[p + "down.weight"].T + w[p + "down.bias"]) * mask
            h = (h + branch * w[p + "scale"]) * mask
        valid = mask[:, 0]

        def masked_max(m):
            return h[m].max(0) if m.any() else np.zeros(h.shape[1])

        def masked_mean(m):
            return h[m].sum(0) / max(1, int(m.sum()))

        scores = h @ w["attention.weight"][:, :, 0].T + w["attention.bias"]  # (L, heads)
        attended = []
        for head in range(scores.shape[1]):
            s = scores[valid, head]
            a = np.exp(s - s.max())
            attended.append((a / a.sum()) @ h[valid])
        parts = [masked_max(valid), masked_mean(valid), *attended]
        if c["component_pools"]:
            for role in (ROLE_SUBDOMAIN, ROLE_NAME, ROLE_SUFFIX):
                m = (roles == role) & valid
                parts += [masked_mean(m), masked_max(m)]
        else:
            parts.append(masked_mean((roles == ROLE_NAME) & valid))
        hidden = np.maximum(w["dense.weight"] @ np.concatenate(parts) + w["dense.bias"], 0)
        logit = float((w["out.weight"] @ hidden + w["out.bias"])[0])
        if c.get("wide_buckets"):  # H12 Wide & Deep path, see models.wide_term
            ids = ngram_buckets(chars, roles, c["wide_max_n"], c["wide_buckets"])
            logit += float(w["wide.weight"][ids, 0].sum()) / math.sqrt(max(1, len(ids)))
        return logit

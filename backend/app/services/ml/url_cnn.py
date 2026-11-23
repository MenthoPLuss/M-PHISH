"""Character-level CNN over the target hostname. Inference only, pure numpy.

Training happens offline in ``train_url_cnn.py`` (PyTorch). The trained weights are
exported to a checksum-pinned JSON artifact, the same deployment rule as the linear
model: no pickle/torch files are ever loaded by the application.

Three artifact formats are supported (architecture must match the training script exactly):

``m-phish-charcnn-v1``
    chars -> embedding(E) -> parallel Conv1d(kernel k, F filters) + ReLU
          -> global max pool -> concat -> Linear(H) + ReLU -> Linear(1) -> logit

``m-phish-charcnn-v2`` (deep residual, optional ensemble)
    chars + domain-structure role of each char (subdomain / name / public suffix)
          -> char embedding + role embedding (E)
          -> stem Conv1d(C, k) + ReLU
          -> N residual blocks: ReLU(x + Conv(d) -> ReLU -> Conv(d)), dilations d = 1, 2, 4, ...
          -> global max pool ++ global mean pool (2C) -> Linear(H) + ReLU -> Linear(1)
    Batch norm is folded into the convolutions at export. Ensemble members' logits are
    averaged. Weights are stored as base64 little-endian float32 inside the JSON.

``m-phish-charcnn-v3`` (masked gated residual CNN)
    Full hostname -> char + role embeddings -> stem -> dilated depthwise/gated blocks
    with per-position layer normalization -> masked max, mean, attention and domain-name
    pooling -> classifier. Padding is masked throughout, including between convolutions.

The logit is Platt-calibrated on a held-out split, and a validation-selected
threshold decides whether the model contributes a warning floor.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from ..observations import host_parts
from .model import Prediction

ARTIFACT_FORMAT = "m-phish-charcnn-v1"
FEATURE_VERSION = "hostname-chars-v1"
ARTIFACT_FORMAT_V2 = "m-phish-charcnn-v2"
FEATURE_VERSION_V2 = "hostname-chars-roles-v1"
ARTIFACT_FORMAT_V3 = "m-phish-charcnn-v3"
FEATURE_VERSION_V3 = "hostname-chars-roles-masked-v1"
NORM_EPS = 1e-5
# Index 0 = padding, 1 = unknown character, then this alphabet.
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-._"
PAD, UNKNOWN = 0, 1
VOCAB_SIZE = len(ALPHABET) + 2
# Role of each character inside the hostname (v2). 0 = padding.
ROLE_SUBDOMAIN, ROLE_NAME, ROLE_SUFFIX = 1, 2, 3
ROLE_COUNT = 4
MAX_ARTIFACT_BYTES = 64_000_000
MAX_MEMBERS = 8
_INDEX = {char: i + 2 for i, char in enumerate(ALPHABET)}


def model_input(hostname: str) -> str:
    """The exact text the CNN sees. Shared by training and inference.

    A leading "www." is dropped: it carries no phishing signal and is a known
    dataset artifact (legitimate samples are often www.* homepages).
    """
    host = (hostname or "").strip().rstrip(".").lower()
    with contextlib.suppress(UnicodeError):
        host = host.encode("idna").decode("ascii")
    return host.removeprefix("www.")


def encode(text: str, max_length: int) -> np.ndarray:
    """Right-aligned indices: long hostnames keep their end (domain + TLD)."""
    indices = [_INDEX.get(char, UNKNOWN) for char in text[-max_length:]]
    encoded = np.full(max_length, PAD, dtype=np.int64)
    if indices:
        encoded[max_length - len(indices) :] = indices
    return encoded


def char_roles(text: str) -> list[int]:
    """Per-character role: subdomain (incl. its trailing dot), name, or public suffix
    (incl. its leading dot). "login.example.co.uk" -> 6 x SUBDOMAIN, 7 x NAME, 6 x SUFFIX.
    """
    subdomain, name, suffix = host_parts(text)
    suffix_length = len(suffix) + 1 if suffix else 0
    if suffix_length > len(text) or not text.endswith(suffix):
        return [ROLE_NAME] * len(text)
    name_length = min(len(name), len(text) - suffix_length)
    subdomain_length = len(text) - suffix_length - name_length
    return (
        [ROLE_SUBDOMAIN] * subdomain_length
        + [ROLE_NAME] * name_length
        + [ROLE_SUFFIX] * suffix_length
    )


def encode_roles(text: str, max_length: int) -> np.ndarray:
    """Right-aligned role indices, aligned position-by-position with ``encode``."""
    roles = char_roles(text)[-max_length:]
    encoded = np.full(max_length, PAD, dtype=np.int64)
    if roles:
        encoded[max_length - len(roles) :] = roles
    return encoded


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-700.0, min(700.0, value))))


def _array(values, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"Invalid shape or nonfinite values for {name}")
    return array


def _packed(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Decode a base64 little-endian float32 tensor and validate its shape and values."""
    if not isinstance(value, str):
        raise ValueError(f"Expected packed float32 data for {name}")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid packed data for {name}") from exc
    if len(raw) != 4 * math.prod(shape):
        raise ValueError(f"Invalid shape or nonfinite values for {name}")
    return _array(np.frombuffer(raw, dtype="<f4").reshape(shape), shape, name)


def conv_same(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, dilation: int) -> np.ndarray:
    """Zero-padded 'same' Conv1d over (L, Cin) -> (L, Cout); matches PyTorch for odd k."""
    cout, cin, k = weight.shape
    pad = dilation * (k - 1) // 2
    length = x.shape[0]
    padded = np.pad(x, ((pad, pad), (0, 0)))
    columns = np.concatenate(
        [padded[j * dilation : j * dilation + length] for j in range(k)], axis=1
    )  # (L, k * Cin), column = j * Cin + c
    return columns @ weight.transpose(2, 1, 0).reshape(k * cin, cout) + bias


class UrlCnnModel:
    def __init__(
        self,
        artifact_path: str | None = None,
        expected_checksum: str | None = None,
        *,
        allow_ungated: bool = False,
    ):
        """``allow_ungated`` is for offline tooling (e.g. parity checks), never the app."""
        self.artifact: dict | None = None
        self.ungated = False
        self.reason = "No URL CNN artifact configured."
        path = artifact_path or os.getenv("ML_CNN_ARTIFACT_PATH")
        checksum = expected_checksum or os.getenv("ML_CNN_ARTIFACT_SHA256")
        if not path:
            return
        try:
            if not checksum or len(checksum) != 64:
                raise ValueError("An explicit artifact SHA-256 is required")
            file = Path(path)
            if file.stat().st_size > MAX_ARTIFACT_BYTES:
                raise ValueError("Artifact exceeds size limit")
            raw = file.read_bytes()
            if hashlib.sha256(raw).hexdigest() != checksum.lower():
                raise ValueError("Artifact checksum mismatch")
            data = json.loads(raw)
            version = {
                ARTIFACT_FORMAT: FEATURE_VERSION,
                ARTIFACT_FORMAT_V2: FEATURE_VERSION_V2,
                ARTIFACT_FORMAT_V3: FEATURE_VERSION_V3,
            }.get(data.get("format"))
            if version is None or data.get("alphabet") != ALPHABET:
                raise ValueError("Incompatible CNN artifact format or alphabet")
            if data.get("feature_version") != version:
                raise ValueError("Incompatible CNN feature version")
            if data.get("release_gate") != "passed":
                # Local experimentation only; never honoured in production.
                allow = os.getenv("ML_CNN_ALLOW_UNGATED", "").lower() == "true"
                if not allow_ungated and (not allow or os.getenv("APP_ENV") == "production"):
                    raise ValueError("Artifact has not passed its scoped detection gates")
                self.ungated = True
            if data["format"] == ARTIFACT_FORMAT_V3:
                self.members = self._load_v3(data)
            elif data["format"] == ARTIFACT_FORMAT_V2:
                self.members = self._load_v2(data)
            else:
                self.weights = self._load_weights(data)
            self._check_calibration(data)
            self.artifact, self.reason = data, None
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
            self.artifact = None
            self.reason = f"URL CNN unavailable: {type(exc).__name__}: {exc}"

    @property
    def feature_version(self) -> str:
        if self.artifact and self.artifact.get("format") == ARTIFACT_FORMAT_V3:
            return FEATURE_VERSION_V3
        if self.artifact and self.artifact.get("format") == ARTIFACT_FORMAT_V2:
            return FEATURE_VERSION_V2
        return FEATURE_VERSION

    @staticmethod
    def _check_calibration(data: dict) -> None:
        for key in ("calibration_slope", "calibration_intercept", "threshold"):
            if not math.isfinite(float(data[key])):
                raise ValueError("Nonfinite calibration parameter")
        if not 0 <= float(data["threshold"]) <= 1:
            raise ValueError("Threshold outside [0, 1]")

    @staticmethod
    def _load_weights(data: dict) -> dict:
        length = int(data["max_length"])
        dim = int(data["embedding_dim"])
        filters = int(data["filters"])
        hidden = int(data["hidden"])
        kernels = [int(k) for k in data["kernel_sizes"]]
        if not (8 <= length <= 253 and 1 <= dim <= 256 and 1 <= filters <= 512):
            raise ValueError("Unsupported CNN dimensions")
        if not (1 <= hidden <= 1024 and kernels and all(1 <= k <= length for k in kernels)):
            raise ValueError("Unsupported CNN dimensions")
        w = data["weights"]
        weights = {
            "embedding": _array(w["embedding"], (VOCAB_SIZE, dim), "embedding"),
            "convs": [
                (
                    _array(conv["weight"], (filters, dim, k), f"conv{k}.weight"),
                    _array(conv["bias"], (filters,), f"conv{k}.bias"),
                )
                for conv, k in zip(w["convs"], kernels, strict=True)
            ],
            "dense": _array(w["dense"]["weight"], (hidden, filters * len(kernels)), "dense"),
            "dense_bias": _array(w["dense"]["bias"], (hidden,), "dense.bias"),
            "out": _array(w["out"]["weight"], (1, hidden), "out"),
            "out_bias": _array(w["out"]["bias"], (1,), "out.bias"),
        }
        return weights

    @staticmethod
    def _load_v2(data: dict) -> list[dict]:
        length = int(data["max_length"])
        dim = int(data["embedding_dim"])
        channels = int(data["channels"])
        k = int(data["kernel_size"])
        hidden = int(data["hidden"])
        dilations = [int(d) for d in data["dilations"]]
        if not (8 <= length <= 253 and 1 <= dim <= 256 and 1 <= channels <= 512):
            raise ValueError("Unsupported CNN dimensions")
        if not (k in (3, 5, 7) and 1 <= hidden <= 1024 and len(dilations) <= 16):
            raise ValueError("Unsupported CNN dimensions")
        if not all(1 <= d <= 64 for d in dilations):
            raise ValueError("Unsupported CNN dimensions")
        members = data["members"]
        if not isinstance(members, list) or not 1 <= len(members) <= MAX_MEMBERS:
            raise ValueError("Unsupported ensemble size")

        def conv(entry, cin, name):
            return (
                _packed(entry["weight"], (channels, cin, k), name + ".weight"),
                _packed(entry["bias"], (channels,), name + ".bias"),
            )

        loaded = []
        for m, member in enumerate(members):
            prefix = f"member{m}."
            blocks = member["blocks"]
            if len(blocks) != len(dilations):
                raise ValueError("Block count does not match dilations")
            loaded.append(
                {
                    "embedding": _packed(member["embedding"], (VOCAB_SIZE, dim), prefix + "emb"),
                    "roles": _packed(member["role_embedding"], (ROLE_COUNT, dim), prefix + "role"),
                    "stem": conv(member["stem"], dim, prefix + "stem"),
                    "blocks": [
                        (
                            conv(block["conv1"], channels, f"{prefix}block{b}.conv1"),
                            conv(block["conv2"], channels, f"{prefix}block{b}.conv2"),
                            d,
                        )
                        for b, (block, d) in enumerate(zip(blocks, dilations, strict=True))
                    ],
                    "dense": _packed(
                        member["dense"]["weight"], (hidden, 2 * channels), prefix + "dense"
                    ),
                    "dense_bias": _packed(
                        member["dense"]["bias"], (hidden,), prefix + "dense.bias"
                    ),
                    "out": _packed(member["out"]["weight"], (1, hidden), prefix + "out"),
                    "out_bias": _packed(member["out"]["bias"], (1,), prefix + "out.bias"),
                }
            )
        return loaded

    @staticmethod
    def _load_v3(data: dict) -> list[dict]:
        def dimension(key, low, high):
            value = data[key]
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Unsupported CNN dimension: {key}")
            return value

        dimension("max_length", 8, 253)
        dim = dimension("embedding_dim", 1, 256)
        channels = dimension("channels", 2, 512)
        hidden = dimension("hidden", 1, 1024)
        k = dimension("kernel_size", 3, 7)
        dilations = data["dilations"]
        if k % 2 == 0 or not isinstance(dilations, list) or not 1 <= len(dilations) <= 16:
            raise ValueError("Unsupported CNN dimensions")
        if any(type(d) is not int or not 1 <= d <= 64 for d in dilations):
            raise ValueError("Unsupported CNN dilations")
        members = data["members"]
        if not isinstance(members, list) or not 1 <= len(members) <= MAX_MEMBERS:
            raise ValueError("Unsupported ensemble size")

        def pair(entry, weight_shape, bias_shape, name):
            return (
                _packed(entry["weight"], weight_shape, name + ".weight"),
                _packed(entry["bias"], bias_shape, name + ".bias"),
            )

        loaded = []
        for member in members:
            if len(member["blocks"]) != len(dilations):
                raise ValueError("Block count does not match dilations")
            loaded.append(
                {
                    "embedding": _packed(member["embedding"], (VOCAB_SIZE, dim), "embedding"),
                    "roles": _packed(member["role_embedding"], (ROLE_COUNT, dim), "roles"),
                    "stem": pair(member["stem"], (channels, dim, k), (channels,), "stem"),
                    "norm": pair(member["norm"], (channels,), (channels,), "norm"),
                    "blocks": [
                        {
                            "depthwise": pair(b["depthwise"], (channels, 1, k), (channels,), "dw"),
                            "norm": pair(b["norm"], (channels,), (channels,), "norm"),
                            "gate": pair(
                                b["gate"], (2 * channels, channels, 1), (2 * channels,), "gate"
                            ),
                            "scale": _packed(b["scale"], (channels,), "scale"),
                            "dilation": d,
                        }
                        for b, d in zip(member["blocks"], dilations, strict=True)
                    ],
                    "attention": pair(member["attention"], (1, channels, 1), (1,), "attention"),
                    "dense": pair(member["dense"], (hidden, 4 * channels), (hidden,), "dense"),
                    "out": pair(member["out"], (1, hidden), (1,), "out"),
                }
            )
        return loaded

    def logit(self, text: str) -> float:
        """Forward pass identical to the PyTorch module(s) used in training."""
        if self.artifact.get("format") in (ARTIFACT_FORMAT_V2, ARTIFACT_FORMAT_V3):
            return float(np.mean([self.member_logit(m, text) for m in range(len(self.members))]))
        weights, length = self.weights, int(self.artifact["max_length"])
        x = weights["embedding"][encode(text, length)]  # (L, E)
        pooled = []
        for weight, bias in weights["convs"]:
            k = weight.shape[2]
            windows = np.lib.stride_tricks.sliding_window_view(x, k, axis=0)  # (L-k+1, E, k)
            feature_map = np.einsum("tek,fek->tf", windows, weight) + bias
            pooled.append(np.maximum(feature_map, 0).max(axis=0))
        hidden = np.maximum(weights["dense"] @ np.concatenate(pooled) + weights["dense_bias"], 0)
        return float((weights["out"] @ hidden + weights["out_bias"])[0])

    def member_logit(self, index: int, text: str) -> float:
        """One deep ensemble member (used directly by the training parity check)."""
        if self.artifact["format"] == ARTIFACT_FORMAT_V3:
            return self._member_logit_v3(index, text)
        w, length = self.members[index], int(self.artifact["max_length"])
        h = w["embedding"][encode(text, length)] + w["roles"][encode_roles(text, length)]
        h = np.maximum(conv_same(h, *w["stem"], 1), 0)
        for (w1, b1), (w2, b2), dilation in w["blocks"]:
            r = np.maximum(conv_same(h, w1, b1, dilation), 0)
            h = np.maximum(h + conv_same(r, w2, b2, dilation), 0)
        pooled = np.concatenate([h.max(axis=0), h.mean(axis=0)])
        hidden = np.maximum(w["dense"] @ pooled + w["dense_bias"], 0)
        return float((w["out"] @ hidden + w["out_bias"])[0])

    def _member_logit_v3(self, index: int, text: str) -> float:
        w, length = self.members[index], int(self.artifact["max_length"])
        length = max(1, min(length, len(text)))
        chars, roles = encode(text, length), encode_roles(text, length)
        mask = (chars != PAD)[:, None]

        def norm(x, params):
            weight, bias = params
            return (x - x.mean(-1, keepdims=True)) / np.sqrt(
                x.var(-1, keepdims=True) + NORM_EPS
            ) * weight + bias

        h = (w["embedding"][chars] + w["roles"][roles]) * mask
        h = np.maximum(norm(conv_same(h, *w["stem"], 1), w["norm"]), 0) * mask
        for block in w["blocks"]:
            weight, bias = block["depthwise"]
            k, dilation = weight.shape[-1], block["dilation"]
            pad = dilation * (k - 1) // 2
            padded = np.pad(h, ((pad, pad), (0, 0)))
            branch = (
                sum(
                    padded[j * dilation : j * dilation + length] * weight[:, 0, j] for j in range(k)
                )
                + bias
            )
            branch = np.maximum(norm(branch, block["norm"]), 0) * mask
            weight, bias = block["gate"]
            value, gate = np.split(branch @ weight[:, :, 0].T + bias, 2, axis=1)
            branch = value / (1 + np.exp(-np.clip(gate, -700, 700)))
            h = (h + branch * block["scale"]) * mask
        valid = h[mask[:, 0]]
        if not len(valid):
            pooled = np.zeros(4 * h.shape[1])
        else:
            weight, bias = w["attention"]
            scores = valid @ weight[0, :, 0] + bias[0]
            attention = np.exp(scores - scores.max())
            attention /= attention.sum()
            name = h[(roles == ROLE_NAME) & mask[:, 0]]
            pooled = np.concatenate(
                [
                    valid.max(0),
                    valid.mean(0),
                    attention @ valid,
                    name.mean(0) if len(name) else np.zeros(h.shape[1]),
                ]
            )
        weight, bias = w["dense"]
        hidden = np.maximum(weight @ pooled + bias, 0)
        weight, bias = w["out"]
        return float((weight @ hidden + bias)[0])

    def predict_hostname(self, hostname: str) -> Prediction:
        text = model_input(hostname)
        if self.artifact is None:
            return Prediction(model_name="url-char-cnn", reason=self.reason)
        if not text:
            return Prediction(model_name="url-char-cnn", reason="No hostname to evaluate.")
        artifact = self.artifact
        margin = self.logit(text)
        calibrated = _sigmoid(
            margin * float(artifact["calibration_slope"]) + float(artifact["calibration_intercept"])
        )
        threshold = float(artifact["threshold"])
        reason = None
        if self.ungated:
            reason = "Experimental: this artifact did not pass its release gate (local use only)."
        return Prediction(
            risk=calibrated,
            model_name="url-char-cnn",
            model_version=str(artifact.get("model_version", "unknown")),
            feature_version=self.feature_version,
            calibrated_probability=calibrated,
            raw_probability=_sigmoid(margin),
            calibration_method="held-out-platt",
            is_phishing=calibrated >= threshold,
            availability="available",
            reason=reason,
            attribution_scale=None,
            base_value=None,
            threshold=threshold,
        )


_global_model: UrlCnnModel | None = None


def get_url_cnn() -> UrlCnnModel:
    global _global_model
    if _global_model is None:
        _global_model = UrlCnnModel()
    return _global_model

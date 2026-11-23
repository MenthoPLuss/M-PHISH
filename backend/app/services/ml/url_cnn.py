"""Character-level CNN over the target hostname. Inference only, pure numpy.

Training happens offline in ``train_url_cnn.py`` (PyTorch). The trained weights are
exported to a checksum-pinned JSON artifact, the same deployment rule as the linear
model: no pickle/torch files are ever loaded by the application.

Architecture (must match the training script exactly):
    chars -> embedding(E) -> parallel Conv1d(kernel k, F filters) + ReLU
          -> global max pool -> concat -> Linear(H) + ReLU -> Linear(1) -> logit
The logit is Platt-calibrated on a held-out split, and a validation-selected
threshold decides whether the model contributes a warning floor.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .model import Prediction

ARTIFACT_FORMAT = "m-phish-charcnn-v1"
FEATURE_VERSION = "hostname-chars-v1"
# Index 0 = padding, 1 = unknown character, then this alphabet.
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-._"
PAD, UNKNOWN = 0, 1
VOCAB_SIZE = len(ALPHABET) + 2
MAX_ARTIFACT_BYTES = 8_000_000
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


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-700.0, min(700.0, value))))


def _array(values, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"Invalid shape or nonfinite values for {name}")
    return array


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
            if data.get("format") != ARTIFACT_FORMAT or data.get("alphabet") != ALPHABET:
                raise ValueError("Incompatible CNN artifact format or alphabet")
            if data.get("feature_version") != FEATURE_VERSION:
                raise ValueError("Incompatible CNN feature version")
            if data.get("release_gate") != "passed":
                # Local experimentation only; never honoured in production.
                allow = os.getenv("ML_CNN_ALLOW_UNGATED", "").lower() == "true"
                if not allow_ungated and (not allow or os.getenv("APP_ENV") == "production"):
                    raise ValueError("Artifact has not passed its scoped detection gates")
                self.ungated = True
            self.weights = self._load_weights(data)
            self.artifact, self.reason = data, None
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.artifact = None
            self.reason = f"URL CNN unavailable: {type(exc).__name__}: {exc}"

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
        for key in ("calibration_slope", "calibration_intercept", "threshold"):
            if not math.isfinite(float(data[key])):
                raise ValueError("Nonfinite calibration parameter")
        if not 0 <= float(data["threshold"]) <= 1:
            raise ValueError("Threshold outside [0, 1]")
        return weights

    def logit(self, text: str) -> float:
        """Forward pass identical to the PyTorch module used in training."""
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
            feature_version=FEATURE_VERSION,
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

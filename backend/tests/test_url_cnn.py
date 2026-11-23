import hashlib
import json

import numpy as np
import pytest

from backend.app.services.ml import url_cnn
from backend.app.services.ml.model import Prediction
from backend.app.services.ml.url_cnn import (
    ALPHABET,
    ARTIFACT_FORMAT,
    FEATURE_VERSION,
    VOCAB_SIZE,
    UrlCnnModel,
    encode,
    model_input,
)

LENGTH, DIM, FILTERS, KERNELS, HIDDEN = 12, 4, 3, [2, 3], 5


def artifact(gate="passed", threshold=0.5, seed=0):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.normal(0, 0.5, shape).round(6).tolist()

    return {
        "format": ARTIFACT_FORMAT,
        "feature_version": FEATURE_VERSION,
        "alphabet": ALPHABET,
        "model_version": "test",
        "release_gate": gate,
        "max_length": LENGTH,
        "embedding_dim": DIM,
        "filters": FILTERS,
        "kernel_sizes": KERNELS,
        "hidden": HIDDEN,
        "weights": {
            "embedding": r(VOCAB_SIZE, DIM),
            "convs": [
                {"kernel_size": k, "weight": r(FILTERS, DIM, k), "bias": r(FILTERS)}
                for k in KERNELS
            ],
            "dense": {"weight": r(HIDDEN, FILTERS * len(KERNELS)), "bias": r(HIDDEN)},
            "out": {"weight": r(1, HIDDEN), "bias": r(1)},
        },
        "calibration_slope": 1.0,
        "calibration_intercept": 0.0,
        "threshold": threshold,
    }


def write(tmp_path, data):
    raw = json.dumps(data).encode()
    path = tmp_path / "url_cnn.json"
    path.write_bytes(raw)
    return str(path), hashlib.sha256(raw).hexdigest()


def reference_logit(data, text):
    """Deliberately naive loops: an independent check of the vectorized forward pass."""
    w = data["weights"]
    x = [w["embedding"][i] for i in encode(text, data["max_length"])]
    pooled = []
    for conv, k in zip(w["convs"], data["kernel_sizes"], strict=True):
        for f in range(data["filters"]):
            best = 0.0
            for t in range(data["max_length"] - k + 1):
                total = conv["bias"][f]
                for e in range(data["embedding_dim"]):
                    for j in range(k):
                        total += conv["weight"][f][e][j] * x[t + j][e]
                best = max(best, total)
            pooled.append(best)
    hidden = [
        max(0.0, sum(a * b for a, b in zip(row, pooled, strict=True)) + bias)
        for row, bias in zip(w["dense"]["weight"], w["dense"]["bias"], strict=True)
    ]
    return (
        sum(a * b for a, b in zip(w["out"]["weight"][0], hidden, strict=True)) + w["out"]["bias"][0]
    )


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "ML_CNN_ARTIFACT_PATH",
        "ML_CNN_ARTIFACT_SHA256",
        "ML_CNN_ALLOW_UNGATED",
        "APP_ENV",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(url_cnn, "_global_model", None)


def test_preprocessing_matches_production_hostnames():
    assert model_input("WWW.PayPal-Login.COM.") == "paypal-login.com"
    assert model_input("bücher.example") == "xn--bcher-kva.example"
    encoded = encode("ab", 5)
    assert encoded.tolist() == [0, 0, 0, 2, 3]  # right-aligned, 0 = padding
    assert encode("a" * 20, 5).tolist() == [2] * 5  # keeps the end of long hostnames
    assert encode("a$", 3).tolist()[-1] == 1  # unknown character


def test_unconfigured_model_is_unavailable():
    prediction = UrlCnnModel().predict_hostname("example.com")
    assert prediction.availability == "unavailable"
    assert prediction.calibrated_probability is None


def test_checksum_and_shape_are_enforced(tmp_path):
    path, sha = write(tmp_path, artifact())
    assert UrlCnnModel(path, "0" * 64).artifact is None
    broken = artifact()
    broken["weights"]["dense"]["bias"] = broken["weights"]["dense"]["bias"][:-1]
    path, sha = write(tmp_path, broken)
    model = UrlCnnModel(path, sha)
    assert model.artifact is None and "Invalid shape" in model.reason


def test_release_gate_and_production_lock(tmp_path, monkeypatch):
    path, sha = write(tmp_path, artifact(gate="failed"))
    assert UrlCnnModel(path, sha).artifact is None
    monkeypatch.setenv("ML_CNN_ALLOW_UNGATED", "true")
    experimental = UrlCnnModel(path, sha).predict_hostname("example.com")
    assert experimental.availability == "available"
    assert "Experimental" in experimental.reason
    monkeypatch.setenv("APP_ENV", "production")
    assert UrlCnnModel(path, sha).artifact is None


def test_numpy_forward_pass_matches_reference(tmp_path):
    data = artifact()
    path, sha = write(tmp_path, data)
    model = UrlCnnModel(path, sha)
    for host in ["example.com", "secure-paypa1-verify.xyz", "a.b", "x" * 40]:
        text = model_input(host)
        assert model.logit(text) == pytest.approx(reference_logit(data, text), abs=1e-9)
    prediction = model.predict_hostname("www.example.com")
    assert prediction.model_name == "url-char-cnn"
    assert 0 <= prediction.calibrated_probability <= 1


def test_cnn_raises_floor_only_when_above_threshold(tmp_path, monkeypatch):
    from backend.app.analysis import build_report

    path, sha = write(tmp_path, artifact(threshold=0.0))  # always above threshold
    monkeypatch.setenv("ML_CNN_ARTIFACT_PATH", path)
    monkeypatch.setenv("ML_CNN_ARTIFACT_SHA256", sha)
    report = build_report("https://example.com", fixture_html="<html></html>", remote=False)
    assert report["url_cnn_prediction"]["availability"] == "available"
    assert report["risk_score"] >= 50
    assert any(e["source"] == "url_cnn" for e in report["evidence"])

    monkeypatch.setattr(url_cnn, "_global_model", None)
    path, sha = write(tmp_path, artifact(threshold=1.0))  # never above threshold
    monkeypatch.setenv("ML_CNN_ARTIFACT_SHA256", sha)
    report = build_report("https://example.com", fixture_html="<html></html>", remote=False)
    assert report["risk_score"] == 0
    assert not any(e["source"] == "url_cnn" for e in report["evidence"])


def test_report_without_cnn_is_unchanged():
    from backend.app.analysis import build_report

    report = build_report("https://example.com", fixture_html="<html></html>", remote=False)
    assert report["url_cnn_prediction"]["availability"] == "unavailable"
    assert report["risk_score"] == 0


def test_multimodal_model_keeps_priority():
    from backend.app.analysis import select_warning_model

    linear_warns = Prediction(calibrated_probability=0.9, threshold=0.5)
    linear_quiet = Prediction(calibrated_probability=0.1, threshold=0.5)
    cnn_warns = Prediction(calibrated_probability=0.8, threshold=0.5)
    cnn_quiet = Prediction(calibrated_probability=0.2, threshold=0.5)
    assert select_warning_model(linear_warns, cnn_warns) is linear_warns
    assert select_warning_model(linear_quiet, cnn_warns) is cnn_warns
    assert select_warning_model(linear_quiet, cnn_quiet) is linear_quiet
    assert select_warning_model(Prediction(), Prediction()).calibrated_probability is None

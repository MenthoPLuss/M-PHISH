"""Training/export regressions, including padding invariance and deployment gates."""

import base64
import hashlib
import json

import numpy as np
import pytest

from backend.app.services.ml import train_url_cnn as training
from backend.app.services.ml import url_cnn
from backend.app.services.ml.offline import metrics, release_gate


def v3_artifact(members, length=253):
    return {
        "format": url_cnn.ARTIFACT_FORMAT_V3,
        "feature_version": url_cnn.FEATURE_VERSION_V3,
        "alphabet": url_cnn.ALPHABET,
        "release_gate": "passed",
        "max_length": length,
        "embedding_dim": 4,
        "channels": 6,
        "kernel_size": 5,
        "dilations": [1, 4],
        "hidden": 8,
        "members": members,
        "calibration_slope": 1.2,
        "calibration_intercept": -0.2,
        "threshold": 0.8,
    }


def load_artifact(tmp_path, artifact):
    raw = json.dumps(artifact).encode()
    path = tmp_path / "cnn.json"
    path.write_bytes(raw)
    return url_cnn.UrlCnnModel(str(path), hashlib.sha256(raw).hexdigest())


def random_member():
    rng = np.random.default_rng(4)

    def pack(shape):
        return base64.b64encode(rng.normal(0, 0.2, shape).astype("<f4").tobytes()).decode()

    def pair(shape, bias):
        return {"weight": pack(shape), "bias": pack(bias)}

    return {
        "embedding": pack((url_cnn.VOCAB_SIZE, 4)),
        "role_embedding": pack((url_cnn.ROLE_COUNT, 4)),
        "stem": pair((6, 4, 5), (6,)),
        "norm": pair((6,), (6,)),
        "blocks": [
            {
                "depthwise": pair((6, 1, 5), (6,)),
                "norm": pair((6,), (6,)),
                "gate": pair((12, 6, 1), (12,)),
                "scale": pack((6,)),
            }
            for _ in range(2)
        ],
        "attention": pair((1, 6, 1), (1,)),
        "dense": pair((8, 24), (8,)),
        "out": pair((1, 8), (1,)),
    }


def test_v3_runtime_without_torch_and_invalid_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("ML_CNN_ALLOW_UNGATED", raising=False)
    data = v3_artifact([random_member(), random_member()])
    loaded = load_artifact(tmp_path, data)
    assert loaded.artifact is not None, loaded.reason
    assert np.isfinite(loaded.logit(""))
    prediction = loaded.predict_hostname("www.example.com")
    assert prediction.feature_version == url_cnn.FEATURE_VERSION_V3
    assert prediction.availability == "available"
    assert 0 <= prediction.calibrated_probability <= 1
    assert loaded.logit("example.com") == pytest.approx(loaded.member_logit(0, "example.com"))
    for override in [
        {"kernel_size": 4},
        {"dilations": []},
        {"dilations": [1, 65]},
        {"channels": 6.5},
        {"members": []},
        {"members": [None]},
        {"release_gate": "failed"},
        {"feature_version": url_cnn.FEATURE_VERSION_V2},
        {"threshold": float("nan")},
    ]:
        assert load_artifact(tmp_path, data | override).artifact is None, override
    data["members"][0]["attention"]["weight"] = "broken base64!"
    assert load_artifact(tmp_path, data).artifact is None
    data = v3_artifact([random_member()])
    data["members"][0]["blocks"][0]["scale"] = base64.b64encode(
        np.full(6, np.nan, dtype="<f4").tobytes()
    ).decode()
    assert load_artifact(tmp_path, data).artifact is None


def test_v3_torch_export_parity_and_padding_invariance(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    torch.manual_seed(18)
    models = [training.build_model_v3(torch, 4, 6, 5, [1, 4], 8, 0.2) for _ in range(2)]
    texts = ["example.com", "a.b.github.io", "x", "", "a." * 115 + "example.co.uk"]
    inputs = torch.from_numpy(training.model_features(texts, "v3", 253))
    expected = []
    for model in models:
        model.train()
        # Exercise learned parameters rather than only initialization.
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(inputs), torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0])
        )
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            full = model(inputs).numpy()
            expected.append(full)
            short = torch.from_numpy(training.model_features(texts[:4], "v3", 32))
            np.testing.assert_allclose(model(short).numpy(), full[:4], atol=2e-6)
            # Other examples in the batch must not change an example's prediction.
            for text, value in zip(texts, full, strict=True):
                alone = torch.from_numpy(training.model_features([text], "v3", 253))
                assert float(model(alone)[0]) == pytest.approx(float(value), abs=2e-6)
    loaded = load_artifact(tmp_path, v3_artifact([training.export_v3(m) for m in models]))
    assert loaded.artifact is not None, loaded.reason
    for text, value in zip(texts, np.mean(expected, axis=0), strict=True):
        assert loaded.logit(text) == pytest.approx(float(value), abs=2e-5)
    # Configured short-context variants still export the same truncation semantics.
    loaded = load_artifact(tmp_path, v3_artifact([training.export_v3(models[0])], length=32))
    with torch.no_grad():
        value = models[0](torch.from_numpy(training.model_features([texts[-1]], "v3", 32)))
    assert loaded.logit(texts[-1]) == pytest.approx(float(value[0]), abs=2e-5)


def test_csv_rejects_unknown_labels_and_removes_conflicts(tmp_path):
    path = tmp_path / "urls.csv"
    path.write_text(
        "URL,label\nhttps://www.example.com/a,good\nexample.com,good\n"
        "conflict.com,good\nconflict.com,bad\nphishing.test,bad\n"
    )
    hosts, labels = training.load_csv(path, "URL", "label", "good", "bad")
    assert hosts == ["example.com", "phishing.test"]
    assert labels.tolist() == [0, 1]
    for invalid in ["", "unknown", "2"]:
        path.write_text(f"URL,label\nexample.com,{invalid}\n")
        with pytest.raises(ValueError, match="Unknown label"):
            training.load_csv(path, "URL", "label", "1", "0")


def test_threshold_respects_confidence_bounds_and_fpr_fallback():
    labels = np.array([0] * 2000 + [1] * 2000)
    probabilities = np.array([0.1] * 2000 + [0.9] * 2000)
    threshold = training.select_threshold(labels, probabilities)
    assert release_gate(metrics(labels, probabilities, threshold))
    # No threshold can meet recall, but do not revert to 0.5 and trigger every negative.
    probabilities = np.array([0.6] * 2000 + [0.7] * 100 + [0.4] * 1900)
    threshold = training.select_threshold(labels, probabilities)
    result = metrics(labels, probabilities, threshold)
    assert result["false_positive_rate"] == 0
    assert result["recall"] == 0.05
    assert not release_gate(result)


@pytest.mark.parametrize(
    "options",
    [
        ["--epochs", "0"],
        ["--batch-size", "0"],
        ["--dropout", "nan"],
        ["--dilations", "0"],
        ["--max-length", "300"],
        ["--lr", "inf"],
        ["--arch", "v1", "--ensemble", "2"],
        ["--phishing-label", "1"],
        ["--char-dropout", "1"],
        ["--label-smoothing", "-0.1"],
        ["--ema-decay", "1.5"],
        ["--reuse-splits", "missing-splits.json"],
    ],
)
def test_bad_configuration_fails_before_output_creation(tmp_path, options):
    output = tmp_path / "run"
    with pytest.raises(SystemExit):
        training.main(["--synthetic-smoke", "--output", str(output), *options])
    assert not output.exists()


def test_character_dropout_keeps_padding_and_roles():
    torch = pytest.importorskip("torch")
    hosts = ["login.example.co.uk", "a.b", "paypal-secure-verify.xyz"] * 200
    values = torch.from_numpy(training.model_features(hosts, "v3", 40))
    augmented = training.augment_characters(torch, values, 0.3, torch.Generator().manual_seed(0))
    chars, new_chars = values[:, 0], augmented[:, 0]
    assert torch.equal(augmented[:, 1], values[:, 1])  # roles untouched
    assert torch.equal(new_chars == url_cnn.PAD, chars == url_cnn.PAD)  # padding untouched
    changed = new_chars != chars
    assert torch.all(new_chars[changed] == url_cnn.UNKNOWN)
    rate = changed.sum().item() / (chars != url_cnn.PAD).sum().item()
    assert 0.25 < rate < 0.35
    assert torch.equal(values[:, 0], chars)  # the input batch itself is not modified


def test_regularised_training_is_reproducible_and_exports_the_average():
    torch = pytest.importorskip("torch")
    hosts, y = training.synthetic_hostnames(160, 1)
    X = training.model_features(hosts, "v3", 32)

    def fit(**recipe):
        torch.manual_seed(5)
        model = training.build_model_v3(torch, 4, 6, 3, [1, 2], 8, 0.1)
        training.train_deep(
            torch,
            model,
            X[:120],
            y[:120],
            X[120:],
            y[120:],
            epochs=3,
            batch_size=16,
            lr=3e-3,
            weight_decay=1e-4,
            patience=0,
            seed=5,
            **recipe,
        )
        return model

    recipe = {"char_dropout": 0.1, "label_smoothing": 0.05, "ema_decay": 0.9}
    first, second, plain = fit(**recipe), fit(**recipe), fit(char_dropout=0.1, label_smoothing=0.05)
    assert first.training_recipe == recipe
    pairs = list(zip(first.parameters(), second.parameters(), strict=True))
    assert all(torch.equal(a, b) for a, b in pairs)  # same seed and recipe -> identical weights
    averaged = list(zip(first.parameters(), plain.parameters(), strict=True))
    assert any(not torch.equal(a, c) for a, c in averaged)  # EMA weights are what is kept
    with pytest.raises(ValueError, match="must be in"):
        fit(ema_decay=1.0)


def test_one_step_training_export_history_and_synthetic_gate(tmp_path):
    pytest.importorskip("torch")
    output = tmp_path / "run"
    report = training.main(
        [
            "--synthetic-smoke",
            "--output",
            str(output),
            "--max-rows",
            "120",
            "--epochs",
            "1",
            "--batch-size",
            "256",
            "--channels",
            "6",
            "--hidden",
            "8",
            "--embedding-dim",
            "4",
            "--dilations",
            "1",
            "2",
            "--threads",
            "2",
            "--class-weight",
            "balanced",
            "--device",
            "cpu",
        ]
    )
    result = report["experiments"][0]
    assert result["release_gate"] == "failed"
    assert result["numpy_torch_max_logit_difference"] < 1e-3
    assert result["experiment_id"] == "CNN3-HOST"
    assert not url_cnn.UrlCnnModel(str(output / "url_cnn.json"), report["model_sha256"]).artifact
    history = json.loads((output / "history.json").read_text())[0]
    assert history["best_epoch"] == 1
    assert history["epochs"][0]["validation_loss"] > 0
    assert history["positive_weight"] > 0
    # v3 trains with the validated regularisation recipe unless told otherwise.
    recipe = training.REGULARISER_DEFAULTS_V3
    config = json.loads((output / "training_config.json").read_text())
    assert {name: config[name] for name in recipe} == recipe
    assert history["training_recipe"] == recipe
    for line in (output / "checksums.txt").read_text().splitlines():
        checksum, name = line.split("  ")
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum
    split_ids = list(json.loads((output / "splits.json").read_text()).values())
    assert len(set(sum(split_ids, []))) == sum(map(len, split_ids))
    with pytest.raises(ValueError, match="Output folder must be empty"):
        training.main(["--synthetic-smoke", "--output", str(output)])

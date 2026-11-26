import base64
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


# ---------------------------------------------------------------- v2 (deep residual)

V2 = {"max_length": 12, "embedding_dim": 4, "channels": 3, "kernel_size": 3, "hidden": 5}
V2_DILATIONS = [1, 2]


def packed(array):
    return base64.b64encode(np.asarray(array, dtype="<f4").tobytes()).decode("ascii")


def v2_member(rng):
    e, c, k, h = V2["embedding_dim"], V2["channels"], V2["kernel_size"], V2["hidden"]

    def r(*shape):
        return rng.normal(0, 0.5, shape).astype(np.float32)

    def conv(cin):
        return {"weight": r(c, cin, k), "bias": r(c)}

    return {
        "embedding": r(VOCAB_SIZE, e),
        "role_embedding": r(url_cnn.ROLE_COUNT, e),
        "stem": conv(e),
        "blocks": [{"conv1": conv(c), "conv2": conv(c)} for _ in V2_DILATIONS],
        "dense": {"weight": r(h, 2 * c), "bias": r(h)},
        "out": {"weight": r(1, h), "bias": r(1)},
    }


def pack_member(member):
    if isinstance(member, dict):
        return {key: pack_member(value) for key, value in member.items()}
    if isinstance(member, list):
        return [pack_member(value) for value in member]
    return packed(member)


def v2_artifact(members, gate="passed", threshold=0.5):
    return V2 | {
        "format": url_cnn.ARTIFACT_FORMAT_V2,
        "feature_version": url_cnn.FEATURE_VERSION_V2,
        "alphabet": ALPHABET,
        "model_version": "test-v2",
        "release_gate": gate,
        "dilations": V2_DILATIONS,
        "members": [pack_member(m) for m in members],
        "calibration_slope": 1.0,
        "calibration_intercept": 0.0,
        "threshold": threshold,
    }


def reference_v2_logit(member, text):
    """Deliberately naive loops: an independent check of the vectorized v2 forward pass."""
    length, k = V2["max_length"], V2["kernel_size"]
    chars, roles = encode(text, length), url_cnn.encode_roles(text, length)
    emb, role = member["embedding"], member["role_embedding"]
    x = [
        [float(emb[chars[t]][i] + role[roles[t]][i]) for i in range(len(emb[0]))]
        for t in range(length)
    ]

    def conv(rows, w, b, d):
        pad = d * (k - 1) // 2
        out = []
        for t in range(length):
            out.append([])
            for o in range(len(b)):
                total = float(b[o])
                for j in range(k):
                    s = t + j * d - pad
                    if 0 <= s < length:
                        total += sum(float(w[o][i][j]) * rows[s][i] for i in range(len(rows[0])))
                out[-1].append(total)
        return out

    def relu(rows):
        return [[max(0.0, v) for v in row] for row in rows]

    h = relu(conv(x, member["stem"]["weight"], member["stem"]["bias"], 1))
    for block, d in zip(member["blocks"], V2_DILATIONS, strict=True):
        r = relu(conv(h, block["conv1"]["weight"], block["conv1"]["bias"], d))
        r = conv(r, block["conv2"]["weight"], block["conv2"]["bias"], d)
        h = relu(
            [[a + b for a, b in zip(hr, rr, strict=True)] for hr, rr in zip(h, r, strict=True)]
        )
    channels = range(V2["channels"])
    pooled = [max(row[c] for row in h) for c in channels]
    pooled += [sum(row[c] for row in h) / length for c in channels]
    dense, out = member["dense"], member["out"]
    hidden = [
        max(0.0, sum(float(a) * b for a, b in zip(w, pooled, strict=True)) + float(bias))
        for w, bias in zip(dense["weight"], dense["bias"], strict=True)
    ]
    return sum(float(a) * b for a, b in zip(out["weight"][0], hidden, strict=True)) + float(
        out["bias"][0]
    )


def test_roles_mark_subdomain_name_and_suffix():
    sub, name, suffix = url_cnn.ROLE_SUBDOMAIN, url_cnn.ROLE_NAME, url_cnn.ROLE_SUFFIX
    assert url_cnn.char_roles("login.example.co.uk") == [sub] * 6 + [name] * 7 + [suffix] * 6
    assert url_cnn.char_roles("example.com") == [name] * 7 + [suffix] * 4
    assert url_cnn.char_roles("1.2.3.4") == [name] * 7  # IP address: no suffix
    assert url_cnn.char_roles("localhost") == [name] * 9
    # Private PSL suffixes (shared hosting such as github.io) count as the suffix.
    assert url_cnn.char_roles("a.b.github.io") == [sub] * 2 + [name] + [suffix] * 10
    roles = url_cnn.encode_roles("example.com", 14).tolist()
    assert roles[:3] == [0, 0, 0] and roles[3:] == [name] * 7 + [suffix] * 4


def test_v2_forward_pass_and_ensemble_match_reference(tmp_path):
    rng = np.random.default_rng(1)
    members = [v2_member(rng), v2_member(rng)]
    path, sha = write(tmp_path, v2_artifact(members))
    model = UrlCnnModel(path, sha)
    assert model.artifact is not None, model.reason
    for host in ["example.com", "login.secure-paypa1.co.uk", "a.b", "x" * 40]:
        text = model_input(host)
        expected = [reference_v2_logit(m, text) for m in members]
        for index, value in enumerate(expected):
            assert model.member_logit(index, text) == pytest.approx(value, abs=1e-6)
        assert model.logit(text) == pytest.approx(float(np.mean(expected)), abs=1e-6)
    prediction = model.predict_hostname("www.example.com")
    assert prediction.feature_version == url_cnn.FEATURE_VERSION_V2
    assert 0 <= prediction.calibrated_probability <= 1


def test_v2_artifact_validation(tmp_path):
    rng = np.random.default_rng(2)
    broken = v2_artifact([v2_member(rng)])
    broken["members"][0]["dense"]["bias"] = packed(np.zeros(V2["hidden"] - 1))
    path, sha = write(tmp_path, broken)
    model = UrlCnnModel(path, sha)
    assert model.artifact is None and "Invalid shape" in model.reason

    for key, value in (("kernel_size", 4), ("dilations", [1, 128]), ("members", [])):
        path, sha = write(tmp_path, v2_artifact([v2_member(rng)]) | {key: value})
        assert UrlCnnModel(path, sha).artifact is None, key

    nonfinite = v2_member(rng)
    nonfinite["out"]["bias"] = np.array([np.inf], dtype=np.float32)
    path, sha = write(tmp_path, v2_artifact([nonfinite]))
    assert "nonfinite" in UrlCnnModel(path, sha).reason

    path, sha = write(tmp_path, v2_artifact([v2_member(rng)], gate="failed"))
    assert UrlCnnModel(path, sha).artifact is None  # release gate still enforced for v2


def test_v2_export_matches_pytorch_with_folded_batch_norm(tmp_path):
    torch = pytest.importorskip("torch")
    from backend.app.services.ml.train_url_cnn import build_model_v2, export_v2, model_features

    torch.manual_seed(0)
    model = build_model_v2(torch, 4, 6, 3, [1, 2, 4], 5, 0.0)
    with torch.no_grad():  # non-trivial batch-norm statistics, so folding is really exercised
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.running_mean.uniform_(-1, 1)
                module.running_var.uniform_(0.5, 2)
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.5, 0.5)
    model.eval()
    artifact = {
        "format": url_cnn.ARTIFACT_FORMAT_V2,
        "feature_version": url_cnn.FEATURE_VERSION_V2,
        "alphabet": ALPHABET,
        "release_gate": "passed",
        "max_length": 64,
        "embedding_dim": 4,
        "channels": 6,
        "kernel_size": 3,
        "dilations": [1, 2, 4],
        "hidden": 5,
        "members": [export_v2(model)],
        "calibration_slope": 1.0,
        "calibration_intercept": 0.0,
        "threshold": 0.5,
    }
    path, sha = write(tmp_path, artifact)
    loaded = UrlCnnModel(path, sha)
    assert loaded.artifact is not None, loaded.reason
    texts = [model_input(h) for h in ["example.com", "secure.paypa1-verify.xyz", "a.b.github.io"]]
    with torch.no_grad():
        expected = model(torch.from_numpy(model_features(texts, "v2", 64))).numpy()
    for text, value in zip(texts, expected, strict=True):
        assert loaded.logit(text) == pytest.approx(float(value), abs=1e-4)


# ---------------------------------------------------------------- v3-wide (Wide & Deep)

V3 = {"max_length": 40, "embedding_dim": 4, "channels": 6, "kernel_size": 3, "hidden": 5}
V3_DILATIONS = [1, 2]
WIDE_BUCKETS, WIDE_MAX_N = 97, 4


def v3_member(rng, wide=None):
    e, c, k, h = V3["embedding_dim"], V3["channels"], V3["kernel_size"], V3["hidden"]

    def r(*shape):
        return rng.normal(0, 0.5, shape).astype(np.float32)

    def pair(weight_shape, bias_shape):
        return {"weight": r(*weight_shape), "bias": r(*bias_shape)}

    member = {
        "embedding": r(VOCAB_SIZE, e),
        "role_embedding": r(url_cnn.ROLE_COUNT, e),
        "stem": pair((c, e, k), (c,)),
        "norm": pair((c,), (c,)),
        "blocks": [
            {
                "depthwise": pair((c, 1, k), (c,)),
                "norm": pair((c,), (c,)),
                "gate": pair((2 * c, c, 1), (2 * c,)),
                "scale": r(c),
            }
            for _ in V3_DILATIONS
        ],
        "attention": pair((1, c, 1), (1,)),
        "dense": pair((h, 4 * c), (h,)),
        "out": pair((1, h), (1,)),
    }
    if wide is not None:
        member["wide"] = wide
    return member


def v3_artifact(members, wide=False, gate="passed"):
    data = V3 | {
        "format": url_cnn.ARTIFACT_FORMAT_V3_WIDE if wide else url_cnn.ARTIFACT_FORMAT_V3,
        "feature_version": (
            url_cnn.FEATURE_VERSION_V3_WIDE if wide else url_cnn.FEATURE_VERSION_V3
        ),
        "alphabet": ALPHABET,
        "model_version": "test-v3",
        "release_gate": gate,
        "dilations": V3_DILATIONS,
        "members": [pack_member(m) for m in members],
        "calibration_slope": 1.0,
        "calibration_intercept": 0.0,
        "threshold": 0.5,
    }
    if wide:
        data |= {"wide_buckets": WIDE_BUCKETS, "wide_max_n": WIDE_MAX_N}
    return data


def reference_wide_term(table, text):
    """Deliberately naive: every unpadded n-gram, hashed exactly as documented."""
    length = min(V3["max_length"], len(text))
    chars, roles = encode(text, length), url_cnn.encode_roles(text, length)
    weights = []
    for n in range(1, WIDE_MAX_N + 1):
        for t in range(length - n + 1):
            if any(chars[t + j] == url_cnn.PAD for j in range(n)):
                continue
            key = int(roles[t])
            for j in range(n):
                key = key * 64 + int(chars[t + j])
            bucket = (key * 8 + n) * url_cnn.HASH_MULTIPLIER % url_cnn.HASH_PRIME % WIDE_BUCKETS
            weights.append(float(table[bucket]))
    return sum(weights) / max(1, len(weights)) ** 0.5


def test_ngram_buckets_skip_padding_and_cover_every_window():
    chars, roles = encode("ab.c", 6), url_cnn.encode_roles("ab.c", 6)
    ids = url_cnn.ngram_buckets(chars, roles, 3, 1000)
    assert len(ids) == 4 + 3 + 2  # 4 characters: unigrams, bigrams, trigrams; no padding
    assert ids.min() >= 0 and ids.max() < 1000
    assert len(url_cnn.ngram_buckets(chars, roles, 8, 1000)) == 4 + 3 + 2 + 1
    assert np.array_equal(ids, url_cnn.ngram_buckets(chars[2:], roles[2:], 3, 1000))


def test_v3_wide_with_zero_table_is_exactly_v3(tmp_path):
    rng = np.random.default_rng(7)
    members = [v3_member(rng), v3_member(rng)]
    zero = np.zeros(WIDE_BUCKETS, dtype=np.float32)
    path, sha = write(tmp_path, v3_artifact(members))
    plain = UrlCnnModel(path, sha)
    (tmp_path / "wide").mkdir()
    path, sha = write(tmp_path / "wide", v3_artifact([m | {"wide": zero} for m in members], True))
    wide = UrlCnnModel(path, sha)
    assert plain.artifact is not None and wide.artifact is not None, wide.reason
    for host in ["example.com", "login.secure-paypa1.co.uk", "a.b", "x" * 60]:
        text = model_input(host)
        assert wide.logit(text) == plain.logit(text)
    assert wide.predict_hostname("example.com").feature_version == url_cnn.FEATURE_VERSION_V3_WIDE


def test_v3_wide_adds_the_reference_wide_term_per_member(tmp_path):
    rng = np.random.default_rng(8)
    tables = [rng.normal(0, 1, WIDE_BUCKETS).astype(np.float32) for _ in range(2)]
    members = [v3_member(rng) for _ in tables]
    path, sha = write(tmp_path, v3_artifact(members))
    plain = UrlCnnModel(path, sha)
    (tmp_path / "wide").mkdir()
    data = v3_artifact([m | {"wide": t} for m, t in zip(members, tables, strict=True)], True)
    path, sha = write(tmp_path / "wide", data)
    wide = UrlCnnModel(path, sha)
    assert wide.artifact is not None, wide.reason
    for host in ["example.com", "login.secure-paypa1.co.uk", "a.b", "x" * 60]:
        text = model_input(host)
        expected = [
            plain.member_logit(i, text) + reference_wide_term(t, text) for i, t in enumerate(tables)
        ]
        for i, value in enumerate(expected):
            assert wide.member_logit(i, text) == pytest.approx(value, abs=1e-9)
        assert wide.logit(text) == pytest.approx(float(np.mean(expected)), abs=1e-9)


def test_v3_wide_artifact_validation(tmp_path):
    rng = np.random.default_rng(9)
    table = np.zeros(WIDE_BUCKETS, dtype=np.float32)
    good = v3_artifact([v3_member(rng, table)], True)
    path, sha = write(tmp_path, good)
    assert UrlCnnModel(path, sha).artifact is not None

    for key, value in (("wide_max_n", 7), ("wide_max_n", 0), ("wide_buckets", 2**25)):
        path, sha = write(tmp_path, good | {key: value})
        assert UrlCnnModel(path, sha).artifact is None, (key, value)

    short = v3_artifact([v3_member(rng, table[:-1])], True)
    path, sha = write(tmp_path, short)
    assert "Invalid shape" in UrlCnnModel(path, sha).reason

    path, sha = write(tmp_path, v3_artifact([v3_member(rng)], True))  # no wide table
    assert UrlCnnModel(path, sha).artifact is None

    path, sha = write(tmp_path, good | {"feature_version": url_cnn.FEATURE_VERSION_V3})
    assert "feature version" in UrlCnnModel(path, sha).reason

    path, sha = write(tmp_path, v3_artifact([v3_member(rng, table)], True, gate="failed"))
    assert UrlCnnModel(path, sha).artifact is None  # release gate still enforced

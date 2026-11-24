"""Research harness tests: v4 NumPy parity and invariances, encodings, metrics, safeguards.

Run from the repository root: ``python -m pytest research -q``.
"""

import json

import numpy as np
import pytest

from research.hostname_cnn import lockbox, metrics, models
from research.hostname_cnn.numpy_v4 import FEATURE_VERSION, FORMAT, V4Numpy

HOSTS = [
    "example.com",
    "login.example.co.uk",
    "a.b.github.io",
    "x",
    "paypal-secure-verify.xyz",
    "a." * 115 + "example.co.uk",
]


def test_structure_ids_mark_labels_from_the_right_and_dots():
    labels, positions = models.structure_ids("login.example.co.uk", 20)
    text = "login.example.co.uk"
    pad = 20 - len(text)
    assert (labels[:pad] == 0).all() and (positions[:pad] == 0).all()
    labels, positions = labels[pad:].tolist(), positions[pad:].tolist()
    # login = label 3 from the right -> slot 4; its dot shares the slot, position slot 17.
    assert labels[:6] == [4] * 6 and positions[:6] == [1, 2, 3, 4, 5, 17]
    assert labels[-2:] == [1, 1] and positions[-2:] == [1, 2]  # "uk" is label 0
    long_label, long_pos = models.structure_ids("a" * 30 + ".com", 34)
    assert long_pos[:30].max() == 16  # offsets past 15 share the last slot
    assert long_label[0] == 2


def train_one_step(torch, model, X):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.train()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(torch.from_numpy(X)), torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    )
    loss.backward()
    optimizer.step()
    model.eval()


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"block": "convnext", "dilations": [1, 1, 2, 2], "kernel_size": 7, "attention_heads": 2},
        {"structure_embeddings": False, "component_pools": False},
    ],
)
def test_v4_numpy_matches_torch_and_is_padding_and_batch_invariant(overrides):
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    config = models.defaults("v4") | {"channels": 8, "embedding_dim": 6, "hidden": 8} | overrides
    model = models.build(torch, "v4", config)
    X = models.features("v4", HOSTS, config)
    train_one_step(torch, model, X)
    with torch.no_grad():
        full = model(torch.from_numpy(X)).numpy()
        for i, host in enumerate(HOSTS):  # other batch members never change a prediction
            alone = model(torch.from_numpy(models.features("v4", [host], config))).numpy()[0]
            assert alone == pytest.approx(full[i], abs=2e-5)
        short = models.features("v4", HOSTS[:5], config | {"max_length": 40})
        np.testing.assert_allclose(model(torch.from_numpy(short)).numpy(), full[:5], atol=2e-5)
    artifact = {
        "format": FORMAT,
        "feature_version": FEATURE_VERSION,
        "config": config,
        "members": [models.export_tensors(model)],
    }
    runtime = V4Numpy(json.loads(json.dumps(artifact)))
    for host, value in zip(HOSTS, full, strict=True):
        assert runtime.logit(host) == pytest.approx(float(value), abs=2e-5)


@pytest.mark.parametrize("arch", ["v3wide", "v4"])
def test_wide_and_deep_numpy_matches_torch_and_is_padding_and_batch_invariant(arch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(5)
    config = models.defaults(arch) | {"channels": 8, "embedding_dim": 6, "hidden": 8}
    config |= {"wide_buckets": 4096, "wide_max_n": 5}
    model = models.build(torch, arch, config)
    X = models.features(arch, HOSTS, config)
    for _ in range(3):  # make the wide weights non-zero
        train_one_step(torch, model, X)
    assert float(model.wide.weight.detach().abs().max()) > 0
    with torch.no_grad():
        full = model(torch.from_numpy(X)).numpy()
        for i, host in enumerate(HOSTS):
            alone = model(torch.from_numpy(models.features(arch, [host], config))).numpy()[0]
            assert alone == pytest.approx(full[i], abs=2e-5)
        short = models.features(arch, HOSTS[:5], config | {"max_length": 40})
        np.testing.assert_allclose(model(torch.from_numpy(short)).numpy(), full[:5], atol=2e-5)
    artifact = {
        "format": FORMAT,
        "feature_version": FEATURE_VERSION,
        "config": models.runtime_config(arch, config),
        "members": [models.export_tensors(model)],
    }
    runtime = V4Numpy(json.loads(json.dumps(artifact)))
    for host, value in zip(HOSTS, full, strict=True):
        assert runtime.logit(host) == pytest.approx(float(value), abs=2e-5)


def test_v3wide_starts_as_exactly_the_v3_control():
    torch = pytest.importorskip("torch")
    config = models.defaults("v3wide")
    torch.manual_seed(42)
    control = models.build(torch, "v3", config)
    after_control = torch.get_rng_state()
    torch.manual_seed(42)
    wide = models.build(torch, "v3wide", config)
    assert torch.equal(torch.get_rng_state(), after_control)  # same later dropout masks
    for name, value in control.state_dict().items():
        assert torch.equal(wide.state_dict()["deep." + name], value)
    X = torch.from_numpy(models.features("v3", HOSTS, config))
    control.eval()
    wide.eval()
    with torch.no_grad():
        assert torch.equal(control(X), wide(X))


def test_pruned_checkpoints_keep_exactly_the_chosen_epoch(tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from research.hostname_cnn.train import choose, train_member

    config = models.defaults("v3") | {"channels": 8, "embedding_dim": 4, "hidden": 16}
    torch.manual_seed(0)
    model = models.build(torch, "v3", config)
    rng = np.random.default_rng(0)
    hosts = [f"{''.join(rng.choice(list('abcxyz'), 8))}.com" for _ in range(64)]
    X = models.features("v3", hosts, config)
    y = np.array([float("x" in h) for h in hosts], dtype=np.float32)  # learnable
    args = SimpleNamespace(
        seed=0, lr=0.01, weight_decay=0, batch_size=16, epochs=6, warmup_frac=0.1,
        min_lr_frac=0.05, ema_decay=0, swa_epochs=0, patience=0, batching="random",
        char_dropout=0, suffix_dropout=0, label_smoothing=0, loss="bce", pauc_weight=0,
        pauc_start_epoch=0, pauc_ramp_epochs=0, prune_checkpoints=True,
    )  # fmt: skip
    history, _ = train_member(torch, model, X, y, X, y, args, tmp_path)
    kept = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
    assert kept == [f"epoch_{choose(history, 'val_bce', False):02d}.pt"]


def test_v4_artifact_is_rejected_by_the_deployed_loader(tmp_path):
    import hashlib

    from backend.app.services.ml.url_cnn import UrlCnnModel

    raw = json.dumps({"format": FORMAT, "feature_version": FEATURE_VERSION}).encode()
    path = tmp_path / "v4.json"
    path.write_bytes(raw)
    model = UrlCnnModel(str(path), hashlib.sha256(raw).hexdigest(), allow_ungated=True)
    assert model.artifact is None and "Incompatible" in model.reason


def test_recall_at_fpr_counts_allowed_false_positives():
    y = np.array([0] * 1000 + [1] * 10)
    score = np.concatenate([np.arange(1000) / 1000, np.full(10, 0.9985)])
    # 1 false positive allowed at 0.1%: cut at the 2nd-highest negative (0.998).
    assert metrics.recall_at_fpr(y, score, 0.001) == 1.0
    assert metrics.recall_at_fpr(y, score, 0.0) == 0.0
    worse = score.copy()
    worse[-5:] = 0.5
    diff = metrics.paired_difference(y, score, worse, bootstrap=50)
    assert diff["recall_at_0.001_fpr"]["difference"] == pytest.approx(-0.5)


def test_extend_splits_keeps_earlier_assignments_and_whole_domains():
    from backend.app.services.ml.offline import canonical_hash
    from research.hostname_cnn.datasets import extend_splits

    hosts = ["a.example.com", "b.example.com", "new-site.org", "x.new-site.org", "other.net"]
    previous = {
        "train": [canonical_hash("a.example.com")],
        "calibration": [],
        "validation": [],
        "test": [],
    }
    splits, kept = extend_splits(hosts, previous)
    where = {hosts[i]: name for name, idx in splits.items() for i in idx}
    assert kept == 1 and where["a.example.com"] == "train" == where["b.example.com"]
    assert where["new-site.org"] == where["x.new-site.org"]
    assert len(where) == len(hosts)


def test_pauc_term_waits_then_ramps_in():
    from types import SimpleNamespace

    from research.hostname_cnn.train import pauc_factor

    args = SimpleNamespace(pauc_start_epoch=2, pauc_ramp_epochs=1)
    assert pauc_factor(199, 100, args) == 0.0  # still in epoch 2
    assert pauc_factor(200, 100, args) == pytest.approx(0.01)
    assert pauc_factor(249, 100, args) == pytest.approx(0.5)
    assert pauc_factor(5000, 100, args) == 1.0
    assert pauc_factor(0, 100, SimpleNamespace(pauc_start_epoch=0, pauc_ramp_epochs=0)) == 1.0


def test_recover_history_recomputes_validation_and_checks_the_log(tmp_path):
    from backend.app.services.ml.train_url_cnn import bce
    from research.hostname_cnn.train import recover_history

    yv = np.array([0, 1, 0, 1, 1])
    logits = np.array([[-1, 1, -2, 0.5, 2], [-2, 2, -3, 1, 3]], dtype=np.float32)
    np.save(tmp_path / "validation_logits_by_epoch.npy", logits)
    lines = [
        f"epoch {e + 1}/30  train_loss=0.5000  val_loss={bce(logits[e].astype(float), yv):.4f}  "
        f"val_pauc=0.7000  lr=1.00e-03  (100.0s)"
        for e in range(2)
    ]
    log = tmp_path / "run.log"
    log.write_text("header\n" + "\n".join(lines) + "\nTraceback ...\n")
    history, has_swa = recover_history(tmp_path, yv, log)
    assert [h["epoch"] for h in history] == [1, 2] and not has_swa
    log.write_text(lines[0].replace("val_loss=", "val_loss=9"))  # disagreeing or missing epochs
    with pytest.raises(ValueError):
        recover_history(tmp_path, yv, log)


def test_lockbox_is_sealed_without_the_phase_11_unlock(monkeypatch):
    monkeypatch.delenv("M_PHISH_LOCKBOX_UNLOCK", raising=False)
    with pytest.raises(PermissionError):
        lockbox.open_lockbox()

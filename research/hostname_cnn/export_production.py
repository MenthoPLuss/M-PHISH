"""Convert a final research build into the app's ``m-phish-charcnn-v3-wide`` format.

Only Wide & Deep builds of the deployed v3 network (``v3wide``) convert: their deep part is
the v3 network the app already runs, and the wide table is one extra tensor per member. The
tensors are copied byte for byte; nothing is retrained or recalibrated. The calibration and
threshold are the build's own.

The converted file is then loaded by the app's own loader (``UrlCnnModel``) and every
calibration and validation hostname in the build's ``scores.npz`` is rescored. The export is
refused unless the app's logits match the build's NumPy logits within ``PARITY_LIMIT`` and no
warn/no-warn decision changes. The test split and the lockbox are not touched.

The output keeps ``release_gate: "failed"`` (like the deployed ``cnn-v3-fresh``), so the app
serves it only with ``ML_CNN_ALLOW_UNGATED=true`` and never with ``APP_ENV=production``.
Nothing changes in the app until ``ML_CNN_ARTIFACT_PATH`` points at the new file.

    python -m research.hostname_cnn.export_production --build F-c13-ensemble3

Writes ``runs/cnn-v3wide-<name>/``: ``url_cnn.json`` and ``export.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from backend.app.services.ml.url_cnn import (
    ALPHABET,
    ARTIFACT_FORMAT_V3_WIDE,
    FEATURE_VERSION_V3_WIDE,
    ROLE_COUNT,
    VOCAB_SIZE,
    UrlCnnModel,
    model_input,
)

from .common import RESEARCH_RUNS, ROOT, git_state
from .numpy_v4 import FEATURE_VERSION, FORMAT

PARITY_LIMIT = 1e-5  # the harness target; the app and the build run the same arithmetic
# The v4 research config that computes exactly the v3 network (see models.runtime_config).
V3_EQUIVALENT = {
    "block": "glu",
    "structure_embeddings": False,
    "component_pools": False,
    "attention_heads": 1,
}
_model = None


def convert(research: dict) -> dict:
    """The production artifact for one research artifact; tensors are reused unchanged."""
    if research.get("format") != FORMAT or research.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Not a v4 research artifact")
    if research.get("alphabet") != ALPHABET:
        raise ValueError("Research artifact uses a different alphabet")
    c = research["config"]
    for key, value in V3_EQUIVALENT.items():
        if c[key] != value:
            raise ValueError(f"Only v3wide builds convert; config has {key}={c[key]!r}")
    if c["stem_kernel"] != c["kernel_size"]:
        raise ValueError("Only v3wide builds convert; stem and block kernels differ")
    if not c.get("wide_buckets"):
        raise ValueError("Build has no wide path; it is a plain v3 network")

    def packed(tensors, name, shape):
        if name not in tensors:
            raise ValueError(f"Member is missing {name}")
        stored_shape, data = tensors[name]
        if list(stored_shape) != list(shape):
            raise ValueError(f"{name} has shape {stored_shape}, expected {list(shape)}")
        return data

    ch, e, k, hidden = c["channels"], c["embedding_dim"], c["kernel_size"], c["hidden"]
    members = []
    # embedding, role embedding, wide, and weight + bias of stem, norm, attention, dense, out;
    # each block has depthwise, norm and gate pairs plus a scale.
    tensor_count = 3 + 2 * 5 + 7 * len(c["dilations"])
    for tensors in research["members"]:
        if len(tensors) != tensor_count:
            raise ValueError("Member does not hold exactly the v3-wide tensors")

        def pair(prefix, weight_shape, bias_shape, t=tensors):
            return {
                "weight": packed(t, prefix + ".weight", weight_shape),
                "bias": packed(t, prefix + ".bias", bias_shape),
            }

        member = {
            "embedding": packed(tensors, "embedding.weight", (VOCAB_SIZE, e)),
            "role_embedding": packed(tensors, "role_embedding.weight", (ROLE_COUNT, e)),
            "stem": pair("stem", (ch, e, k), (ch,)),
            "norm": pair("norm", (ch,), (ch,)),
            "blocks": [
                {
                    "depthwise": pair(f"blocks.{b}.depthwise", (ch, 1, k), (ch,)),
                    "norm": pair(f"blocks.{b}.norm", (ch,), (ch,)),
                    "gate": pair(f"blocks.{b}.gate", (2 * ch, ch, 1), (2 * ch,)),
                    "scale": packed(tensors, f"blocks.{b}.scale", (ch,)),
                }
                for b in range(len(c["dilations"]))
            ],
            "attention": pair("attention", (1, ch, 1), (1,)),
            "dense": pair("dense", (hidden, 4 * ch), (hidden,)),
            "out": pair("out", (1, hidden), (1,)),
            "wide": packed(tensors, "wide.weight", (c["wide_buckets"], 1)),
        }
        members.append(member)
    return {
        "format": ARTIFACT_FORMAT_V3_WIDE,
        "feature_version": FEATURE_VERSION_V3_WIDE,
        "alphabet": ALPHABET,
        "max_length": c["max_length"],
        "embedding_dim": e,
        "channels": ch,
        "kernel_size": k,
        "dilations": list(c["dilations"]),
        "hidden": hidden,
        "wide_buckets": c["wide_buckets"],
        "wide_max_n": c["wide_max_n"],
        "members": members,
        "best_epochs": research.get("best_epochs"),
        "calibration_slope": research["calibration_slope"],
        "calibration_intercept": research["calibration_intercept"],
        "threshold": research["threshold"],
        "release_gate": "failed",  # validation only until the sealed test (Phase 11) passes
    }


def _init(path: str, sha: str):
    global _model
    _model = UrlCnnModel(path, sha, allow_ungated=True)


def _logit(host: str) -> float:
    return _model.logit(host)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", required=True, help="final build id, e.g. F-c13-ensemble3")
    parser.add_argument("--out", type=Path, help="default: runs/cnn-v3wide-<name>")
    parser.add_argument("--processes", type=int, default=6)
    args = parser.parse_args(argv)
    started = time.time()
    source = RESEARCH_RUNS / args.build
    out = args.out or ROOT / "runs" / f"cnn-v3wide-{args.build.removeprefix('F-')}"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists; choose another --out")

    raw = (source / "url_cnn.json").read_bytes()
    record = json.loads((source / "experiment.json").read_text())
    if record.get("kind") != "final build":
        raise SystemExit(f"{args.build} is not a final build")
    if hashlib.sha256(raw).hexdigest() != record["artifact_sha256"]:
        raise SystemExit("The build's url_cnn.json does not match its experiment.json")
    if record["git"]["dirty"]:
        raise SystemExit("The build was made from uncommitted code")
    if record.get("test_split_scored") or record.get("lockbox_opened"):
        raise SystemExit("The build record says the test split or lockbox was used")
    research = json.loads(raw)
    artifact = convert(research) | {
        "model_version": f"cnn-v3wide-{args.build.removeprefix('F-')}",
        "source_build": args.build,
        "source_artifact_sha256": record["artifact_sha256"],
        "data": record["data"],
    }
    exported = json.dumps(artifact, sort_keys=True, allow_nan=False).encode()
    sha = hashlib.sha256(exported).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "url_cnn.json"
    path.write_bytes(exported)

    model = UrlCnnModel(str(path), sha, allow_ungated=True)
    if model.artifact is None:
        path.unlink()
        raise SystemExit(f"The app's loader refuses the export: {model.reason}")
    scores = np.load(source / "scores.npz")
    report = {}
    with Pool(args.processes, initializer=_init, initargs=(str(path), sha)) as pool:
        for split in ("calibration", "validation"):
            hosts = [str(h) for h in scores[f"{split}_hosts"]]
            expected = scores[f"{split}_logit"]
            got = np.array(pool.map(_logit, hosts, chunksize=300))
            slope, intercept = artifact["calibration_slope"], artifact["calibration_intercept"]
            threshold = artifact["threshold"]

            def warns(z, s=slope, i=intercept, t=threshold):
                return 1 / (1 + np.exp(-np.clip(s * z + i, -700, 700))) >= t

            report[split] = {
                "hostnames": len(hosts),
                "max_abs_logit_difference": float(np.abs(got - expected).max()),
                "decisions_changed": int((warns(got) != warns(expected)).sum()),
                "hosts_changed_by_model_input": sum(model_input(h) != h for h in hosts),
            }
    worst = max(r["max_abs_logit_difference"] for r in report.values())
    flips = sum(r["decisions_changed"] for r in report.values())
    if worst > PARITY_LIMIT or flips:
        path.unlink()
        raise SystemExit(f"Export disagrees with the build: max |dz| {worst:.2e}, {flips} flips")

    sample = [str(h) for h in scores["validation_hosts"][:1000]]
    t0 = time.perf_counter()
    for host in sample:
        model.predict_hostname(host)
    latency_ms = (time.perf_counter() - t0) / len(sample) * 1000
    summary = {
        "kind": "production export",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_state(),
        "source_build": args.build,
        "source_artifact_sha256": record["artifact_sha256"],
        "artifact": str(path.relative_to(ROOT)),
        "artifact_sha256": sha,
        "artifact_bytes": len(exported),
        "format": ARTIFACT_FORMAT_V3_WIDE,
        "release_gate": artifact["release_gate"],
        "threshold": artifact["threshold"],
        "parity": report,
        "parity_limit": PARITY_LIMIT,
        "app_latency_ms": latency_ms,
        "wall_minutes": (time.time() - started) / 60,
        "test_split_scored": False,
        "lockbox_opened": False,
    }
    (out / "export.json").write_text(json.dumps(summary, indent=1))
    print(
        f"DONE {path.relative_to(ROOT)}: {len(exported) / 1e6:.1f} MB, sha256 {sha}; "
        f"max |dz| {worst:.1e} over "
        f"{sum(r['hostnames'] for r in report.values())} hostnames, 0 decision changes; "
        f"{latency_ms:.1f} ms/hostname in the app"
    )
    return summary


if __name__ == "__main__":
    main()

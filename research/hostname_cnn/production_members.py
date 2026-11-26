"""Expose each member of the deployed cnn-v3-fresh ensemble as a read-only research run.

The deployed members (seeds 42, 43, 44) were trained by the production trainer with random
batches. The research control ``P0.5-control-random-s42`` reproduced that trainer exactly at
epoch 1, so these members are seed-matched random-batch controls for confirmation runs. Each
member is scored with the app's own NumPy runtime on the calibration and validation splits;
the deployed artifact is only read, never changed.

    python -m research.hostname_cnn.production_members
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from multiprocessing import Pool

import numpy as np

from .common import BASELINE_RUN, RESEARCH_RUNS, git_state, load_dataset, repo_relative
from .metrics import ranking

ARTIFACT = BASELINE_RUN / "url_cnn.json"
SEEDS = (42, 43, 44)
_model = None


def _init():
    global _model
    from backend.app.services.ml.url_cnn import UrlCnnModel

    raw = ARTIFACT.read_bytes()
    _model = UrlCnnModel(str(ARTIFACT), hashlib.sha256(raw).hexdigest(), allow_ungated=True)


def _members(host: str) -> list[float]:
    return [_model.member_logit(i, host) for i in range(len(SEEDS))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processes", type=int, default=6)
    args = parser.parse_args(argv)
    data = load_dataset()
    artifact_sha = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    scored = {}
    for split in ("calibration", "validation"):
        idx = data.indices(split)
        hosts = [data.hosts[i] for i in idx]
        with Pool(args.processes, initializer=_init) as pool:
            scored[split] = (hosts, data.y[idx], np.array(pool.map(_members, hosts, chunksize=300)))
    for member, seed in enumerate(SEEDS):
        out = RESEARCH_RUNS / f"P0-production-member-s{seed}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "scores.npz",
            **{
                f"{split}_{key}": value
                for split, (hosts, y, Z) in scored.items()
                for key, value in (("hosts", np.array(hosts)), ("y", y), ("logit", Z[:, member]))
            },
        )
        hosts, y, Z = scored["validation"]
        record = {
            "experiment_id": out.name,
            "hypothesis": "control",
            "description": f"Deployed cnn-v3-fresh member {member} (seed {seed}), scored as-is",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git": git_state(),
            "source_artifact": repo_relative(ARTIFACT),
            "source_artifact_sha256": artifact_sha,
            "member_index": member,
            "seed": seed,
            "training": {"trainer": "backend train_url_cnn.train_deep", "batching": "random"},
            "data": data.provenance(),
            "validation": {"all": ranking(y, Z[:, member])},
            "test_split_scored": False,
        }
        (out / "experiment.json").write_text(json.dumps(record, indent=1, default=float))
        v = record["validation"]["all"]
        print(
            f"{out.name}: val ROC-AUC {v['roc_auc']:.4f} pAUC {v['pauc_1pct']:.4f} "
            f"recall@1% {v['recall_at_0.01_fpr']:.4f}"
        )


if __name__ == "__main__":
    main()

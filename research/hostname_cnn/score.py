"""Score a research run's exported artifact on another dataset's validation split (NumPy only).

Used when two runs have different validation sets (H1 adds hostnames): the control is scored
on the candidate's extra validation hostnames so both are compared on identical hostnames.

    python -m research.hostname_cnn.score --run P0.5-control-bucketed-s42 \
        --dataset data/feeds/research/h1-v2/combined.csv --splits data/feeds/research/h1-v2/splits.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import RESEARCH_RUNS, load_dataset

_model = None


def _init(path: str):
    global _model
    from backend.app.services.ml.url_cnn import UrlCnnModel

    from .numpy_v4 import FORMAT, V4Numpy

    raw = Path(path).read_bytes()
    artifact = json.loads(raw)
    if artifact["format"] == FORMAT:
        _model = V4Numpy(artifact)
    else:
        _model = UrlCnnModel(path, hashlib.sha256(raw).hexdigest(), allow_ungated=True)


def _logit(host: str) -> float:
    return _model.logit(host)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--split", default="validation", choices=["calibration", "validation"])
    parser.add_argument(
        "--only-new", action="store_true", help="skip hostnames the run already scored"
    )
    parser.add_argument("--processes", type=int, default=6)
    args = parser.parse_args(argv)
    folder = RESEARCH_RUNS / args.run
    data = load_dataset(Path(args.dataset), Path(args.splits))
    idx = data.indices(args.split)
    hosts = [data.hosts[i] for i in idx]
    y = data.y[idx]
    if args.only_new:
        known = set(np.load(folder / "scores.npz")[f"{args.split}_hosts"].tolist())
        keep = [i for i, h in enumerate(hosts) if h not in known]
        hosts, y = [hosts[i] for i in keep], y[keep]
    with Pool(args.processes, initializer=_init, initargs=(str(folder / "url_cnn.json"),)) as pool:
        logits = np.array(pool.map(_logit, hosts, chunksize=200))
    name = Path(args.dataset).parent.name
    target = folder / f"extra_scores__{name}__{args.split}.npz"
    np.savez_compressed(target, hosts=np.array(hosts), y=y, logit=logits)
    print(
        f"Scored {len(hosts)} hostnames with {args.run} -> {target} ({os.path.getsize(target)} bytes)"
    )


if __name__ == "__main__":
    main()

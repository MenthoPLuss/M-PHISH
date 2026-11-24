"""Character n-gram logistic regression (H4): reference model and candidate ensemble member.

TF-IDF over character 1-5-grams of ``<subdomain|name|suffix>`` (component markers), logistic
regression with C chosen on the calibration split. Writes a research run folder with the
same ``scores.npz`` layout as the CNN trainer, so ``compare.py`` can stack or average them.

    python -m research.hostname_cnn.ngram --exp-id E4-ngram-markers
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from backend.app.services.ml.offline import canonical_hash
from backend.app.services.observations import _suffix_extractor

from .common import BASELINE_RUN, FEED, RESEARCH_RUNS, git_state, load_dataset, snapshot_code
from .metrics import evaluate


def marked(host: str) -> str:
    parts = _suffix_extractor()(host)
    if not parts.suffix or not parts.domain:
        return f"<|{host}|>"
    return f"<{parts.subdomain}|{parts.domain}|{parts.suffix}>"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--dataset", default=str(FEED / "combined.csv"))
    parser.add_argument("--splits", default=str(BASELINE_RUN / "splits.json"))
    parser.add_argument("--text", choices=["markers", "plain"], default="markers")
    parser.add_argument("--c-values", default="0.5,1,2,4")
    parser.add_argument("--output-root", default=str(RESEARCH_RUNS))
    args = parser.parse_args(argv)
    out = Path(args.output_root) / args.exp_id
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} exists; experiment ids are never reused")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    git = git_state()
    snapshot_code(out / "code")
    data = load_dataset(Path(args.dataset), Path(args.splits))
    text = marked if args.text == "markers" else (lambda h: f"<{h}>")
    idx = {n: data.indices(n) for n in ("train", "calibration", "validation")}
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(1, 5), min_df=3, sublinear_tf=True, dtype=np.float32
    )
    X = {"train": vectorizer.fit_transform([text(data.hosts[i]) for i in idx["train"]])}
    for name in ("calibration", "validation"):
        X[name] = vectorizer.transform([text(data.hosts[i]) for i in idx[name]])
    y = {n: data.y[i] for n, i in idx.items()}
    search = []
    best = None
    for c in (float(v) for v in args.c_values.split(",")):
        model = LogisticRegression(C=c, max_iter=3000, solver="liblinear").fit(
            X["train"], y["train"]
        )
        auc = roc_auc_score(y["calibration"], model.decision_function(X["calibration"]))
        search.append({"C": c, "calibration_roc_auc": auc})
        if best is None or auc > best[0]:
            best = (auc, c, model)
    model = best[2]
    z = {n: model.decision_function(X[n]) for n in ("calibration", "validation")}
    from backend.app.services.ml.train_url_cnn import calibrate, select_threshold

    calibrator = calibrate(z["calibration"], y["calibration"])
    p = calibrator.predict_proba(z["validation"].reshape(-1, 1))[:, 1]
    threshold = select_threshold(y["validation"], p)
    hosts_v = [data.hosts[i] for i in idx["validation"]]
    np.savez_compressed(
        out / "scores.npz",
        validation_hosts=np.array(hosts_v),
        validation_y=y["validation"],
        validation_logit=z["validation"],
        calibration_hosts=np.array([data.hosts[i] for i in idx["calibration"]]),
        calibration_y=y["calibration"],
        calibration_logit=z["calibration"],
    )
    record = {
        "experiment_id": args.exp_id,
        "hypothesis": "H4",
        "description": f"char 1-5-gram TF-IDF + logistic regression ({args.text})",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git,
        "data": data.provenance(),
        "validation_hosts_sha256": canonical_hash(sorted(hosts_v)),
        "arch": "ngram-logistic",
        "config": {
            "text": args.text,
            "features": int(X["train"].shape[1]),
            "C": best[1],
            "search": search,
        },
        "parameters": int(X["train"].shape[1]) + 1,
        "threshold": float(threshold),
        "validation": {
            "all": evaluate(y["validation"], z["validation"], p, threshold, bootstrap=200)
        },
        "wall_minutes": (time.time() - started) / 60,
        "test_split_scored": False,
    }
    (out / "experiment.json").write_text(json.dumps(record, indent=1, default=float))
    v = record["validation"]["all"]
    print(
        f"DONE {args.exp_id}: C={best[1]}, val ROC-AUC {v['roc_auc']:.4f}, pAUC {v['pauc_1pct']:.4f}, "
        f"recall@0.5% {v['recall_at_0.005_fpr']:.4f}, recall@1% {v['recall_at_0.01_fpr']:.4f}"
    )


if __name__ == "__main__":
    main()

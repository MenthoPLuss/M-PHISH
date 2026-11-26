"""H2a: does choosing the checkpoint by a low-FPR metric beat choosing by validation BCE?

Selection must not be judged on the data it was chosen on. Every saved epoch is scored on
the *calibration* split; an epoch is chosen there by each rule, and the chosen epochs are
compared on the *validation* split (already saved per epoch by the trainer).

    python -m research.hostname_cnn.checkpoint_select --run P0.5-control-bucketed-s42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend.app.services.ml.train_url_cnn import bce

from . import models
from .common import RESEARCH_RUNS, load_dataset
from .metrics import paired_difference, partial_auc, ranking, recall_at_fpr
from .train import score


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    folder = RESEARCH_RUNS / args.run
    record = json.loads((folder / "experiment.json").read_text())
    data = load_dataset(
        Path(record["data"]["dataset"]).resolve()
        if Path(record["data"]["dataset"]).is_absolute()
        else Path(__file__).resolve().parents[2] / record["data"]["dataset"],
        Path(__file__).resolve().parents[2] / record["data"]["splits"],
    )
    cal = data.indices("calibration")
    X = models.features(record["arch"], [data.hosts[i] for i in cal], record["config"])
    yc = data.y[cal]
    model = models.build(torch, record["arch"], record["config"])
    by_epoch = []
    val_logits = np.load(folder / "validation_logits_by_epoch.npy").astype(np.float64)
    yv = np.load(folder / "scores.npz")["validation_y"]
    for h in record["history"]:
        state = torch.load(folder / "checkpoints" / f"epoch_{h['epoch']:02d}.pt", weights_only=True)
        model.load_state_dict(state)
        zc = score(torch, model, X)
        by_epoch.append(
            {
                "epoch": h["epoch"],
                "calibration_bce": bce(zc, yc),
                "calibration_pauc_1pct": partial_auc(yc, zc),
                "calibration_recall_at_0.005_fpr": recall_at_fpr(yc, zc, 0.005),
                "validation": ranking(yv, val_logits[h["epoch"] - 1]),
            }
        )
        print(
            f"epoch {h['epoch']}: cal BCE {by_epoch[-1]['calibration_bce']:.4f} pAUC {by_epoch[-1]['calibration_pauc_1pct']:.4f}",
            flush=True,
        )
    rules = {
        "production rule (validation BCE)": record["chosen_epoch"],
        "calibration BCE": min(by_epoch, key=lambda e: e["calibration_bce"])["epoch"],
        "calibration pAUC (FPR<=1%)": max(by_epoch, key=lambda e: e["calibration_pauc_1pct"])[
            "epoch"
        ],
        "calibration recall at 0.5% FPR": max(
            by_epoch, key=lambda e: e["calibration_recall_at_0.005_fpr"]
        )["epoch"],
    }
    base = rules["production rule (validation BCE)"]
    report = {"run": args.run, "rules": {}, "by_epoch": by_epoch}
    for name, epoch in rules.items():
        diff = paired_difference(yv, val_logits[base - 1], val_logits[epoch - 1], bootstrap=300)
        report["rules"][name] = {
            "epoch": epoch,
            "validation": ranking(yv, val_logits[epoch - 1]),
            "vs_production_rule": diff,
        }
        v = report["rules"][name]["validation"]
        print(
            f"{name:34s} epoch {epoch:2d}: val pAUC {v['pauc_1pct']:.4f} recall@0.5% {v['recall_at_0.005_fpr']:.4f} "
            f"recall@1% {v['recall_at_0.01_fpr']:.4f} AUC {v['roc_auc']:.4f}  "
            f"diff recall@1% {diff['recall_at_0.01_fpr']['difference'] * 100:+.2f} pts "
            f"[{diff['recall_at_0.01_fpr']['ci95'][0] * 100:+.2f}, {diff['recall_at_0.01_fpr']['ci95'][1] * 100:+.2f}]"
        )
    out = RESEARCH_RUNS / "comparisons" / f"E2a__{args.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

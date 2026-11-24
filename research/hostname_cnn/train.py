"""Research trainer: one seeded member per run, validation-only evaluation, full provenance.

With default options it reproduces the production v3 recipe (``train_url_cnn.train_deep``):
AdamW, 10% warm-up, cosine to 5% of peak, clipping 1.0, character dropout 0.1, label
smoothing 0.05, EMA 0.999, early stopping on validation BCE with patience 5. Each option
below switches on one hypothesis from the research proposal.

The test split is never scored here. Example (from the repository root)::

    python -m research.hostname_cnn.train --exp-id P0.5-control-s42 --hypothesis P0.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

from backend.app.services.ml.offline import canonical_hash
from backend.app.services.ml.train_url_cnn import (
    batch_input,
    bce,
    calibrate,
    export_v3,
    select_threshold,
)
from backend.app.services.ml.url_cnn import (
    ALPHABET,
    ARTIFACT_FORMAT_V3,
    FEATURE_VERSION_V3,
    PAD,
    ROLE_SUFFIX,
    UNKNOWN,
    UrlCnnModel,
)

from . import models, numpy_v4
from .common import (
    BASELINE_RUN,
    FEED,
    RESEARCH_RUNS,
    Dataset,
    describe,
    git_state,
    load_dataset,
    snapshot_code,
    snapshot_sha256,
    stem_unseen_mask,
)
from .metrics import evaluate, partial_auc, subgroup_table


def batches(torch, lengths, batch_size, generator, mode):
    """Yield index tensors. ``bucketed`` sorts 50-batch chunks by length, then shuffles batches."""
    order = torch.randperm(len(lengths), generator=generator)
    if mode == "random":
        return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    chunk = batch_size * 50
    out = []
    for start in range(0, len(order), chunk):
        part = order[start : start + chunk]
        part = part[torch.argsort(lengths[part], stable=True)]
        out.extend(part[i : i + batch_size] for i in range(0, len(part), batch_size))
    perm = torch.randperm(len(out), generator=generator)
    return [out[i] for i in perm.tolist()]


def score(torch, model, X, batch=512):
    """Logits in input order; batches sorted by length for speed (results are order-free)."""
    model.eval()
    lengths = (X[:, 0] != PAD).sum(1)
    order = np.argsort(lengths, kind="stable")
    out = np.empty(len(X), dtype=np.float64)
    device = next(model.parameters()).device
    with torch.inference_mode():
        for i in range(0, len(X), batch):
            idx = order[i : i + batch]
            values = batch_input(model, torch.from_numpy(X[idx]), device)
            out[idx] = model(values).cpu().numpy()
    return out


def loss_terms(torch, logits, target, hard, weights, args, pauc_scale=1.0):
    """Per-batch training loss for the configured objective.

    The pAUC hinge spreads its gradient over only the k hardest negatives of a batch, about
    ``2 * weight / k`` per negative against about ``0.5 / batch_size`` per sample from BCE.
    At weight 0.25 and k = 7 that is ~18x BCE from the first step, which killed every hidden
    unit of the dense head in E2b-pauc0.25-s42. ``pauc_scale`` (from ``pauc_factor``) lets
    the term start after BCE has formed features and ramp in.
    """
    functional = torch.nn.functional
    if args.loss == "focal":
        ce = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * hard + (1 - p) * (1 - hard)
        per = (1 - p_t) ** args.focal_gamma * ce
        loss = (per * weights).mean() if weights is not None else per.mean()
    elif weights is not None:
        ce = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (ce * weights).mean()
    else:
        loss = functional.binary_cross_entropy_with_logits(logits, target)
    if args.pauc_weight and pauc_scale > 0:
        positives, negatives = logits[hard == 1], logits[hard == 0]
        if len(positives) and len(negatives):
            k = max(1, math.ceil(args.pauc_alpha * len(negatives)))
            top = torch.topk(negatives, k).values
            gap = args.pauc_margin - (positives[:, None] - top[None, :])
            loss = loss + pauc_scale * args.pauc_weight * torch.relu(gap).pow(2).mean()
    return loss


def pauc_factor(step, steps_per_epoch, args):
    """0 before ``--pauc-start-epoch``, then a linear ramp to 1 over ``--pauc-ramp-epochs``."""
    start = args.pauc_start_epoch * steps_per_epoch
    if step < start:
        return 0.0
    ramp = args.pauc_ramp_epochs * steps_per_epoch
    return 1.0 if ramp <= 0 else min(1.0, (step - start + 1) / ramp)


def train_member(torch, model, X, y, Xv, yv, args, out: Path, sample_weight=None):
    generator = torch.Generator().manual_seed(args.seed)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0},
        ],
        lr=args.lr,
    )
    steps = math.ceil(len(X) / args.batch_size)
    total_steps = args.epochs * steps
    warmup = max(1, int(args.warmup_frac * total_steps))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup - 1))
        return args.min_lr_frac + (1 - args.min_lr_frac) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    wt = None if sample_weight is None else torch.from_numpy(sample_weight.astype(np.float32))
    lengths = (Xt[:, 0] != PAD).sum(1)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if args.ema_decay else None
    swa, swa_count = None, 0
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    history, val_logits = [], []
    best_loss, stale, step, best_epoch = float("inf"), 0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, started = 0.0, time.perf_counter()
        for index in batches(torch, lengths, args.batch_size, generator, args.batching):
            optimizer.zero_grad(set_to_none=True)
            values = Xt[index]
            if args.char_dropout:
                values = values.clone()
                chars = values[:, 0]
                noise = torch.rand(chars.shape, generator=generator) < args.char_dropout
                chars[noise & (chars != PAD)] = UNKNOWN
            if args.suffix_dropout:
                # H5b: hide the public suffix of a random subset of hostnames (roles kept), so
                # the model must find evidence in the name and subdomain.
                values = values.clone()
                chosen = torch.rand(len(values), generator=generator) < args.suffix_dropout
                suffix = (values[:, 1] == ROLE_SUFFIX) & chosen[:, None]
                values[:, 0][suffix] = UNKNOWN
            hard = yt[index]
            target = hard * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
            logits = model(batch_input(model, values, "cpu"))
            loss = loss_terms(
                torch,
                logits,
                target,
                hard,
                None if wt is None else wt[index],
                args,
                pauc_factor(step, steps, args),
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            if ema is not None:
                rate = min(args.ema_decay, (1 + step) / (10 + step))
                with torch.no_grad():
                    for key, value in model.state_dict().items():
                        if value.dtype.is_floating_point:
                            ema[key].mul_(rate).add_(value.detach(), alpha=1 - rate)
                        else:
                            ema[key].copy_(value)
            total += loss.item() * len(index)
        live = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if args.swa_epochs and epoch > args.epochs - args.swa_epochs:
            swa_count += 1
            if swa is None:
                swa = {k: v.clone() for k, v in live.items()}
            else:
                for key, value in live.items():
                    if value.dtype.is_floating_point:
                        swa[key].mul_((swa_count - 1) / swa_count).add_(value / swa_count)
        if ema is not None:
            model.load_state_dict(ema)
        logits_v = score(torch, model, Xv)
        val_loss = bce(logits_v, yv)
        torch.save(model.state_dict(), checkpoints / f"epoch_{epoch:02d}.pt")
        val_logits.append(logits_v.astype(np.float32))
        record = {
            "epoch": epoch,
            "train_loss": total / len(Xt),
            "validation_loss": val_loss,
            "validation_pauc_1pct": partial_auc(yv, logits_v),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={record['train_loss']:.4f}  "
            f"val_loss={val_loss:.4f}  val_pauc={record['validation_pauc_1pct']:.4f}  "
            f"lr={record['learning_rate']:.2e}  ({record['seconds']:.1f}s)",
            flush=True,
        )
        model.load_state_dict(live)
        if float(np.std(logits_v)) < 1e-6:
            # A constant output (e.g. every ReLU of the head dead) has zero gradient and
            # never recovers; stop instead of spending the remaining epochs.
            raise RuntimeError(f"Collapsed at epoch {epoch}: validation output is constant")
        improved = val_loss < best_loss - 1e-4  # the production rule, as in choose()
        if args.prune_checkpoints:  # keep only the checkpoint choose("val_bce") can return
            stale_file = checkpoints / f"epoch_{(best_epoch if improved else epoch):02d}.pt"
            if best_epoch != epoch:
                stale_file.unlink(missing_ok=True)
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch, 0
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"Early stop: no validation improvement for {args.patience} epochs")
                break
    if swa is not None:
        torch.save(swa, checkpoints / "swa.pt")
    np.save(out / "validation_logits_by_epoch.npy", np.stack(val_logits))
    return history, swa is not None


EPOCH_LINE = re.compile(
    r"epoch (\d+)/\d+\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+val_pauc=([\d.]+)\s+"
    r"lr=([\d.e+-]+)\s+\(([\d.]+)s\)"
)


def recover_history(out: Path, yv, log_path: Path):
    """Rebuild the history of a run that trained fully but crashed before writing its record.

    Validation loss and pAUC are recomputed from the saved per-epoch validation logits.
    Training loss, learning rate and epoch time come from the run's log, whose rounded
    validation loss must agree with the recomputed one.
    """
    val_logits = np.load(out / "validation_logits_by_epoch.npy").astype(np.float64)
    rows = [m.groups() for m in EPOCH_LINE.finditer(log_path.read_text(encoding="utf-8"))]
    if len(rows) != len(val_logits):
        raise ValueError(f"{log_path} has {len(rows)} epochs, saved logits {len(val_logits)}")
    history = []
    for (epoch, train_loss, logged, _, lr, seconds), z in zip(rows, val_logits, strict=True):
        record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "validation_loss": bce(z, yv),
            "validation_pauc_1pct": partial_auc(yv, z),
            "learning_rate": float(lr),
            "seconds": float(seconds),
        }
        if abs(record["validation_loss"] - float(logged)) > 1e-4:
            raise ValueError(f"epoch {epoch}: log {logged} vs saved {record['validation_loss']}")
        history.append(record)
    return history, (out / "checkpoints" / "swa.pt").exists()


def choose(history, rule, has_swa):
    if rule == "swa":
        if not has_swa:
            raise ValueError("--select swa needs --swa-epochs and a run that reached the end")
        return "swa"
    if rule == "last":
        return history[-1]["epoch"]
    if rule == "pauc":
        return max(history, key=lambda h: h["validation_pauc_1pct"])["epoch"]
    # Production rule (train_deep): an epoch replaces the best only if it improves by > 1e-4.
    best_loss, best_epoch = float("inf"), history[0]["epoch"]
    for h in history:
        if h["validation_loss"] < best_loss - 1e-4:
            best_loss, best_epoch = h["validation_loss"], h["epoch"]
    return best_epoch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--dataset", default=str(FEED / "combined.csv"))
    parser.add_argument("--splits", default=str(BASELINE_RUN / "splits.json"))
    parser.add_argument("--arch", default="v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.1)
    parser.add_argument("--min-lr-frac", type=float, default=0.05)
    parser.add_argument("--char-dropout", type=float, default=0.1)
    parser.add_argument("--suffix-dropout", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--batching", choices=["random", "bucketed"], default="random")
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pauc-weight", type=float, default=0.0)
    parser.add_argument("--pauc-alpha", type=float, default=0.1)
    parser.add_argument("--pauc-margin", type=float, default=1.0)
    parser.add_argument("--pauc-start-epoch", type=float, default=0.0)
    parser.add_argument("--pauc-ramp-epochs", type=float, default=0.0)
    parser.add_argument(
        "--prune-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="keep only the checkpoint the val_bce rule selects; default: on for networks "
        "with a wide table (about 4 MB per checkpoint) under --select val_bce",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="finish a run that trained but crashed before writing experiment.json",
    )
    parser.add_argument("--swa-epochs", type=int, default=0)
    parser.add_argument("--select", choices=["val_bce", "pauc", "last", "swa"], default="val_bce")
    parser.add_argument("--sample-weights", help="JSON {hostname: weight} for training rows")
    parser.add_argument("--max-train", type=int, default=0, help="subsample training (smoke)")
    parser.add_argument("--threads", type=int, default=7)
    parser.add_argument("--config", default="{}", help="JSON overrides of the architecture config")
    parser.add_argument("--output-root", default=str(RESEARCH_RUNS))
    args = parser.parse_args(argv)
    if args.prune_checkpoints and args.select != "val_bce":
        parser.error("--prune-checkpoints keeps only the val_bce choice")
    out = Path(args.output_root) / args.exp_id
    if args.finalize:
        if (out / "experiment.json").exists() or not (
            out / "validation_logits_by_epoch.npy"
        ).exists():
            parser.error(f"{out} has no finished training waiting for its record")
    elif out.exists() and any(out.iterdir()):
        parser.error(f"{out} exists; experiment ids are never reused")
    import torch

    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    git = git_state()
    if args.finalize:  # the result belongs to the code that trained it, saved in out/code
        git = git | {
            "code_sha256": snapshot_sha256(out / "code"),
            "finalize_code_sha256": git["code_sha256"],
        }
    else:
        snapshot_code(out / "code")
    data: Dataset = load_dataset(Path(args.dataset), Path(args.splits))
    config = models.defaults(args.arch) | json.loads(args.config)
    if args.prune_checkpoints is None:  # resolved here so the record shows the value used
        args.prune_checkpoints = bool(config.get("wide_buckets")) and args.select == "val_bce"
    X = models.features(args.arch, data.hosts, config)
    train_idx, cal_idx, val_idx = (data.indices(n) for n in ("train", "calibration", "validation"))
    if args.max_train:
        rng = np.random.default_rng(args.seed)
        train_idx = np.sort(
            rng.choice(train_idx, min(args.max_train, len(train_idx)), replace=False)
        )
    weights = None
    if args.sample_weights:
        table = json.loads(Path(args.sample_weights).read_text())
        weights = np.array([float(table.get(data.hosts[i], 1.0)) for i in train_idx])
        weights = weights / weights.mean()
    y = data.y
    torch.manual_seed(args.seed)
    model = models.build(torch, args.arch, config)
    parameters = sum(p.numel() for p in model.parameters())
    print(
        f"{args.exp_id}: {args.arch}, {parameters} parameters, {len(train_idx)} training hostnames, "
        f"seed {args.seed}, git {git['commit'][:7]}{' (dirty)' if git['dirty'] else ''}",
        flush=True,
    )
    if args.finalize:
        history, has_swa = recover_history(
            out, y[val_idx], Path(args.output_root) / f"{args.exp_id}.log"
        )
    else:
        history, has_swa = train_member(
            torch, model, X[train_idx], y[train_idx], X[val_idx], y[val_idx], args, out, weights
        )
    chosen = choose(history, args.select, has_swa)
    name = "swa.pt" if chosen == "swa" else f"epoch_{chosen:02d}.pt"
    model.load_state_dict(torch.load(out / "checkpoints" / name, weights_only=True))
    logits = {
        "calibration": score(torch, model, X[cal_idx]),
        "validation": score(torch, model, X[val_idx]),
    }
    calibrator = calibrate(logits["calibration"], y[cal_idx])
    slope, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])

    def probability(z):
        return 1 / (1 + np.exp(-np.clip(slope * z + intercept, -700, 700)))

    yv, zv = y[val_idx], logits["validation"]
    pv = probability(zv)
    threshold = select_threshold(yv, pv)
    hosts_v = [data.hosts[i] for i in val_idx]
    groups = {
        dim: np.array([describe(h)[dim] for h in hosts_v])
        for dim in ("shape", "platform", "length", "labels", "tld")
    }
    groups["source"] = data.source[val_idx]
    groups["stem"] = np.where(stem_unseen_mask(data, val_idx), "stem unseen", "stem seen")
    validation = {
        "all": evaluate(yv, zv, pv, threshold, bootstrap=200, seed=args.seed),
        "subgroups": {k: subgroup_table(yv, zv, pv, threshold, g) for k, g in groups.items()},
    }

    common_fields = {
        "release_gate": "failed",  # research artifacts never claim a passed gate
        "model_version": f"research-{args.exp_id}",
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "threshold": float(threshold),
        "best_epochs": [chosen if chosen != "swa" else -1],
    }
    if args.arch == "v3":  # the deployed format, loadable by the app's own loader
        artifact = common_fields | {
            "format": ARTIFACT_FORMAT_V3,
            "feature_version": FEATURE_VERSION_V3,
            "alphabet": ALPHABET,
            "max_length": config["max_length"],
            "embedding_dim": config["embedding_dim"],
            "channels": config["channels"],
            "kernel_size": config["kernel_size"],
            "dilations": config["dilations"],
            "hidden": config["hidden"],
            "members": [export_v3(model)],
        }
    else:
        artifact = common_fields | {
            "format": numpy_v4.FORMAT,
            "feature_version": numpy_v4.FEATURE_VERSION,
            "alphabet": ALPHABET,
            "config": models.runtime_config(args.arch, config),
            "members": [models.export_tensors(model)],
        }
    raw = json.dumps(artifact, sort_keys=True, allow_nan=False).encode()
    (out / "url_cnn.json").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    if args.arch == "v3":
        loaded = UrlCnnModel(str(out / "url_cnn.json"), sha, allow_ungated=True)
        if loaded.artifact is None:
            raise RuntimeError(loaded.reason)
    else:
        loaded = numpy_v4.V4Numpy(json.loads(raw))
    sample = list(range(min(1000, len(val_idx))))
    t0 = time.perf_counter()
    numpy_logits = np.array([loaded.logit(hosts_v[i]) for i in sample])
    latency_ms = (time.perf_counter() - t0) / len(sample) * 1000
    parity = float(np.max(np.abs(numpy_logits - zv[sample])))
    if parity > 1e-3:
        raise RuntimeError(f"NumPy/PyTorch parity failed: {parity}")

    np.savez_compressed(
        out / "scores.npz",
        validation_hosts=np.array(hosts_v),
        validation_y=yv,
        validation_logit=zv,
        calibration_hosts=np.array([data.hosts[i] for i in cal_idx]),
        calibration_y=y[cal_idx],
        calibration_logit=logits["calibration"],
    )
    record = {
        "experiment_id": args.exp_id,
        "hypothesis": args.hypothesis,
        "description": args.description,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git,
        "command": " ".join(
            [sys.executable, "-m", "research.hostname_cnn.train", *(argv or sys.argv[1:])]
        ),
        "data": data.provenance() | {"training_hostnames": int(len(train_idx))},
        "validation_hosts_sha256": canonical_hash(sorted(hosts_v)),
        "arch": args.arch,
        "config": config,
        "training": {k: v for k, v in vars(args).items() if k != "config"},
        "parameters": parameters,
        "chosen_epoch": chosen,
        "history": history,
        "calibration": {"method": "platt", "slope": slope, "intercept": intercept},
        "threshold": float(threshold),
        "validation": validation,
        "artifact_sha256": sha,
        "numpy_parity_max_abs": parity,
        "numpy_latency_ms": latency_ms,
        "wall_minutes": (time.time() - started) / 60
        + (sum(h["seconds"] for h in history) / 60 if args.finalize else 0),
        "test_split_scored": False,
    }
    if args.finalize:
        record["finalized"] = {
            "when": record["created"],
            "how": "history rebuilt by recover_history from the saved per-epoch validation "
            "logits and the run log; chosen checkpoint re-scored; artifact and scores rewritten",
        }
    (out / "experiment.json").write_text(json.dumps(record, indent=1, default=float))
    v = validation["all"]
    print(
        f"DONE {args.exp_id}: epoch {chosen}, val ROC-AUC {v['roc_auc']:.4f}, "
        f"pAUC {v['pauc_1pct']:.4f}, recall@0.5% {v['recall_at_0.005_fpr']:.4f}, "
        f"recall@1% {v['recall_at_0.01_fpr']:.4f}, parity {parity:.1e}, "
        f"{record['wall_minutes']:.0f} min",
        flush=True,
    )
    return record


if __name__ == "__main__":
    main()

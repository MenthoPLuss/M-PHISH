# Hostname CNN v3: architecture and training

The default trainer now builds `m-phish-charcnn-v3`, a masked, gated residual CNN.
It exports the same checksum-pinned JSON deployment contract as earlier versions.
The API still uses NumPy only; PyTorch is an offline training dependency.

## Findings from the existing model

The saved `runs/cnn-1/results.json` records 219,036 unique PhiUSIIL hostnames, with
33,430 in the domain-disjoint test split. Its v1 model achieved ROC-AUC 0.91418,
precision 94.91%, recall 71.88% and FPR 2.54% at threshold 0.5. It failed the
release gate. These are historical measurements, not a fresh evaluation of v1.

The working tree already contained a v2 residual CNN and ensemble implementation.
The upgrade preserves those changes and both existing artifact formats.

| Finding | v3 change |
| --- | --- |
| v1 was still the default | v3 is the default; `--arch v1` and `--arch v2` remain available |
| Only the last 64 characters were retained | Default context is 253 characters; truncation count is recorded |
| v2 convolutions and pooling included padding activations; batch normalization included padded positions | Every stage masks padding; normalization operates across channels at each character |
| Global pooling could lose the distinction between a subdomain and the registered name | Character roles plus a dedicated registered-name mean pool |
| Fixed max/mean pooling could miss distributed evidence | Learned attention pooling alongside max and mean pooling |
| Full convolutions made a deeper stack expensive | Dilated depthwise convolution and pointwise gated residual branches |
| All non-legitimate CSV values silently became phishing | Both label values are explicit; missing and unknown labels fail |
| A failed validation threshold search reverted to 0.5 | Select under confidence-bound constraints; report infeasibility and retain an FPR-constrained fallback |
| Custom CSVs inherited PhiUSIIL licensing metadata | Custom provenance no longer claims the UCI dataset license |
| No saved learning curves or device controls | Saved configuration/history, bounded batches, CPU/CUDA selection and deterministic operations |

## Architecture

```text
normalized hostname (lowercase, IDNA, one leading www. removed)
  -> character embedding + public-suffix/domain-role embedding
  -> stem convolution + per-position LayerNorm + ReLU + mask
  -> five residual blocks, dilation 1, 2, 4, 8, 16:
       depthwise convolution -> LayerNorm -> ReLU -> mask
       -> pointwise gated linear unit -> dropout -> learned residual scale
       -> residual addition -> mask
  -> concatenate masked max, mean, attention and registered-name mean pools
  -> Dense + ReLU + dropout -> scalar logit
  -> mean logits across independently seeded members (when enabled)
  -> held-out Platt calibration -> validation-selected threshold
```

Defaults: embedding 32, channels 64, kernel size 5, hidden width 128, dropout 0.3,
89,442 parameters per member. The dilated stack has a 129-character receptive
field before pooling. Attention pools evidence from the entire retained hostname.
This is attention pooling, not a Transformer or an explanation of causality.

Padding is masked before and between convolutions and excluded from pooling.
LayerNorm has no batch statistics. Shared leading padding can therefore be trimmed
from each training batch, and NumPy inference processes only the actual retained
characters. This keeps full-hostname coverage without always computing 253 positions.
Tests compare predictions across padding widths and batch compositions.

## Train

From the repository root in PowerShell, with the backend/evaluation dependencies
already installed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-cnn.txt

# Small functional check; synthetic data can never pass the release gate.
.\.venv\Scripts\python.exe -m backend.app.services.ml.train_url_cnn --synthetic-smoke --max-rows 500 --epochs 2 --output runs\v3-smoke

# Full real-data run; downloads only when the dataset is absent.
.\.venv\Scripts\python.exe -m backend.app.services.ml.train_url_cnn --download-phiusiil data\phiusiil --output runs\v3-full

# Optional ensemble; requires roughly three times the training and inference work.
.\.venv\Scripts\python.exe -m backend.app.services.ml.train_url_cnn --download-phiusiil data\phiusiil --ensemble 3 --device auto --output runs\v3-ensemble

# Custom CSV with conventional 0=legitimate / 1=phishing labels.
.\.venv\Scripts\python.exe -m backend.app.services.ml.train_url_cnn --csv data\custom.csv --legit-label 0 --phishing-label 1 --output runs\v3-custom
```

Every output directory must be new or empty. Default training uses at most 30
epochs with patience 5, batches of 128, AdamW, warmup plus cosine learning-rate
decay, gradient clipping and best-validation-loss restoration. The validation
loss is unweighted BCE. `--class-weight balanced` optionally weights the training
loss using the training split's negative/positive ratio; calibration and evaluation
retain the observed class distribution. Weighting is off by default because it
changes the precision/recall tradeoff and should be evaluated for the dataset.

`--device auto` selects CUDA when available and otherwise CPU. CUDA requires a
compatible PyTorch installation; CPU is the tested platform in this workspace.
Use `--threads` to control CPU use. Fixed seeds and deterministic operations help
repeat a run on the same software/hardware; cross-platform bitwise reproducibility
is not guaranteed. Training duration depends on data, context lengths and hardware.

The implementation follows PyTorch's [LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html),
[AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) and
[BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
semantics. The NumPy implementation is separately checked against exported PyTorch weights.

## Evaluation and artifacts

Normalization, hostname deduplication and conflicting-label removal happen before
splitting. Registrable-domain groups (including private PSL suffixes) remain
disjoint across train, validation, calibration and test. All four splits must have
both classes. Training sees only the training split; validation selects epochs
and the operating threshold; calibration fits Platt scaling. Test metrics do not
select weights or thresholds.

Threshold selection maximizes validation recall among thresholds meeting the
release confidence bounds: precision lower 95% bound >= 95%, recall lower bound
>= 90%, and FPR upper bound <= 1%. If no threshold qualifies, it chooses under the
FPR bound where possible, otherwise the empirical 1% FPR limit. This fallback
does not qualify the model for release. Both validation and test must pass; real
labels, absence of detected trivial shortcuts, and successful export parity are
also required. The existing Wilson bounds treat hostnames as observations; they
do not account for all campaign or collection correlations.

| Output | Purpose |
| --- | --- |
| `url_cnn.json` | Packed finite float32 parameters, calibration and deployment gate |
| `results.json` | Test and validation metrics, intervals, baseline, parity and latency |
| `history.json` | Per-member epochs, train/validation losses, validation PR-AUC, LR and duration |
| `training_config.json` | Arguments, seed and selected device |
| `manifest.json` | Dataset checksum/provenance, label mapping and truncation count |
| `splits.json` | Hashed hostname membership of each split |
| `release_targets.json` | Required confidence bounds |
| `checksums.txt` | SHA-256 of every output file |

Only a successfully parity-checked export can be marked passed. Failed artifacts
remain blocked by default. The existing local-only ungated override is unchanged;
no environment configuration is changed automatically by training.

The Research page reads the existing `evaluation-v2` format. Load an accepted
artifact through `ML_CNN_ARTIFACT_PATH` and `ML_CNN_ARTIFACT_SHA256`; use
`ML_RESEARCH_RESULTS` for its report, then restart the API and worker.

## Limits

This model sees hostnames, not paths, page content, credentials, or browser actions.
It cannot reliably distinguish benign and phishing pages on the same compromised
hostname. Collection artifacts in PhiUSIIL can inflate apparent quality, and an
absence of shortcut warnings does not prove their absence. Domain-disjoint random
holdout does not establish temporal or cross-dataset robustness. A deeper model
or an ensemble must earn deployment through measured performance; architecture
alone does not make the detector production-ready.

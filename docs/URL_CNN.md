# Domain-name CNN

**Current default: v3.** See [the v3 architecture, analysis and training guide](URL_CNN_V3.md).
The architecture and timing below describe the original v1 baseline; the commands
explicitly select v1 for reproducibility. v1 and v2 artifacts remain supported.

A character-level convolutional neural network that scores a website's **hostname**
(for example `secure-paypa1-verify.com`) for phishing-like patterns. It complements
the rule engine and the multimodal linear model. It never marks a site safe.

| Part | File |
| --- | --- |
| Training (offline, PyTorch) | `backend/app/services/ml/train_url_cnn.py` |
| Inference (app, numpy only) | `backend/app/services/ml/url_cnn.py` |
| Wiring into reports | `backend/app/analysis.py` (`url_cnn_prediction`, `select_warning_model`) |
| Dashboard panel | `web/components/report.tsx` (`DomainCnnPanel`) |
| Tests (no PyTorch needed) | `backend/tests/test_url_cnn.py` |

## How it works

```
hostname ──► lowercase, punycode, drop leading "www." ──► last 64 characters, right-aligned
        ──► embedding (41 symbols × 32)
        ──► 4 parallel Conv1d (kernel 2, 3, 4, 5; 64 filters each) + ReLU
        ──► global max pool ──► concat (256) ──► Dense 64 + ReLU ──► Dense 1 ──► logit
        ──► Platt calibration (held-out split) ──► probability vs. validation threshold
```

About 47,000 parameters. It trains on a normal CPU and predicts in under a millisecond per hostname.

The convolution kernels act as learned n-gram detectors. They pick up patterns such as
`paypa1`, `-verify`, `secure-login`, digit runs, unusual TLDs and random-looking
strings, wherever they appear in the name.

## Safety rules (the same as the project's linear model)

- **No pickle or `.pt` files in the app.** Training exports the weights to JSON. The app
  loads it only when its SHA-256 matches `ML_CNN_ARTIFACT_SHA256`, then validates every
  shape and checks that all values are finite.
- **Release gate.** On the held-out test split, the 95% interval lower bounds must reach
  precision ≥ 95% and recall ≥ 90%, and the upper bound of the false-positive rate must be
  ≤ 1%. There must also be no single-feature shortcut, and the data must be real labelled
  data. Otherwise the artifact is marked `failed` and the app refuses it.
  `ML_CNN_ALLOW_UNGATED=true` overrides this for local experiments only, and is ignored
  when `APP_ENV=production`.
- **Warning floor only.** When the CNN's probability reaches its threshold, the report
  can rise to at least HIGH (score 50). The CNN never adds to rule scores, never lowers
  them, and never produces "safe". If the multimodal model is also configured and
  warning, that model takes priority.
- **Grouped splits.** Hostnames are grouped by registrable domain (for example,
  `login.evil.com` and `pay.evil.com` stay in the same split), so the test score is not
  inflated by near-duplicates.
- **Parity check.** After export, the numpy forward pass is compared with PyTorch on up to
  2,000 test hostnames. Training aborts if any logit differs by more than 1e-3.

## Training on Windows (PowerShell, from the `M-PHISH` folder)

1. Install PyTorch (CPU, about 200 MB):
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r backend\requirements-cnn.txt
   ```
2. Optional one-minute smoke test on generated data. Its gate always fails, by design:
   ```powershell
   .\.venv\Scripts\python.exe -m backend.app.services.ml.train_url_cnn --arch v1 --synthetic-smoke --epochs 4 --output runs\smoke
   ```
3. Train on PhiUSIIL. This downloads the dataset from UCI on first use. Training is light: this sandbox
   processed about 8,000 hostnames per second on CPU, so an epoch takes well under a minute or two:
   ```powershell
   .\.venv\Scripts\python.exe -m backend.app.services.ml.train_url_cnn --arch v1 --download-phiusiil data\phiusiil --output runs\cnn-1
   ```
   For a quick first try, add `--max-rows 30000 --epochs 3`. Every run needs a new
   `--output` folder; earlier runs are never overwritten.

The script prints the test precision, recall and false-positive rate, the release gate
result, the parity check, and the exact settings to use. It also refuses to train if
famous legitimate sites (google.com, wikipedia.org, …) appear to be labelled as phishing,
which protects against an inverted label column. PhiUSIIL uses `label` = 1 for
legitimate; `--legit-label` changes this.

VS Code: **Terminal → Run Task → "ML: install PyTorch for CNN training"**, then
**"ML: train URL CNN (PhiUSIIL)"**.

## Using the trained model

Add the printed values to `.env`. The VS Code launch configurations read this file.
If you start servers by hand, set them in each terminal instead (`$env:NAME = "value"`):

```
ML_CNN_ARTIFACT_PATH=C:\...\M-PHISH\runs\cnn-1\url_cnn.json
ML_CNN_ARTIFACT_SHA256=<printed sha256>
ML_RESEARCH_RESULTS=C:\...\M-PHISH\runs\cnn-1\results.json
```

Restart the **API** and the **local worker**, because both build reports. Each new report
then shows a **Domain-name CNN** panel. The **Research** page shows the CNN next to a
trivial baseline (a logistic model on length, dots, hyphens, digits and `.com`).

## Reading the results honestly

- Public URL datasets contain collection artifacts. If the trivial baseline scores almost
  as well as the CNN, or the run prints shortcut warnings, the high accuracy partly comes
  from how the data was collected rather than from real phishing patterns.
- The model sees only the hostname. Phishing hosted on legitimate or compromised domains
  (`sites.google.com/...`, hacked blogs) cannot be caught by it. That is the job of the
  page and form-destination checks.
- Phishing domains change over time. Retrain periodically, and treat any accuracy figure
  as specific to this dataset and split, not as real-world performance.

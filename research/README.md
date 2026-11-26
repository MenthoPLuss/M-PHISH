# Hostname CNN research

Research toward the next version of M-PHISH's hostname model. The deployed model
(`cnn-v3-fresh`) is unchanged, and the app never imports anything in this folder.

## Status (2026-11-26)

- **Phase 1 audit:** done. See `phase1/`.
- **Wave 1 (one seed per idea):** finished.
  - Promoted:
    - More legitimate training data (H1).
    - Stacking the CNN with a character n-gram model (H4). Superseded by H12.
  - Not confirmed on 3 seeds: suffix dropout (H5b).
  - Rejected: schedule and SWA (H3), checkpoint selection (H2a), focal loss (H2c), and
    structure inputs (H5a).
- **Wave 2:** finished.
  - **Wide & Deep CNN (H12): confirmed on 3 seeds.** It adds a linear path over hashed
    character n-grams to the v3 CNN.
    - Recall at 1% false-positive rate: +3.9 points.
    - One model beats the deployed 3-member ensemble.
- **Waves 3–4:** finished.
  - **Wide & Deep on the H1 data:**
    - Confirmed: false alarms on legitimate hosting-platform sites fall from about 37% to 14%.
    - The cost: about 0.6 points of recall on the original validation set. That marginally
      misses H1's pre-registered limit of 0.5, so adopting it is a product decision.
    - Weighting the added data at one half does not change this trade-off.
  - **Suffix dropout:** rejected on Wide & Deep too.
- **Calibration (H10, 2026-11-26):** Platt calibration stays.
  - One website with many user subdomains (`yupoo.com`) causes a quarter of validation
    false alarms. Measured by site, the realistic false-alarm rate at the threshold is
    0.6-1.4%.
- **Final build (2026-11-26):** `F-c13-ensemble3`, the 3-seed ensemble of Wide & Deep on the
  H1 data.
  - Recall at 1% false-positive rate: +2.3 points over the deployed model (validation only).
  - 18 MB; 2.9 ms per hostname.
- **Backend loader (2026-11-27):** the app can now run the final build, but does not by
  default.
  - New format `m-phish-charcnn-v3-wide`: the v3 network plus one n-gram table per member.
  - `export_production.py` writes `runs/cnn-v3wide-c13-ensemble3/url_cnn.json`. The app's
    logits are identical to the build's on all 152,476 calibration and validation
    hostnames.
  - The app still serves `cnn-v3-fresh`. To switch, see `docs/URL_CNN_V3.md`.
- **Final test:** run once on a sealed set no earlier than 2026-12-08.

Every decision, with its evidence, is in [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

## Layout

| Path | Contents |
| --- | --- |
| `RESEARCH_LOG.md` | Dated decision log: results, confidence intervals, failures and fixes |
| `phase1/` | Audit reports, the audit script and its output |
| `hostname_cnn/train.py` | Trainer; its defaults reproduce the production recipe |
| `hostname_cnn/models.py`, `numpy_v4.py` | Candidate networks (v3, v4, Wide & Deep) and their NumPy runtime |
| `hostname_cnn/compare.py`, `metrics.py` | Paired bootstrap comparisons, fixed-recall read-outs, ensembles and stacking |
| `hostname_cnn/datasets.py`, `ngram.py`, `lockbox.py` | Dataset variants, the n-gram baseline, and the sealed test set |
| `hostname_cnn/calibration.py` | H10: calibrator comparison, threshold stability, false alarms by site |
| `hostname_cnn/final_build.py` | Merges finished runs into one ensemble artifact and checks it on every hostname |
| `hostname_cnn/export_production.py` | Converts a final build to the app's `m-phish-charcnn-v3-wide` format and checks the app's loader on every hostname |
| `hostname_cnn/tests/` | Tests for parity, invariances, metrics and safeguards |

## Running

From the repository root:

```
python -m pytest research -q
python -m research.hostname_cnn.train --exp-id <id> --hypothesis <H> [options]
python -m research.hostname_cnn.compare diff --control <run> --candidates <run> ...
python -m research.hostname_cnn.calibration --name <name> --runs <run> ...
python -m research.hostname_cnn.final_build --name <name> --runs <run> ...
python -m research.hostname_cnn.export_production --build F-<name>
```

Runs are written to `runs/research/<id>/`. That folder and the datasets under `data/feeds/`
are not in git: they are built locally from public sources.

## Rules the code enforces

- **Test split:** never scored. Screening reads the validation split only.
- **Sealed set:** the final test set opens only with `M_PHISH_LOCKBOX_UNLOCK=phase-11`.
- **Provenance:** each run records:
  - the git commit and a hash of its exact source files, with a copy of those files;
  - the dataset and split hashes;
  - its full configuration.
- **Parity:** every exported model must match its NumPy runtime within 1e-3 (target 1e-5).

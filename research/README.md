# Hostname CNN research

Research toward the next version of M-PHISH's hostname model. The deployed model
(`cnn-v3-fresh`) is unchanged, and the app never imports anything in this folder.

## Status (2026-11-25)

- **Phase 1 audit:** done. See `phase1/`.
- **Wave 1 (one seed per idea):** finished.
  - Promoted:
    - More legitimate training data (H1).
    - Suffix dropout (H5b).
    - Stacking the CNN with a character n-gram model (H4).
  - Rejected: schedule and SWA (H3), checkpoint selection (H2a), focal loss (H2c), and
    structure inputs (H5a).
- **In progress:**
  - 3-seed confirmations.
  - A Wide & Deep CNN (H12).
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
| `hostname_cnn/tests/` | Tests for parity, invariances, metrics and safeguards |

## Running

From the repository root:

```
python -m pytest research -q
python -m research.hostname_cnn.train --exp-id <id> --hypothesis <H> [options]
python -m research.hostname_cnn.compare diff --control <run> --candidates <run> ...
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

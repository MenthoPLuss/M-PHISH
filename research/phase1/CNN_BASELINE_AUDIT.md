# CNN baseline audit: v3-fresh

Phase 1, 24 Nov 2026. Read-only. No file in the M-PHISH repository was changed. All
numbers below were read from run artifacts or re-measured by
`scripts/phase1_audit.py`, which scores only the validation split.

**Verdict.** `runs/cnn-v3-fresh` is fully reconstructable from its artifact, dataset and
split checksums. Every historical claim checked against the artifacts holds. Three
reproducibility gaps must close before Phase 5:
- The run does not record its git commit.
- The comparison scripts behind several claims live in a temporary folder.
- The test split has been looked at too often to serve as a locked final test.

## 1. Identity of the baseline (immutable reference)

| Item | Value |
| --- | --- |
| Run folder | `runs/cnn-v3-fresh` (git-ignored; must never be modified) |
| Artifact | `url_cnn.json`, 1.44 MB, format `m-phish-charcnn-v3`, feature version `hostname-chars-roles-masked-v1` |
| Artifact SHA-256 | `0b7742a96f59a26c32a7f7c0dd6da284347b27005febb0a368a8c86f381fc7fb` |
| Git commit | **Not recorded by the trainer.** Training started 24 Nov 12:20 IST from `110e279` plus the then-uncommitted `--reuse-splits` change, committed unchanged as `54b5d74` at 17:03. This is reconstructed from the session record, not provable from the artifact. |
| Dataset | `data/feeds/2026-11-24/combined.csv`, version `csv-75ef558ae717` |
| Dataset SHA-256 | `75ef558ae717dc504b74e6cb89e76f3f1ffea34bf27c09ce9a3b3dda6173a16e` |
| Dataset sources | PhiUSIIL CSV `a236549c…d06d1c6`; Phishing.Database ACTIVE list `70c6c2e2…e37326` (MIT); CrUX 202610 `410165bb…ad14911` (CC BY 4.0); builder seed 42 |
| Rows | 435,795 rows -> 419,036 unique hostnames (184,244 phishing, 234,792 legitimate); 58 conflicting hostnames dropped |
| Splits | `splits.json` SHA-256 `ac6b872e3797ed7d53d769bf75a43eeaa76b8bdf4e7f8bda3f8772379b14656f`; PhiUSIIL assignments reused from `cnn-v3-ens3` (`8433e1b9…d969e55a`) |
| Split sizes | train 231,431 · calibration 63,071 · validation 62,045 · test 62,489 |
| Parameters | 3 members x 89,442 = 268,326 |
| Training | 3 members, seeds 42/43/44, best epochs 16/21/14 of 21/26/19 run; 272 min of epoch time (227 to 290 s per epoch) on 14 CPU threads |
| Inference | 2.21 ms per hostname (NumPy, ensemble, single hostname); NumPy/PyTorch max logit difference 2.18e-6 |
| Calibration | Platt on the calibration split: p = sigmoid(1.2675 z + 0.3826) |
| Threshold | 0.8931, chosen on the validation split (Wilson upper bound on FPR <= 1%, recall maximised) |
| Environment | Windows 11, Python 3.12.10, torch 2.15.0+cpu, NumPy 2.5.4, scikit-learn 1.9.1, tldextract 5.4.0 (bundled PSL), 20 logical cores, 15.6 GB RAM, no GPU |

### Recorded test result (from `results.json`, computed once at training time)

| Metric | Value | 95% interval |
| --- | ---: | --- |
| Precision | 98.14% | 97.93 to 98.33% |
| Recall | 62.78% | 62.21 to 63.36% |
| False-positive rate | 0.92% | 0.82 to 1.02% |
| ROC-AUC | 0.9118 | |
| PR-AUC | 0.9176 | |
| Brier score | 0.1064 | |
| Support | 35,257 legitimate, 27,232 phishing | |

The release gate failed on recall (lower bound 62.2% against 90%). The shortcut audit on
the training split found no single trivial feature with balanced accuracy >= 0.97. The
5-feature logistic baseline (length, dots, hyphens, digits, `.com`) scored ROC-AUC 0.789
on the same test split.

## 2. Verification of the historical claims

| Claim in the brief | Artifact evidence | Verdict |
| --- | --- | --- |
| v1 about 0.914 ROC-AUC | `runs/cnn-1/results.json`: 0.91418 on the PhiUSIIL test split (33,430 hostnames) | Confirmed |
| v3 about 0.915 | `runs/cnn-v3-phiusiil/results.json`: 0.9150; 1 member, 10 epochs, previous defaults | Confirmed (single seed) |
| 3-member ensemble about 0.923 | `runs/cnn-v3-ens3/results.json`: 0.9225, same split checksum `8433e1b9…` as v1 | Confirmed |
| Fresh-data model about 0.899 on PhiUSIIL | Session evaluation output `eval_fresh_3.json`: 0.8990 on the 33,430 PhiUSIIL test hostnames | Confirmed, but by a script outside the repo |
| Fresh-test about 0.924 | Same file: 0.9238 on 29,059 fresh test hostnames | Confirmed, same caveat |
| Serious false positives on legitimate subdomains | Same file: `cnn-v3-ens3` flagged 73.3% of 5,595 unseen CrUX subdomain sites | Confirmed |
| Fresh data reduced them | Same file: 3.1% for `cnn-v3-fresh` | Confirmed |
| Bare-domain phishing remains hard | Same file: 65.7% of 9,254 fresh bare phishing caught; re-measured on validation in `CNN_FAILURE_ANALYSIS.md` | Confirmed |
| Width gave limited improvement | Session log `exp_results.jsonl`: 128 channels 0.9161 vs 0.9140 to 0.9148 at 64 channels (validation) | Directionally confirmed; **single seed**, so the 0.0013 to 0.0021 gain is within plausible seed noise |
| Averaging members improved validation | Docs and session output: 3 seeds averaged 0.9187 vs best single 0.9148 | Confirmed; the 0.004 gain is about 5 times the seed spread (0.0008) |

## 3. Reproducibility gaps

1. **No git commit in run metadata.** `manifest.json`, `training_config.json` and the artifact
   have no commit field. A run cannot be tied to its code without the session record.
2. **Evaluation code outside the repo.** The cross-run comparisons (`eval_fresh.py`,
   `ens.py`, `testcmp.py`, `drift.py`) are in a temporary session folder that can be wiped.
   Four of the claims above depend on them.
3. **The test split is no longer a locked test.** The trainer scores the test split in
   every run, and it was used in two decisions: v1 -> v3 x3 on 24 Nov, 07:29, and
   v3 x3 -> v3-fresh at 17:02. It remains valid for *reporting* but not as an untouched
   final test. Phase 1 therefore scored only the validation split.
4. **Provenance mislabel.** `cnn-v3-ens3` was trained with `--csv` on the PhiUSIIL file, so
   its manifest calls PhiUSIIL "Operator-supplied; not independently verified" instead of
   recording the UCI licence.
5. **Oldest run is incomplete.** `runs/cnn-1` has no `training_config.json` or
   `history.json`. That predates the v3 trainer.
6. **Validation is used twice.** Early stopping and threshold selection both use the
   validation split, so validation metrics are slightly optimistic. This is acceptable for
   diagnosis, but the final comparison needs a split used for neither.

## 4. Baseline re-measured on the validation split (62,045 hostnames)

See `results/phase1_audit.json` for every number. Headline values are in
`CNN_FAILURE_ANALYSIS.md` §2, with bootstrap 95% intervals.

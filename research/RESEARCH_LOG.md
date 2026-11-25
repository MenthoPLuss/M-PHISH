# Hostname CNN v4 research log

Newest entries last. Each entry records what was decided and why, and points to the run or
file that holds the evidence. Numbers come from `runs/research/*/experiment.json`.

## 2026-11-24

### Phase 1 approved (user: "approve with your recommendations")

- **Order:** data and objective (H1, H2, H3) before architecture (H5, H6), then pretraining.
- **Release gate:** unchanged in the app. Research adds a separate, informational score on
  attacker-registered hostnames.
- **n-gram member (H4):** a candidate. It must earn its place by the same promotion rule.
- **Code:** branch `cnn-v4-research`, new package `research/hostname_cnn/`, no edits to
  `backend/`. Backend lint and the 92 backend tests still pass. Nothing is committed yet;
  each run stores a copy of its source files and their SHA-256 (`code_sha256`).
- **Locked test:** wait for snapshots taken on or after 2026-12-08.

### P0 prerequisites done

| Item | Result |
| --- | --- |
| Harness | `common`, `metrics`, `train`, `compare`, `score`, `datasets`, `ngram`, `tags`, `lockbox`, `models`, `numpy_v4`; 8 research tests pass |
| Provenance | every run records git commit, branch, dirty flag, code hash, dataset and splits SHA-256, full config, history, chosen epoch |
| Lockbox part 1 | 200,000 CrUX 2026-10 legitimate hostnames (49,127 with subdomains, 2,013 on hosting platforms) sharing no hostname, registered domain or stem with training data; SHA-256 `b6f49803…3c557d`; readable only with `M_PHISH_LOCKBOX_UNLOCK=phase-11` |
| Lockbox parts 2–3 | reference snapshots recorded (Phishing.Database `70c6c2e2…`, CrUX 202610 `410165bb…`); build after 2026-12-08 |
| Control | `P0.5-control-random-s42` reproduces production epoch 1 exactly (train 0.5101, validation 0.4044 vs 0.4043) |

### Decisions made while preparing wave 1

1. **Length-bucketed batching under test.** Epochs take 163 s vs 287 s (1.8x). A seed-matched
   pair of controls (random vs bucketed) decides whether it is accuracy-neutral. All wave-1
   screens use bucketing and are compared only with the bucketed control, so their
   comparisons are valid either way.
2. **Multi-scale linear stem dropped from H6.** Parallel linear convolutions with kernels
   3/5/7, summed or fused by a 1x1, are exactly one kernel-7 convolution. They add no
   capacity. Multi-scale remains only where a nonlinearity follows (kernel-7 depthwise blocks).
3. **Parameter matching.** Default v4 had 131,266 parameters vs 89,442 because component
   pools widen the head input. H5a uses hidden 64 (about 94,000 parameters) and H6 uses
   C = 48 (about 92,000), so any gain is not just "more parameters".
4. **H8 training run dropped.** The first rule (popular *registered domain*) wrongly tagged
   attacker-chosen subdomains on free hosting and dynamic DNS, such as `*.work.gd` and
   `*.liveblog365.com`. The narrowed rule (the hostname itself is a CrUX site or Tranco domain,
   or has an institutional suffix) tags 884 training phishing rows (0.9%). That is too few for
   down-weighting to produce a measurable effect. The tags stay as an evaluation lens.
5. **H1 dataset `h1-v2`.**
   - Added 90,120 legitimate hostnames: 5,101 hosting-platform, 16,173 longer than 30
     characters, 68,846 with 4+ labels.
   - Removed 3,292 linked to the phishing list and 16,580 touching the lockbox.
   - New registered domains are split by a hash of the domain (55/15/15/15).
   - All 419,036 earlier hostnames keep their split.

### Wave 1 queue (seed 42, bucketed, 7 threads, validation only)

| Lane | Experiment | Hypothesis | Change |
| --- | --- | --- | --- |
| A | E3a-cosine20-last-s42 | H3 | cosine over 20 epochs to 1%, keep last epoch |
| A | E5b-suffixdrop0.2-s42 | H5 | suffix dropout 0.2 (v3) |
| A | E1-h1v2-s42 | H1 | dataset h1-v2 |
| A | E2b-pauc0.25-s42 | H2 | + one-way pAUC hinge |
| B | E5a-v4-structure-components-s42 | H5 | label/position embeddings + component pools |
| B | E6-v4-convnext-s42 | H6 | ConvNeXt-1D blocks |
| B | E3b-cosine20-swa5-s42 | H3 | SWA of last 5 epochs |
| B | E2c-focal1-s42 | H2 | focal loss, gamma 1 |
| — | E4-ngram-markers | H4 | done: validation ROC-AUC 0.9162, pAUC 0.764, recall at 0.5% FPR 55.1%, at 1% FPR 61.0% |
| — | E2a | H2 | computed from the controls' saved per-epoch logits (no training) |

### 19:20 — control (bucketed) and H2a

- `P0.5-control-bucketed-s42`: best epoch 19 of 24; validation ROC-AUC 0.9020, pAUC 0.741,
  recall at 0.5% FPR 53.4%, at 1% FPR 59.4%; NumPy parity 5.0e-6; 66 min. The production
  member with the same seed scored ROC-AUC 0.9024 and recall at 1% FPR 59.4% on the same
  hostnames, so the bucketed control reproduces production.
- **H2a rejected.** Every epoch was rescored on the calibration split and an epoch chosen
  there; the choice was judged on validation (`comparisons/E2a__P0.5-control-bucketed-s42.json`).

  | Rule | Epoch | Validation recall at 1% FPR | vs production rule (95% CI) |
  | --- | ---: | ---: | --- |
  | production (validation BCE) | 19 | 59.4% | — |
  | calibration BCE | 24 | 59.5% | +0.03 pts [−0.44, +0.50] |
  | calibration pAUC / recall at 0.5% FPR | 17 | 58.7% | −0.70 pts [−1.22, −0.36] |

  Epoch-to-epoch swings in low-FPR metrics are noise that does not transfer between splits.
  The production selection rule stays. H2 continues only through the loss (E2b, E2c).

### 20:15 — faithful control; training noise; bucketing verdict

`P0.5-control-random-s42` (faithful): best epoch 19 of 24, validation ROC-AUC 0.9034,
pAUC 0.747, recall at 0.5% FPR 52.7%, at 1% FPR 59.0%; 116 min.

**Training noise measured.** Four runs of the identical recipe with random batches: the
three production members (seeds 42–44, 14 threads) and the research control (seed 42,
7 threads). All are scored on the same 62,045 validation hostnames.

| Metric | Mean of 4 | SD | Bucketed control |
| --- | ---: | ---: | ---: |
| ROC-AUC | 0.9031 | 0.0006 | 0.9020 |
| pAUC (FPR <= 1%) | 0.7495 | 0.0035 | 0.7413 |
| Recall at 0.1% FPR | 36.8% | 3.1 pts | 28.5% |
| Recall at 0.25% FPR | 47.3% | 1.3 pts | 45.2% |
| Recall at 0.5% FPR | 53.7% | 0.6 pts | 53.3% |
| Recall at 1% FPR | 59.3% | 0.6 pts | 59.4% |

- The paired bootstrap over validation hostnames ignores this training noise, which dominates
  at the tail. **Decision rule updated:** a screen is promoted only if its difference from
  the seed-matched control exceeds 2 training SDs *and* the paired-bootstrap 95% interval
  excludes 0. Recall at 0.1% FPR is reported but never decides on one seed.
- **Bucketing is not proven neutral.** At 0.5–1% FPR it matches; at 0.1% FPR, pAUC and
  ROC-AUC it sits below all four random-batch runs (about 2–3 SD). Wave 1 keeps it for cheap
  *screening*, compared only with the bucketed control. Every promoted candidate is confirmed
  with random batches over 3 seeds against random-batch controls.

### 23:45 — wave 1 screens read; three failures diagnosed and fixed

All screens: seed 42, bucketed batches, paired bootstrap (500) against `P0.5-control-bucketed-s42`
on the 62,045 shared validation hostnames (`runs/research/comparisons/diff__*.json`).

| Screen | Hyp. | ROC-AUC | pAUC | Recall at 0.5% FPR | Recall at 1% FPR | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| E5b suffix dropout 0.2 | H5 | +0.0030 [+0.0017, +0.0045] | +0.0137 [+0.0079, +0.0197] | +1.77 [+0.50, +2.88] | +1.17 [+0.26, +1.96] | **promoted** to 3-seed confirmation |
| E1 dataset h1-v2 | H1 | −0.0059 [−0.0075, −0.0044] | +0.0139 [+0.0068, +0.0214] | +1.07 [−0.26, +2.54] | +0.12 [−0.47, +1.26] | **promoted** on its pre-registered metric (below) |
| E6 ConvNeXt-1D | H6 | +0.0027 [+0.0013, +0.0041] | +0.0131 [+0.0068, +0.0195] | +0.44 [−0.65, +1.54] | −0.30 [−1.00, +0.65] | not promoted: ranking gain, no operating-point gain |
| E5a v4 structure + component pools | H5 | +0.0012 [−0.0002, +0.0027] | +0.0031 [−0.0040, +0.0112] | +0.03 [−1.34, +1.36] | +0.07 [−0.77, +0.84] | null; not promoted |
| E3b cosine 20 + SWA 5 | H3 | −0.0012 [−0.0025, +0.0000] | +0.0034 [−0.0028, +0.0095] | −0.97 [−2.35, +0.22] | −1.26 [−1.87, −0.50] | rejected |
| E3a cosine 20, last epoch | H3 | −0.0013 [−0.0027, −0.0001] | +0.0027 [−0.0035, +0.0085] | −1.31 [−2.52, +0.24] | −1.25 [−1.95, −0.53] | rejected |
| E2b pAUC hinge 0.25 | H2 | — | — | — | — | invalid run (collapsed); corrected rerun queued |

Recall columns are percentage points.

- **The bucketed control is an unlucky draw at the tail.** Every screen beats it on recall at
  0.1% FPR (+6.5 to +10.8 pts, including the rejected E3b) and most on pAUC. It sat 2.3 SD
  below the four random-batch runs there. Screens are therefore decided on ROC-AUC and recall
  at 0.5% and 1% FPR, where it matched the random runs; tail metrics wait for the 3-seed
  confirmation against random-batch controls.
- **E5b mechanism matches H5.** Gains concentrate where evidence lies outside the suffix:
  recall at 1% FPR within subdomain hostnames +8.4 pts, hosting platform +6.1 pts (ROC-AUC
  +0.024), longer than 50 characters +13.8 pts; bare `.com` +0.9 pts. No subgroup loses more
  than 0.9 pts. Against the mean of the four random-batch runs it is still +3.2 SD
  (ROC-AUC), +2.3 SD (recall at 0.5%) and +2.2 SD (recall at 1%).
- **E1 (H1), pre-registered metric: FPR at fixed recall on platform, long and deep
  legitimate hostnames.** Thresholds give 59% recall on the shared validation phishing
  (about today's 1% FPR point). The control was scored on the 13,916 added validation
  hostnames with `score --only-new` (`comparisons/fixed_recall__*.json`).

  | Group | Control | E1 | Difference [95% CI] |
  | --- | ---: | ---: | --- |
  | Legitimate, hosting platform (already in validation, n=137) | 43.8% FPR | 16.1% | −27.7 pts [−35.0, −20.4] |
  | Legitimate, hosting platform (added, n=774) | 43.0% | 18.9% | −24.2 pts [−27.7, −21.1] |
  | Legitimate, longer than 30 (already in validation, n=438) | 13.0% | 3.7% | −9.4 pts [−12.1, −6.4] |
  | Legitimate, 4+ labels (added, n=11,979) | 3.7% | 1.5% | −2.2 pts [−2.5, −1.8] |
  | Other legitimate (n=35,308) | 0.95% | 0.90% | −0.05 pts [−0.16, +0.06] |
  | Phishing, hosting platform (n=8,799) | 98.5% recall | 94.4% | −4.1 pts [−4.7, −3.6] |
  | Phishing, longer than 30 (n=6,575) | 94.5% recall | 90.8% | −3.6 pts [−4.3, −2.9] |

  The overall primary metric stays within the pre-registered −0.5 pts (recall at 0.5% FPR
  +1.07, at 1% +0.12). The platform-phishing recall loss is the trade H1 named in advance. The
  ROC-AUC drop on the original validation is expected: that split carries the same
  source–label confound (8,799 platform phishing vs 137 platform legitimate), so it rewards the
  shortcut H1 removes.
- **E5b on the same added hostnames** cuts platform-legitimate FPR only from 43.0% to 37.7%
  (−5.3 pts [−7.5, −3.0]) at almost no recall cost. H1 changes the learned prior; H5 changes
  what evidence the model uses. They look complementary, so **E1 + E5b is the next combined
  candidate**.
- **H3 rejected.** E3a crashed at epoch 8 (disk full). Its trajectory equals E3b's exactly
  through epoch 7 (SWA only averages passively; deterministic training), so E3a's result is
  E3b's epoch-20 checkpoint (`compare diff ... E3b-cosine20-swa5-s42@20`); no rerun. Finishing
  the anneal costs 1.25 pts of recall at 1% FPR: the risk the proposal named (the train/
  validation gap widens at a low learning rate).
- **E2b collapsed in epoch 1.** Every hidden ReLU of the dense head was negative for every
  input, so the output was a constant bias with zero gradient. The hinge spreads its gradient
  over the k ≈ 7 hardest negatives per batch (≈ 2w/k each against ≈ 0.5/128 per sample from
  BCE), so at w = 0.25 the hardest negatives got about 18x the BCE gradient from step 0. This is
  an optimisation failure, not evidence about H2. **Deviation from the pre-registered range
  (0.1–0.5):** the rerun E2b2 uses w = 0.05 (≈ 3.7x BCE on those negatives), off for the 3
  warm-up epochs, then ramped in over 2. The trainer now stops a run whose validation output
  is constant.
- **H4, n-gram (E4).** Alone it matches the deployed 3-member CNN ensemble at the operating
  point and beats it on ROC-AUC. Stacking (logistic regression fitted on the calibration
  split) adds to both. Its logit correlation with a CNN member is 0.93, against 0.965 between
  CNN seeds.

  | Model (validation) | ROC-AUC | pAUC | Recall at 0.5% FPR | Recall at 1% FPR |
  | --- | ---: | ---: | ---: | ---: |
  | Deployed ensemble (mean of 3 members) | 0.9098 | 0.7636 | 55.4% | 61.1% |
  | n-gram E4 | 0.9162 | 0.7638 | 55.1% | 61.0% |
  | Stack: ensemble + n-gram (n-gram weight 0.63) | 0.9190 | 0.7723 | 56.9% | 62.9% |

  A paired interval for the stack vs the ensemble is pending.

**Infrastructure fixes.**
- **Disk.** C: is full: 200 of 201 GB, about 930 MB free. It is not this work: runs total
  228 MB and the session's scratchpad 22 MB. This is reported to the user, and nothing
  outside the work was deleted.
- **Queue scripts.** They logged `$?` after `$(date)`, so every run showed exit 0. Fixed in
  wave 1b with `rc=$?`.
- **E1 crash.** E1 crashed while writing its record (a relative dataset path in
  `provenance`). Fixed with `repo_relative`. The new `train --finalize` rebuilds the record
  from the saved per-epoch validation logits (the log must agree) and records the
  training-time code hash from the run's code snapshot. Re-finalizing with the original 7
  threads gives a byte-identical artifact (`542262f4…`); 5 threads changed the last digits of
  the Platt fit.
- **Production members.** `production_members.py` exposes the three deployed members
  (seeds 42–44) as read-only runs `P0-production-member-s*`. These are the seed-matched
  random-batch controls for confirmation. They reproduce the earlier numbers.
- **Tests.** Two new research tests: the pAUC schedule and the recovery consistency check.
  10 pass.

**Queued (wave 1b, 7 threads per lane).**
- Lane A2: `C5b-suffixdrop0.2-random-s42`, then `-s43`.
- Lane B2, after E2c: `E2b2-pauc0.05-start3-ramp2-s42`, then `C5b-suffixdrop0.2-random-s44`.

### 00:00 — H2c rejected; H4 promoted; new hypothesis H12 (Wide & Deep)

- **E2c focal γ = 1 rejected.** ROC-AUC −0.0028 [−0.0041, −0.0013], beyond the pre-registered
  limit of −0.002. Recall at 0.5% FPR +0.18 [−1.05, +1.40], at 1% −0.69 [−1.47, +0.14]. Its
  pAUC gain (+0.012) is in the range every screen shows against the unlucky control.
- **E2b2 delayed start verified.** Epoch 1 of `E2b2-pauc0.05-start3-ramp2-s42` equals the
  control's to every printed digit (train 0.5285, validation 0.4050, pAUC 0.6968). The pAUC
  term is therefore the only difference after epoch 3.
- **H4 promoted.** Combinations are now saved as runs (`compare combine --save` writes
  `X-<name>`), so they get the same paired bootstrap as single models. Stacking the deployed
  ensemble with E4 vs the deployed ensemble (`diff__X-prod-ensemble-mean__*.json`):

  | Metric | Difference [95% CI] |
  | --- | --- |
  | ROC-AUC | +0.0092 [+0.0082, +0.0100] |
  | pAUC | +0.0088 [+0.0060, +0.0117] |
  | Recall at 0.5% FPR | +1.53 pts [+0.44, +2.40] |
  | Recall at 1% FPR | +1.80 pts [+1.09, +2.35] |

  The ensemble's training noise is smaller than one member's (about SD/√3), so +1.80 is
  about 5 SD.
- **Memorisation check (`comparisons/H4_stem_seen_unseen.json`).** 57,076 of 62,045
  validation hostnames have a stem never seen in training.

  | Subset | n-gram alone vs ensemble, recall at 1% | Stack vs ensemble, recall at 1% |
  | --- | --- | --- |
  | Stem seen (4,969) | +3.18 [+0.67, +5.21] | +1.16 [+0.33, +3.27] |
  | Stem unseen (57,076) | −0.82 [−1.72, +0.19] | +1.56 [+0.99, +2.33] |

  The n-gram's standalone strength at the operating point is memorised stems. The stack's
  gain is not: it holds on unseen stems, so the two models carry complementary evidence.
- **Caveat: E4's C was chosen on the calibration split, which also fits the stack.** The stack
  weights are mildly optimistic in-sample. Validation, where everything is judged, is
  untouched.

**H12 — Wide & Deep hostname CNN (new; follows from H4).**

- **Limitation:** the CNN's 89k parameters must both compose patterns and memorise lexical
  items (brand tokens, platform and TLD strings). A linear model with about 10^6 n-gram weights
  ties the 3-member ensemble at the operating point, and stacking adds 1.8 pts, including
  1.56 on unseen stems. The CNN is missing information it could use. Post-hoc stacking learns
  only one weight per model, and a second model breaks the single-artifact inference
  contract.
- **Why it should help:** f(x) = g_deep(x) + w·φ(x), where φ(x) is the hashed, role-tagged
  character 1–5-gram indicator vector divided by √(number of n-grams). That approximates E4's
  L2-normalised TF-IDF row. Training both jointly fits the CNN to what the linear path does not
  already explain, so each part can specialise: memorisation in w, composition in g. Post-hoc
  stacking is a restricted case (w trained alone, then one mixing weight), so the joint
  optimum's training loss is at least as low. The table starts at zero and is created without
  drawing random numbers, so step 0 is exactly the seed-42 control: the same parameters, the
  same random-number state, the same dropout masks (`test_v3wide_starts_as_exactly_the_v3_control`).
- **Evidence:** the H4 stack (above). Logit correlation n-gram vs CNN is 0.93, against 0.965
  between CNN seeds.
- **Experiment:** `E12-v3wide-s42`.
  - Setup: 2^20 buckets; 1,138,018 parameters (89,442 deep); bucketed screen against
    `P0.5-control-bucketed-s42`.
  - Also compared with the post-hoc stack of that control and E4, to test whether joint
    training beats stacking.
  - Implementation: `models.with_wide`, `models.wide_term` / `ngram_buckets` (the same
    integer hash in int64 for PyTorch and NumPy), and the v4 runtime (v3 equals v4 with
    structure and component pools off).
  - Parity is 4.0e-7 on a smoke run. NumPy latency is 2.2 ms per hostname. The artifact is
    about 6 MB (v3: 0.48 MB).
  - Checkpoints are about 4 MB each, so the trainer now keeps only the one the val_bce rule
    selects for wide networks (`--prune-checkpoints`, tested).
- **Could go wrong:**
  1. The wide path memorises training stems. Read the stem-unseen subset now, and the
     temporal lockbox later.
  2. The wide path learns faster and the CNN under-trains. Check by ablating the trained
     model's wide path.
  3. Hash collisions, from millions of distinct n-grams in 2^20 buckets.
  4. Deployment needs a new artifact format and loader: a later decision for the user, since
     the app is out of scope here.
- **Metric:** primary metrics vs the seed-matched control, recall at 1% FPR on stem-unseen
  hostnames, and the same metrics vs the post-hoc stack (control + E4).

**Lane B2 (relaunched before it started).** E2b2, then `E12-v3wide-s42`, then
`C5b-suffixdrop0.2-random-s44`. Research tests: 14 pass. No file under `backend/`,
`frontend/` or `extension/` is modified.

### 00:15 — E2b2 not evaluable under the production stopping rule; E12 early epochs

- **E2b2.** Once the pAUC term turned on (epoch 4), validation pAUC rose faster than the
  control's: 0.7508 at epoch 8, against the control's best of 0.7413 at epoch 19. Validation
  BCE also rose, from 0.385 to 0.410, because the hinge changes the logit scale. The proposal
  anticipated this ("calibration may degrade; Platt refitting covers it"). The val-BCE early
  stop ended the run at epoch 8 and chose epoch 3, before the term was active, so the record
  says nothing about H2b. **Next:** rerun with a fixed schedule (`--patience 0 --select
  last`), so the stopping rule does not penalise the logit scale the objective changes.
- **E12 (Wide & Deep).** Validation BCE at epoch 5 is 0.3514, already below the control's best
  over all epochs (0.3610 at epoch 19; at epoch 5 the control had 0.3775). This is not yet a
  verdict.
- **Leftover queue process.** The first lane-B2 shell, stopped before relaunch, outlived
  TaskStop. It re-read the rewritten script after its wait loop, hit a syntax error and
  exited without running anything. The run-id guard would have refused duplicates anyway.

## 2026-11-25

### 01:10 — H12 Wide & Deep: largest gain so far; C5b seed 42 weak; work moved to E:

**E12-v3wide-s42** (screen, bucketed, seed 42). Early stop at epoch 18; epoch 13 chosen. NumPy
parity is 1.1e-6. Paired bootstrap on the 62,045 validation hostnames; recall in percentage points.

| E12 vs | ROC-AUC | pAUC | Recall at 0.5% FPR | Recall at 1% FPR |
| --- | --- | --- | --- | --- |
| Seed-matched control (identical at step 0) | +0.0140 [+0.0122, +0.0153] | +0.0324 [+0.0263, +0.0385] | +4.23 [+3.25, +5.44] | +2.98 [+2.26, +3.90] |
| Deployed 3-member ensemble | +0.0062 [+0.0047, +0.0074] | +0.0101 [+0.0060, +0.0139] | +2.19 [+1.40, +3.14] | +1.33 [+0.59, +2.14] |
| Post-hoc stack (control + E4 n-gram) | −0.0026 [−0.0035, −0.0017] | +0.0049 [+0.0014, +0.0084] | +1.60 [+0.69, +2.74] | +0.10 [−0.62, +0.92] |

- **Size of the gain.** Against the control it is 5–7 training SDs at the operating point, far
  beyond the bucketed control's tail luck. One Wide & Deep model beats the 3-member ensemble.
- **Joint training vs stacking.** Joint training matches or beats post-hoc stacking at the
  operating point with one artifact, so the single-artifact inference contract holds.
- **Memorisation check passed (`comparisons/H12_memorisation_and_ablation.json`).** On the
  57,076 stem-unseen hostnames, E12 vs the ensemble gives recall +2.17 pts [+0.99, +3.04] at
  0.5% FPR and +1.04 [+0.27, +1.98] at 1% FPR. Vs the control it gives +4.04 and +3.14. The
  wide path memorises seen stems (vs the control, recall at 0.5% FPR +32 pts on stem-seen),
  but the gain on unseen stems stands on its own.
- **Ablation.** The two paths specialised; the deep path is not lazy.

  | Path | ROC-AUC | Recall at 1% FPR |
  | --- | ---: | ---: |
  | Deep path alone | 0.873 | 54.0% |
  | Wide path alone | 0.894 | 52.9% |
  | Control CNN | 0.902 | 59.4% |
  | Both paths | 0.916 | 62.4% |

  Logit correlation between the paths is 0.70. **Risk:** the deep path now relies on the wide
  path's lexical memory. If phishing vocabulary drifts, both could degrade together. The
  temporal lockbox (≥ 2026-12-08) is the test.
- **Platform false alarms barely change.** At 59% recall: shared legitimate FPR 0.95% → 0.63%
  (−0.32 [−0.42, −0.24]); platform-legitimate (added, n=774) 43.0% → 38.8%; long-legitimate
  11.4% → 7.4%. H1's data fix brought platform-legitimate to 18.9%, so the two address
  different failures. **Next: E13 = Wide & Deep on h1-v2 (forward selection).**

**C5b-suffixdrop0.2-random-s42** (confirmation, random batches). The screen's gain mostly
does not reproduce.

| vs | ROC-AUC | pAUC | Recall at 0.5% FPR | Recall at 1% FPR |
| --- | --- | --- | --- | --- |
| P0.5-control-random-s42 | −0.0020 [−0.0031, −0.0006] | +0.0048 [−0.0008, +0.0108] | +0.45 [−0.83, +1.68] | +1.06 [+0.07, +1.72] |
| P0-production-member-s42 | −0.0010 [−0.0022, +0.0002] | +0.0057 [−0.0002, +0.0112] | −0.62 [−2.05, +1.00] | +0.71 [−0.16, +1.39] |

This is the inflation the 23:45 caveat warned about: E5b was judged against the unlucky
bucketed control. **H5b is downgraded to "not confirmed; at most a small effect at 1% FPR".**
Its seeds 43 and 44 move behind the Wide & Deep work. Suffix dropout can still be tried as an
add-on to Wide & Deep in forward selection.

**Wave 2 (from E:, both runs record git d7bf21c, clean):**
- Lane A: `C12-v3wide-random-s42`, then `-s43`.
- Lane B: `E13-v3wide-h1v2-s42` (screen vs E12, both bucketed seed 42), then
  `C12-v3wide-random-s44`.

**Housekeeping.**
- The workspace moved to `E:\M-PHISH-vscode (1)`; C: was full.
- The final sync copied the runs: 30 files, 0 failed. E: holds everything; it also keeps 217
  completed-run checkpoints that were removed from C: as verified duplicates.
- This log moved into the repository (`research/RESEARCH_LOG.md`).
- Commit `d7bf21c` pushed as `mine/cnn-v4-research`; CI "Verify and package" passed.

### 05:05 — H12 confirmed on 3 seeds; E13 (H12 + H1) fixes platform false alarms

All four wave-2 runs exited 0 and record git `d7bf21c` with clean code. NumPy parity is
1.0–1.7e-6. Paired bootstrap (500) on the 62,045 validation hostnames; recall in points.

**H12 Wide & Deep is confirmed (random batches, seeds 42–44).**

| Seed | Chosen epoch | ROC-AUC | pAUC | Recall at 0.5% FPR | Recall at 1% FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 16 | 0.9170 | 0.7793 | 58.8% | 63.8% |
| 43 | 15 | 0.9179 | 0.7771 | 58.1% | 63.2% |
| 44 | 14 | 0.9162 | 0.7773 | 58.1% | 62.4% |
| Mean (SD) | | 0.9170 (0.0009) | 0.7779 (0.0012) | 58.3% (0.4) | 63.2% (0.7) |
| Random-batch controls, mean of 4 (SD) | | 0.9031 (0.0006) | 0.7495 (0.0035) | 53.7% (0.6) | 59.3% (0.6) |

Against each seed-matched control (`diff__P0-production-member-s*__C12-*.json`):

| Seed | ROC-AUC | pAUC | Recall at 0.5% FPR | Recall at 1% FPR |
| --- | --- | --- | --- | --- |
| 42 vs production member | +0.0146 [+0.0129, +0.0161] | +0.0331 [+0.0265, +0.0399] | +5.00 [+3.53, +6.46] | +4.45 [+3.69, +5.36] |
| 42 vs research control | +0.0137 [+0.0120, +0.0151] | +0.0323 [+0.0259, +0.0388] | +6.07 [+4.72, +7.24] | +4.79 [+3.95, +5.60] |
| 43 vs production member | +0.0141 [+0.0124, +0.0157] | +0.0257 [+0.0202, +0.0324] | +3.90 [+2.78, +5.25] | +2.99 [+2.40, +3.88] |
| 44 vs production member | +0.0134 [+0.0117, +0.0149] | +0.0240 [+0.0194, +0.0285] | +4.14 [+3.19, +4.98] | +3.76 [+3.06, +4.61] |

- **Verdict: promoted; Wide & Deep becomes the base for forward selection.** All 3 seeds clear
  the rule on every primary metric. The mean gains are +0.0139 ROC-AUC (about 23 training SDs),
  +4.6 pts at 0.5% FPR (about 8 SDs) and +3.9 pts at 1% FPR (about 6 SDs). The bucketed screen
  (E12: +4.23 and +2.98) was not inflated.
- **Ensembles (`combine__c12-ensemble-mean.json`).**

  | Comparison | ROC-AUC | Recall at 0.5% FPR | Recall at 1% FPR |
  | --- | --- | --- | --- |
  | One Wide & Deep model (s42) vs deployed 3-member ensemble | +0.0072 [+0.0056, +0.0084] | +3.44 [+2.40, +4.46] | +2.75 [+2.03, +3.63] |
  | 3-member Wide & Deep ensemble vs deployed 3-member ensemble | +0.0086 [+0.0070, +0.0098] | +2.69 [+1.99, +4.11] | +2.73 [+1.99, +3.53] |
  | 3-member Wide & Deep ensemble vs deployed ensemble + n-gram stack | −0.0006 [−0.0015, +0.0003] | +1.15 [+0.37, +2.63] | +0.94 [+0.34, +1.74] |

  Seed-ensembling Wide & Deep adds almost nothing: ROC-AUC 0.9184 vs best single 0.9179, and
  recall at 1% FPR 63.8% vs 63.8%. The members agree too closely: pairwise logit correlation is
  0.990, against 0.963–0.968 for the deployed members. A likely reason: for a fixed deep path
  the linear path's loss is convex and starts at zero for every seed, so it ends in nearly the
  same place. The seeds then differ mainly through the deep path. This is not tested.
  **Consequence:** one Wide & Deep artifact (about 6 MB) does the work of a 3-member ensemble.
  That is relevant to the final ensemble phase and to deployment, which stays the user's
  decision.

**E13-v3wide-h1v2-s42 (screen: Wide & Deep on h1-v2, vs E12, both bucketed seed 42).**

- **Shared hostnames: neutral.** ROC-AUC −0.0012 [−0.0019, −0.0006]; pAUC −0.0002
  [−0.0039, +0.0038]; recall at 0.5% FPR −0.20 [−1.53, +0.76]; at 1% FPR +0.10 [−0.63, +0.75].
  The AUC cost is about 2 training SDs. On the v3 CNN the same data cost −0.0059 (E1), so Wide &
  Deep absorbs most of H1's ranking cost. E13's own validation ROC-AUC (0.9274) includes the
  added legitimate hostnames and is not comparable.
- **Fixed recall 59% on the shared phishing (`fixed_recall__E12-v3wide-s42__E13-v3wide-h1v2-s42.json`).**

  | False positive rate | Bucketed control | E1 (v3 + h1-v2) | E12 (Wide & Deep) | E13 (both) | E13 − E12 [95% CI] |
  | --- | ---: | ---: | ---: | ---: | --- |
  | Shared legitimate (35,308) | 0.95% | 0.90% | 0.63% | 0.62% | −0.01 [−0.08, +0.07] |
  | Added legitimate, hosting platform (774) | 43.0% | 18.9% | 38.8% | **16.4%** | −22.35 [−25.19, −19.44] |
  | Shared legitimate, hosting platform (137) | 43.8% | 16.1% | 35.8% | 11.7% | −24.09 [−31.39, −16.79] |
  | Added legitimate, longer than 30 (2,484) | 11.4% | 4.9% | 7.4% | 2.6% | −4.79 [−5.72, −3.90] |
  | Added legitimate, 4+ labels (11,979) | — | — | 2.4% | 1.2% | −1.19 [−1.38, −0.97] |

  At 50% recall: platform-legitimate (added) 19.9% → 6.5%.
- **The cost is where expected.** Recall is fixed overall, so it moves between groups.
  - Phishing on hosting platforms (8,799, a third of the shared phishing): 98.7% → 95.4%
    (−3.26 [−3.63, −2.81]). E1 paid the same (98.5% → 94.4%).
  - Long phishing (6,575): 94.1% → 92.2%.
  - 4+ label phishing (4,461): 91.0% → 87.8%.

  Before, being on a hosting platform was almost enough to be flagged. Now the subdomain itself
  has to look like phishing.
- **Verdict: promoted to 3-seed confirmation.** H1's pre-registered target is met: platform
  false alarms fall by more than half with recall held. E13 also beats E1 on that target, and
  keeps E12's operating-point gains on the shared hostnames.

**Wave 3 (from E:, git `d7bf21c`, clean code; started 05:05):**
- Lane A: `C13-v3wide-h1v2-random-s42`, then `-s43`.
- Lane B: `C13-v3wide-h1v2-random-s44`, then the next forward-selection screen,
  `E14-v3wide-h1v2-suffixdrop0.2-s42` (E13 + suffix dropout 0.2; bucketed, vs E13). Its
  rationale: the audit found that the TLD alone swings decisions (.xyz +0.60). Suffix dropout
  hides the suffix characters from both paths for 20% of hostnames. On v3 it was not confirmed
  (C5b).
- C13 seed *N* is judged against C12 seed *N*: the same recipe and seed, only the data differs.
  C12 has been scored on the hostnames h1-v2 adds (`score --only-new`), so the fixed-recall
  read-out is paired per seed.

### 08:45 — C13 breaches H1's recall guardrail on 2 of 3 seeds; E14 suffix dropout recovers it in the screen

All runs exited 0, git `d7bf21c`, clean code, parity ≤ 1.5e-6. Seed 43 of C13 is still running.

**C13 seed *N* vs C12 seed *N*** (random batches; only the dataset differs). Primary metrics on
the 62,045 original validation hostnames (`diff__C12-*__C13-*.json`):

| Seed | ROC-AUC | pAUC | Recall at 0.5% FPR | Recall at 1% FPR |
| --- | --- | --- | --- | --- |
| 42 | −0.0023 [−0.0031, −0.0016] | −0.0039 [−0.0074, −0.0000] | −1.13 [−2.09, +0.08] | −1.24 [−1.81, −0.55] |
| 44 | −0.0028 [−0.0035, −0.0020] | −0.0064 [−0.0106, −0.0022] | −1.62 [−2.35, −0.42] | −0.43 [−1.30, +0.17] |

Fixed recall 59% on the shared phishing (`fixed_recall__C12-*__C13-*.json`):

| Seed | Shared legitimate FPR | Platform legitimate (added, 774) | Long legitimate (added, 2,484) | Platform phishing recall |
| --- | --- | --- | --- | --- |
| 42 | 0.51% → 0.61% (+0.09 [+0.02, +0.17]) | 35.0% → 14.2% (−20.8) | 6.9% → 2.3% | 98.4% → 95.1% |
| 44 | 0.58% → 0.69% (+0.11 [+0.03, +0.19]) | 38.6% → 16.2% (−22.5) | 7.0% → 2.8% | 98.6% → 95.6% |

- **The platform fix replicates exactly.** False alarms on legitimate hosting-platform sites
  fall by about 21 pts on both seeds.
- **H1's pre-registered guardrail is breached on both seeds.** The rule: recall at the
  operating point may fall by at most 0.5 pts on the original validation. Seed 42 is at −1.13
  and −1.24; seed 44 is at −1.62 at 0.5% FPR. The bucketed screen (E13 vs E12) had shown no
  cost, so random batches reveal one the screen hid.
- **Caveat, not an excuse.** The original validation has only 137 legitimate platform sites
  among 35,308. Its FPR therefore barely counts the population H1 fixes. At 59% recall, the
  seed-42 trade breaks even if just 0.5% of legitimate traffic is on hosting platforms:
  0.10 pts more shared FPR against 20.8 pts less platform FPR. Real browsing likely exceeds
  that, but no traffic data here measures it. The guardrail stands as registered: H1 on Wide &
  Deep is **not promoted alone**. Accepting the trade would be a product decision for the user.
- Verdict after seed 43.

**E14-v3wide-h1v2-suffixdrop0.2-s42 vs E13** (screen, both bucketed seed 42, both trained on
h1-v2; scratch analysis `e14_vs_e13.py` with the added hostnames read as in `fixed-recall`;
`diff__E13-*__E14-*__original-hosts.json`, `fixed_recall__E13-*__E14-*.json`).

| Metric (original 62,045 hostnames) | E14 − E13 [95% CI] |
| --- | --- |
| ROC-AUC | −0.0014 [−0.0019, −0.0007] |
| pAUC | +0.0049 [+0.0020, +0.0077] |
| Recall at 0.5% FPR | +1.55 [+0.49, +2.51] |
| Recall at 1% FPR | +0.61 [+0.02, +1.21] |

| At 59% recall | E13 | E14 | Difference [95% CI] |
| --- | ---: | ---: | --- |
| Shared legitimate FPR | 0.62% | 0.52% | −0.10 [−0.16, −0.03] |
| Platform legitimate (added) | 16.4% | 13.2% | −3.23 [−4.78, −1.16] |
| 4+ labels legitimate (added) | 1.19% | 0.42% | −0.78 [−0.94, −0.60] |
| Platform phishing recall | 95.4% | 94.9% | −0.50 [−0.85, −0.23] |

- On the rule, recall at 0.5% FPR passes (+1.55 > 2 SD, CI excludes 0); at 1% it does not
  (+0.61). ROC-AUC dips by about 2 SD, like C5b's (−0.0020). This is the same pattern as E5b,
  whose screen did not survive confirmation. So it is not promoted on the screen alone.
- **Why it is confirmed now anyway.** It works against exactly what breaches H1's guardrail.
  The shared-legitimate FPR that h1-v2 raised by 0.1 pts falls by 0.1 pts. Hiding the suffix
  for 20% of hostnames is meant to stop the model leaning on the TLD. Whether that is the
  mechanism behind h1-v2's cost is a hypothesis, not tested.
- The scratch analysis is saved next to its output as
  `runs/research/comparisons/e14_vs_e13.py`. It becomes a `compare` option once no run is
  in flight, since a code change now would mark the queued run as dirty.

**09:25: C13 seed 43, and the 3-seed verdict on H1 over Wide & Deep.** Seed 43 vs C12 seed 43:
ROC-AUC −0.0025 [−0.0033, −0.0017]; recall at 0.5% FPR +0.93 [−0.83, +1.74]; at 1% FPR +0.08
[−0.77, +0.77]. At 59% recall: shared legitimate FPR 0.54% → 0.50%; platform legitimate
36.6% → 12.9%; platform phishing recall 98.5% → 94.8%.

| C13 − C12, original validation | Seed 42 | Seed 43 | Seed 44 | Mean (SE) |
| --- | ---: | ---: | ---: | ---: |
| ROC-AUC | −0.0023 | −0.0025 | −0.0028 | −0.0025 (0.0001) |
| Recall at 0.5% FPR | −1.13 | +0.93 | −1.62 | −0.61 (0.78) |
| Recall at 1% FPR | −1.24 | +0.08 | −0.43 | −0.53 (0.38) |
| Shared legitimate FPR at 59% recall | +0.09 | −0.04 | +0.11 | +0.05 (0.05) |
| Platform legitimate FPR at 59% recall | −20.8 | −23.6 | −22.5 | −22.3 (0.8) |
| Platform phishing recall at 59% recall | −3.27 | −3.74 | −2.97 | −3.33 (0.22) |

- **Verdict: H1 over Wide & Deep is a marginal fail of its guardrail, not promoted alone.** The
  3-seed mean is past the −0.5 pts limit at both operating points (−0.61, −0.53), but within
  its own uncertainty. The platform fix is large and stable (−22 pts, SE 0.8). The ROC-AUC
  cost is small but certain. The decision moves to C14 (with suffix dropout).

**Wave 4 (started 08:45; git `d7bf21c`, clean code):** confirm E14 with random batches.
- Lane B: `C14-v3wide-h1v2-suffixdrop0.2-random-s42`, then `-s44`.
- Lane A: after C13 seed 43, `C14-...-s43`.
- Each C14 seed *N* is judged against C12 seed *N*, to test whether the full combination
  passes H1's guardrail against the confirmed base. It is also judged against C13 seed *N*, to
  isolate suffix dropout.

### 11:25 — C14 seed 43: suffix dropout hurts; H1 at half weight queued

`C14-v3wide-h1v2-suffixdrop0.2-random-s43` (chosen epoch 13; parity 2.2e-6).

| C14 seed 43 vs | ROC-AUC | pAUC | Recall at 0.5% FPR | Recall at 1% FPR | Shared legitimate FPR at 59% recall |
| --- | --- | --- | --- | --- | --- |
| C12 seed 43 (guardrail) | −0.0033 [−0.0042, −0.0025] | −0.0036 [−0.0078, +0.0002] | −0.73 [−2.06, +0.27] | −1.38 [−2.16, −0.75] | 0.54% → 0.63% |
| C13 seed 43 (suffix dropout alone) | −0.0008 [−0.0014, −0.0002] | −0.0045 [−0.0077, −0.0014] | −1.66 [−2.26, −0.45] | −1.46 [−2.10, −0.85] | 0.50% → 0.63% |

- Against C13, suffix dropout moves every primary metric the wrong way, beyond 2 SD at both
  operating points. It also undoes part of the platform fix: 12.9% → 15.5%. The E14 screen's
  +1.55 does not reproduce, the same as E5b → C5b on v3. Seeds 42 and 44 are still running,
  so the verdict waits. If they agree, this is the second time a suffix-dropout screen has
  failed to replicate. Bucketed screens are not reliable for this kind of regulariser.

**12:15: C14 seed 42** (chosen epoch 21; parity 1.6e-6).
- Against C12 seed 42, it breaches the guardrail: ROC-AUC −0.0033 [−0.0042, −0.0025]; recall
  at 0.5% FPR −0.77 [−2.15, +0.37]; at 1% FPR −1.62 [−2.41, −0.92]. Platform legitimate FPR is
  35.0% → 12.1%.
- Against C13 seed 42 (suffix dropout alone) it is neutral: ROC-AUC −0.0010 [−0.0016, −0.0004];
  recall at 0.5% FPR +0.36 [−0.71, +1.05]; at 1% FPR −0.37 [−1.08, +0.23]. Platform FPR is
  14.2% → 12.1%.
- With seed 43 negative and seed 42 neutral, suffix dropout cannot reach a 3-seed gain.
  Seed 44 is finishing for completeness.

**E15-v3wide-h1v2-w0.5-random-s42 (lane A, 11:25; git `d7bf21c`, clean code).**
- **Limitation:** H1 brings a cost. Phishing recall falls in exactly the groups the added
  legitimate rows resemble:
  - 4+ labels: −2.3 to −4.4 pts.
  - Long: −2.0 to −2.5 pts.
  - Platform: −3.0 to −3.7 pts.

  Most of the 49,360 added training rows are 4+-label or long legitimate hostnames under ICANN
  suffixes (46,573). Only 2,787 are on hosting platforms. The platform fix needs the
  platform rows; the volume of the others may be what moves the boundary for deep and long
  phishing.
- **Why weighting should help:** halving the loss weight of the added rows halves their pull on
  the decision boundary. Their evidence still points the same way. Weights are normalised to
  mean 1: original rows 1.19, added 0.60. The training step size is unchanged, and the weights
  enter the loss only, so the random-number stream is identical to C13 seed 42.
- **Experiment:** C13 seed 42 with `--sample-weights data/feeds/research/h1-v2/added-train-weights-0.5.json`.
  It is judged against C12 seed 42 for the guardrail (C13 there: −1.13 and −1.24) and C13
  seed 42 for the platform fix (14.2%).
- **Could go wrong:** the platform fix shrinks along with the cost. Weighting is also a knob
  tuned on validation. If it is adopted, the weight must be fixed before the lockbox and not
  re-tuned on it.
- **Metric:** recall at 0.5% and 1% FPR vs C12 seed 42 (limit −0.5 pts), and platform-legitimate
  FPR at 59% recall vs C12 seed 42 (H1's target: halve it, so at most about 17.5%).

### 13:45 — wave 4 verdicts: suffix dropout rejected on Wide & Deep; H1 weighting does not help

All runs exited 0, git `d7bf21c`, clean code, parity ≤ 2.2e-6. No run is in flight.

**H5b suffix dropout on Wide & Deep + h1-v2: rejected.** C14 seed 44 (chosen epoch 14):
- vs C12 seed 44: ROC-AUC −0.0027; recall at 0.5% FPR −1.23 [−1.99, −0.06]; at 1% −0.25.
- vs C13 seed 44: ROC-AUC +0.0001; recall at 0.5% FPR +0.39 [−0.32, +1.20]; at 1% +0.17.
  Platform FPR unchanged (16.2%).

| C14 − C13 (suffix dropout alone) | Seed 42 | Seed 43 | Seed 44 | Mean |
| --- | ---: | ---: | ---: | ---: |
| ROC-AUC | −0.0010 | −0.0008 | +0.0001 | −0.0006 |
| Recall at 0.5% FPR | +0.36 | −1.66 | +0.39 | −0.30 |
| Recall at 1% FPR | −0.37 | −1.46 | +0.17 | −0.55 |
| Platform legitimate FPR at 59% recall | −2.1 | +2.6 | 0.0 | +0.2 |

There is no gain on any seed beyond noise, and one seed is clearly worse. Against the guardrail
base C12, C14 averages −0.91 and −1.08: worse than C13. The E14 screen's +1.55 was the second
suffix-dropout screen to fail to replicate. **Lesson recorded:** bucketed one-seed screens
over-state regularisers. For future regulariser ideas, screen with random batches against
an existing seed-matched random run, as E15 does.

**E15 (H1 at half weight, seed 42, chosen epoch 16): not useful.**

| E15 vs | ROC-AUC | Recall at 0.5% FPR | Recall at 1% FPR | Platform legitimate FPR at 59% recall | Platform phishing recall |
| --- | --- | --- | --- | --- | --- |
| C12 seed 42 (no h1-v2) | −0.0022 [−0.0028, −0.0015] | −0.41 [−1.35, +0.70] | −1.23 [−1.88, −0.73] | 35.0% → 19.4% | 98.4% → 96.4% |
| C13 seed 42 (full weight) | +0.0002 [−0.0003, +0.0007] | +0.71 [−0.02, +1.36] | +0.01 [−0.58, +0.35] | 14.2% → 19.4% | 95.1% → 96.4% |

- Half weight gives back a quarter of the platform fix: 19.4% misses H1's "halve it" target
  of 17.5%.
- The recall cost at 1% FPR stays the same (−1.23 vs −1.24).
- 4+-label phishing recall recovers (−0.45 vs −2.31 at full weight), as the hypothesis
  predicted. That does not reach the primary metric.
- **Not pursued.** The cost at the operating point does not scale with the added rows' weight,
  so it is not simply their volume. Weighting only moves along the same trade-off.

**State of the candidates (validation only; the lockbox is still sealed).**

| Recipe | Status | vs deployed members, recall at 1% FPR | Platform legitimate FPR at 59% recall |
| --- | --- | ---: | ---: |
| C12: Wide & Deep, original data | **Confirmed** (3 seeds) | +3.9 pts | ≈ 37% |
| C13: Wide & Deep + h1-v2 | Platform fix confirmed (−22 pts); marginally fails H1's recall guardrail (−0.61 / −0.53) | ≈ +3.3 pts | ≈ 14% |

Whether the platform fix is worth about 0.6 pts of recall on the original validation, a set
with 137 legitimate platform sites among 35,308, is a product decision. **It is put to the
user.** The guardrail is not quietly relaxed.

**Decision (user, 13:50): recipe C13 (Wide & Deep + h1-v2) goes forward to the final stage.**
- The user accepted the trade over H1's pre-registered limit of −0.5 pts: the 3-seed mean is
  −0.61 at 0.5% FPR and −0.53 at 1% FPR.
- Reason: fewer false alarms on legitimate hosting-platform sites people actually visit.
- This was decided before the lockbox is opened. The lockbox is not used to choose between C12
  and C13.
- No more training today. Next: calibration (H10) and the final build on C13. Then the pAUC
  objective (H2b, with `--patience 0 --select last` and a matching control), if wanted.

## 2026-11-26

### Plan — H10 calibration on C13 (post-hoc, no training), written before any result

- **Scorers:** each C13 seed (42, 43, 44) and their 3-seed mean of logits. Scores come from
  each run's saved `scores.npz`. Nothing is retrained.
- **Calibrators:** none (σ(z)), temperature (σ(z/T)), Platt (σ(az+b), the format the app
  already reads), and isotonic. Each is fitted on the calibration split (76,515 hostnames)
  and judged on validation (75,961).
- **Metrics:** validation log loss (primary), Brier score, ECE with 15 equal-width bins and
  with 15 equal-count bins. Also per structural group: predicted vs observed phishing rate,
  and ECE.
- **Decision rule:** keep Platt unless another calibrator lowers validation log loss on all
  three seeds, with the paired 95% interval excluding zero, and also lowers ECE. Isotonic
  would also need a new artifact field, so its gain must justify a format change.
- **Threshold under shift:** the production threshold rule is unchanged. Report:
  - the 95% bootstrap interval of the threshold, re-selected on 1,000 resamples;
  - the out-of-bag FPR and recall of those re-selected thresholds;
  - the FPR at the chosen threshold on the calibration split. That split played no part in
    choosing the threshold, so this is an independent check.
  These are informational: they say how precisely an FPR can be promised. They are not a
  gate.
- **Ensemble:** the same read-out says whether the 3-seed mean is worth three times the size
  (about 18 MB) for the final build. For C12, seed-ensembling added almost nothing.

### H10 results — Platt kept; one website drives much of the operating-point noise

Code: `research/hostname_cnn/calibration.py` (4 new tests, 18 pass). Report:
`comparisons/H10_calibration__c13.json`. Run time 1.5 min, no training.

**Calibrators (validation log loss; paired 95% interval vs Platt).**

| Scorer | None | Temperature | Platt | Isotonic | Platt ECE → isotonic ECE |
| --- | ---: | ---: | ---: | ---: | --- |
| Seed 42 | 0.3078 | 0.3061 | **0.3053** | 0.3066 (+0.0014 [+0.0008, +0.0019]) | 0.010 → 0.008 |
| Seed 43 | 0.3031 | 0.3013 | 0.3006 | **0.3001** (−0.0006 [−0.0012, +0.0000]) | 0.013 → 0.007 |
| Seed 44 | 0.3086 | 0.3069 | **0.3053** | 0.3058 (+0.0005 [−0.0001, +0.0011]) | 0.009 → 0.008 |
| Mean of 3 | 0.3034 | 0.3012 | **0.2999** | 0.3002 (+0.0003 [−0.0003, +0.0008]) | 0.011 → 0.008 |

- **Verdict: Platt kept, by the pre-registered rule.** Isotonic does not lower log loss on all
  three seeds, and it is worse on seed 42. Its lower ECE does not justify a format change.
- Temperature is worse than Platt on every scorer: the fitted intercept matters. The Platt
  slope is about 1.2, so the raw model is slightly under-confident.
- **Caveat for the app:** these probabilities are calibrated to the dataset's mix, which is
  35% phishing. Real browsing has far less phishing, so a score of 0.85 is not an 85% chance
  that a site is phishing. The threshold is what the app relies on, and it is not affected.

**Threshold (Platt, mean of 3; the production rule).**
- The threshold is 0.847. Validation FPR is 0.91% and recall 59.3%.
- Re-selected on 1,000 resamples, the threshold stays in [0.834, 0.859]. The out-of-bag FPR of
  the re-selected thresholds averages 0.92% [0.75, 1.12]; it is above 1% in 17% of resamples.
- **Hostname resampling understates the uncertainty.** At the same threshold, the calibration
  split gives an FPR of **0.59%** against 0.91% on validation. A hostname-level 95% interval
  of [0.83, 0.99] cannot explain that gap.
- **Cause: one website.** `yupoo.com`, a photo-hosting site with a subdomain per user, has
  262 legitimate validation hostnames (0.5% of validation negatives). Splits go by registered
  domain, so all of them are in validation; the calibration split has one yupoo hostname.
  - Yupoo hostnames are **113 of 448 false alarms** for the mean of 3, and 191 of 448 for
    seed 42.
  - Without yupoo, validation FPR is 0.68%. The calibration split has no comparable cluster:
    its 10 largest domains give 13 of 286 false alarms.
  - Resampling registered domains instead of hostnames gives an FPR interval of
    **[0.62, 1.40]**.
- **What can be promised:** on held-out sites like these, the threshold gives an FPR of about
  0.6–0.9%. A single site with many user subdomains can push it past 1%. The threshold rule
  stays as pre-registered. It is conservative here, because it was tuned on the yupoo-heavy
  validation.

**Yupoo and the earlier decisions.**
- Seed differences at fixed FPR are partly a yupoo effect. Seed 42 puts 204 of 262 yupoo
  hostnames above its 1%-FPR cut; seed 43 puts 77. Without yupoo, seed 42's recall at 1% FPR
  is 61.6% instead of 57.6%.
- **The C13 decision is unaffected.** On the original validation hostnames (25 of them yupoo),
  the 3-seed C13 − C12 means change little without yupoo: from −0.61 to −0.58 at 0.5% FPR,
  and from −0.53 to −0.47 at 1% FPR.
- **Lesson:** report fixed-FPR results with a by-site check. A single site can move them by
  several points.

**Ensemble vs single model (`diff__X-c13-ensemble-mean__...json`).** Each seed vs the 3-seed
mean of logits, recall at 1% FPR:

| Seed vs ensemble | All validation | Without yupoo |
| --- | ---: | ---: |
| Seed 42 | −2.45 [−3.19, −1.86] | −0.43 [−0.83, −0.01] |
| Seed 43 | +0.84 [+0.44, +1.35] | −0.08 [−0.48, +0.41] |
| Seed 44 | −1.29 [−1.63, −0.55] | −1.57 [−1.94, −1.04] |

- The ensemble has the best ROC-AUC and log loss of all four scorers. Without yupoo, it
  matches or beats every seed.
- Seed 43's lead over the ensemble is a yupoo effect. Choosing it would select on validation
  luck.
- Unlike C12, ensembling C13 helps. Against a typical seed, the gain is about 0.7 pts of
  recall at 1% FPR without yupoo, and it is steadier across sites.
- **Cost:** three times the size (about 18 MB vs 6 MB) and the scoring time (about 6.6 ms vs
  2.2 ms per hostname). The model is loaded by the backend only.
- **Put to the user:** single model or 3-seed ensemble for the final build.

**Decision (user, 2026-11-26): the final build is the 3-seed ensemble** (C13 seeds 42, 43
and 44, mean of logits), with Platt calibration and the production threshold rule. It was
chosen before the lockbox is opened.

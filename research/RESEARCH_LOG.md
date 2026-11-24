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

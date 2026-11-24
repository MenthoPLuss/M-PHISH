# CNN research proposal: hostname CNN v4

Phase 1 deliverable, 24 Nov 2026. For review before any implementation.

## Summary

Phase 1 changes the research question. A linear character n-gram model already ties the
deployed 3-member CNN (validation ROC-AUC 0.915 vs 0.910; recall at 1% FPR 61% vs 61%).
So "a deeper, bigger CNN" is the wrong bet. The evidence points elsewhere:

1. **Data, not depth.** Labels are confounded with source. Legitimate hosting-platform,
   long and deep hostnames are scarce, and some "phishing" labels are compromised
   legitimate domains. These cause the worst failures, including 47.5% false alarms on
   legitimate platform sites.
2. **Objective.** Training and model selection optimise average log loss, while the app
   operates at <= 1% FPR. Members are selected before the learning rate anneals.
3. **Structure n-grams cannot express.** The CNN only earns its place by learning
   non-additive, component-level evidence: a brand in the subdomain, a bait word plus a
   suffix, name "wordness". This is where the v4 architecture work focuses.
4. **Diversity.** Seed ensembles are saturated (correlation 0.965). A member with a
   different inductive bias adds more than two extra seeds.

The proposal ranks 12 hypotheses and 4 prerequisites by expected value per CPU hour. It
holds everything to one protocol (§1), and every new component must earn its place in an
ablation. **The 90% recall release target is very likely out of reach for any hostname-only
model**, because part of the labelled phishing sits on legitimate domains. §5 asks you to
decide how the gate should be scoped.

## 1. Research protocol (applies to every experiment)

### 1.1 Data splits and the locked test

| Set | Contents | Use |
| --- | --- | --- |
| train | 231,431 hostnames (`cnn-v3-fresh` splits) | fitting |
| calibration | 63,071 | Platt / temperature fitting only |
| validation | 62,045 | early stopping, model selection, thresholds; all screening results |
| reported test | 62,489 | already seen in 4 runs; reported once per finalist, never used to choose |
| **locked test (new)** | built before Phase 5, never scored until Phase 11 | the only basis for the final claim |

The new locked test has three parts, all disjoint by registrable domain from every
training, calibration and validation hostname:
1. **Temporal phishing holdout:** hostnames that appear on the Phishing.Database ACTIVE
   list in a snapshot taken 2 or more weeks after 24 Nov 2026 and were absent on 24 Nov.
   This is the only honest test of "new phishing".
2. **Temporal legitimate holdout:** sites in the next CrUX monthly snapshot (2026-11)
   that are not in 2026-10.
3. **Low-FPR legitimate bank:** about 200,000 CrUX 2026-10 hostnames never sampled for
   training. Legitimate hostnames are plentiful, so FPR can be measured far more precisely
   than recall. At 200,000 negatives, an observed FPR of 0.10% has a 95% Wilson interval
   of about 0.09 to 0.11%. At the current 35,000, it is about 0.07 to 0.14%.

The checksums of all three are recorded in the experiment registry before any candidate
is trained, and the scoring script refuses to run on them before Phase 11.

### 1.2 Screening and promotion

Training cost is the constraint: about 75 to 95 minutes per member on the full training
split with 14 threads. Two experiments run in parallel at 7 threads each for throughput.

1. **Screen:** one seed (42), full training split, the candidate vs a baseline rerun with
   the same seed and schedule. Read on validation only.
2. **Promote** a candidate only if it beats the seed-matched baseline by more than the
   seed spread measured in Phase 1 (validation ROC-AUC range 0.0008 across 3 seeds),
   on the primary metric, with no subgroup losing more than 2 points of recall at 1% FPR.
3. **Confirm:** 3 seeds; report mean, standard deviation, and a paired bootstrap 95%
   interval on the difference vs the baseline, using the same validation hostnames.
4. **Combine:** promoted components are combined one at a time (forward selection) and
   each combination is re-confirmed, so every component earns its place in the final
   model.

### 1.3 Metrics

- **Primary:** recall at 0.5% and 1% FPR on validation, and partial ROC-AUC for FPR <= 1%
  (McClish-standardised). These are the operating region of the app.
- **Secondary:**
  - ROC-AUC and PR-AUC
  - recall at 0.1% and 0.25% FPR, reported with intervals and the number of false
    positives allowed (35 at 0.1% on validation)
  - Brier score and ECE
  - per-subgroup recall and FPR: source, bare vs subdomain, hosting platform, length,
    TLD class, unseen name stem
  - worst-group recall at 1% FPR
- **Cost:**
  - parameters and artifact size
  - NumPy latency per hostname, p50 and p95
  - NumPy/PyTorch parity: must be under 1e-5; the hard limit is 1e-3
- No metric is reported to more digits than its interval supports.

### 1.4 Experiment registry

Every run writes `experiment.json` with:
- the experiment ID and hypothesis ID
- the git commit, with a dirty-tree flag
- the dataset SHA-256 and the splits SHA-256
- the seed and the full configuration
- loss, optimiser, scheduler and regularisers
- wall time, parameters and the artifact SHA-256
- validation metrics

A single `compare.py` builds every table in the reports from these files. No number is
typed by hand.

### 1.5 Rules

- The test sets never drive a decision. All decisions are made on validation.
- No synthetic hostname enters training with a label unless its label is guaranteed by
  construction. Synthetic probes are used for diagnosis only.
- The baseline artifact, its run folder and the deployed `.env` stay untouched.
- Research code lives on a separate branch (`cnn-v4-research`) and in new modules. The
  existing v1/v2/v3 paths, tests and artifact formats keep working, and a v4 artifact
  gets its own format string so older code refuses it cleanly.

## 2. Prerequisites (no model change)

| ID | Work | Why it comes first | Cost |
| --- | --- | --- | --- |
| P0.1 | Benchmark harness and experiment registry (§1.3, §1.4); move the Phase 1 scripts into the repo | Four historical claims rest on scripts in a temporary folder | 0.5 day |
| P0.2 | Trainer records git commit + dirty flag; saves per-epoch EMA checkpoints | Needed for reproducibility and for H2a (choose checkpoints by low-FPR metric at no training cost) | 0.5 day |
| P0.3 | Locked test: low-FPR legitimate bank now; temporal phishing and CrUX holdouts from the next snapshots | Only honest test of new phishing, and of FPR at 0.1% | 0.5 day now + wait for snapshots |
| P0.4 | Stem-grouped evaluation subset: validation hostnames whose name stem never occurs in training | Stem-seen hostnames score 0.982 vs 0.900; this removes the optimism | 1 hour |
| P0.5 | Re-baseline: retrain the v3-fresh recipe, seed 42, with checkpoints saved | The seed-matched control every screen compares against | 1.5 CPU hours |

## 3. Hypotheses

Each states the limitation, the mechanism, the evidence, the experiment, what could go wrong
and the decision metric. "Primary" = recall at 0.5% and 1% FPR and partial ROC-AUC
(FPR <= 1%) on validation (§1.3). Costs are CPU hours at 14 threads (one member on the full
split is about 1.3 h).

### H1 · Rebalance the data the model learns its priors from — priority 1

- **Limitation:** 48% of training rows come from single-label sources. Legitimate
  platform-hosted (137 in validation), long and deep hostnames are rare. The model learned
  "platform / long / deep ⇒ phishing", which flags 47.5% of legitimate platform sites.
- **Why it should help:** empirical risk minimisation minimises loss under the *training*
  distribution. When a feature f (platform suffix) has P(y=1 | f) = 0.985 in training, the
  Bayes-optimal score under that distribution is near 1 for every f-host. Making the training
  conditional P(y | f) closer to deployment's changes the optimum itself. No architecture
  can fix a prior the data teaches.
- **Evidence:** the previous data fix cut CrUX subdomain false alarms from 73% to 3%,
  the same mechanism. 38 of the 40 highest-scoring legitimate hostnames are
  platform/infrastructure (`CNN_FAILURE_ANALYSIS.md` §6).
- **Experiment:** dataset v2.
  - Add CrUX hostnames on PSL private suffixes, plus long (>30 characters) and deep
    (4+ labels) CrUX hostnames, beyond the 100,000 random sample (about 860,000 CrUX
    hostnames are unused).
  - Remove any of them whose hostname or registered domain is on a phishing list.
  - Keep `--reuse-splits`, and retrain one seed.
- **Could go wrong:**
  - CrUX "legitimate" includes abused platform hosts, so this adds label noise. Report the
    share removed by the list filter.
  - Recall on platform phishing will drop. That is the intended trade, and it is reported.
- **Metric:** FPR at fixed recall on platform-legitimate, long-legitimate and
  deep-legitimate hostnames. Primary metric overall: within −0.5 points, or better.
- **Cost:** 2 h.

### H2 · Optimise and select for the operating region, not the average — priority 1

- **Limitation:**
  - Early stopping and checkpoint choice use mean validation BCE.
  - Recall falls 18 points from 1% to 0.1% FPR, and the top negatives are structured
    (platform/infra), not noise.
- **Why it should help:**
  - BCE weights all negatives equally. Partial AUC over FPR in [0, α] depends only on
    negatives in the top α-quantile.
  - (a) Choosing the checkpoint by validation pAUC aligns selection with the metric for free.
  - (b) A one-way partial-AUC surrogate concentrates the gradient on the hardest negatives:
    a pairwise squared hinge between each positive and the top-k negatives in each batch
    (Narasimhan and Agarwal, 2013).
  - (c) Focal or asymmetric losses down-weight easy examples (Lin et al., 2017;
    Ridnik et al., 2021).
- **Evidence:** §2 and §10 of the failure analysis.
- **Experiment:**
  - H2a: re-select checkpoints of P0.5 by pAUC (0 h).
  - H2b: pAUC surrogate added to BCE (weight 0.1 to 0.5), one seed.
  - H2c: focal loss, γ = 1 to 2.
- **Could go wrong:**
  - Few hard negatives per 128-batch make the surrogate noisy. Mitigate with a per-epoch
    hard-negative memory.
  - Optimising pAUC on the validation set that also selects checkpoints overfits it. The
    final read is on the locked test.
  - Calibration may degrade; Platt refitting covers it.
- **Metric:** primary metrics; ROC-AUC may fall by at most 0.002.
- **Cost:** 3 h.

### H3 · Finish the learning-rate schedule — priority 1

- **Limitation:** members are kept while the learning rate is still 29–66% of peak.
- **Why it should help:** the SGD noise scale grows with learning rate / batch size.
  Annealing to a small learning rate, or averaging weights along the tail (SWA; Izmailov et
  al., 2018), reaches lower-loss, flatter solutions. Today's 0.55-epoch EMA horizon is
  short.
- **Evidence:** best epochs 14–21 of a 30-epoch cosine; validation loss swings ±0.003 between
  epochs (`history.json`).
- **Experiment:** cosine over 20 epochs to 1% of peak, with no early-stop truncation. A second
  arm runs SWA over the last 5 epochs.
- **Could go wrong:** more low-learning-rate epochs widen the existing 0.03 train/validation
  gap. Watch the stem-unseen subset.
- **Metric:** validation BCE and primary metrics, seed-matched.
- **Cost:** 3 h.

### H4 · Add a member with a different inductive bias — priority 1

- **Limitation:** seed members share 81–84% of errors (logit correlation 0.965).
- **Why it should help:** for an average of M members with correlation ρ, the variance of the
  mean error is σ²(ρ + (1−ρ)/M). With ρ = 0.965, adding seeds barely helps. The character
  n-gram model has ρ = 0.94 with the CNN and a different hypothesis class (additive n-grams
  vs convolution).
- **Evidence:** the untuned standardised sum already gives ROC-AUC 0.9177 vs 0.9098 and recall
  at 1% FPR 63.0% vs 61.1% (`CNN_FAILURE_ANALYSIS.md` §7).
- **Experiment:**
  - Logit stacking of CNN members + n-gram logistic regression, with weights fit on the
    calibration split.
  - Later, the same with v4 members.
  - The NumPy form of the n-gram model is a sparse dot product: about 400,000 weights plus
    a hashed or explicit n-gram vocabulary, a few MB.
- **Could go wrong:**
  - The brief says the CNN must remain central. This makes the CNN the main model with a
    small linear auxiliary. If you prefer, it stays a research reference only (§5).
  - The n-gram model memorises stems more. Check the stem-unseen subset.
- **Metric:** primary metrics, plus the stem-unseen ROC-AUC.
- **Cost:** 0.5 h plus inference work.

### H5 · Component-aware representation and pooling — priority 2 (the core v4 architecture change)

- **Limitation:** the head sees global max/mean/attention pools and the name mean. There is
  no separate suffix or subdomain vector, and no label index or position-in-label input.
  The TLD swings decisions (+0.60 for `.xyz`), acting as a near-independent prior.
- **Why it should help:** a linear n-gram model can only *add* evidence. With per-component
  pooled vectors (subdomain, name, suffix) feeding an MLP head, the model can represent
  *interactions*: "brand token in the subdomain AND an unrelated registered name", or "bait
  word AND cheap suffix". An additive model cannot express these. This is the CNN's one
  structural advantage over the n-gram ceiling.
  - Label-index and position-in-label embeddings make "third label from the right" and
    "label length 25" directly available. Today they must be inferred from dots through a
    5-wide stem.
- **Evidence:** component boundary markers alone improve the n-gram model (recall at 0.1%
  FPR +2.4 points). Roles already exist, so this extends a working mechanism.
- **Experiment:**
  - H5a: v3 + per-component masked mean and max pools + label-index (0..7 from the right)
    and position-in-label (0..15) embeddings.
  - H5b: H5a + *suffix dropout*: zero the suffix pooled vector, and replace suffix
    characters with UNKNOWN, with p = 0.2 during training. This forces name and subdomain
    evidence and reduces the TLD shortcut.
- **Could go wrong:** an explicit suffix vector could *strengthen* TLD reliance, worsening
  evasion. H5b is the control for that; measure the TLD-swap probe.
- **Metric:** primary metrics; recall at 1% FPR on bare `.com` (44% today); |Δp| on the
  suffix-swap probe (lower is better); stem-unseen ROC-AUC.
- **Cost:** 3 h.

### H6 · Re-allocate the receptive field; ConvNeXt-style channel mixing — priority 3

- **Limitation:** two of five blocks are degenerate for the median 17-character hostname
  (A1). Each block's channel mixing is one gated linear map on ReLU outputs (A2).
- **Why it should help:**
  - Parallel kernels k ∈ {3, 5, 7} at dilation 1–2 match the 3–15-character scale where the
    signal is (the n-gram model works with 1–5-grams).
  - An inverted bottleneck (C → 3C → GELU → C; Liu et al., 2022) gives each position a
    two-layer MLP. GLU gating (Dauphin et al., 2017) is kept as an arm, not assumed.
- **Evidence:** arithmetic only (A1, A2). The n-gram tie predicts a *small* effect, hence
  priority 3.
- **Experiment:** at matched parameters (C = 48, 4 blocks, dilations 1, 1, 2, 2):
  - H6a: multi-scale stem.
  - H6b: inverted-bottleneck block vs the GLU block.
  - H6c: both.
- **Could go wrong:**
  - No gain, because of the ceiling.
  - More NumPy operations per hostname raise latency; budget p95 <= 2× today.
- **Metric:** primary metrics, by length bucket; latency.
- **Cost:** 4 h.

### H7 · Self-supervised pretraining on unlabelled hostnames — priority 2

- **Limitation:**
  - Novel names generalise worse (stem-unseen 0.900 vs seen 0.982).
  - The hardest population is short `.com` names, where only the name's "wordness",
    language and brand content can help. 231,000 labels are too few to learn lexicons.
- **Why it should help:** masked character modelling (BERT-style, at character level as in
  CANINE; Clark et al., 2022) learns p(character | context). Its representations encode
  segmentation into words, language and pronounceability from millions of hostnames. The
  fine-tuning task then needs fewer labels to use those features.
- **Evidence:** indirect (the weak subgroups in §3 of the failure analysis). Unused corpora:
  about 860,000 CrUX, 1,000,000 Tranco and 278,000 Phishing.Database hostnames.
- **Experiment:**
  - Pretrain the v4 encoder with 15% span masking on the unlabelled corpus. Exclude every
    validation and test registered domain *and name stem* (leakage).
  - Fine-tune, and compare with random initialisation at equal fine-tuning epochs, over
    3 seeds.
  - Also evaluate source-held-out: fine-tune on PhiUSIIL only, test on fresh.
- **Could go wrong:**
  - The corpus is about 87% legitimate-popular, so the representation may *encode popularity*
    and amplify the source confounding.
  - Pretraining cost is 8–12 CPU hours.
  - Negative transfer is possible.
- **Metric:** stem-unseen ROC-AUC, recall at 1% FPR on bare `.com`, and the cross-source gap.
  Must improve *downstream* metrics, not reconstruction loss.
- **Cost:** 15 h.

### H8 · Separate "malicious hostname" from "phishing on a legitimate domain" — priority 2

- **Limitation:** some phishing labels sit on compromised or shared legitimate domains:
  - government, university and charity hostnames among the lowest-scored phishing;
  - 3.6% of PhiUSIIL phishing on a registered domain that serves a CrUX site;
  - 272 phishing labels on CrUX top-1M hosts, including `accounts.google.com`.
- **Why it should help:** these positives are unlearnable from the hostname. Under BCE they
  pull up the scores of similar *legitimate* names, raising FPR at a fixed threshold. They
  also cap recall below the 90% target.
- **Experiment:**
  - Tag positives as "likely legitimate-domain" using external evidence only: their
    registered domain is on CrUX or Tranco top lists, or their suffix is `gov`, `edu`, `mil`,
    `ac.*`, `go.*`.
  - (a) Down-weight them in training. (b) Report every metric on the full set and on the
    attacker-registered subset.
  - Never filter by model score.
- **Could go wrong:** popular-list membership is a crude proxy; some attacker domains make
  the Tranco top 1M briefly. Report the tagged share and sample 100 tags for your manual
  check.
- **Metric:** primary metrics on the attacker-registered subset; FPR at fixed recall overall.
- **Cost:** 2 h.

### H9 · Group-robust training — priority 3 (after H1)

- **Limitation:** worst-group FPR is 47.5% (platform legitimate) against 0.89% overall.
- **Why it should help:** GroupDRO (Sagawa et al., 2020) minimises the maximum group loss with
  exponentiated-gradient group weights. That discourages features that help only on average.
  - Groups must contain both labels, so they are defined by structure:
    shape × platform × length bucket, *not* by source.
- **Could go wrong:** lower average performance. It needs strong L2 or early stopping, as
  Sagawa et al. report. After H1, the gap it targets may already be closed.
- **Metric:** worst-group recall at 1% FPR and worst-group FPR, plus the overall primary
  metric trade-off.
- **Cost:** 2 h.

### H10 · Calibration and the threshold under shift — priority 2 (post-hoc, cheap)

- **Experiment:** compare Platt, temperature and isotonic calibration on the calibration
  split. Report ECE overall and per structural group, and the stability of the threshold
  across bootstrap resamples.
- **Why:** a threshold chosen to the third decimal (0.893) on 35,308 negatives moves with
  resampling. Knowing its variance tells us how precisely FPR can be promised.
- **Could go wrong:** isotonic regression overfits small tails.
- **Metric:** ECE, Brier, and the 95% interval of FPR at the chosen threshold.
- **Cost:** 0.2 h.

### H11 · Homoglyph-aware input for punycode — priority 3 (correctness)

- **Limitation:** IDNA turns homoglyph attacks into opaque `xn--` strings. Validation has 54
  such hostnames (ROC-AUC 0.61).
- **Experiment:** add a channel that decodes punycode and maps each code point to its Unicode
  TR39 confusable skeleton, so `pаypal` reads as `paypal`. Evaluate on a *diagnostic* set of
  synthetic homoglyph variants of brand domains. That set is never used for training.
- **Could go wrong:** too little real data to measure accuracy, so it is a robustness claim
  only, and is stated as such.
- **Cost:** 1 h.

### H12 · Hard-negative emphasis — merged into H1 and H2

Hard legitimate hostnames are exactly the platform/infrastructure population. H1 supplies
more of them, and H2b emphasises them in the loss. Labels are verified against the phishing
lists, never against model scores.

## 4. CNN v4 design for review

This is the *starting* architecture, the union of H5 + H6 arms. Each component is removed one
at a time in the ablation, and the final v4 keeps only components that earned their place.

```text
hostname -> same normalisation; optional TR39 skeleton channel (H11)
  inputs per character: char (41) + role (4) + label index from right (0..7)
                        + position in label (0..15)             [summed embeddings, 32-d]
  stem: parallel depthwise-separable k = 3, 5, 7 (dilation 1) -> concat -> 1x1 -> C = 48
  body: 4 ConvNeXt-1D blocks, depthwise k = 7, dilations 1, 1, 2, 2:
        LN -> 1x1 (C -> 3C) -> GELU (tanh approximation, NumPy parity) -> 1x1 (3C -> C)
        -> layer scale -> residual -> mask
  pools: global max, mean, 2-head attention
         + per component (subdomain, name, suffix): masked mean and max
         [suffix dropout p = 0.2 in training]
  head:  dropout -> Linear -> GELU -> Linear -> logit
  ensemble: 2-3 seed members (+ n-gram member if H4 is accepted), logits averaged
            with weights fit on the calibration split
  calibration: chosen by H10; threshold on validation
```

Compatibility:
- New artifact format `m-phish-charcnn-v4` with its own feature version. v1–v3 artifacts and
  tests are untouched.
- NumPy inference uses only matmul, depthwise convolution, LayerNorm, tanh-GELU, softmax and
  masked pooling. Parity is tested at < 1e-5, with masking, padding, batch invariance, the
  253-character maximum, 1-character hostnames and ensembles covered.

Deliberately **not** proposed:
- **A Transformer encoder.** The n-gram tie shows the data, not context length, limits
  accuracy, and attention over 17 characters adds operations without a mechanism to exploit.
  Attention stays as pooling only, where it is cheap and measurable.
- **Larger width.** It gained about 0.002 ROC-AUC (single seed) at 2.6× training time.
- **Synthetic labelled hostnames.**
- **Brand lists.** They are maintenance-heavy and leak into evaluation.

## 5. Experiment matrix

Ordered by expected value per CPU hour. "Screen" = 1 seed vs the P0.5 control.

| Order | ID | Hypothesis | Change | Screen cost | Promote if | Confirm cost |
| ---: | --- | --- | --- | ---: | --- | ---: |
| 1 | P0.1–P0.5 | — | harness, registry, locked test, stem subset, re-baseline | 1.5 h + 1.5 days | — | — |
| 2 | E2a | H2 | checkpoint chosen by pAUC | 0 h | pAUC gain > seed spread | 0 h |
| 3 | E10 | H10 | Platt vs temperature vs isotonic | 0.2 h | lower ECE, stable threshold | — |
| 4 | E4 | H4 | CNN + n-gram stacking | 0.5 h | primary +1 point | 0.5 h |
| 5 | E1 | H1 | dataset v2 (platform, long, deep legitimate) | 2 h | platform-legitimate FPR halves, primary ≥ −0.5 | 4 h |
| 6 | E3 | H3 | 20-epoch cosine to 1%; SWA arm | 3 h | primary > seed spread | 4 h |
| 7 | E8 | H8 | down-weight legitimate-domain phishing | 2 h | attacker-subset primary +1 point | 4 h |
| 8 | E5a/b | H5 | component pools + label position; suffix dropout | 3 h | primary +1 point or bare `.com` +2 | 4 h |
| 9 | E2b/c | H2 | pAUC surrogate; focal | 3 h | recall at 0.5% FPR +1.5 | 4 h |
| 10 | E7 | H7 | masked character pretraining | 15 h | stem-unseen +0.005 ROC-AUC | 8 h |
| 11 | E6a–c | H6 | multi-scale stem; ConvNeXt block | 4 h | primary > seed spread at ≤ 2× latency | 4 h |
| 12 | E9 | H9 | GroupDRO | 2 h | worst-group gain, primary ≥ −0.5 | 4 h |
| 13 | E11 | H11 | punycode skeleton channel | 1 h | probe robustness only | — |
| 14 | F | all | combine winners (forward selection), 3-seed ensemble, calibration, locked test, NumPy parity | — | — | 10 h |

Estimated total: about 60–75 CPU hours, which is 4–6 days of this PC working mostly
unattended. Waves 1–2 (orders 1–9) take about 2 days and answer the highest-value questions
before any long run.

## 6. Decisions needed before Phase 5

1. **Approve the priorities**, or change them. In particular, data-first (H1, H8) and
   objective-first (H2, H3) come before new architecture (H5–H7).
2. **Release-gate scope.** The 90% recall target is very likely unreachable for hostname-only
   detection, because part of labelled phishing uses legitimate domains (H8). Options:
   - keep the gate as is and report against it;
   - add a scoped target on attacker-registered hostnames;
   - lower recall and keep the FPR bound.

   This is a product decision.
3. **n-gram member (H4):** allowed in the deployed ensemble as a small auxiliary, or research
   reference only?
4. **Where the research code lives.** The recommendation is a new branch `cnn-v4-research`
   with new modules and no edits to existing CNN files until a candidate wins. The
   alternative is keeping everything outside the repo.
5. **Machine time:** about 60–75 CPU hours over the next week, using 14 of 20 threads.
6. **Temporal locked test:** wait 2+ weeks for new snapshots (recommended). The alternative
   is to proceed with only the legitimate bank plus the current reported test, and state the
   weaker claim.

## 7. References

Cited by author, year and title. Each method is re-checked against the original paper before
it is implemented.

- Clark, Garrette, Turc, Wieting (2022). *CANINE: Pre-training an Efficient Tokenization-Free
  Encoder for Language Representation.*
- Dauphin, Fan, Auli, Grangier (2017). *Language Modeling with Gated Convolutional Networks.*
- Devlin, Chang, Lee, Toutanova (2019). *BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding.*
- Guo, Pleiss, Sun, Weinberger (2017). *On Calibration of Modern Neural Networks.*
- Izmailov, Podoprikhin, Garipov, Vetrov, Wilson (2018). *Averaging Weights Leads to Wider
  Optima and Better Generalization.*
- Lakshminarayanan, Pritzel, Blundell (2017). *Simple and Scalable Predictive Uncertainty
  Estimation using Deep Ensembles.*
- Le, Pham, Sahoo, Hoi (2018). *URLNet: Learning a URL Representation with Deep Learning for
  Malicious URL Detection.* (Reference baseline family for character-level URL models.)
- Lin, Goyal, Girshick, He, Dollár (2017). *Focal Loss for Dense Object Detection.*
- Liu, Mao, Wu, Feichtenhofer, Darrell, Xie (2022). *A ConvNet for the 2020s.*
- Narasimhan, Agarwal (2013). *A Structural SVM Based Approach for Optimizing Partial AUC.*
- Ridnik et al. (2021). *Asymmetric Loss for Multi-Label Classification.*
- Sagawa, Koh, Hashimoto, Liang (2020). *Distributionally Robust Neural Networks for Group
  Shifts.*
- Unicode Technical Standard #39. *Unicode Security Mechanisms* (confusable skeletons).

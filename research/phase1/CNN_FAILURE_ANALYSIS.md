# CNN failure analysis: v3-fresh baseline

Phase 1, 24 Nov 2026. Read-only. Measured by `scripts/phase1_audit.py` on the
**validation split only** (62,045 hostnames; the test split stays locked). Raw output:
`results/phase1_audit.json`. "Recall at x% FPR" is computed from the ensemble logit; the
95% intervals are percentile bootstrap (300 resamples).

## Headline findings

1. **The architecture is not the bottleneck today.** A linear model on character 1–5-grams,
   trained on the same split in under a minute, matches the 3-member CNN:
   - ROC-AUC 0.9142 to 0.9151, against 0.9098 for the CNN.
   - Recall at 1% FPR 60.5 to 61.9%, against 61.1% for the CNN.
   - The two models' scores correlate at 0.94.

   The CNN currently works as an n-gram detector. It only adds value if it learns what n-grams
   cannot express.
2. **The training data teaches source, not just phishing.** 48% of training rows come from
   a single-label source: every Phishing.Database hostname is phishing, and every CrUX
   hostname is legitimate. On PhiUSIIL, the only source with both labels, the fresh model is
   *worse* than its predecessor: ROC-AUC 0.888 against 0.919.
3. **Legitimate sites on hosting platforms are flagged about half the time.** Only 137 of
   8,936 validation hostnames on platform suffixes (`web.app`, `github.io`, `workers.dev`…)
   are legitimate, and 47.5% of those are flagged. 38 of the 40 highest-scoring legitimate
   hostnames are platform or cloud-infrastructure names.
4. **The top-level domain alone swings decisions.**
   - Re-registering legitimate bare domains under `.xyz` raises their flag rate from 0.5% to 24.5%.
   - Moving caught bare-domain phishing to `.com` drops its flag rate from 56.9% to 25.7%.
5. **Missed bare-domain phishing looks like ordinary legitimate domains,** and many of the
   lowest-scored "phishing" hostnames are government, university and charity sites. These are
   probably compromised legitimate domains, which no hostname-only model should flag. Part
   of the recall gap is a label-definition problem, not a model problem.
6. **The 3 ensemble members make nearly the same errors.** Their logits correlate at
   0.963–0.968, and 85.5% of ensemble errors are made by all three.

## 1. Data and representation facts

| Fact | Value |
| --- | --- |
| Hostname length, all 419,036 | p5 9 · p25 13 · **median 17** · p75 24 · p95 40 · p99 70 · max 249 |
| Median length by class | phishing 23 · legitimate 15 |
| Hostnames longer than 64 (v1's limit) | 4,799 |
| Hostnames longer than the receptive field (129) | 97 |
| Validation hostnames with characters outside the alphabet | 0 |
| Validation hostnames whose roles fall back to all-NAME (no parsable suffix) | 93 |
| Validation punycode (`xn--`) hostnames | 54 |

At the median length of 17, the dilation-16 block sees only its own position and the
dilation-8 block sees at most 3 of its 5 taps (`CNN_ARCHITECTURE_ANALYSIS.md` A1).

### Split composition

| Split | PhiUSIIL phishing | PhiUSIIL legitimate | Phishing.Database (all phishing) | CrUX (all legitimate) |
| --- | ---: | ---: | ---: | ---: |
| train | 46,349 | 74,142 | 55,846 | 55,094 |
| validation | 12,273 | 20,275 | 14,464 | 15,033 |

## 2. Baseline on validation

| Metric | `cnn-v3-fresh` | 95% interval | `cnn-v3-ens3` (same hostnames) |
| --- | ---: | --- | ---: |
| ROC-AUC | 0.9098 | 0.9075–0.9120 | 0.8786 |
| PR-AUC | 0.9143 | | 0.8553 |
| Partial ROC-AUC, FPR <= 1% (McClish) | 0.764 | | 0.559 |
| Recall at 0.1% FPR (35 false positives allowed) | 42.7% | 39.7–44.6% | 2.2% |
| Recall at 0.25% FPR (88 allowed) | 50.4% | 47.9–51.7% | 6.4% |
| Recall at 0.5% FPR (176 allowed) | 55.4% | 54.1–56.6% | 12.9% |
| Recall at 1% FPR (353 allowed) | 61.1% | 60.0–62.3% | 23.0% |
| At the deployed threshold: recall / FPR | 60.1% / 0.89% | | 74.9% / 14.9% |
| Brier score / ECE | 0.108 / 0.009 | | 0.160 / 0.092 |

Recall falls by 18 points between 1% and 0.1% FPR. The negatives that outscore phishing
at the top are not random: they are the platform and infrastructure hostnames in §6. The
0.1% figure rests on 35 false positives, so its interval is 5 points wide. A bank of
about 200,000 held-out legitimate hostnames would tighten it (`CNN_RESEARCH_PROPOSAL.md` §1.1).

### By source

| Source (validation) | Hostnames | ROC-AUC | Recall at 1% FPR | Recall at threshold | FPR at threshold | Mean predicted p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PhiUSIIL (both labels) | 32,548 | 0.888 (ens3: 0.919) | 60.6% (ens3: 69.1%) | 51.6% | 0.09% | 0.365 (observed 0.377) |
| Phishing.Database (all phishing) | 14,464 | | | 67.3% | | 0.804 |
| CrUX (all legitimate) | 15,033 | | | | 1.98% | 0.210 |

## 3. Where it fails: subgroups

All rows are `cnn-v3-fresh` on validation. Recall at 1% FPR uses each subgroup's own
negatives.

| Subgroup | Hostnames | Phishing | ROC-AUC | Recall at 1% FPR | Recall at threshold | FPR at threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bare registered domain | 47,027 | 17,662 | 0.889 | 59.7% | 56.5% | 0.52% |
| With subdomain | 15,018 | 9,075 | 0.945 | 51.2% | 67.1% | 2.76% |
| ICANN suffix | 53,109 | 17,938 | 0.868 | 44.4% | 41.1% | 0.71% |
| **Hosting-platform suffix** | 8,936 | 8,799 | 0.941 | 52.0% | 98.9% | **47.5%** (65 of 137) |
| Length 1–10 | 6,973 | 1,562 | 0.844 | 35.5% | 30.6% | 0.48% |
| **Length 11–15** | 18,031 | 4,590 | **0.810** | 31.7% | 25.8% | 0.34% |
| Length 16–20 | 16,171 | 5,788 | 0.870 | 47.1% | 43.0% | 0.66% |
| Length 21–30 | 13,857 | 8,222 | 0.935 | 60.3% | 68.7% | 2.01% |
| Length 31–50 | 5,457 | 5,023 | 0.970 | 62.3% | 94.0% | **13.8%** |
| **2 labels** (`name.tld`) | 35,146 | 9,584 | **0.809** | 33.0% | 26.8% | 0.38% |
| 3 labels | 20,445 | 12,692 | 0.954 | 67.8% | 74.6% | 1.97% |
| 4 labels | 5,057 | 3,238 | 0.982 | 58.0% | 87.7% | 3.19% |
| 5+ labels | 1,397 | 1,223 | 0.997 | 89.8% | 97.4% | 4.02% |
| **`.com`** | 26,054 | 9,547 | **0.821** | 44.4% | 41.8% | 0.73% |
| `.net` / `.org` | 5,638 | 1,806 | 0.908 | 61.4% | 61.2% | 0.94% |
| ccTLD | 20,842 | 8,149 | 0.939 | 68.2% | 64.7% | 0.54% |
| Other gTLD | 9,495 | 7,228 | 0.960 | 65.4% | 78.8% | 4.06% |
| No digit | 49,182 | 16,778 | 0.876 | 50.4% | 46.1% | 0.57% |
| No hyphen | 49,031 | 16,923 | 0.877 | 50.5% | 46.5% | 0.64% |
| Punycode | 54 | 25 | 0.61 | n too small | 28% | 13.8% |
| Name stem seen in training (other suffix) | 5,042 | 2,832 | **0.982** | 83.4% | 86.8% | 1.90% |
| Name stem unseen | 57,003 | 23,905 | 0.900 | 58.7% | 56.9% | 0.83% |

The weak region is one coherent population: short, 2-label `.com` names with no digits or
hyphens. The model then falls back on the length, platform and TLD priors. Those priors work
in aggregate but are costly for the rare legitimate hostnames that are long, deep or on a
platform. In those groups, FPR at the threshold is 4 to 50 times the overall 0.89%.

## 4. Shortcuts: structural probes

Each probe takes 3,000 validation bare domains, applies one edit and re-scores. Probes
measure sensitivity, not correctness: `login.example.com` is often legitimate.

| Edit | Legitimate: flagged before → after | Mean Δp | Phishing: flagged before → after | Mean Δp | `ens3` legitimate after |
| --- | --- | ---: | --- | ---: | ---: |
| prefix `shop.` | 0.5% → 0.2% | −0.09 | 56.9% → 53.6% | −0.07 | 69.4% |
| prefix `login.` | 0.5% → 0.8% | 0.00 | 56.9% → 57.9% | 0.00 | 83.6% |
| prefix random 6-character label | 0.5% → 14.6% | +0.33 | 56.9% → 70.5% | +0.14 | 89.6% |
| name + `-login` | 0.5% → 6.3% | +0.34 | 56.9% → 68.8% | +0.16 | 27.1% |
| name + digit `7` | 0.5% → 0.4% | +0.01 | 56.9% → 56.3% | 0.00 | 7.1% |
| suffix → `.xyz` | 0.5% → **24.5%** | **+0.60** | 56.9% → 65.4% | +0.19 | 100% |
| suffix → `.com` | 0.5% → 0.2% | +0.03 | 56.9% → **25.7%** | **−0.18** | 2.8% |

- The fresh data removed the subdomain shortcut. A benign `shop.` or `login.` prefix no
  longer moves the score; the previous model flagged 69–84% of such variants.
- The TLD is now the dominant single cue. That is partly legitimate, since abuse rates differ
  by TLD. But it gives attackers a cheap evasion (register under `.com`), and it flags
  legitimate sites on cheaper TLDs.
- A random-looking subdomain label raises the score strongly. That is often right for
  phishing kits, but cloud and CDN hostnames also use random labels (§6).

## 5. Generalisation and memorisation

| Set | Hostnames | ROC-AUC | Log loss | Recall at 1% FPR |
| --- | ---: | ---: | ---: | ---: |
| Training sample (fresh) | 30,000 | 0.940 | 0.283 | 72.4% |
| Validation (fresh) | 62,045 | 0.910 | 0.343 | 61.1% |
| Training sample (`ens3`, PhiUSIIL only) | 15,547 | 0.942 | 0.259 | 74.9% |
| Validation (`ens3`, PhiUSIIL only) | 32,548 | 0.919 | 0.301 | 69.1% |

- The train/validation gap is moderate: 0.03 ROC-AUC and 11 points of recall at 1% FPR.
- Name stems recurring under another suffix are scored far better: ROC-AUC 0.982 against
  0.900 for unseen stems. Splitting groups by registered domain keeps `paypal-verify.com` and
  `paypal-verify.net` apart. So 8% of validation hostnames share a campaign stem with
  training, and validation is slightly optimistic.
- **Source shift is large.** The PhiUSIIL-only model scores 0.819 on fresh data, while the
  fresh model loses 0.03 on PhiUSIIL. Neither generalises across collection processes.

## 6. Dataset quality

### Confounding between source and label

In the fresh additions, the label is a function of the source. Any feature that tells
"CrUX-style" from "Phishing.Database-style" hostnames lowers the loss, whether or not it
relates to phishing. CrUX means *popular*, not *benign*. Among the 40 highest-scoring
legitimate validation hostnames, 38 are on platforms or cloud infrastructure: Azure Front
Door, Cloudflare Workers, Heroku, Vercel, Firebase, `x.yupoo.com`, S3. A few look abusive
themselves, for example `mail.gov-naverworks.com`, `illanes-encuesta.ddns.net` and
`parasut-e-documents.s3.amazonaws.com`. So some "legitimate" labels are probably wrong.

### Label conflicts with external lists

| Check | Count | Share |
| --- | ---: | ---: |
| PhiUSIIL legitimate hostnames on the Phishing.Database list | 30 of 134,792 | 0.02% |
| PhiUSIIL phishing hostnames that are CrUX top-1M sites (e.g. `accounts.google.com`, `2ip.ua`) | 272 of 84,244 | 0.32% |
| PhiUSIIL phishing hostnames whose registered domain serves a CrUX site | 3,038 of 84,244 | 3.6% |

The last row is a lower bound on phishing hosted on compromised or shared legitimate
domains. Small compromised sites are not in the CrUX top 1M.

### Hard examples

- **20** validation legitimate hostnames score p > 0.99. All are CrUX platform or
  infrastructure names (above).
- **88** validation phishing hostnames score p < 0.05. The lowest-scored are:
  - `med.siam.edu`
  - `bps.dti.gov.ph`
  - `vetdolakha.gov.np`
  - `president-office.gov.mm`
  - `snum.or.kr`
  - `stmarouns.nsw.edu.au`

  These are institutional domains with phishing content. Their hostnames are legitimate,
  and only the page or path is malicious.

### Missed vs caught bare-domain phishing

| Bare domains (validation) | n | Median length | Has digit | Has hyphen | `.com` | Platform suffix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phishing caught | 9,975 | 26 | 48.7% | 56.1% | 24.2% | 73.9% |
| **Phishing missed** | 7,687 | **15** | **10.4%** | **14.0%** | **57.0%** | **1.3%** |
| Legitimate passed | 29,213 | 14 | 6.8% | 8.4% | 49.6% | 0.2% |
| Legitimate flagged | 152 | 19 | 32.9% | 32.9% | 26.3% | 33.6% |

Missed bare phishing has nearly the same surface profile as legitimate domains. What
remains is information in the name itself: words, brands, language and pronounceability.
Some of it may be learnable from far more unlabelled text (H7 in the proposal). Some of it
is not in the hostname at all.

## 7. The n-gram reference: is the architecture the bottleneck?

TF-IDF over character 1–5-grams (with boundary marks) feeds a logistic regression. It was
trained on the same 231,431 training hostnames, with C chosen on the calibration split and
scored on validation. The CNN had an advantage here: it used validation for early stopping.

| Model | Features | ROC-AUC | Recall at 1% FPR | Recall at 0.1% FPR | ROC-AUC on PhiUSIIL part |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN ensemble (`cnn-v3-fresh`) | 268,326 parameters | 0.9098 | 61.1% | 42.7% | 0.888 |
| n-gram LR, plain | 387,926 | 0.9142 | 61.9% | 38.7% | 0.900 |
| n-gram LR, component markers `<sub\|name\|suffix>` | 399,965 | **0.9151** | 60.5% | 41.1% | 0.902 |
| CNN + n-gram, standardised score sum (no fitting) | | **0.9177** | **63.0%** | | |

- The n-gram model is at least as good overall, slightly worse at 0.1% FPR, and better on
  PhiUSIIL. With 0.94 correlation, the two learned nearly the same function.
- Marking component boundaries helps the n-gram model (ROC-AUC +0.001, recall at 0.1% FPR
  +2.4 points). Structure carries some signal beyond raw n-grams.
- An untuned sum gains +0.008 ROC-AUC and +1.9 points of recall at 1% FPR. One different
  model adds more than the two extra seed members added (+0.007 ROC-AUC).

**Consequence:** deeper or longer-range convolution is unlikely to pay off. The CNN can only
beat this reference with:
- non-additive structure (which component holds a brand; brand combined with suffix),
- better use of unlabelled data,
- a training objective aimed at the low-FPR region,
- cleaner data.

## 8. Ensemble diversity

| Measure | `cnn-v3-fresh` |
| --- | ---: |
| Member ROC-AUC | 0.9024 / 0.9038 / 0.9028 → ensemble 0.9098 |
| Member recall at 1% FPR | 59.4% / 60.2% / 58.7% → ensemble 61.1% |
| Pairwise logit correlation | 0.963–0.968 |
| Pairwise disagreement at the threshold | 3.2–3.7% |
| P(both wrong given one wrong) | 0.81–0.84 |
| Ensemble errors made by all 3 members | 85.5% |
| Error rate: ensemble / oracle (any member right) | 17.7% / 15.1% |

Seed diversity is nearly exhausted: a fourth seed member would add little. Real diversity
has to come from a different inductive bias.

## 9. Calibration

Overall calibration is good: ECE 0.009, Brier 0.108. Conditional calibration is not:
- PhiUSIIL subdomain hostnames: ECE 0.175. They are 85% phishing, but get lower probabilities.
- CrUX legitimate hostnames: mean p 0.21.
- Phishing.Database hostnames: mean p 0.80.

A single global Platt fit averages over sources whose phishing base rates are 0%, 38% and
100%. A probability of 0.9 therefore means different things for different kinds of hostname.
Group-specific calibration is **not** the fix, because it would re-encode the shortcuts. The
fix is data whose base rates resemble deployment (H1).

## 10. Training dynamics

- Members were selected at epochs 14–21, when the learning rate was still 29–66% of its
  peak (5.8e-4 to 1.3e-3, against 2e-3). The 30-epoch cosine never completes. EMA averages
  over about 1,000 steps (0.55 epoch), which only partly compensates.
- Validation BCE drives both early stopping and checkpoint choice. It is not the operating
  metric; recall at <= 1% FPR is.
- Members differ only by seed and data order.

## 11. Findings mapped to the brief's questions

| Brief | Question | Finding | Evidence | Hypothesis |
| --- | --- | --- | --- | --- |
| A | Representation adequate? | Character and role inputs are lossless for ASCII hostnames. There is no label position or length, and homoglyphs are lost in punycode | §1, n-gram markers gain | H5, H11 |
| B | Generalisable or memorised? | Mixed: stem memorisation (0.982 vs 0.900), source style learned | §5, §6 | H1, H7, P0 split |
| C | Convolutions expressive enough? | Expressive *enough*: a linear n-gram model ties. Capacity is misallocated (A1), and non-additive structure is under-used | §7 | H5, H6 |
| D | Categories under-represented? | Yes: legitimate platform hostnames (137 in validation), long and deep legitimate hostnames, punycode | §3 | H1, H9 |
| E | Objective fit for low FPR? | No: selection by BCE, and recall falls 18 points from 1% to 0.1% FPR | §2, §10 | H2, H3 |
| F | Fails on unseen sources, domains, time? | Sources: yes (0.819 / 0.888 cross-source). Time: **untestable with current data** (no dates) | §5 | P0 locked temporal test |
| G | Members make similar errors? | Yes: correlation 0.965, and 85.5% of errors are shared | §8 | H4 |
| H | Probabilities trustworthy? | Globally yes, conditionally no | §9 | H10, H1 |

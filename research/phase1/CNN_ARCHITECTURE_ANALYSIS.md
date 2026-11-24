# CNN architecture analysis: v3-fresh baseline

Phase 1, 24 Nov 2026. Read-only. Reconstructed from `backend/app/services/ml/url_cnn.py`
(NumPy inference), `train_url_cnn.py` (PyTorch training), `observations.py` (domain
parsing) at commit `54b5d74`, and the artifact `runs/cnn-v3-fresh/url_cnn.json`.

**Summary.** The baseline is a well-engineered, padding-invariant gated CNN. Its design
spends its capacity in the wrong places for hostnames:
- Two of the five residual blocks have receptive fields far longer than a typical hostname.
- Each block has only one thin cross-channel layer.
- The head never sees the public suffix or the subdomain as separate pooled features.

None of these is proven to limit accuracy yet. Each becomes a hypothesis with an ablation in
`CNN_RESEARCH_PROPOSAL.md`.

## 1. Exact reconstruction

```text
hostname
  -> model_input: strip, drop trailing ".", lowercase, IDNA -> punycode, drop one leading "www."
  -> characters: 39-symbol alphabet [a-z0-9-._] + PAD(0) + UNKNOWN(1)  -> vocab 41
  -> roles via bundled PSL (tldextract 5.4.0, private suffixes ON):
       SUBDOMAIN (incl. its trailing dot) | NAME | SUFFIX (incl. its leading dot) | PAD
  -> right-aligned, max 253 characters (DNS maximum; nothing is truncated)
  -> E_char[c] + E_role[r]            (32-d each, summed, masked)
  -> stem Conv1d k=5 (32 -> 64) -> LayerNorm(channel) -> ReLU -> mask
  -> 5 blocks, dilation 1, 2, 4, 8, 16:
       depthwise Conv1d k=5 -> LayerNorm -> ReLU -> mask
       -> 1x1 Conv (64 -> 128) -> GLU (value * sigmoid(gate)) -> dropout 0.3
       -> h + scale_c * branch  (scale initialised 0.1, per channel) -> mask
  -> pools (all masked): max | mean | single-query attention | mean over NAME positions
  -> concat (256) -> dropout -> Linear 256 -> 128 -> ReLU -> dropout -> Linear 128 -> 1
  -> ensemble: mean of 3 member logits (seeds 42, 43, 44)
  -> Platt: p = sigmoid(1.2675 * z + 0.3826)   (fit on the calibration split)
  -> flag if p >= 0.8931                       (chosen on the validation split)
```

Training: AdamW (lr 2e-3, weight decay 1e-4 on matrices only), batch 128, 10% linear
warm-up then cosine to 5% of peak over 30 epochs, gradient clipping at 1.0, early stopping
on validation BCE with patience 5, best-epoch restore. Regularisers: character dropout
0.1 (characters -> UNKNOWN; padding and roles untouched), symmetric label smoothing 0.05,
EMA 0.999 with warm-up `min(0.999, (1+t)/(10+t))`; the EMA weights are validated and
exported. Unweighted BCE (class weight "none"), deterministic algorithms, 14 CPU threads.

## 2. Parameters per member (89,442)

| Component | Shape | Parameters | Share |
| --- | --- | ---: | ---: |
| Character embedding | 41 x 32 | 1,312 | 1.5% |
| Role embedding | 4 x 32 | 128 | 0.1% |
| Stem convolution + LayerNorm | 64 x 32 x 5 (+64), LN 128 | 10,432 | 11.7% |
| 5 x depthwise conv (k=5) | 64 x 1 x 5 (+64) each | 1,920 | 2.1% |
| 5 x block LayerNorm | 128 each | 640 | 0.7% |
| 5 x pointwise GLU projection | 128 x 64 (+128) each | 41,600 | 46.5% |
| 5 x residual scale | 64 each | 320 | 0.4% |
| Attention pooling score | 64 (+1) | 65 | 0.1% |
| Head dense | 128 x 256 (+128) | 32,896 | 36.8% |
| Output | 1 x 128 (+1) | 129 | 0.1% |

The feature extractor that sees characters has 56,352 parameters, and 74% of them are the
five 1x1 GLU projections. The head, which only sees four pooled vectors, holds 37% of the
model.

## 3. Compute

Per character and member: stem 10,240 multiply-adds, each block 320 (depthwise) + 8,192
(pointwise), attention 64. That is about 52,900 MACs per character, plus 32,900 for the head.
A median-length hostname costs about 1 M MACs per member, or 3 M for the ensemble.

The deployed NumPy path measures 2.2 ms per hostname for the ensemble (`results.json`).
At about 3 M MACs, arithmetic is a small part of that; per-operation Python/NumPy overhead
dominates. So a moderately wider or deeper candidate would cost little extra latency. The
latency budget limits the *number of NumPy operations*, not FLOPs. This matters for
choosing between "more layers" and "wider layers".

Training cost is the practical constraint: 227 to 290 s per epoch per member on 231,431
hostnames with 14 threads, so about 1.2 to 1.6 h per member and 4.5 h per 3-member run.

## 4. Architectural findings

Each finding states the evidence available now. Phase 1 measurements that test it are in
`CNN_FAILURE_ANALYSIS.md`.

### A1. Most of the receptive field is spent on padding

Receptive field = 1 + 4 x (1 [stem] + 1 + 2 + 4 + 8 + 16) = 129 characters. The median
hostname is far shorter (see `CNN_FAILURE_ANALYSIS.md` §1). For a 17-character hostname,
every tap of the dilation-16 block except the centre falls outside the string and is
masked to zero. The dilation-8 block keeps at most three of its five taps. Those two
blocks then act mostly as per-position gated MLPs, not as convolutions. The depth that is
meant for "long-range structure" adds little for most inputs.

### A2. Channel mixing is one gated linear map per block

The only cross-channel operation in a block is a 1x1 projection to 2C followed by a GLU.
ConvNeXt-style blocks use an inverted bottleneck (C -> 4C -> GELU -> C), which gives each
position a two-layer MLP. Here, each block's non-linear channel transform is one
multiplicative gate. Also, the projection only ever receives ReLU outputs, so it never
sees a negative input: the sign information that LayerNorm produced is discarded before
channels are mixed.

### A3. The suffix and the subdomain never reach the head on their own

The head sees global max, global mean, attention-weighted mean and the NAME-role mean.
The public suffix (`.com` vs `.xyz` vs `.web.app`) is one of the strongest single cues in
phishing data. It reaches the head only mixed into global pools, where every other
character competes for the same channels. The same holds for "what the subdomain says".

### A4. Structure is a 3-way role, added to the character embedding

Roles are summed into the character vector. Nothing encodes:
- which label a character is in (the 1st, 2nd or 5th from the right),
- a character's position inside its label,
- the label's length,
- how many labels the hostname has.

The CNN has to infer these from dots, using a stem whose kernel spans 5 characters. The
roles do separate the three examples in the brief correctly:
- `login.example.com`: subdomain `login`, name `example`.
- `example-login.com`: name only.
- `example.com.login-security.xyz`: subdomain `example.com`, name `login-security`, suffix `xyz`.

Their structure is available. What's missing is any numeric notion of position or length
within it.

### A5. One attention query

The attention pool is a single linear scoring vector (65 parameters). It can emphasise one
kind of position per hostname: for example the brand token or the bait word, but not both.

### A6. Homoglyph information is lost before the model sees it

IDNA turns `pаypal.com` (Cyrillic а) into `xn--pypal-4ve.com`. The model sees an
arbitrary-looking punycode string and cannot compare it visually with `paypal`. The
validation split has few punycode hostnames (see the failure analysis). So this is a
correctness gap for a known attack, not a measurable accuracy problem in the current data.

### A7. Ensemble members differ only by seed

Same architecture, same data, same recipe, different initialisation and order. Deep-ensemble
theory says seed diversity helps. But members that share inductive biases tend to make
correlated errors, and the size of that correlation is measured in the failure analysis.

### A8. Things that are right and should be kept

- Masking at every stage and LayerNorm without batch statistics. These make predictions
  independent of padding width and batch composition, and the tests enforce it.
- Right alignment. The registered domain and suffix are always at fixed offsets from the
  end.
- Private PSL suffixes (`web.app`, `github.io`). On a hosting platform, the customer label
  counts as the registered name, which is the right unit for both splitting and roles.
- The JSON + SHA-256 artifact with format and feature-version checks, and NumPy/PyTorch
  parity under 1e-5.

## 5. What this analysis does not show

Capacity being spent badly isn't proof that it limits accuracy. The near-identical ROC-AUC
of v1, v2 and v3 (within 0.01) suggests the data, not the architecture, sets the ceiling.
That would make A1 to A5 cheap to fix but small in effect. The decisive Phase 1 test is a
non-neural character n-gram model trained on the same split
(`CNN_FAILURE_ANALYSIS.md` §7). If it matches the CNN, the hostname's information content is
the binding limit. Architecture work should then target the low-FPR region and
generalisation, not overall ROC-AUC.

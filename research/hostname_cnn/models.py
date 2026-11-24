"""Model registry for research candidates.

``v3`` is the deployed architecture, built by the production builder, so the research control
is exactly the baseline network. ``v4`` is a configurable family for hypotheses H5 and H6:

* ``structure_embeddings``: label index from the right (0..7) and position inside the label
  (0..15, dots separate) are embedded and summed with the character and role embeddings;
* ``component_pools``: masked mean and max over each of subdomain, name and suffix, so the
  head sees the components separately (replaces v3's single name mean);
* ``block``: ``glu`` (the v3 block) or ``convnext`` (LN -> 1x1 expand -> GELU -> 1x1 project);
* ``attention_heads``: independent single-vector attention pools.

A parallel multi-scale *linear* stem is deliberately absent: summing or 1x1-fusing linear
convolutions with kernels 3/5/7 equals one kernel-7 convolution, so it adds no capacity.

``v3wide`` (H12, Wide & Deep; Cheng et al., 2016) is the deployed v3 network plus a linear
"wide" path: one weight per hashed character n-gram (n = 1..5, tagged with the hostname
component of its first character), summed and divided by sqrt(n-gram count), added to the
CNN logit. The wide table starts at zero and is created without drawing random numbers, so at
step 0 a v3wide run is exactly the v3 control with the same seed. Any v4 config can carry the
same path through ``wide_buckets``.
"""

from __future__ import annotations

import numpy as np

from backend.app.services.ml.train_url_cnn import build_model_v3, model_features
from backend.app.services.ml.url_cnn import (
    NORM_EPS,
    PAD,
    ROLE_COUNT,
    ROLE_NAME,
    ROLE_SUBDOMAIN,
    ROLE_SUFFIX,
    VOCAB_SIZE,
    encode,
    encode_roles,
)

LABEL_SLOTS = 9  # 0 = padding, 1..8 = label index 0..7 from the right (7 = "7 or more")
POSITION_SLOTS = 18  # 0 = padding, 1..16 = offset 0..15 in the label, 17 = dot

V3_DEFAULTS = {
    "embedding_dim": 32,
    "channels": 64,
    "kernel_size": 5,
    "dilations": [1, 2, 4, 8, 16],
    "hidden": 128,
    "dropout": 0.3,
    "max_length": 253,
}
V4_DEFAULTS = V3_DEFAULTS | {
    "stem_kernel": 5,
    "block": "glu",
    "expansion": 3,
    "structure_embeddings": True,
    "component_pools": True,
    "attention_heads": 1,
    "wide_buckets": 0,
    "wide_max_n": 5,
}
WIDE_DEFAULTS = {"wide_buckets": 2**20, "wide_max_n": 5}
# n-gram key = ((role * 64 + c0) * 64 + c1 ...) * 8 + n  (< 3.5e10 for n <= 5), then
# bucket = key * 48271 mod (2^31 - 1) mod buckets: exact in int64 for torch and NumPy alike.
HASH_MULTIPLIER, HASH_PRIME = 48271, 2_147_483_647


def ngram_buckets(chars: np.ndarray, roles: np.ndarray, max_n: int, buckets: int) -> np.ndarray:
    """Wide-path bucket ids of one encoded hostname (NumPy twin of ``wide_term``)."""
    chars, roles = chars.astype(np.int64), roles.astype(np.int64)
    valid = chars != PAD
    out = []
    for n in range(1, min(max_n, len(chars)) + 1):
        windows = np.lib.stride_tricks.sliding_window_view(chars, n)
        key = roles[: len(chars) - n + 1].copy()
        for j in range(n):
            key = key * 64 + windows[:, j]
        ok = np.lib.stride_tricks.sliding_window_view(valid, n).all(1)
        out.append(((key * 8 + n) * HASH_MULTIPLIER % HASH_PRIME % buckets)[ok])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def wide_term(table, chars, roles, max_n: int, buckets: int):
    """Wide-path logit for a batch: sum of n-gram weights / sqrt(number of n-grams)."""
    import torch

    mask = chars != PAD
    length = chars.shape[1]
    total = torch.zeros(chars.shape[0], dtype=table.weight.dtype, device=chars.device)
    count = torch.zeros(chars.shape[0], dtype=torch.int64, device=chars.device)
    for n in range(1, min(max_n, length) + 1):
        windows = chars.unfold(1, n, 1)
        key = roles[:, : length - n + 1].long()
        for j in range(n):
            key = key * 64 + windows[..., j]
        ok = mask.unfold(1, n, 1).all(-1)
        weights = table((key * 8 + n) * HASH_MULTIPLIER % HASH_PRIME % buckets).squeeze(-1)
        total = total + (weights * ok).sum(1)
        count = count + ok.sum(1)
    return total / count.clamp_min(1).to(total.dtype).sqrt()


def with_wide(torch, deep, config: dict):
    """Wrap a deep network with the zero-initialised wide path (no random numbers drawn)."""
    nn = torch.nn
    buckets, max_n = config["wide_buckets"], config["wide_max_n"]

    class WideAndDeep(nn.Module):
        padding_invariant = True

        def __init__(self):
            super().__init__()
            self.deep = deep
            self.wide = nn.Embedding(buckets, 1, _weight=torch.zeros(buckets, 1))

        def forward(self, x):
            return self.deep(x) + wide_term(self.wide, x[:, 0], x[:, 1], max_n, buckets)

    return WideAndDeep()


def runtime_config(arch: str, config: dict) -> dict:
    """The ``V4Numpy`` config that computes the same function as ``build(arch, config)``.

    v3 equals v4 with structure embeddings, component pools and extra heads switched off
    (same layers, same parameter names), so v3wide runs on the v4 research runtime.
    """
    if arch == "v4":
        return dict(config)
    if arch == "v3wide":
        return (
            V4_DEFAULTS
            | {k: config[k] for k in V3_DEFAULTS}
            | {
                "stem_kernel": config["kernel_size"],
                "block": "glu",
                "structure_embeddings": False,
                "component_pools": False,
                "attention_heads": 1,
                "wide_buckets": config["wide_buckets"],
                "wide_max_n": config["wide_max_n"],
            }
        )
    raise ValueError(f"{arch!r} uses the deployed format, not the research runtime")


def structure_ids(text: str, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-aligned (label index slot, position slot) per character, matching ``encode``."""
    labels = text.split(".")
    total = len(labels)
    label_slots: list[int] = []
    position_slots: list[int] = []
    for li, label in enumerate(labels):
        slot = min(total - 1 - li, 7) + 1
        for j in range(len(label)):
            label_slots.append(slot)
            position_slots.append(min(j, 15) + 1)
        if li < total - 1:
            label_slots.append(slot)
            position_slots.append(POSITION_SLOTS - 1)
    out = np.zeros((2, max_length), dtype=np.int64)
    kept_l, kept_p = label_slots[-max_length:], position_slots[-max_length:]
    if kept_l:
        out[0, max_length - len(kept_l) :] = kept_l
        out[1, max_length - len(kept_p) :] = kept_p
    return out[0], out[1]


def v4_input(text: str, max_length: int) -> np.ndarray:
    labels, positions = structure_ids(text, max_length)
    return np.stack([encode(text, max_length), encode_roles(text, max_length), labels, positions])


def features(arch: str, hosts: list[str], config: dict) -> np.ndarray:
    if arch in ("v3", "v3wide"):
        return model_features(hosts, "v3", config["max_length"])
    if arch == "v4":
        return np.stack([v4_input(h, config["max_length"]) for h in hosts])
    raise ValueError(f"Unknown architecture {arch!r}")


def defaults(arch: str) -> dict:
    if arch == "v3":
        return dict(V3_DEFAULTS)
    if arch == "v3wide":
        return V3_DEFAULTS | WIDE_DEFAULTS
    if arch == "v4":
        return dict(V4_DEFAULTS)
    raise ValueError(f"Unknown architecture {arch!r}")


def build(torch, arch: str, config: dict):
    if arch in ("v3", "v3wide"):
        model = build_model_v3(
            torch,
            config["embedding_dim"],
            config["channels"],
            config["kernel_size"],
            config["dilations"],
            config["hidden"],
            config["dropout"],
        )
        return with_wide(torch, model, config) if arch == "v3wide" else model
    if arch == "v4":
        model = build_v4(torch, config)
        return with_wide(torch, model, config) if config.get("wide_buckets") else model
    raise ValueError(f"Unknown architecture {arch!r}")


def build_v4(torch, c: dict):
    nn = torch.nn
    functional = torch.nn.functional
    C, E, k = c["channels"], c["embedding_dim"], c["kernel_size"]

    class GluBlock(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            self.depthwise = nn.Conv1d(
                C, C, k, groups=C, padding=dilation * (k - 1) // 2, dilation=dilation
            )
            self.norm = nn.LayerNorm(C, eps=NORM_EPS)
            self.gate = nn.Conv1d(C, 2 * C, 1)
            self.scale = nn.Parameter(torch.full((C,), 0.1))
            self.dropout = nn.Dropout(c["dropout"])

        def forward(self, h, mask):
            branch = self.depthwise(h).transpose(1, 2)
            branch = torch.relu(self.norm(branch)).transpose(1, 2) * mask
            branch = functional.glu(self.gate(branch), dim=1)
            return (h + self.dropout(branch) * self.scale[None, :, None]) * mask

    class NextBlock(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            self.depthwise = nn.Conv1d(
                C, C, k, groups=C, padding=dilation * (k - 1) // 2, dilation=dilation
            )
            self.norm = nn.LayerNorm(C, eps=NORM_EPS)
            self.up = nn.Linear(C, c["expansion"] * C)
            self.down = nn.Linear(c["expansion"] * C, C)
            self.scale = nn.Parameter(torch.full((C,), 0.1))
            self.dropout = nn.Dropout(c["dropout"])

        def forward(self, h, mask):
            branch = self.norm(self.depthwise(h).transpose(1, 2))
            branch = self.down(functional.gelu(self.up(branch), approximate="tanh"))
            branch = branch.transpose(1, 2) * mask
            return (h + self.dropout(branch) * self.scale[None, :, None]) * mask

    Block = {"glu": GluBlock, "convnext": NextBlock}[c["block"]]
    heads = c["attention_heads"]

    class CharCNNv4(nn.Module):
        padding_invariant = True

        def __init__(self):
            super().__init__()
            self.config = dict(c)
            self.embedding = nn.Embedding(VOCAB_SIZE, E, padding_idx=PAD)
            self.role_embedding = nn.Embedding(ROLE_COUNT, E, padding_idx=PAD)
            if c["structure_embeddings"]:
                self.label_embedding = nn.Embedding(LABEL_SLOTS, E, padding_idx=0)
                self.position_embedding = nn.Embedding(POSITION_SLOTS, E, padding_idx=0)
            self.stem = nn.Conv1d(E, C, c["stem_kernel"], padding=(c["stem_kernel"] - 1) // 2)
            self.norm = nn.LayerNorm(C, eps=NORM_EPS)
            self.blocks = nn.ModuleList(Block(d) for d in c["dilations"])
            self.attention = nn.Conv1d(C, heads, 1)
            pooled = C * (2 + heads) + (6 * C if c["component_pools"] else C)
            self.dropout = nn.Dropout(c["dropout"])
            self.dense = nn.Linear(pooled, c["hidden"])
            self.out = nn.Linear(c["hidden"], 1)

        def forward(self, x):
            chars, roles = x[:, 0], x[:, 1]
            mask = (chars != PAD).unsqueeze(1)
            e = self.embedding(chars) + self.role_embedding(roles)
            if c["structure_embeddings"]:
                e = e + self.label_embedding(x[:, 2]) + self.position_embedding(x[:, 3])
            h = self.stem(e.transpose(1, 2) * mask).transpose(1, 2)
            h = torch.relu(self.norm(h)).transpose(1, 2) * mask
            for block in self.blocks:
                h = block(h, mask)
            lowest = torch.finfo(h.dtype).min

            def masked_max(m):
                value = h.masked_fill(~m, lowest).amax(2)
                return torch.where(m.any(2), value, torch.zeros_like(value))

            def masked_mean(m):
                return (h * m).sum(2) / m.sum(2).clamp_min(1)

            scores = self.attention(h).masked_fill(~mask, lowest)
            weights = torch.softmax(scores, dim=2) * mask
            attended = torch.einsum("bhl,bcl->bhc", weights, h).reshape(h.shape[0], -1)
            parts = [masked_max(mask), masked_mean(mask), attended]
            if c["component_pools"]:
                for role in (ROLE_SUBDOMAIN, ROLE_NAME, ROLE_SUFFIX):
                    m = (roles == role).unsqueeze(1) & mask
                    parts += [masked_mean(m), masked_max(m)]
            else:
                parts.append(masked_mean((roles == ROLE_NAME).unsqueeze(1) & mask))
            hidden = torch.relu(self.dense(self.dropout(torch.cat(parts, dim=1))))
            return self.out(self.dropout(hidden)).squeeze(1)

    return CharCNNv4()


def export_tensors(model) -> dict:
    """All parameters as {name: [shape, base64 little-endian float32]} (research artifacts).

    A Wide & Deep wrapper's deep parameters lose their ``deep.`` prefix, so the runtime reads
    the same names for every v4-compatible network.
    """
    import base64

    out = {}
    for name, tensor in model.state_dict().items():
        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
        out[name.removeprefix("deep.")] = [
            list(array.shape),
            base64.b64encode(array.tobytes()).decode("ascii"),
        ]
    return out

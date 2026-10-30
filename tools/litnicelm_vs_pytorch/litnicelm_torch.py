"""PyTorch version of a LitniceLM model, and LitniceLM's data windows.

Shared by reference_torch.py (parity) and bench_torch.py (speed).

The model follows an export's config.json exactly: pre-LayerNorm blocks
(LN, attention, residual; LN, MLP, residual), fused QKV projection, ReLU MLP,
a final LayerNorm and an untied LM head without bias; positions either a
learned table (wpe) added to the token embeddings, or rotary embeddings on Q
and K ("half_split": element i paired with i + head_dim/2, the "rotate half"
form, position = index within the window). Attention is
F.scaled_dot_product_attention(is_causal=True).

Weights load from model.safetensors with the export's layout: c_attn, c_proj
and c_fc weights are [in, out] (used as x @ W), lm_head.weight is
[vocab, d_model].
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

DATASET_MAGIC = 0x4C4E4750  # "LNGP"
DATASET_HEADER_BYTES = 12   # magic, version, vocab size (u32 each), then u32 tokens


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def rope_tables(seq_len, head_dim, theta, device):
    """cos and sin of shape [seq_len, head_dim / 2], computed as LitniceLM does
    (inv_freq in float32: theta ** (-2 i / head_dim))."""
    half = head_dim // 2
    i = torch.arange(half, dtype=torch.float32, device=device)
    inv_freq = torch.pow(torch.tensor(theta, dtype=torch.float32, device=device),
                         -2.0 * i / head_dim)
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    angle = pos[:, None] * inv_freq[None, :]
    return torch.cos(angle), torch.sin(angle)


def apply_rope(x, cos, sin):
    """x: [batch, heads, seq, head_dim]; rotates (x[i], x[i + half])."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, f = cfg["n_embd"], cfg["n_inner"]
        eps = cfg["layer_norm_epsilon"]
        self.n_head = cfg["n_head"]
        self.ln_1 = nn.LayerNorm(d, eps=eps)
        self.ln_2 = nn.LayerNorm(d, eps=eps)
        # [in, out] as exported (GPT-2 Conv1D layout).
        self.c_attn_w = nn.Parameter(torch.empty(d, 3 * d))
        self.c_attn_b = nn.Parameter(torch.empty(3 * d))
        self.c_proj_w = nn.Parameter(torch.empty(d, d))
        self.c_proj_b = nn.Parameter(torch.empty(d))
        self.c_fc_w = nn.Parameter(torch.empty(d, f))
        self.c_fc_b = nn.Parameter(torch.empty(f))
        self.mlp_proj_w = nn.Parameter(torch.empty(f, d))
        self.mlp_proj_b = nn.Parameter(torch.empty(d))

    def forward(self, x, rope):
        b, s, d = x.shape
        h = self.n_head
        qkv = self.ln_1(x) @ self.c_attn_w + self.c_attn_b
        q, k, v = (t.view(b, s, h, d // h).transpose(1, 2) for t in qkv.split(d, dim=-1))
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + a.transpose(1, 2).reshape(b, s, d) @ self.c_proj_w + self.c_proj_b
        m = F.relu(self.ln_2(x) @ self.c_fc_w + self.c_fc_b)
        return x + m @ self.mlp_proj_w + self.mlp_proj_b


class LitniceLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, v = cfg["n_embd"], cfg["vocab_size"]
        self.position_encoding = cfg["position_encoding"]
        if cfg["activation_function"] != "relu":
            raise ValueError("only activation_function relu is supported")
        if self.position_encoding == "rope" and cfg.get("rope_layout") != "half_split":
            raise ValueError("only rope_layout half_split is supported")
        self.wte = nn.Parameter(torch.empty(v, d))
        self.wpe = (nn.Parameter(torch.empty(cfg["n_positions"], d))
                    if self.position_encoding == "learned" else None)
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg["n_layer"]))
        self.ln_f = nn.LayerNorm(d, eps=cfg["layer_norm_epsilon"])
        self.lm_head = nn.Parameter(torch.empty(v, d))  # [vocab, d_model]
        self._rope = {}

    def rope(self, seq_len, device):
        if self.position_encoding != "rope":
            return None
        key = (seq_len, str(device))
        if key not in self._rope:
            head_dim = self.cfg["n_embd"] // self.cfg["n_head"]
            self._rope[key] = rope_tables(seq_len, head_dim, self.cfg["rope_theta"], device)
        return self._rope[key]

    def forward(self, ids):
        """ids: [batch, seq] int64 -> logits [batch, seq, vocab]."""
        x = self.wte[ids]
        if self.wpe is not None:
            x = x + self.wpe[: ids.shape[1]]
        rope = self.rope(ids.shape[1], ids.device)
        for block in self.h:
            x = block(x, rope)
        return F.linear(self.ln_f(x), self.lm_head)

    def init_weights(self, seed=0):
        """Random weights for benchmarking (values do not affect speed)."""
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name.endswith(("ln_1.weight", "ln_2.weight", "ln_f.weight")):
                    p.fill_(1.0)
                elif name.endswith(("_b", ".bias")):
                    p.zero_()
                else:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.02)

    # Export tensor name -> parameter name.
    def export_names(self):
        names = {"transformer.wte.weight": "wte", "lm_head.weight": "lm_head",
                 "transformer.ln_f.weight": "ln_f.weight", "transformer.ln_f.bias": "ln_f.bias"}
        if self.wpe is not None:
            names["transformer.wpe.weight"] = "wpe"
        for i in range(len(self.h)):
            p, q = f"transformer.h.{i}.", f"h.{i}."
            names.update({
                p + "ln_1.weight": q + "ln_1.weight", p + "ln_1.bias": q + "ln_1.bias",
                p + "attn.c_attn.weight": q + "c_attn_w", p + "attn.c_attn.bias": q + "c_attn_b",
                p + "attn.c_proj.weight": q + "c_proj_w", p + "attn.c_proj.bias": q + "c_proj_b",
                p + "ln_2.weight": q + "ln_2.weight", p + "ln_2.bias": q + "ln_2.bias",
                p + "mlp.c_fc.weight": q + "c_fc_w", p + "mlp.c_fc.bias": q + "c_fc_b",
                p + "mlp.c_proj.weight": q + "mlp_proj_w", p + "mlp.c_proj.bias": q + "mlp_proj_b",
            })
        return names


def load_export(export_dir, device):
    """The model of a `litnicelm export` directory, weights loaded."""
    from safetensors.torch import load_file

    export_dir = Path(export_dir)
    cfg = json.loads((export_dir / "config.json").read_text())
    model = LitniceLM(cfg)
    tensors = load_file(str(export_dir / "model.safetensors"))
    names = model.export_names()
    missing = set(names) - set(tensors)
    extra = set(tensors) - set(names)
    if missing or extra:
        raise ValueError(f"export does not match config.json: missing {sorted(missing)}, "
                         f"unexpected {sorted(extra)}")
    params = dict(model.named_parameters())
    with torch.no_grad():
        for export_name, param_name in names.items():
            src = tensors[export_name]
            if tuple(src.shape) != tuple(params[param_name].shape):
                raise ValueError(f"{export_name}: shape {tuple(src.shape)}, "
                                 f"model expects {tuple(params[param_name].shape)}")
            params[param_name].copy_(src)
    return model.to(device), cfg


# ---------------------------------------------------------------------------
# Data: LitniceLM's tokenized dataset and its validation windows
# ---------------------------------------------------------------------------

def read_config(path):
    return yaml.safe_load(Path(path).read_text())


def read_tokens(path):
    """The u32 token ids of a LitniceLM .bin dataset."""
    raw = np.fromfile(path, dtype=np.uint32)
    if raw.size < 3 or raw[0] != DATASET_MAGIC:
        raise ValueError(f"{path}: not a LitniceLM dataset (magic LNGP)")
    return raw[DATASET_HEADER_BYTES // 4:].astype(np.int64)


def validation_windows(num_tokens, training):
    """Start offsets of the validation windows, as DatasetSplit and TextDataset
    build them: every round(1/fraction)-th chunk of validation_chunk_tokens
    (starting at chunk group-1) is held out, cut into back-to-back windows of
    train_seq_len + 1 tokens, rounded down to whole batches."""
    seq = training["train_seq_len"]
    batch = training["batch_size"]
    chunk = training["validation_chunk_tokens"]
    group = int(round(1.0 / training["validation_fraction"]))
    num_chunks = (num_tokens + chunk - 1) // chunk
    starts = []
    for i in range(group - 1, num_chunks, group):
        begin, end = i * chunk, min(i * chunk + chunk, num_tokens)
        start = begin
        while start + seq + 1 <= end:
            starts.append(start)
            start += seq
    return starts[: len(starts) // batch * batch]


def window_batch(tokens, starts, seq):
    """ids and targets [len(starts), seq] for the given window starts."""
    idx = np.asarray(starts)[:, None] + np.arange(seq + 1)[None, :]
    w = torch.from_numpy(tokens[idx])
    return w[:, :-1], w[:, 1:]


def mean_val_loss(model, cfg_yaml, device, autocast_dtype=None):
    """The validation loss as Trainer::evaluate computes it: the mean over
    batches of each batch's mean cross-entropy. Returns (loss, batches,
    tokens)."""
    training = cfg_yaml["training"]
    tokens = read_tokens(cfg_yaml["tokenization"]["output_binary"])
    starts = validation_windows(len(tokens), training)
    seq, batch = training["train_seq_len"], training["batch_size"]
    losses = []
    model.eval()
    with torch.no_grad():
        for b in range(0, len(starts), batch):
            ids, targets = window_batch(tokens, starts[b:b + batch], seq)
            ids, targets = ids.to(device), targets.to(device)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits = model(ids)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   targets.reshape(-1))
            losses.append(loss.item())
    return float(np.mean(losses)), len(losses), len(starts) * seq


def parameter_groups(model, weight_decay):
    """AdamW groups as LitniceLM applies weight decay: on embeddings and weight
    matrices, not on LayerNorm parameters or biases."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        is_matrix = p.dim() == 2
        (decay if is_matrix else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def learning_rate(completed_steps, total_steps, training):
    """LitniceLM's LearningRateSchedule::at: constant, or linear warmup over
    warmup_steps (1-based step) then cosine down to learning_rate * min_lr_ratio
    at total_steps."""
    peak = training["learning_rate"]
    if training.get("lr_schedule", "constant") != "cosine":
        return peak
    step = completed_steps + 1
    warmup = training.get("warmup_steps", 0)
    if step <= warmup:
        return peak * step / warmup
    if total_steps <= warmup:
        return peak
    floor = peak * training.get("min_lr_ratio", 0.0)
    progress = min(1.0, (step - warmup) / (total_steps - warmup))
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

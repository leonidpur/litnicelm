#!/usr/bin/env python3
"""Training-speed benchmark of the PyTorch reference model, set up like a
LitniceLM training run, for comparison with LitniceLM on the same config.

From the LitniceLM config: model shapes, batch_size, train_seq_len,
batches_per_step (gradient accumulation), AdamW betas and weight decay
(weight decay on embeddings and matrices only, as LitniceLM applies it),
grad_clip (global norm), learning rate and schedule; the vocabulary size
and training windows come from the tokenized dataset. Implementation:
F.scaled_dot_product_attention(is_causal=True), torch.optim.AdamW(fused=True,
eps=1e-8 as LitniceLM's AdamW uses), torch.nn.utils.clip_grad_norm_,
TF32 off (PyTorch's default for matmul, like LitniceLM's SGEMM).

--precision bf16 runs forward and loss under torch.autocast(bfloat16) with
FP32 master weights and optimizer state (the counterpart of bf16_mixed).
Each batch's loss is read on the host (loss.item()), as LitniceLM reads it;
--no-sync reads the losses only at the end.

Prints one JSON object: tokens/s, per-step milliseconds (median, min, max),
allocator peak memory.

Example:
  tools/litnicelm_vs_pytorch/bench_torch.py --config conf/run.yaml --precision bf16 --steps 214
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import litnicelm_torch as lt  # noqa: E402


def model_config(cfg_yaml, vocab_size):
    """config.json-style hyperparameters from a LitniceLM training config."""
    m = cfg_yaml["model"]
    pos = cfg_yaml.get("model_algo", {}).get("position_encoding", "learned")
    cfg = {"vocab_size": vocab_size, "n_positions": m["max_seq_len"], "n_embd": m["d_model"],
           "n_layer": m["n_layers"], "n_head": m["n_heads"], "n_inner": m["d_ff"],
           "layer_norm_epsilon": 1e-5, "activation_function": "relu",
           "position_encoding": pos}
    if pos == "rope":
        cfg.update({"rope_theta": 10000.0, "rope_layout": "half_split"})
    return cfg


def training_windows(num_tokens, training, seed):
    """Shuffled training-window starts (every window_stride tokens)."""
    seq, stride = training["train_seq_len"], training.get("window_stride", training["train_seq_len"])
    starts = np.arange(0, num_tokens - seq - 1, stride)
    np.random.default_rng(seed).shuffle(starts)
    return starts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="LitniceLM training config")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--steps", type=int, default=200, help="measured optimizer steps")
    parser.add_argument("--warmup", type=int, default=5, help="unmeasured optimizer steps first")
    parser.add_argument("--no-sync", action="store_true",
                        help="read the losses at the end instead of after each batch")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg_yaml = lt.read_config(args.config)
    training = cfg_yaml["training"]
    raw = np.fromfile(cfg_yaml["tokenization"]["output_binary"], dtype=np.uint32)
    vocab_size = int(raw[2])
    tokens = lt.read_tokens(cfg_yaml["tokenization"]["output_binary"])
    device = "cuda"

    model = lt.LitniceLM(model_config(cfg_yaml, vocab_size))
    model.init_weights(args.seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        lt.parameter_groups(model, training["weight_decay"]), lr=training["learning_rate"],
        betas=(training["beta1"], training["beta2"]), eps=1e-8, fused=True)
    autocast = torch.bfloat16 if args.precision == "bf16" else None

    batch, seq = training["batch_size"], training["train_seq_len"]
    accum = training.get("batches_per_step", 1)
    clip = training.get("grad_clip", 0.0)
    total_steps = args.warmup + args.steps
    starts = training_windows(len(tokens), training, args.seed)
    needed = total_steps * accum * batch
    starts = np.resize(starts, needed)

    losses_on_device = []
    step_events = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cursor = 0
    for step in range(total_steps):
        if step == args.warmup:
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(accum):
            ids, targets = lt.window_batch(tokens, starts[cursor:cursor + batch], seq)
            cursor += batch
            ids, targets = ids.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=autocast, enabled=autocast is not None):
                logits = model(ids)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                       targets.reshape(-1))
            (loss / accum).backward()
            if args.no_sync:
                losses_on_device.append(loss.detach())
            else:
                loss.item()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        lr = lt.learning_rate(step, total_steps, training)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        end_ev.record()
        if step >= args.warmup:
            step_events.append((start_ev, end_ev))
    torch.cuda.synchronize()
    wall = time.perf_counter() - wall_start
    if args.no_sync:
        torch.stack(losses_on_device).sum().item()

    step_ms = np.array([s.elapsed_time(e) for s, e in step_events])
    tokens_per_step = accum * batch * seq
    result = {
        "engine": "pytorch",
        "precision": args.precision,
        "sync": "end" if args.no_sync else "per batch",
        "steps": args.steps,
        "tokens_per_step": tokens_per_step,
        "tokens_per_s": tokens_per_step * args.steps / wall,
        "step_ms_median": float(np.median(step_ms)),
        "step_ms_min": float(step_ms.min()),
        "step_ms_max": float(step_ms.max()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()

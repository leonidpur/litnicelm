# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Fresh-process PyTorch training worker; reads one JSON request file."""

import json
import math
from pathlib import Path
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from ..corpus.corpus import CorpusManager, training_windows, window_batch
from ..run import SPEC
from .torch_model import LitniceLM

WORKLOAD = SPEC.workload


def _profile_summary(profiler: torch.profiler.profile) -> tuple[str, str]:
    """Classify SDPA strictly from observed CUDA kernel names."""
    totals: dict[str, float] = defaultdict(float)
    for event in profiler.events():
        if "cuda" not in str(event.device_type).lower():
            continue
        duration = float(
            getattr(event, "self_device_time_total", 0.0)
            or getattr(event, "self_cuda_time_total", 0.0)
        )
        if duration > 0:
            totals[event.name] += duration
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    names = " ".join(name.lower() for name, _ in ranked)
    if "flash" in names:
        backend = "flash"
    elif any(token in names for token in ("fmha", "cutlass", "efficient_attention")):
        backend = "memory-efficient (fmha/cutlass)"
    elif "cudnn" in names and any(token in names for token in ("attention", "sdpa")):
        backend = "cudnn"
    elif "softmax" in names and any(token in names for token in ("gemm", " matmul", "_mm")):
        backend = "math fallback"
    else:
        backend = "unknown from kernel names"
    total = sum(duration for _, duration in ranked)
    top = "; ".join(
        f"{name} ({100 * duration / total:.1f}%)"
        for name, duration in ranked[:10]
    )
    return backend, top or "no CUDA kernels recorded"


def _model_config(vocab_size: int) -> dict[str, object]:
    config: dict[str, object] = {
        "vocab_size": vocab_size,
        "n_positions": WORKLOAD.seq_len,
        "n_embd": WORKLOAD.d_model,
        "n_layer": WORKLOAD.n_layers,
        "n_head": WORKLOAD.n_heads,
        "n_inner": WORKLOAD.d_ff,
        "layer_norm_epsilon": 1e-5,
        "activation_function": "relu",
        "position_encoding": WORKLOAD.position_encoding,
    }
    if WORKLOAD.position_encoding == "rope":
        config.update({"rope_theta": 10000.0, "rope_layout": "half_split"})
    return config


def _parameter_groups(model: LitniceLM) -> list[dict[str, object]]:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.dim() == 2 else no_decay).append(parameter)
    return [
        {"params": decay, "weight_decay": WORKLOAD.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _learning_rate(completed_steps: int, total_steps: int) -> float:
    if WORKLOAD.lr_schedule != "cosine":
        return WORKLOAD.learning_rate
    step = completed_steps + 1
    if step <= WORKLOAD.warmup_steps:
        return WORKLOAD.learning_rate * step / WORKLOAD.warmup_steps
    if total_steps <= WORKLOAD.warmup_steps:
        return WORKLOAD.learning_rate
    floor = WORKLOAD.learning_rate * WORKLOAD.min_lr_ratio
    progress = min(
        1.0, (step - WORKLOAD.warmup_steps) / (total_steps - WORKLOAD.warmup_steps)
    )
    return floor + (WORKLOAD.learning_rate - floor) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def train(request: dict[str, object]) -> dict[str, object]:
    """Execute one old-style PyTorch timing row entirely in this process."""
    precision = str(request["precision"])
    loss_read = str(request["loss_read"])
    compile_mode = str(request["compile_mode"])
    profile_only = bool(request.get("profile_only", False))
    if compile_mode != "none" and loss_read != "end":
        raise ValueError("compiled PyTorch variants require end-only loss reads")
    measured_steps = int(request["steps"])
    benchmark_warmup = int(request["benchmark_warmup_steps"])
    corpus = CorpusManager().load(SPEC)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    eager_model = LitniceLM(_model_config(corpus.vocab_size))
    eager_model.init_weights(WORKLOAD.seed)
    eager_model.to(device).train()
    if compile_mode != "none" and WORKLOAD.position_encoding == "rope":
        # The fixed-shape benchmark uses one RoPE table. Build it eagerly so
        # the model cache never retains an output owned by a CUDA graph.
        eager_model.rope(WORKLOAD.seq_len, eager_model.wte.device)
    model = (
        torch.compile(eager_model, mode=compile_mode)
        if compile_mode != "none"
        else eager_model
    )
    def new_optimizer() -> torch.optim.AdamW:
        return torch.optim.AdamW(
            _parameter_groups(eager_model),
            lr=WORKLOAD.learning_rate,
            betas=(WORKLOAD.beta1, WORKLOAD.beta2),
            eps=WORKLOAD.epsilon,
            fused=True,
        )

    optimizer = new_optimizer()
    if compile_mode != "none":
        for parameter in eager_model.parameters():
            parameter.grad = torch.zeros_like(parameter)
    autocast_dtype = {"fp32": None, "bf16": torch.bfloat16}[precision]
    total_steps = benchmark_warmup + measured_steps
    starts = training_windows(len(corpus.tokens), SPEC)
    batches_per_epoch = len(starts)
    starts = np.resize(
        starts, total_steps * WORKLOAD.accumulation_steps * WORKLOAD.batch_size
    )

    losses_on_device = []
    losses_on_host = []
    step_events = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cursor = 0
    compile_warmup_seconds = 0.0
    active_profiler = None
    for step in range(total_steps):
        if step == benchmark_warmup:
            # Warm-up populates CUDA/kernel caches but must not advance the
            # matched training state used for loss equivalence.
            eager_model.init_weights(WORKLOAD.seed)
            optimizer = new_optimizer()
            cursor = 0
            losses_on_device.clear()
            losses_on_host.clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            if profile_only:
                active_profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                )
                active_profiler.__enter__()
            wall_start = time.perf_counter()
        if step == 0 and compile_mode != "none":
            torch.cuda.synchronize()
            compile_started = time.perf_counter()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(WORKLOAD.accumulation_steps):
            batch_starts = starts[cursor : cursor + WORKLOAD.batch_size].tolist()
            input_ids, targets = window_batch(corpus.tokens, batch_starts, SPEC)
            cursor += WORKLOAD.batch_size
            input_ids = torch.from_numpy(input_ids).to(device, non_blocking=True)
            targets = torch.from_numpy(targets).to(device, non_blocking=True)
            if compile_mode != "none":
                # Each accumulated microbatch is one complete invocation of
                # the compiled graph: its backward pass finishes before the
                # next invocation begins.
                torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(
                "cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None
            ):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
                )
            (loss / WORKLOAD.accumulation_steps).backward()
            if loss_read == "end":
                losses_on_device.append(loss.detach())
            else:
                losses_on_host.append(loss.item())
        if WORKLOAD.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), WORKLOAD.grad_clip)
        measured_step = max(0, step - benchmark_warmup)
        learning_rate = _learning_rate(measured_step, measured_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=compile_mode == "none")
        end_event.record()
        if step == 0 and compile_mode != "none":
            torch.cuda.synchronize()
            compile_warmup_seconds = time.perf_counter() - compile_started
        if step >= benchmark_warmup:
            step_events.append((start_event, end_event))
            if active_profiler is not None:
                active_profiler.step()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    profile_metrics: dict[str, str] = {}
    if active_profiler is not None:
        active_profiler.__exit__(None, None, None)
        backend, kernels = _profile_summary(active_profiler)
        profile_metrics = {
            "attention_backend": backend,
            "attention_cuda_kernels": kernels,
        }
    if losses_on_device:
        final_train_loss = torch.stack(losses_on_device[-batches_per_epoch:]).mean().item()
    else:
        final_train_loss = float(np.mean(losses_on_host[-batches_per_epoch:]))

    step_ms = np.array([start.elapsed_time(end) for start, end in step_events])
    return {
        "precision": precision,
        "sync": "end" if loss_read == "end" else "per batch",
        "steps": measured_steps,
        "tokens_per_step": WORKLOAD.tokens_per_step,
        "tokens_per_s": WORKLOAD.tokens_per_step * measured_steps / wall_seconds,
        "step_ms_mean": 1000 * wall_seconds / measured_steps,
        "step_ms_median": float(np.median(step_ms)),
        "step_ms_min": float(step_ms.min()),
        "step_ms_max": float(step_ms.max()),
        "engine_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(),
        "n_layers": WORKLOAD.n_layers,
        "n_heads": WORKLOAD.n_heads,
        "d_model": WORKLOAD.d_model,
        "d_ff": WORKLOAD.d_ff,
        "final_train_loss": final_train_loss,
        **(
            {"compile_mode": compile_mode, "compile_warmup_seconds": compile_warmup_seconds}
            if compile_mode != "none"
            else {}
        ),
        **profile_metrics,
    }


def main() -> None:
    """Read one persistent request file, train once, and emit one JSON result."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m vs_bench.engines.torch_worker <request.json>")
    request = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(train(request)))


if __name__ == "__main__":
    main()

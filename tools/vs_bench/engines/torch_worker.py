"""Fresh-process PyTorch training worker; reads one JSON request from stdin."""

import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..corpus.corpus import CorpusManager, training_windows, window_batch
from ..run import SPEC
from .torch_model import LitniceLM


def _model_config(vocab_size: int) -> dict[str, object]:
    config: dict[str, object] = {
        "vocab_size": vocab_size,
        "n_positions": SPEC.seq_len,
        "n_embd": SPEC.d_model,
        "n_layer": SPEC.n_layers,
        "n_head": SPEC.n_heads,
        "n_inner": SPEC.d_ff,
        "layer_norm_epsilon": 1e-5,
        "activation_function": "relu",
        "position_encoding": SPEC.position_encoding,
    }
    if SPEC.position_encoding == "rope":
        config.update({"rope_theta": 10000.0, "rope_layout": "half_split"})
    return config


def _parameter_groups(model: LitniceLM) -> list[dict[str, object]]:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.dim() == 2 else no_decay).append(parameter)
    return [
        {"params": decay, "weight_decay": SPEC.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _learning_rate(completed_steps: int, total_steps: int) -> float:
    if SPEC.lr_schedule != "cosine":
        return SPEC.learning_rate
    step = completed_steps + 1
    if step <= SPEC.warmup_steps:
        return SPEC.learning_rate * step / SPEC.warmup_steps
    if total_steps <= SPEC.warmup_steps:
        return SPEC.learning_rate
    floor = SPEC.learning_rate * SPEC.min_lr_ratio
    progress = min(
        1.0, (step - SPEC.warmup_steps) / (total_steps - SPEC.warmup_steps)
    )
    return floor + (SPEC.learning_rate - floor) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def train(request: dict[str, object]) -> dict[str, object]:
    """Execute one old-style PyTorch timing row entirely in this process."""
    precision = str(request["precision"])
    loss_read = str(request["loss_read"])
    measured_steps = int(request["steps"])
    benchmark_warmup = int(request["benchmark_warmup_steps"])
    corpus = CorpusManager().load(SPEC)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    model = LitniceLM(_model_config(corpus.vocab_size))
    model.init_weights(SPEC.seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        _parameter_groups(model),
        lr=SPEC.learning_rate,
        betas=(SPEC.beta1, SPEC.beta2),
        eps=SPEC.epsilon,
        fused=True,
    )
    autocast_dtype = {"fp32": None, "bf16": torch.bfloat16}[precision]
    total_steps = benchmark_warmup + measured_steps
    starts = training_windows(len(corpus.tokens), SPEC)
    starts = np.resize(
        starts, total_steps * SPEC.accumulation_steps * SPEC.batch_size
    )

    losses_on_device = []
    step_events = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cursor = 0
    for step in range(total_steps):
        if step == benchmark_warmup:
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(SPEC.accumulation_steps):
            batch_starts = starts[cursor : cursor + SPEC.batch_size].tolist()
            input_ids, targets = window_batch(corpus.tokens, batch_starts, SPEC)
            cursor += SPEC.batch_size
            input_ids = torch.from_numpy(input_ids).to(device, non_blocking=True)
            targets = torch.from_numpy(targets).to(device, non_blocking=True)
            with torch.autocast(
                "cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None
            ):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
                )
            (loss / SPEC.accumulation_steps).backward()
            if loss_read == "end":
                losses_on_device.append(loss.detach())
            else:
                loss.item()
        if SPEC.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), SPEC.grad_clip)
        learning_rate = _learning_rate(step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        end_event.record()
        if step >= benchmark_warmup:
            step_events.append((start_event, end_event))
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    if losses_on_device:
        torch.stack(losses_on_device).sum().item()

    step_ms = np.array([start.elapsed_time(end) for start, end in step_events])
    return {
        "precision": precision,
        "sync": "end" if loss_read == "end" else "per batch",
        "steps": measured_steps,
        "tokens_per_step": SPEC.tokens_per_step,
        "tokens_per_s": SPEC.tokens_per_step * measured_steps / wall_seconds,
        "step_ms_mean": 1000 * wall_seconds / measured_steps,
        "step_ms_median": float(np.median(step_ms)),
        "step_ms_min": float(step_ms.min()),
        "step_ms_max": float(step_ms.max()),
        "engine_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(),
    }


def main() -> None:
    """Read one request, train once, and emit one JSON result."""
    request = json.load(sys.stdin)
    print(json.dumps(train(request)))


if __name__ == "__main__":
    main()

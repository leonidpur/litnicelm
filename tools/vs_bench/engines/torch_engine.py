# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""PyTorch adapter for exported-model evaluation."""

import json
from pathlib import Path
from types import MappingProxyType
import sys

import numpy as np
import torch
import torch.nn.functional as F

from ..corpus.corpus import CorpusManager, validation_windows, window_batch
from ..measure.process import run_sampled_process, write_process_request
from ..spec import ComparisonSpec
from .base import Engine, EngineResult, EngineVariant
from .torch_model import load_export


REPOSITORY = Path(__file__).resolve().parents[3]
ALL_COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)
NATIVE_SETTINGS = MappingProxyType(
    {
        "device": "cuda",
        "attention": "torch.nn.functional.scaled_dot_product_attention",
        "optimizer": "torch.optim.AdamW(fused=True)",
        "mixed_precision": "torch.autocast(bfloat16)",
        "master_weights": "fp32",
        "matmul_tf32": False,
        "benchmark_warmup_steps": 5,
        "pytorch.profile_steps": 3,
        "pytorch.compile_modes": ALL_COMPILE_MODES,
        "worker": "python -m vs_bench.engines.torch_worker",
    }
)


class TorchEngine(Engine):
    """Translate the comparison spec into a PyTorch model and optimizer setup."""

    @staticmethod
    def _native_precision(precision: str) -> str:
        """Translate a shared precision name to the PyTorch execution mode."""
        return {"fp32": "plain FP32", "bf16": "torch.autocast(bfloat16)"}[precision]

    @property
    def name(self) -> str:
        return "PyTorch"

    def describe_setup(self, spec: ComparisonSpec) -> tuple[str, ...]:
        precisions = (self._native_precision(value) for value in spec.workload.precisions)
        settings = tuple(f"native setting {key}: {value}" for key, value in NATIVE_SETTINGS.items())
        return (
            "construct an equivalent PyTorch model and optimizer from the shared spec",
            "use the same corpus windows and optimizer-step count",
            f"translate shared precisions: {', '.join(precisions)}",
        ) + settings

    def supports(self, operation: str) -> bool:
        return operation in {"parity", "train_speed"}

    def variants(
        self, spec: ComparisonSpec, vertical: str, precision: str
    ) -> tuple[EngineVariant, ...]:
        del spec
        if vertical == "parity":
            return (
                EngineVariant(
                    self.name,
                    precision,
                    "eager reference",
                    {"loss_read": "end", "compile_mode": "none"},
                ),
            )
        if vertical == "train_speed":
            compile_modes = tuple(NATIVE_SETTINGS["pytorch.compile_modes"])
            unknown = set(compile_modes) - set(ALL_COMPILE_MODES)
            if unknown:
                raise ValueError(f"unknown pytorch.compile_modes: {sorted(unknown)}")
            eager = (
                EngineVariant(
                    self.name,
                    precision,
                    "eager / per-batch loss",
                    {"loss_read": "per_batch", "compile_mode": "none"},
                ),
                EngineVariant(
                    self.name,
                    precision,
                    "eager / end loss",
                    {"loss_read": "end", "compile_mode": "none"},
                ),
            )
            compiled = tuple(
                EngineVariant(
                    self.name,
                    precision,
                    f"compiled / {mode}",
                    {"loss_read": "end", "compile_mode": mode},
                )
                for mode in compile_modes
            )
            return eager + compiled
        return ()

    def run_speed(
        self, spec: ComparisonSpec, variant: EngineVariant, steps: int
    ) -> EngineResult:
        loss_read = variant.options["loss_read"]
        compile_mode = variant.options["compile_mode"]
        request = json.dumps(
            {
                "precision": variant.precision,
                "loss_read": loss_read,
                "steps": steps,
                "benchmark_warmup_steps": NATIVE_SETTINGS["benchmark_warmup_steps"],
                "compile_mode": compile_mode,
            }
        )
        request_path = write_process_request("pytorch", request)
        command = [
            sys.executable,
            "-m",
            "vs_bench.engines.torch_worker",
            str(request_path),
        ]
        measured = run_sampled_process(command, REPOSITORY / "tools")
        measured.gpu.verify_clock(
            spec.environment.clock_mhz,
            f"PyTorch precision={variant.precision} "
            f"variant={variant.label} options={dict(variant.options)}",
        )
        metrics = json.loads(measured.output.strip().splitlines()[-1])
        metrics.update(
            {
                "process_wall_seconds": measured.wall_seconds,
                "process_peak_mib": measured.gpu.process_peak_mib,
                "sm_clock_mhz": measured.gpu.mean_clock_mhz,
                "sm_clock_min_mhz": measured.gpu.min_clock_mhz,
                "sm_clock_max_mhz": measured.gpu.max_clock_mhz,
                "sm_clock_sample_count": len(measured.gpu.clock_samples_mhz),
                "sm_clock_outlier_count": measured.gpu.clock_outlier_count(
                    spec.environment.clock_mhz
                ),
                "sm_clock_stable": True,
            }
        )
        return EngineResult(engine=self.name, operation="train_speed", metrics=metrics)

    def profile_variant(
        self, spec: ComparisonSpec, vertical: str, variant: EngineVariant
    ) -> dict[str, str]:
        """Profile this PyTorch variant in a separate, untimed worker process."""
        if vertical != "train_speed":
            return {}
        request = json.dumps(
            {
                "precision": variant.precision,
                "loss_read": variant.options["loss_read"],
                "steps": int(NATIVE_SETTINGS["pytorch.profile_steps"]),
                "benchmark_warmup_steps": NATIVE_SETTINGS["benchmark_warmup_steps"],
                "compile_mode": variant.options["compile_mode"],
                "profile_only": True,
            }
        )
        request_path = write_process_request("pytorch_profile", request)
        command = [
            sys.executable,
            "-m",
            "vs_bench.engines.torch_worker",
            str(request_path),
        ]
        measured = run_sampled_process(command, REPOSITORY / "tools")
        metrics = json.loads(measured.output.strip().splitlines()[-1])
        return {
            "attention_backend": str(metrics["attention_backend"]),
            "attention_backend_source": "torch.profiler CUDA kernel names",
            "attention_cuda_kernels": str(metrics["attention_cuda_kernels"]),
        }

    def run(
        self, spec: ComparisonSpec, operation: str, variant: EngineVariant | None = None
    ) -> EngineResult:
        if operation != "parity":
            raise NotImplementedError(f"PyTorch operation is not implemented: {operation}")
        if variant is None:
            raise ValueError("PyTorch parity requires an engine variant")
        return self._evaluate_export(spec, variant)

    @staticmethod
    def _autocast_dtype(precision: str) -> torch.dtype | None:
        """Translate the shared precision to PyTorch's autocast dtype."""
        return {"fp32": None, "bf16": torch.bfloat16}[precision]

    def _evaluate_export(
        self, spec: ComparisonSpec, variant: EngineVariant
    ) -> EngineResult:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, config = load_export(spec.parity.export_dir, device)
        corpus = CorpusManager().load(spec)
        starts = validation_windows(len(corpus.tokens), spec)
        autocast_dtype = self._autocast_dtype(variant.precision)
        losses = []
        model.eval()
        with torch.no_grad():
            for offset in range(0, len(starts), spec.workload.batch_size):
                input_ids, targets = window_batch(
                    corpus.tokens,
                    starts[offset : offset + spec.workload.batch_size],
                    spec,
                )
                input_ids = torch.from_numpy(input_ids).to(device)
                targets = torch.from_numpy(targets).to(device)
                with torch.autocast(
                    "cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None
                ):
                    logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
                )
                losses.append(loss.item())
        return EngineResult(
            engine=self.name,
            operation="parity",
            metrics={
                "precision": variant.precision,
                "loss": float(np.mean(losses)),
                "batches": len(losses),
                "tokens": len(starts) * spec.workload.seq_len,
                "n_layers": int(config["n_layer"]),
                "d_model": int(config["n_embd"]),
                "position_encoding": str(config["position_encoding"]),
                "device": str(device),
            },
        )


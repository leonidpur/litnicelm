"""PyTorch adapter for exported-model evaluation."""

import json
from pathlib import Path
from types import MappingProxyType
import sys

import numpy as np
import torch
import torch.nn.functional as F

from ..corpus.corpus import CorpusManager, validation_windows, window_batch
from ..measure.process import run_sampled_process
from ..spec import ComparisonSpec
from .base import Engine, EngineResult, SpeedVariant
from .torch_model import load_export


REPOSITORY = Path(__file__).resolve().parents[3]
NATIVE_SETTINGS = MappingProxyType(
    {
        "device": "cuda",
        "attention": "torch.nn.functional.scaled_dot_product_attention",
        "optimizer": "torch.optim.AdamW(fused=True)",
        "mixed_precision": "torch.autocast(bfloat16)",
        "master_weights": "fp32",
        "matmul_tf32": False,
        "benchmark_warmup_steps": 5,
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

    def speed_variants(self, spec: ComparisonSpec) -> tuple[SpeedVariant, ...]:
        return tuple(
            SpeedVariant(precision, loss_read)
            for precision in spec.workload.precisions
            for loss_read in ("per_batch", "end")
        )

    def run_speed(
        self, spec: ComparisonSpec, variant: SpeedVariant, steps: int
    ) -> EngineResult:
        request = json.dumps(
            {
                "precision": variant.precision,
                "loss_read": variant.loss_read,
                "steps": steps,
                "benchmark_warmup_steps": NATIVE_SETTINGS["benchmark_warmup_steps"],
            }
        )
        command = [sys.executable, "-m", "vs_bench.engines.torch_worker"]
        measured = run_sampled_process(command, REPOSITORY / "tools", request)
        metrics = json.loads(measured.output.strip().splitlines()[-1])
        metrics.update(
            {
                "process_wall_seconds": measured.wall_seconds,
                "process_peak_mib": measured.gpu.process_peak_mib,
                "sm_clock_mhz": measured.gpu.mean_clock_mhz,
            }
        )
        return EngineResult(engine=self.name, operation="train_speed", metrics=metrics)

    def run(self, spec: ComparisonSpec, operation: str) -> EngineResult:
        if operation != "parity":
            raise NotImplementedError(f"PyTorch operation is not implemented: {operation}")
        return self._evaluate_export(spec)

    @staticmethod
    def _autocast_dtype(precision: str) -> torch.dtype | None:
        """Translate the shared precision to PyTorch's autocast dtype."""
        return {"fp32": None, "bf16": torch.bfloat16}[precision]

    def _evaluate_export(self, spec: ComparisonSpec) -> EngineResult:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, config = load_export(spec.parity.export_dir, device)
        corpus = CorpusManager().load(spec)
        starts = validation_windows(len(corpus.tokens), spec)
        autocast_dtype = self._autocast_dtype(spec.parity.precision)
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
                "precision": spec.parity.precision,
                "loss": float(np.mean(losses)),
                "batches": len(losses),
                "tokens": len(starts) * spec.workload.seq_len,
                "n_layers": int(config["n_layer"]),
                "d_model": int(config["n_embd"]),
                "position_encoding": str(config["position_encoding"]),
                "device": str(device),
            },
        )


"""PyTorch adapter for exported-model evaluation."""

import numpy as np
import torch
import torch.nn.functional as F

from ..corpus.corpus import CorpusManager, validation_windows, window_batch
from ..spec import ComparisonSpec
from .base import Engine, EngineResult
from .torch_model import load_export


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
        precisions = (self._native_precision(value) for value in spec.precisions)
        return (
            "construct an equivalent PyTorch model and optimizer from the shared spec",
            "use the same corpus windows and optimizer-step count",
            f"translate shared precisions: {', '.join(precisions)}",
        )

    def supports(self, operation: str) -> bool:
        return operation == "parity"

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
        model, config = load_export(spec.export_dir, device)
        corpus = CorpusManager().load(spec)
        starts = validation_windows(len(corpus.tokens), spec)
        autocast_dtype = self._autocast_dtype(spec.parity_precision)
        losses = []
        model.eval()
        with torch.no_grad():
            for offset in range(0, len(starts), spec.batch_size):
                input_ids, targets = window_batch(
                    corpus.tokens, starts[offset : offset + spec.batch_size], spec
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
                "precision": spec.parity_precision,
                "loss": float(np.mean(losses)),
                "batches": len(losses),
                "tokens": len(starts) * spec.seq_len,
                "n_layers": int(config["n_layer"]),
                "d_model": int(config["n_embd"]),
                "position_encoding": str(config["position_encoding"]),
                "device": str(device),
            },
        )


"""PyTorch adapter skeleton with no PyTorch dependency yet."""

from ..spec import ComparisonSpec
from .base import Engine, EngineResult


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

    def run(self, spec: ComparisonSpec, operation: str) -> EngineResult:
        raise NotImplementedError("PyTorch execution is not part of skeleton step 1")


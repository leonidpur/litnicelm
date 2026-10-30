"""PyTorch adapter skeleton with no PyTorch dependency yet."""

from engines.base import Engine, EngineResult
from spec import ComparisonSpec


class TorchEngine(Engine):
    """Translate the comparison spec into a PyTorch model and optimizer setup."""

    @property
    def name(self) -> str:
        return "PyTorch"

    def describe_setup(self, spec: ComparisonSpec) -> tuple[str, ...]:
        return (
            "construct an equivalent PyTorch model and optimizer from the shared spec",
            "use the same corpus windows and optimizer-step count",
            f"translate shared precisions: {', '.join(spec.precisions)}",
        )

    def run(self, spec: ComparisonSpec, operation: str) -> EngineResult:
        raise NotImplementedError("PyTorch execution is not part of skeleton step 1")


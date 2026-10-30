"""LitniceLM adapter skeleton."""

from engines.base import Engine, EngineResult
from spec import ComparisonSpec


class LitniceLMEngine(Engine):
    """Translate the comparison spec into YAML and drive the LitniceLM binary."""

    @staticmethod
    def _native_precision(precision: str) -> str:
        """Translate a shared precision name to the LitniceLM config value."""
        return {"fp32": "fp32", "bf16": "bf16_mixed"}[precision]

    @property
    def name(self) -> str:
        return "LitniceLM"

    def describe_setup(self, spec: ComparisonSpec) -> tuple[str, ...]:
        precisions = (self._native_precision(value) for value in spec.precisions)
        return (
            "generate a temporary LitniceLM YAML from the shared spec",
            "drive the LitniceLM binary as a subprocess",
            f"request precisions: {', '.join(precisions)}",
        )

    def run(self, spec: ComparisonSpec, operation: str) -> EngineResult:
        raise NotImplementedError("LitniceLM execution is not part of skeleton step 1")


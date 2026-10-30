"""LitniceLM adapter skeleton."""

from engines.base import Engine, EngineResult
from spec import ComparisonSpec


class LitniceLMEngine(Engine):
    """Translate the comparison spec into YAML and drive the LitniceLM binary."""

    @property
    def name(self) -> str:
        return "LitniceLM"

    def describe_setup(self, spec: ComparisonSpec) -> tuple[str, ...]:
        return (
            "generate a temporary LitniceLM YAML from the shared spec",
            "drive the LitniceLM binary as a subprocess",
            f"request precisions: {', '.join(spec.precisions)}",
        )

    def run(self, spec: ComparisonSpec, operation: str) -> EngineResult:
        raise NotImplementedError("LitniceLM execution is not part of skeleton step 1")


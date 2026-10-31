"""Interface implemented by engines participating in comparisons."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..spec import ComparisonSpec


@dataclass(frozen=True)
class EngineResult:
    """Structured result returned by an engine for one requested operation."""

    engine: str
    operation: str
    metrics: dict[str, bool | int | float | str]


@dataclass(frozen=True)
class EngineVariant:
    """One engine expansion inside a vertical's neutral precision group."""

    precision: str
    loss_read: str = "end"
    compile_mode: str = "none"


class Engine(ABC):
    """Translate an engine-neutral spec into one engine's native execution."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable engine name used in plans and results."""

    @abstractmethod
    def describe_setup(self, spec: ComparisonSpec) -> tuple[str, ...]:
        """Describe how this engine will realize the shared specification."""

    def supports(self, operation: str) -> bool:
        """Return whether this adapter implements the named vertical operation."""
        return False

    @property
    def determines_step_count(self) -> bool:
        """Whether this engine's native run determines peer measured steps."""
        return False

    def variants(
        self, spec: ComparisonSpec, vertical: str, precision: str
    ) -> tuple[EngineVariant, ...]:
        """Expand only this engine's axes inside one neutral precision group."""
        return ()

    def run_speed(
        self, spec: ComparisonSpec, variant: EngineVariant, steps: int
    ) -> EngineResult:
        """Run one isolated training-speed variant."""
        raise NotImplementedError(f"{self.name} does not implement training speed")

    def run_differential(
        self,
        spec: ComparisonSpec,
        variant: EngineVariant,
        step_counts: tuple[int, int] | None,
    ) -> tuple[EngineResult, EngineResult]:
        """Run the two whole-process endpoints used for differential timing."""
        if step_counts is None:
            raise ValueError(f"{self.name} requires explicit differential step counts")
        return tuple(
            self.run_speed(spec, variant, steps) for steps in step_counts
        )  # type: ignore[return-value]

    @abstractmethod
    def run(
        self, spec: ComparisonSpec, operation: str, variant: EngineVariant | None = None
    ) -> EngineResult:
        """Execute one vertical operation and return structured measurements."""


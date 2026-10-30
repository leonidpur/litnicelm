"""Interface implemented by engines participating in comparisons."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..spec import ComparisonSpec


@dataclass(frozen=True)
class EngineResult:
    """Structured result returned by an engine for one requested operation."""

    engine: str
    operation: str
    metrics: dict[str, int | float | str]


class Engine(ABC):
    """Translate an engine-neutral spec into one engine's native execution."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable engine name used in plans and results."""

    @abstractmethod
    def describe_setup(self, spec: ComparisonSpec) -> tuple[str, ...]:
        """Describe how this engine will realize the shared specification."""

    @abstractmethod
    def run(self, spec: ComparisonSpec, operation: str) -> EngineResult:
        """Execute one vertical operation and return structured measurements."""


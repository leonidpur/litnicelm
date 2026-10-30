"""Interface for one independently reportable kind of comparison."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from engines.base import Engine, EngineResult
from spec import ComparisonSpec


@dataclass(frozen=True)
class EnginePlan:
    """Planned actions for one engine within one vertical."""

    engine: str
    actions: tuple[str, ...]


@dataclass(frozen=True)
class VerticalPlan:
    """Structured plan returned by a vertical in plan mode."""

    vertical: str
    purpose: str
    engines: tuple[EnginePlan, ...]


@dataclass(frozen=True)
class VerticalResult:
    """Structured results returned after a vertical executes all engines."""

    vertical: str
    engines: tuple[EngineResult, ...]


class Vertical(ABC):
    """Compare engines along one dimension under a shared specification."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable vertical name used in plans and results."""

    @abstractmethod
    def plan(self, spec: ComparisonSpec, engines: Sequence[Engine]) -> VerticalPlan:
        """Return the work this vertical would perform without executing it."""

    @abstractmethod
    def run(self, spec: ComparisonSpec, engines: Sequence[Engine]) -> VerticalResult:
        """Execute this comparison for every engine and return structured results."""


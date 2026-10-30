"""Numerical-parity vertical skeleton."""

from typing import Sequence

from ..engines.base import Engine
from ..spec import ComparisonSpec
from .base import EnginePlan, Vertical, VerticalPlan, VerticalResult


class ParityVertical(Vertical):
    """Compare engine outputs and losses on identical weights and data."""

    @property
    def name(self) -> str:
        return "parity"

    def plan(self, spec: ComparisonSpec, engines: Sequence[Engine]) -> VerticalPlan:
        plans = tuple(
            EnginePlan(
                engine=engine.name,
                actions=engine.describe_setup(spec)
                + (
                    "load identical exported weights",
                    "evaluate identical corpus windows and report loss differences",
                ),
            )
            for engine in engines
        )
        return VerticalPlan(
            vertical=self.name,
            purpose="verify numerical equivalence under explicitly matched conditions",
            engines=plans,
        )

    def run(self, spec: ComparisonSpec, engines: Sequence[Engine]) -> VerticalResult:
        raise NotImplementedError("parity execution is not part of skeleton step 1")


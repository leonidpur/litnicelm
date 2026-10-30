"""Training-speed vertical skeleton."""

from typing import Sequence

from ..engines.base import Engine
from ..spec import ComparisonSpec
from .base import EnginePlan, Vertical, VerticalPlan, VerticalResult


class TrainSpeedVertical(Vertical):
    """Compare optimizer-step throughput and memory under matched workloads."""

    @property
    def name(self) -> str:
        return "train_speed"

    def plan(self, spec: ComparisonSpec, engines: Sequence[Engine]) -> VerticalPlan:
        plans = tuple(
            EnginePlan(
                engine=engine.name,
                actions=engine.describe_setup(spec)
                + (
                    f"run {spec.steps} optimizer steps, repeated {spec.repeats} times",
                    "measure external differential time, engine time, and GPU memory",
                ),
            )
            for engine in engines
        )
        return VerticalPlan(
            vertical=self.name,
            purpose="compare training speed and memory with an engine-independent timer",
            engines=plans,
        )

    def run(self, spec: ComparisonSpec, engines: Sequence[Engine]) -> VerticalResult:
        raise NotImplementedError("training execution is not part of skeleton step 1")


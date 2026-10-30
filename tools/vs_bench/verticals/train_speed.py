"""Training-speed vertical skeleton."""

from typing import Sequence

from ..engines.base import Engine, EngineResult
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
                    f"run each supported speed variant, repeated {spec.repeats} times",
                    "launch every row in a fresh sampled subprocess",
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
        sources = tuple(engine for engine in engines if engine.determines_step_count)
        peers = tuple(engine for engine in engines if not engine.determines_step_count)
        if not sources:
            raise ValueError("training speed requires an engine that determines measured steps")
        results = []
        for repeat in range(1, spec.repeats + 1):
            measured_steps = spec.steps
            for engine in sources + peers:
                for variant in engine.speed_variants(spec):
                    result = engine.run_speed(spec, variant, measured_steps)
                    metrics = {**result.metrics, "repeat": repeat}
                    results.append(
                        EngineResult(
                            engine=result.engine,
                            operation=result.operation,
                            metrics=metrics,
                        )
                    )
                    if engine.determines_step_count:
                        measured_steps = int(metrics["steps"])
        return VerticalResult(vertical=self.name, engines=tuple(results))


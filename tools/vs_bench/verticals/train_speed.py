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
                    f"run each supported speed variant, repeated {spec.train_speed.repeats} times",
                    "launch every row in a fresh sampled subprocess",
                    "derive whole-process timing from the difference between two run lengths",
                    "report realized model dimensions, tokens per step, and matched-step loss",
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
        for repeat in range(1, spec.train_speed.repeats + 1):
            measured_steps = spec.train_speed.steps
            differential_steps: dict[str, tuple[int, int]] = {}
            for engine in sources:
                for variant in engine.speed_variants(spec):
                    result = engine.run_speed(spec, variant, measured_steps)
                    endpoints = engine.run_differential(spec, variant, None)
                    steps = tuple(int(endpoint.metrics["steps"]) for endpoint in endpoints)
                    previous = differential_steps.setdefault(variant.precision, steps)
                    if previous != steps:
                        raise ValueError(
                            f"differential step counts disagree for {variant.precision}: "
                            f"{previous} vs {steps}"
                        )
                    metrics = {
                        **result.metrics,
                        **self._differential_metrics(endpoints),
                        "repeat": repeat,
                    }
                    results.append(
                        EngineResult(
                            engine=result.engine,
                            operation=result.operation,
                            metrics=metrics,
                        )
                    )
                    measured_steps = int(metrics["steps"])
            for engine in peers:
                for variant in engine.speed_variants(spec):
                    result = engine.run_speed(spec, variant, measured_steps)
                    endpoints = engine.run_differential(
                        spec, variant, differential_steps[variant.precision]
                    )
                    metrics = {
                        **result.metrics,
                        **self._differential_metrics(endpoints),
                        "repeat": repeat,
                    }
                    results.append(
                        EngineResult(
                            engine=result.engine,
                            operation=result.operation,
                            metrics=metrics,
                        )
                    )
        return VerticalResult(
            vertical=self.name,
            engines=self._check_equivalence(spec, sources, tuple(results)),
        )

    @staticmethod
    def _differential_metrics(
        endpoints: tuple[EngineResult, EngineResult],
    ) -> dict[str, int | float]:
        low, high = endpoints
        low_steps = int(low.metrics["steps"])
        high_steps = int(high.metrics["steps"])
        step_delta = high_steps - low_steps
        wall_delta = float(high.metrics["process_wall_seconds"]) - float(
            low.metrics["process_wall_seconds"]
        )
        if step_delta <= 0 or wall_delta <= 0:
            raise ValueError(
                f"invalid differential endpoints: steps {low_steps}/{high_steps}, "
                f"wall {low.metrics['process_wall_seconds']}/{high.metrics['process_wall_seconds']}"
            )
        tokens_per_step = int(low.metrics["tokens_per_step"])
        return {
            "differential_low_steps": low_steps,
            "differential_high_steps": high_steps,
            "differential_low_wall_seconds": float(low.metrics["process_wall_seconds"]),
            "differential_high_wall_seconds": float(high.metrics["process_wall_seconds"]),
            "differential_step_ms": 1000 * wall_delta / step_delta,
            "differential_tokens_per_s": tokens_per_step * step_delta / wall_delta,
        }

    @staticmethod
    def _check_equivalence(
        spec: ComparisonSpec,
        sources: Sequence[Engine],
        results: tuple[EngineResult, ...],
    ) -> tuple[EngineResult, ...]:
        expected_dims = {
            "n_layers": spec.workload.n_layers,
            "n_heads": spec.workload.n_heads,
            "d_model": spec.workload.d_model,
            "d_ff": spec.workload.d_ff,
            "tokens_per_step": spec.workload.tokens_per_step,
        }
        source_names = {engine.name for engine in sources}
        references: dict[tuple[int, str], float] = {}
        for row in results:
            for key, expected in expected_dims.items():
                if int(row.metrics.get(key, -1)) != expected:
                    raise ValueError(
                        f"{row.engine} reports {key}={row.metrics.get(key)}, "
                        f"expected {expected}"
                    )
            if "final_train_loss" not in row.metrics:
                raise ValueError(f"{row.engine} did not report final training loss")
            if row.engine in source_names:
                references[(int(row.metrics["repeat"]), str(row.metrics["precision"]))] = (
                    float(row.metrics["final_train_loss"])
                )

        checked = []
        for row in results:
            key = (int(row.metrics["repeat"]), str(row.metrics["precision"]))
            if key not in references:
                raise ValueError(f"no LitniceLM loss reference for repeat/precision {key}")
            difference = abs(
                float(row.metrics["final_train_loss"]) - references[key]
            )
            passed = difference <= spec.train_speed.equivalence_loss_tolerance
            if not passed:
                raise ValueError(
                    f"{row.engine} final loss differs from LitniceLM by {difference:g}, "
                    f"above tolerance {spec.train_speed.equivalence_loss_tolerance:g}"
                )
            checked.append(
                EngineResult(
                    engine=row.engine,
                    operation=row.operation,
                    metrics={
                        **row.metrics,
                        "loss_reference": references[key],
                        "loss_difference": difference,
                        "loss_tolerance": spec.train_speed.equivalence_loss_tolerance,
                        "equivalence_passed": passed,
                    },
                )
            )
        return tuple(checked)


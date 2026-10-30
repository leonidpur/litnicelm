"""Numerical-parity vertical skeleton."""

import re
from typing import Sequence

from ..engines.base import Engine, EngineResult
from ..spec import ComparisonSpec
from .base import EnginePlan, Vertical, VerticalPlan, VerticalResult


class ParityVertical(Vertical):
    """Compare engine outputs and losses on identical weights and data."""

    @property
    def name(self) -> str:
        return "parity"

    @staticmethod
    def _card_validation_loss(spec: ComparisonSpec) -> float | None:
        """Read LitniceLM's best validation loss from the export model card."""
        card = (spec.parity.export_dir / "README.md").read_text()
        match = re.search(r"\| best val loss \| ([0-9.eE+-]+)", card)
        return float(match.group(1)) if match else None

    def plan(self, spec: ComparisonSpec, engines: Sequence[Engine]) -> VerticalPlan:
        supported = tuple(engine for engine in engines if engine.supports(self.name))
        plans = tuple(
            EnginePlan(
                engine=engine.name,
                actions=engine.describe_setup(spec)
                + (
                    f"load existing export: {spec.parity.export_dir}",
                    f"evaluate identical corpus windows as {spec.parity.precision}",
                    f"compare validation loss with tolerance {spec.parity.tolerance:g}",
                ),
            )
            for engine in supported
        )
        return VerticalPlan(
            vertical=self.name,
            purpose="verify numerical equivalence under explicitly matched conditions",
            engines=plans,
        )

    def run(self, spec: ComparisonSpec, engines: Sequence[Engine]) -> VerticalResult:
        reference = self._card_validation_loss(spec)
        if reference is None:
            raise ValueError(
                f"no best validation loss in {spec.parity.export_dir / 'README.md'}"
            )
        results = []
        for engine in engines:
            if not engine.supports(self.name):
                continue
            result = engine.run(spec, self.name)
            metrics = dict(result.metrics)
            difference = abs(float(metrics["loss"]) - reference)
            metrics.update(
                {
                    "reference_loss": reference,
                    "difference": difference,
                    "tolerance": spec.parity.tolerance,
                    "passed": difference <= spec.parity.tolerance,
                }
            )
            results.append(
                EngineResult(engine=result.engine, operation=result.operation, metrics=metrics)
            )
        if not results:
            raise ValueError("no engine supports the parity vertical")
        return VerticalResult(vertical=self.name, engines=tuple(results))


# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Locked-clock LitniceLM speed and profile passes for the gate."""

from dataclasses import replace
from pathlib import Path

from ..engines.base import EngineResult, EngineVariant
from ..engines.litnicelm import LitniceLMEngine
from ..report.json_report import write_json
from ..spec import ComparisonSpec
from ..verticals.base import VerticalResult

GATE_REPEATS = 5


class LitniceLMGateRun:
    """Run every LitniceLM variant GATE_REPEATS times plus one profile pass.

    Only steady-state speed runs are made: no differential endpoints, no
    parity, no PyTorch. Rows use the train_speed result layout so baselines
    from full comparison runs remain comparable.
    """

    def __init__(self, engine: LitniceLMEngine, repeats: int = GATE_REPEATS) -> None:
        self.engine = engine
        self.repeats = repeats

    def run(self, spec: ComparisonSpec) -> tuple[EngineResult, ...]:
        """Speed rows for every precision, variant and repeat, with profiles.

        Repeats are the outer loop so slow drift spreads over all variants.
        """
        speed_rows = []
        for repeat in range(1, self.repeats + 1):
            for variant in self._variants(spec):
                try:
                    result = self.engine.run_speed(spec, variant, spec.train_speed.steps)
                except RuntimeError as error:
                    raise RuntimeError(
                        f"repeat={repeat} precision={variant.precision} "
                        f"variant={variant.label}: {error}"
                    ) from error
                speed_rows.append((variant, repeat, result))
        profiles = {
            (variant.precision, variant.label): self.engine.profile_variant(
                spec, "train_speed", variant
            )
            for variant in self._variants(spec)
        }
        return tuple(
            EngineResult(
                engine=result.engine,
                operation=result.operation,
                metrics={
                    **result.metrics,
                    "variant_label": variant.label,
                    "variant_options": ", ".join(
                        f"{key}={value}" for key, value in variant.options.items()
                    ),
                    "repeat": repeat,
                    **profiles[(variant.precision, variant.label)],
                },
            )
            for variant, repeat, result in speed_rows
        )

    def write_results(
        self, spec: ComparisonSpec, rows: tuple[EngineResult, ...], path: Path
    ) -> None:
        """Write rows in the results-JSON layout with vertical train_speed."""
        recorded = replace(
            spec, train_speed=replace(spec.train_speed, repeats=self.repeats)
        )
        write_json(
            path,
            recorded,
            (VerticalResult(vertical="train_speed", engines=rows),),
            path.parent,
        )

    def _variants(self, spec: ComparisonSpec) -> tuple[EngineVariant, ...]:
        return tuple(
            variant
            for precision in spec.workload.precisions
            for variant in self.engine.variants(spec, "train_speed", precision)
        )

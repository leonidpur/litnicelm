# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Steady-state LitniceLM throughput for the gate's PyTorch ratio."""

from collections import defaultdict
from typing import Sequence

from ..engines.base import EngineVariant
from ..engines.litnicelm import LitniceLMEngine
from ..spec import ComparisonSpec
from ..verticals.train_speed import TrainSpeedVertical

STEADY_REPEATS = 3


class LitniceLMSteadyState:
    """Steady-state throughput per variant, for comparison with PyTorch.

    Each sample is the difference of two whole-epoch runs of different
    lengths (spec.train_speed.differential_epochs) without validation or
    checkpoints, so one-time costs (graph builds, kernel plan timing,
    allocation) cancel, as the PyTorch worker's warm-up steps exclude them on
    its side. The gate's all-in timed runs include them.
    """

    def __init__(self, engine: LitniceLMEngine, repeats: int = STEADY_REPEATS) -> None:
        self.engine = engine
        self.repeats = repeats

    def measure(
        self, spec: ComparisonSpec, variants: Sequence[EngineVariant]
    ) -> dict[tuple[str, str], tuple[dict[str, int | float], ...]]:
        """Differential metrics per (precision, variant label), one per repeat.

        Repeats are the outer loop so slow drift spreads over all variants.
        The fields and formula are the full comparison's train_speed ones
        (differential_step_ms, differential_tokens_per_s, ...).
        """
        samples: dict[tuple[str, str], list[dict[str, int | float]]] = defaultdict(list)
        for repeat in range(1, self.repeats + 1):
            for variant in variants:
                try:
                    endpoints = self.engine.run_differential(spec, variant, None)
                except RuntimeError as error:
                    raise RuntimeError(
                        f"steady-state repeat={repeat} precision={variant.precision} "
                        f"variant={variant.label}: {error}"
                    ) from error
                samples[(variant.precision, variant.label)].append(
                    TrainSpeedVertical._differential_metrics(endpoints)
                )
        return {key: tuple(values) for key, values in samples.items()}

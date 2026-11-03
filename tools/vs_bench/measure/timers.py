# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Wall-clock and differential timing boundaries for later implementation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DifferentialTiming:
    """Whole-process timings used to remove constant startup cost."""

    steps_1: int
    seconds_1: float
    steps_2: int
    seconds_2: float

    @property
    def seconds_per_step(self) -> float:
        return (self.seconds_2 - self.seconds_1) / (self.steps_2 - self.steps_1)


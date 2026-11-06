# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Locked-clock LitniceLM speed and profile passes for the gate."""

from pathlib import Path

from ..engines.base import EngineResult
from ..engines.litnicelm import LitniceLMEngine
from ..spec import ComparisonSpec

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
        """Speed rows for every precision, variant and repeat, with profiles."""
        raise NotImplementedError

    def write_results(
        self, spec: ComparisonSpec, rows: tuple[EngineResult, ...], path: Path
    ) -> None:
        """Write rows in the results-JSON layout with vertical train_speed."""
        raise NotImplementedError

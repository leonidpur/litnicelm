# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Median throughput per LitniceLM variant and the 0.5% regression check."""

from dataclasses import dataclass
from pathlib import Path

MAX_DROP_PCT = 0.5


@dataclass(frozen=True)
class GateKey:
    """One gated cell: a neutral precision and a LitniceLM variant label."""

    precision: str
    variant: str


@dataclass(frozen=True)
class GateRow:
    """Steady-state throughput samples of one cell, their median, and the
    per-op breakdown of its profile pass ("" when the variant is unprofiled)."""

    key: GateKey
    samples_tokens_per_s: tuple[float, ...]
    op_breakdown: str

    @property
    def median_tokens_per_s(self) -> float:
        """Median of the steady-state samples."""
        raise NotImplementedError


@dataclass(frozen=True)
class GateComparison:
    """One cell of the current run against the same cell of the baseline."""

    current: GateRow
    baseline: GateRow
    max_drop_pct: float

    @property
    def change_pct(self) -> float:
        """Signed median change in percent; negative is slower."""
        raise NotImplementedError

    @property
    def passed(self) -> bool:
        """Whether the median dropped by at most max_drop_pct."""
        raise NotImplementedError


@dataclass(frozen=True)
class PyTorchReference:
    """Best eager and best compiled PyTorch medians for one precision."""

    precision: str
    best_eager_label: str
    best_eager_tokens_per_s: float
    best_compiled_label: str
    best_compiled_tokens_per_s: float


def load_litnicelm_rows(results_path: Path) -> dict[GateKey, GateRow]:
    """Read LitniceLM train_speed rows from a gate or full-run results JSON."""
    raise NotImplementedError


def load_pytorch_references(results_path: Path) -> dict[str, PyTorchReference]:
    """Read the best eager and compiled PyTorch medians from a full-run JSON."""
    raise NotImplementedError


def compare(
    current: dict[GateKey, GateRow],
    baseline: dict[GateKey, GateRow],
    max_drop_pct: float = MAX_DROP_PCT,
) -> tuple[GateComparison, ...]:
    """Compare every baseline cell; a cell missing from the current run fails."""
    raise NotImplementedError

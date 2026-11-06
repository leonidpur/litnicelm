# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Plain-text gate report."""

from typing import Sequence

from .compare import GateComparison, GateRow, PyTorchReference


def render_gate_report(
    comparisons: Sequence[GateComparison],
    references: dict[str, PyTorchReference],
) -> str:
    """Before/after table, verdict, before/after per-op breakdown, PyTorch ratios."""
    raise NotImplementedError


def render_pytorch_ratios(
    rows: Sequence[GateRow], references: dict[str, PyTorchReference]
) -> str:
    """LitniceLM vs best eager and best compiled PyTorch, per precision."""
    raise NotImplementedError

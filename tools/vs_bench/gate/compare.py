# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Median throughput per LitniceLM variant and the 0.5% regression check."""

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import statistics

MAX_DROP_PCT = 0.5


@dataclass(frozen=True)
class GateKey:
    """One gated cell: a neutral precision and a LitniceLM variant label."""

    precision: str
    variant: str


@dataclass(frozen=True)
class GateRow:
    """Throughput samples of one cell and the per-op breakdown of its profile
    pass ("" when the variant is unprofiled).

    samples_tokens_per_s are the all-in timed runs the gate compares against
    its baseline: whole training runs with validation, checkpoints and
    one-time setup. samples_steady_tokens_per_s are [SteadyState] figures
    after the warm-up optimizer steps (in older results, steady-state
    differentials); empty for results recorded before either existed.
    host_state_flagged: some row ran under a host power state other than
    the spec's.
    """

    key: GateKey
    samples_tokens_per_s: tuple[float, ...]
    op_breakdown: str
    samples_steady_tokens_per_s: tuple[float, ...] = ()
    host_state_flagged: bool = False

    @property
    def median_tokens_per_s(self) -> float:
        """Median of the all-in samples."""
        return statistics.median(self.samples_tokens_per_s)

    @property
    def steady_tokens_per_s(self) -> float | None:
        """Median of the steady-state samples, or None when none were recorded."""
        if not self.samples_steady_tokens_per_s:
            return None
        return statistics.median(self.samples_steady_tokens_per_s)


@dataclass(frozen=True)
class GateComparison:
    """One cell of the current run against the same cell of the baseline."""

    current: GateRow
    baseline: GateRow
    max_drop_pct: float

    @property
    def change_pct(self) -> float:
        """Signed median change in percent; negative is slower."""
        before = self.baseline.median_tokens_per_s
        return 100.0 * (self.current.median_tokens_per_s - before) / before

    @property
    def passed(self) -> bool:
        """Whether the median dropped by at most max_drop_pct."""
        return self.change_pct >= -self.max_drop_pct


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
    samples: dict[GateKey, list[float]] = defaultdict(list)
    steady: dict[GateKey, list[float]] = defaultdict(list)
    breakdowns: dict[GateKey, str] = {}
    host_flagged: dict[GateKey, bool] = defaultdict(bool)
    for row in _train_speed_rows(results_path):
        if row["engine"] != "LitniceLM":
            continue
        key = GateKey(str(row["precision"]), str(row["variant_label"]))
        samples[key].append(float(row["tokens_per_s"]))
        # [SteadyState] after warm-up; older results only have the differential.
        if "steady_tokens_per_s" in row:
            steady[key].append(float(row["steady_tokens_per_s"]))
        elif "differential_tokens_per_s" in row:
            steady[key].append(float(row["differential_tokens_per_s"]))
        host_flagged[key] |= bool(row.get("host_state_flagged"))
        breakdowns[key] = str(row.get("attention_cuda_kernels", ""))
    if not samples:
        raise ValueError(f"{results_path}: no LitniceLM train_speed rows")
    return {
        key: GateRow(
            key, tuple(values), breakdowns[key], tuple(steady[key]), host_flagged[key]
        )
        for key, values in samples.items()
    }


def load_pytorch_references(results_path: Path) -> dict[str, PyTorchReference]:
    """Read the best eager and compiled PyTorch medians from a full-run JSON."""
    samples: dict[tuple[str, str, bool], list[float]] = defaultdict(list)
    for row in _train_speed_rows(results_path):
        if row["engine"] != "PyTorch":
            continue
        eager = "compile_mode=none" in str(row["variant_options"])
        samples[(str(row["precision"]), str(row["variant_label"]), eager)].append(
            float(row["tokens_per_s"])
        )
    medians = {key: statistics.median(values) for key, values in samples.items()}
    references = {}
    for precision in sorted({key[0] for key in medians}):
        def best(eager: bool) -> tuple[str, float]:
            candidates = [
                (label, value)
                for (row_precision, label, row_eager), value in medians.items()
                if row_precision == precision and row_eager == eager
            ]
            if not candidates:
                kind = "eager" if eager else "compiled"
                raise ValueError(f"{results_path}: no {kind} PyTorch rows for {precision}")
            return max(candidates, key=lambda item: item[1])

        eager_label, eager_value = best(True)
        compiled_label, compiled_value = best(False)
        references[precision] = PyTorchReference(
            precision, eager_label, eager_value, compiled_label, compiled_value
        )
    if not references:
        raise ValueError(f"{results_path}: no PyTorch train_speed rows")
    return references


def compare(
    current: dict[GateKey, GateRow],
    baseline: dict[GateKey, GateRow],
    max_drop_pct: float = MAX_DROP_PCT,
) -> tuple[GateComparison, ...]:
    """Compare every baseline cell; a cell missing from the current run fails.

    Cells only in the current run (new variants) have nothing to compare
    against and are not gated.
    """
    missing = sorted(
        f"{key.precision}/{key.variant}" for key in baseline if key not in current
    )
    if missing:
        raise ValueError(f"gate cells missing from the current run: {', '.join(missing)}")
    return tuple(
        GateComparison(current[key], baseline[key], max_drop_pct)
        for key in sorted(baseline, key=lambda item: (item.precision, item.variant))
    )


def new_cells(
    current: dict[GateKey, GateRow], baseline: dict[GateKey, GateRow]
) -> tuple[GateRow, ...]:
    """Current cells without a baseline (new variants), reported but not gated."""
    return tuple(
        current[key]
        for key in sorted(current, key=lambda item: (item.precision, item.variant))
        if key not in baseline
    )


def _train_speed_rows(results_path: Path) -> list[dict[str, object]]:
    payload = json.loads(results_path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"{results_path}: status is {payload.get('status')!r}, not 'complete'")
    for vertical in payload["verticals"]:
        if vertical["name"] == "train_speed":
            return vertical["rows"]
    raise ValueError(f"{results_path}: no train_speed vertical")

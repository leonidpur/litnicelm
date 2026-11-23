# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Plain-text gate report."""

from typing import Sequence

from .compare import GateComparison, GateRow, PyTorchReference


def render_gate_report(
    comparisons: Sequence[GateComparison],
    references: dict[str, PyTorchReference],
    new_rows: Sequence[GateRow] = (),
) -> str:
    """Before/after table, verdict, before/after per-op breakdown, PyTorch ratios."""
    lines = [
        "| precision | variant | baseline tokens/s | current tokens/s "
        "| current range | change | gate |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for item in comparisons:
        samples = item.current.samples_tokens_per_s
        lines.append(
            f"| {item.current.key.precision} | {item.current.key.variant} | "
            f"{item.baseline.median_tokens_per_s:,.0f} | "
            f"{item.current.median_tokens_per_s:,.0f} | "
            f"{min(samples):,.0f}–{max(samples):,.0f} | "
            f"{item.change_pct:+.2f}% | {'pass' if item.passed else 'FAIL'}"
            f"{'; FLAG host state' if item.current.host_state_flagged else ''} |"
        )
    for row in new_rows:
        samples = row.samples_tokens_per_s
        lines.append(
            f"| {row.key.precision} | {row.key.variant} | — | "
            f"{row.median_tokens_per_s:,.0f} | "
            f"{min(samples):,.0f}–{max(samples):,.0f} | — | new, not gated |"
        )
    failed = [item for item in comparisons if not item.passed]
    max_drop = comparisons[0].max_drop_pct if comparisons else 0.0
    lines += [
        "",
        (
            f"Gate FAILED: {len(failed)} of {len(comparisons)} cells dropped by "
            f"more than {max_drop:g}%."
            if failed
            else f"Gate passed: no cell dropped by more than {max_drop:g}%."
        ),
        "",
        "Per-op breakdown (profile pass):",
    ]
    for item in comparisons:
        key = item.current.key
        lines += [
            f"- {key.precision} / {key.variant}",
            f"  - before: {item.baseline.op_breakdown or 'not recorded'}",
            f"  - after:  {item.current.op_breakdown or 'not recorded'}",
        ]
    for row in new_rows:
        lines += [
            f"- {row.key.precision} / {row.key.variant} (new)",
            f"  - after:  {row.op_breakdown or 'not recorded'}",
        ]
    lines += [
        "",
        render_pytorch_ratios(
            [item.current for item in comparisons] + list(new_rows), references
        ),
    ]
    return "\n".join(lines)


PYTORCH_RATIO_NOTE = (
    "Ratios use LitniceLM steady-state tokens/s (batches after its warm-up "
    "optimizer steps, which cover cuDNN plan selection; older results use the "
    "epoch differential) against PyTorch tokens/s after its warm-up steps. "
    "\"all-in\" is the gate's timed run, which also counts validation, "
    "checkpoints and one-time setup; a row marked (all-in) has no steady-state "
    "samples and uses the all-in figure."
)


def render_pytorch_ratios(
    rows: Sequence[GateRow], references: dict[str, PyTorchReference]
) -> str:
    """LitniceLM vs best eager and best compiled PyTorch, per precision.

    Steady state against steady state; the all-in ratio to best compiled
    PyTorch is shown alongside.
    """
    lines = [
        "| precision | variant | steady-state tokens/s | vs best eager PyTorch "
        "| vs best compiled PyTorch | all-in vs best compiled |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        reference = references.get(row.key.precision)
        if reference is None:
            continue
        steady = row.steady_tokens_per_s
        value = steady if steady is not None else row.median_tokens_per_s
        marker = "" if steady is not None else " (all-in)"
        lines.append(
            f"| {row.key.precision} | {row.key.variant} | "
            f"{value:,.0f}{marker} | "
            f"{value / reference.best_eager_tokens_per_s:.3f}× "
            f"({reference.best_eager_label}) | "
            f"{value / reference.best_compiled_tokens_per_s:.3f}× "
            f"({reference.best_compiled_label}) | "
            f"{row.median_tokens_per_s / reference.best_compiled_tokens_per_s:.3f}× |"
        )
    return "\n".join(lines + ["", PYTORCH_RATIO_NOTE])

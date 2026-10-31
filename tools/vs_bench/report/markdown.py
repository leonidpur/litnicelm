"""Markdown tables for parity and training-speed results."""

import statistics
from typing import Sequence

from ..verticals.base import VerticalResult


def render_markdown(results: Sequence[VerticalResult]) -> str:
    """Render parity rows and median speed rows with observed ranges."""
    sections = []
    for result in results:
        if result.vertical == "parity":
            lines = [
                "## Parity",
                "",
                "| engine | precision | loss | reference | difference | tolerance | passed |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
            for row in result.engines:
                m = row.metrics
                lines.append(
                    f"| {row.engine} | {m['precision']} | {float(m['loss']):.9f} | "
                    f"{float(m['reference_loss']):.9f} | {float(m['difference']):.3g} | "
                    f"{float(m['tolerance']):.3g} | {str(m['passed']).lower()} |"
                )
            sections.append("\n".join(lines))
        elif result.vertical == "train_speed":
            groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
            for row in result.engines:
                metrics = row.metrics
                key = (row.engine, str(metrics["precision"]), str(metrics["sync"]))
                groups.setdefault(key, []).append(metrics)
            lines = [
                "## Training speed",
                "",
                "| engine | precision | loss read | profiler tokens/s | process wall (s) | differential tokens/s | differential step (ms) | process peak (MiB) | engine memory (MiB) | SM clock (MHz) |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for (engine, precision, sync), rows in groups.items():
                median = lambda key: statistics.median(float(row[key]) for row in rows)
                throughputs = [float(row["tokens_per_s"]) for row in rows]
                engine_memory = [row["engine_memory_mib"] for row in rows]
                memory = (
                    f"{statistics.median(float(value) for value in engine_memory):.0f}"
                    if all(isinstance(value, (int, float)) for value in engine_memory)
                    else "unknown"
                )
                lines.append(
                    f"| {engine} | {precision} | {sync} | {median('tokens_per_s'):,.0f} "
                    f"({min(throughputs):,.0f}–{max(throughputs):,.0f}) | "
                    f"{median('process_wall_seconds'):.2f} | "
                    f"{median('differential_tokens_per_s'):,.0f} | "
                    f"{median('differential_step_ms'):.1f} | "
                    f"{median('process_peak_mib'):.0f} | "
                    f"{memory} | {median('sm_clock_mhz'):.0f} |"
                )
            equivalence = [
                "## Training equivalence",
                "",
                "| engine | precision | loss read | model (L/H/D/FF) | tokens/step | final train loss | vs LitniceLM | tolerance | passed |",
                "|---|---|---|---|---:|---:|---:|---:|---|",
            ]
            for (engine, precision, sync), rows in groups.items():
                first = rows[0]
                equivalence.append(
                    f"| {engine} | {precision} | {sync} | "
                    f"{first['n_layers']}/{first['n_heads']}/{first['d_model']}/{first['d_ff']} | "
                    f"{first['tokens_per_step']} | "
                    f"{statistics.median(float(row['final_train_loss']) for row in rows):.4f} | "
                    f"{statistics.median(float(row['loss_difference']) for row in rows):.4f} | "
                    f"{float(first['loss_tolerance']):.3g} | "
                    f"{str(all(bool(row['equivalence_passed']) for row in rows)).lower()} |"
                )
            sections.append("\n".join(lines) + "\n\n" + "\n".join(equivalence))
    return "\n\n".join(sections)


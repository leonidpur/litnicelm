# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Markdown tables for parity and training-speed results."""

import statistics
from typing import Sequence

from ..verticals.base import VerticalResult


def _steady_tokens_per_s(row: dict[str, object]) -> float:
    """Throughput after warm-up: LitniceLM's [SteadyState] figure when the row
    has one, otherwise tokens_per_s (PyTorch measures after its warm-up)."""
    return float(row.get("steady_tokens_per_s", row["tokens_per_s"]))


def render_markdown(results: Sequence[VerticalResult]) -> str:
    """Render parity rows and median speed rows with observed ranges."""
    sections = []
    for result in results:
        if result.vertical == "parity":
            precisions = dict.fromkeys(str(row.metrics["precision"]) for row in result.engines)
            for precision in precisions:
                lines = [
                    f"## Parity — {precision}",
                    "",
                    "| engine | variant | options | loss | reference | difference | tolerance | passed |",
                    "|---|---|---|---:|---:|---:|---:|---|",
                ]
                for row in result.engines:
                    m = row.metrics
                    if str(m["precision"]) != precision:
                        continue
                    lines.append(
                        f"| {row.engine} | {m['variant_label']} | "
                        f"{m['variant_options'] or '—'} | "
                        f"{float(m['loss']):.9f} | {float(m['reference_loss']):.9f} | "
                        f"{float(m['difference']):.3g} | {float(m['tolerance']):.3g} | "
                        f"{str(m['passed']).lower()} |"
                    )
                sections.append("\n".join(lines))
        elif result.vertical == "train_speed":
            groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
            for row in result.engines:
                metrics = row.metrics
                key = (
                    str(metrics["precision"]),
                    row.engine,
                    str(metrics["variant_label"]),
                    str(metrics["variant_options"]),
                )
                groups.setdefault(key, []).append(metrics)
            for precision in dict.fromkeys(key[0] for key in groups):
                precision_groups = {key: rows for key, rows in groups.items() if key[0] == precision}
                medians = {
                    key: statistics.median(_steady_tokens_per_s(row) for row in rows)
                    for key, rows in precision_groups.items()
                }
                pytorch_keys = [key for key in precision_groups if key[1] == "PyTorch"]
                best_pytorch = max(pytorch_keys, key=medians.__getitem__)
                litnice_keys = [key for key in precision_groups if key[1] == "LitniceLM"]
                lines = [
                    f"## Training speed — {precision}",
                    "",
                    "| engine | variant | options | attention backend | top CUDA kernels by time | best PyTorch | steady-state tokens/s | all-in tokens/s | differential tokens/s | cross-check | differential step (ms) | compile time (s) | plan selection (s) | process peak (MiB) | engine memory (MiB) | clock (MHz) |",
                    "|---|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
                ]
                for key, rows in precision_groups.items():
                    _, engine, label, options = key
                    median = lambda field: statistics.median(float(row[field]) for row in rows)
                    throughputs = [_steady_tokens_per_s(row) for row in rows]
                    memory_values = [row["engine_memory_mib"] for row in rows]
                    memory = (
                        f"{statistics.median(float(value) for value in memory_values):.0f}"
                        if all(isinstance(value, (int, float)) for value in memory_values)
                        else "unknown"
                    )
                    compile_time = (
                        f"{median('compile_warmup_seconds'):.2f}"
                        if "compile_warmup_seconds" in rows[0]
                        else "—"
                    )
                    steady = statistics.median(throughputs)
                    all_in = (
                        f"{median('tokens_per_s'):,.0f}"
                        if "steady_tokens_per_s" in rows[0]
                        else "—"
                    )
                    plan_selection = (
                        f"{median('plan_selection_seconds'):.3f}"
                        if "plan_selection_seconds" in rows[0]
                        else "—"
                    )
                    valid = [row for row in rows if "differential_tokens_per_s" in row]
                    valid_note = (
                        f" ({len(valid)}/{len(rows)} valid)" if len(valid) < len(rows) else ""
                    )
                    if valid:
                        differential = statistics.median(
                            float(row["differential_tokens_per_s"]) for row in valid
                        )
                        differential_step = statistics.median(
                            float(row["differential_step_ms"]) for row in valid
                        )
                        disagreement = 100.0 * abs(steady - differential) / steady
                        cross_check = (
                            f"FLAG {disagreement:.1f}%"
                            if disagreement > 3.0 or valid_note
                            else f"ok ({disagreement:.1f}%)"
                        ) + valid_note
                        differential_text = f"{differential:,.0f}"
                        differential_step_text = f"{differential_step:.1f}"
                    else:
                        cross_check = "FLAG invalid differential"
                        differential_text = "invalid"
                        differential_step_text = "—"
                    flagged_host = sum(bool(row.get("host_state_flagged")) for row in rows)
                    if flagged_host:
                        cross_check += f"; FLAG host state ({flagged_host}/{len(rows)})"
                    lines.append(
                        f"| {engine} | {label} | {options or '—'} | "
                        f"{rows[0]['attention_backend']} | "
                        f"{rows[0]['attention_cuda_kernels']} | "
                        f"{'yes' if key == best_pytorch else ''} | "
                        f"{steady:,.0f} ({min(throughputs):,.0f}–{max(throughputs):,.0f}) | "
                        f"{all_in} | "
                        f"{differential_text} | {cross_check} | "
                        f"{differential_step_text} | "
                        f"{compile_time} | {plan_selection} | "
                        f"{median('process_peak_mib'):.0f} | {memory} | "
                        f"{median('sm_clock_mhz'):.0f} |"
                    )
                verdict = "\n".join(
                    f"LitniceLM {key[2]} / best PyTorch by steady-state throughput: "
                    f"**{medians[key] / medians[best_pytorch]:.3f}×** "
                    f"({'ahead' if medians[key] > medians[best_pytorch] else 'behind'}; "
                    f"best PyTorch is {best_pytorch[2]} [{best_pytorch[3]}])."
                    for key in litnice_keys
                )
                litnice_attention = (
                    "LitniceLM attention backends are stated from each "
                    "variant's implementation, not inferred by a profiler."
                )
                equivalence = [
                    f"### Training equivalence — {precision}",
                    "",
                    "| engine | variant | options | model (L/H/D/FF) | tokens/step | final loss | vs LitniceLM | passed |",
                    "|---|---|---|---|---:|---:|---:|---|",
                ]
                for key, rows in precision_groups.items():
                    _, engine, label, options = key
                    first = rows[0]
                    equivalence.append(
                        f"| {engine} | {label} | {options or '—'} | "
                        f"{first['n_layers']}/{first['n_heads']}/{first['d_model']}/{first['d_ff']} | "
                        f"{first['tokens_per_step']} | "
                        f"{statistics.median(float(row['final_train_loss']) for row in rows):.4f} | "
                        f"{statistics.median(float(row['loss_difference']) for row in rows):.4f} | "
                        f"{str(all(bool(row['equivalence_passed']) for row in rows)).lower()} |"
                    )
                host_flags = [
                    f"- {key[1]} {key[2]} repeat {row.get('repeat')}: "
                    f"AC online={row.get('host_ac_online')}, "
                    f"platform profile {row.get('host_platform_profile')}"
                    for key, rows in precision_groups.items()
                    for row in rows
                    if row.get("host_state_flagged")
                ]
                host_note = (
                    "Rows measured under a host power state other than the spec's:\n"
                    + "\n".join(host_flags)
                    if host_flags
                    else "Host power state matched the spec in every row."
                )
                sections.append(
                    "\n".join(lines)
                    + "\n\n"
                    + host_note
                    + "\n\n"
                    + verdict
                    + "\n\n"
                    + litnice_attention
                    + "\n\n"
                    + "\n".join(equivalence)
                )
    return "\n\n".join(sections)


"""Markdown tables for parity and training-speed results."""

import statistics
from typing import Sequence

from ..verticals.base import VerticalResult


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
                    key: statistics.median(float(row["differential_tokens_per_s"]) for row in rows)
                    for key, rows in precision_groups.items()
                }
                pytorch_keys = [key for key in precision_groups if key[1] == "PyTorch"]
                best_pytorch = max(pytorch_keys, key=medians.__getitem__)
                litnice_key = next(key for key in precision_groups if key[1] == "LitniceLM")
                ratio = medians[litnice_key] / medians[best_pytorch]
                lines = [
                    f"## Training speed — {precision}",
                    "",
                    "| engine | variant | options | attention backend | top CUDA kernels by time | best PyTorch | profiler tokens/s | differential tokens/s | differential step (ms) | compile time (s) | process peak (MiB) | engine memory (MiB) | clock (MHz) |",
                    "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
                for key, rows in precision_groups.items():
                    _, engine, label, options = key
                    median = lambda field: statistics.median(float(row[field]) for row in rows)
                    throughputs = [float(row["tokens_per_s"]) for row in rows]
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
                    lines.append(
                        f"| {engine} | {label} | {options or '—'} | "
                        f"{rows[0]['attention_backend']} | "
                        f"{rows[0]['attention_cuda_kernels']} | "
                        f"{'yes' if key == best_pytorch else ''} | "
                        f"{median('tokens_per_s'):,.0f} ({min(throughputs):,.0f}–{max(throughputs):,.0f}) | "
                        f"{median('differential_tokens_per_s'):,.0f} | "
                        f"{median('differential_step_ms'):.1f} | "
                        f"{compile_time} | "
                        f"{median('process_peak_mib'):.0f} | {memory} | "
                        f"{median('sm_clock_mhz'):.0f} |"
                    )
                verdict = (
                    f"LitniceLM / best PyTorch: **{ratio:.3f}×** "
                    f"({'ahead' if ratio > 1 else 'behind'}; best PyTorch is "
                    f"{best_pytorch[2]} [{best_pytorch[3]}])."
                )
                litnice_attention = (
                    "LitniceLM attention is implemented as unfused per-head "
                    "GEMMs plus softmax; this is an implementation statement, "
                    "not a PyTorch-profiler inference."
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
                sections.append(
                    "\n".join(lines)
                    + "\n\n"
                    + verdict
                    + "\n\n"
                    + litnice_attention
                    + "\n\n"
                    + "\n".join(equivalence)
                )
    return "\n\n".join(sections)


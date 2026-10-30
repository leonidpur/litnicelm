"""Friendly console output for comparison phases."""

from typing import Sequence

from ..spec import ComparisonSpec
from ..verticals.base import VerticalPlan


def print_plan(spec: ComparisonSpec, plans: Sequence[VerticalPlan]) -> None:
    """Render the skeleton's plan without executing any benchmark work."""
    workload = spec.workload
    print("LitniceLM comparison plan")
    print("===========================")
    print(
        f"Model: {workload.n_layers} layers, {workload.n_heads} heads, "
        f"d_model {workload.d_model}, d_ff {workload.d_ff}"
    )
    print(f"Corpus: {workload.corpus_id}")
    print(
        f"Training: seq {workload.seq_len}, batch {workload.batch_size}, "
        f"accumulation {workload.accumulation_steps}, {workload.tokens_per_step} tokens/step"
    )
    print(f"Run: {spec.train_speed.steps} steps, {spec.train_speed.repeats} repeats")
    print(f"Precisions: {', '.join(workload.precisions)}")
    clock_mhz = spec.environment.clock_mhz
    clock = f"{clock_mhz} MHz" if clock_mhz else "not selected"
    print(f"Locked GPU clock: {clock}")

    for plan in plans:
        print(f"\n[{plan.vertical}] {plan.purpose}")
        for engine in plan.engines:
            print(f"  {engine.engine}:")
            for action in engine.actions:
                print(f"    - {action}")

    print("\nPlan only: no commands were executed.")


"""Friendly console output for comparison phases."""

from typing import Sequence

from spec import ComparisonSpec
from verticals.base import VerticalPlan


def print_plan(spec: ComparisonSpec, plans: Sequence[VerticalPlan]) -> None:
    """Render the skeleton's plan without executing any benchmark work."""
    print("LitniceLM comparison plan")
    print("===========================")
    print(
        f"Model: {spec.n_layers} layers, {spec.n_heads} heads, "
        f"d_model {spec.d_model}, d_ff {spec.d_ff}"
    )
    print(f"Corpus: {spec.corpus_id}")
    print(
        f"Training: seq {spec.seq_len}, batch {spec.batch_size}, "
        f"accumulation {spec.accumulation_steps}, {spec.tokens_per_step} tokens/step"
    )
    print(f"Run: {spec.steps} steps, {spec.repeats} repeats")
    print(f"Precisions: {', '.join(spec.precisions)}")
    clock = f"{spec.clock_mhz} MHz" if spec.clock_mhz else "not selected"
    print(f"Locked GPU clock: {clock}")

    for plan in plans:
        print(f"\n[{plan.vertical}] {plan.purpose}")
        for engine in plan.engines:
            print(f"  {engine.engine}:")
            for action in engine.actions:
                print(f"    - {action}")

    print("\nPlan only: no commands were executed.")


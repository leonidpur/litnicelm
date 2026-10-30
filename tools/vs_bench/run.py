#!/usr/bin/env python3
"""Print the plan for the hard-coded LitniceLM comparison specification."""

from .engines.litnicelm import LitniceLMEngine
from .engines.torch_engine import TorchEngine
from .report.console import print_plan
from .spec import ComparisonSpec
from .verticals.parity import ParityVertical
from .verticals.train_speed import TrainSpeedVertical


SPEC = ComparisonSpec(
    n_layers=4,
    n_heads=4,
    d_model=256,
    d_ff=1024,
    corpus_id="conan-doyle-clean-spm512",
    seq_len=1024,
    batch_size=1,
    accumulation_steps=4,
    steps=300,
    precisions=("fp32", "bf16"),
    repeats=3,
    clock_mhz=0,  # Replace with the clock selected for the verified run.
)


def main() -> None:
    """Build the comparison objects, print their plan, and exit."""
    engines = (LitniceLMEngine(), TorchEngine())
    verticals = (ParityVertical(), TrainSpeedVertical())
    plans = tuple(vertical.plan(SPEC, engines) for vertical in verticals)
    print_plan(SPEC, plans)


if __name__ == "__main__":
    main()


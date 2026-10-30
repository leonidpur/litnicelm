#!/usr/bin/env python3
"""Print the plan for the hard-coded LitniceLM comparison specification."""

from pathlib import Path

from .engines.litnicelm import LitniceLMEngine
from .engines.torch_engine import TorchEngine
from .report.console import print_plan
from .spec import (
    ComparisonSpec,
    EnvironmentSpec,
    ParitySettings,
    TrainSpeedSettings,
    WorkloadSpec,
)
from .verticals.parity import ParityVertical
from .verticals.train_speed import TrainSpeedVertical


SPEC = ComparisonSpec(
    workload=WorkloadSpec(
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=1024,
        position_encoding="rope",
        corpus_id="conan-doyle-clean-spm512",
        tokenized_corpus_path=Path(
            "/mnt/ext_ssd/litnicelm/data/conan-doyle-clean-spm512_tokenized_dataset.bin"
        ),
        seq_len=1024,
        batch_size=1,
        validation_fraction=0.1,
        validation_chunk_tokens=8192,
        window_stride=1024,
        accumulation_steps=4,
        seed=0,
        learning_rate=0.0006,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        weight_decay=0.1,
        grad_clip=0.06,
        lr_schedule="cosine",
        warmup_steps=107,
        min_lr_ratio=0.1,
        precisions=("fp32", "bf16"),
    ),
    environment=EnvironmentSpec(clock_mhz=0),
    parity=ParitySettings(
        export_dir=Path(
            "/mnt/ext_ssd/litnicelm/doyle/exports/"
            "doyle_clean_spm512_seq1024_l4_d256_b1_rope_val_cosine_e12_accum4_fp32"
        ),
        precision="fp32",
        tolerance=1e-5,
    ),
    train_speed=TrainSpeedSettings(steps=300, repeats=3),
)


def main() -> None:
    """Build the comparison objects, print their plan, and exit."""
    engines = (LitniceLMEngine(), TorchEngine())
    verticals = (ParityVertical(), TrainSpeedVertical())
    plans = tuple(vertical.plan(SPEC, engines) for vertical in verticals)
    print_plan(SPEC, plans)


if __name__ == "__main__":
    main()


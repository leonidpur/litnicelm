#!/usr/bin/env python3
"""Run the hard-coded LitniceLM comparison specification."""

from datetime import date
from pathlib import Path

from .engines.litnicelm import LitniceLMEngine
from .engines.torch_engine import TorchEngine
from .device.clock import LockedGpuClock
from .measure.process import configure_raw_logs, raw_artifact_path
from .report.console import print_phase_result, print_phase_start, print_plan
from .report.json_report import write_json
from .report.markdown import render_markdown
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
    environment=EnvironmentSpec(clock_mhz=1500),
    parity=ParitySettings(
        export_dir=Path(
            "/mnt/ext_ssd/litnicelm/doyle/exports/"
            "doyle_clean_spm512_seq1024_l4_d256_b1_rope_val_cosine_e12_accum4_fp32"
        ),
        precision="fp32",
        tolerance=1e-5,
    ),
    train_speed=TrainSpeedSettings(
        steps=300,
        repeats=3,
        differential_epochs=(1, 3),
        equivalence_loss_tolerance=0.5,
    ),
)

RAW_LOG_ROOT = Path("/mnt/ext_ssd/litnicelm/bench")


def main() -> None:
    """Explain, execute, and report every configured comparison vertical."""
    engines = (LitniceLMEngine(), TorchEngine())
    verticals = (ParityVertical(), TrainSpeedVertical())
    plans = tuple(vertical.plan(SPEC, engines) for vertical in verticals)
    print_plan(SPEC, plans)
    raw_log_directory = RAW_LOG_ROOT / date.today().isoformat()
    configure_raw_logs(raw_log_directory)
    results_path = raw_artifact_path("results.json")
    results = []
    with LockedGpuClock(SPEC.environment.clock_mhz) as clock:
        for vertical, plan in zip(verticals, plans, strict=True):
            print_phase_start(plan)
            result = vertical.run(SPEC, engines)
            results.append(result)
            print_phase_result(result)
        clock.verify_under_load(results)
    write_json(results_path, SPEC, results, raw_log_directory)
    print(f"\nResults JSON: {results_path}")
    print(f"Raw logs: {raw_log_directory}\n")
    print(render_markdown(results))


if __name__ == "__main__":
    main()


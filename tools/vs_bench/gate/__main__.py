# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Run the LitniceLM performance gate against a baseline results JSON.

Usage, from tools/:
    python3 -m vs_bench.gate --baseline <results.json>
        [--pytorch-reference <full-run results.json>]

Exit status 1 when any baseline cell's median steady-state throughput drops
by more than 0.5%.
"""

import argparse
from datetime import date
from pathlib import Path
import subprocess
import sys

from ..device.clock import LockedGpuClock
from ..engines.litnicelm import LitniceLMEngine
from ..measure.process import configure_raw_logs, raw_artifact_path
from ..run import RAW_LOG_ROOT, SPEC
from ..verticals.base import VerticalResult
from .compare import compare, load_litnicelm_rows, load_pytorch_references, new_cells
from .report import render_gate_report
from .runner import LitniceLMGateRun


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python3 -m vs_bench.gate")
    parser.add_argument(
        "--baseline", type=Path, required=True,
        help="gate or full-run results JSON of the previous commit",
    )
    parser.add_argument(
        "--pytorch-reference", type=Path,
        help="full-run results JSON with PyTorch rows (default: --baseline)",
    )
    return parser.parse_args()


def _working_tree_note() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    return "working tree has uncommitted changes" if status.stdout.strip() else "working tree clean"


def main() -> int:
    """Load the references, measure under the locked clock, compare, report."""
    arguments = _arguments()
    # Fail on unusable references before spending GPU time.
    baseline = load_litnicelm_rows(arguments.baseline)
    references = load_pytorch_references(arguments.pytorch_reference or arguments.baseline)

    configure_raw_logs(RAW_LOG_ROOT / date.today().isoformat())
    results_path = raw_artifact_path("gate_results.json")
    gate = LitniceLMGateRun(LitniceLMEngine())
    with LockedGpuClock(SPEC.environment.clock_mhz) as clock:
        rows = gate.run(SPEC)
        clock.verify_under_load((VerticalResult(vertical="train_speed", engines=rows),))
    gate.write_results(SPEC, rows, results_path)

    current = load_litnicelm_rows(results_path)
    comparisons = compare(current, baseline)
    print(f"\nBaseline: {arguments.baseline}")
    print(f"Gate results JSON: {results_path} ({_working_tree_note()})\n")
    print(render_gate_report(comparisons, references, new_cells(current, baseline)))
    return 0 if all(item.passed for item in comparisons) else 1


if __name__ == "__main__":
    sys.exit(main())

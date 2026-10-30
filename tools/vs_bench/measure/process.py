"""One process-launch boundary shared by every benchmark engine."""

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
import time

from .gpu import GpuSampleSummary, GpuSampler


@dataclass(frozen=True)
class ProcessMeasurement:
    """Captured child output, external wall time, and GPU samples."""

    output: str
    wall_seconds: float
    gpu: GpuSampleSummary


def run_sampled_process(
    command: list[str], cwd: Path, input_text: str | None = None
) -> ProcessMeasurement:
    """Launch one fresh child, sample its PID, and capture combined output."""
    print(f"\033[32m$ {shlex.join(command)}\033[0m", file=sys.stderr, flush=True)
    start = time.perf_counter()
    with GpuSampler() as sampler:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        sampler.process_id = process.pid
        output, _ = process.communicate(input=input_text)
        wall_seconds = time.perf_counter() - start
    if process.returncode != 0:
        raise RuntimeError(
            f"child failed ({process.returncode}): {shlex.join(command)}\n{output[-2000:]}"
        )
    return ProcessMeasurement(output=output, wall_seconds=wall_seconds, gpu=sampler.result())

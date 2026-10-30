"""One process-launch boundary shared by every benchmark engine."""

from dataclasses import dataclass
from datetime import datetime
import itertools
from pathlib import Path
import shlex
import subprocess
import sys
import time

from .gpu import GpuSampleSummary, GpuSampler


_raw_log_directory: Path | None = None
_log_sequence = itertools.count(1)
_request_sequence = itertools.count(1)
_log_prefix = "run"


def configure_raw_logs(directory: Path) -> None:
    """Send subsequent child-process output to uniquely numbered log files."""
    global _raw_log_directory, _log_sequence, _request_sequence, _log_prefix
    directory.mkdir(parents=True, exist_ok=True)
    _raw_log_directory = directory
    _log_sequence = itertools.count(1)
    _request_sequence = itertools.count(1)
    _log_prefix = datetime.now().strftime("run_%H%M%S")


def raw_artifact_path(suffix: str) -> Path:
    """Return a unique path belonging to the configured raw benchmark run."""
    if _raw_log_directory is None:
        raise RuntimeError("raw benchmark directory is not configured")
    return _raw_log_directory / f"{_log_prefix}_{suffix}"


def write_process_request(engine: str, request: str) -> Path:
    """Persist a rerunnable worker request beside the raw process logs."""
    path = raw_artifact_path(f"{next(_request_sequence):02d}_{engine}_request.json")
    path.write_text(request + "\n")
    return path


def _write_raw_log(command: list[str], output: str) -> Path | None:
    if _raw_log_directory is None:
        return None
    executable = Path(command[0]).name.replace(".", "_")
    path = _raw_log_directory / f"{_log_prefix}_{next(_log_sequence):02d}_{executable}.log"
    path.write_text(f"$ {shlex.join(command)}\n\n{output}")
    return path


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
    raw_log = _write_raw_log(command, output)
    if process.returncode != 0:
        location = f"\nraw log: {raw_log}" if raw_log else ""
        raise RuntimeError(
            f"child failed ({process.returncode}): {shlex.join(command)}\n"
            f"{output[-2000:]}{location}"
        )
    return ProcessMeasurement(output=output, wall_seconds=wall_seconds, gpu=sampler.result())

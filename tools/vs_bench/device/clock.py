"""Noninteractive NVIDIA GPU clock locking and under-load verification."""

from dataclasses import dataclass
import subprocess
from typing import Sequence

from ..verticals.base import VerticalResult


@dataclass(frozen=True)
class ClockState:
    """Observed GPU clock configuration and measured clock under load."""

    requested_mhz: int
    observed_mhz: int | None = None


class LockedGpuClock:
    """Lock one GPU clock with ``sudo -n`` and always restore defaults."""

    def __init__(self, clock_mhz: int) -> None:
        self.clock_mhz = clock_mhz
        self.state = ClockState(requested_mhz=clock_mhz)

    @staticmethod
    def _run_sudo(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["sudo", "-n", "/usr/bin/nvidia-smi", *arguments],
            capture_output=True,
            text=True,
            timeout=10,
        )

    @staticmethod
    def _query_clock() -> int:
        completed = subprocess.run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"cannot query GPU clock: {completed.stderr.strip()}")
        values = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
        if len(values) != 1:
            raise RuntimeError(f"expected one GPU clock value, got {values}")
        return values[0]

    def __enter__(self) -> "LockedGpuClock":
        try:
            completed = self._run_sudo(
                ["-lgc", f"{self.clock_mhz},{self.clock_mhz}"]
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("GPU clock lock timed out; sudo was not allowed to prompt") from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                "cannot lock GPU clock noninteractively. Allow only "
                "'/usr/bin/nvidia-smi' through sudoers, or run "
                f"'sudo /usr/bin/nvidia-smi -lgc {self.clock_mhz},{self.clock_mhz}' "
                f"yourself. sudo -n said: {detail}"
            )
        self.state = ClockState(self.clock_mhz, self._query_clock())
        print(
            f"GPU clock lock requested: {self.clock_mhz} MHz; "
            f"current clock after lock: {self.state.observed_mhz} MHz",
            flush=True,
        )
        return self

    def verify_under_load(self, results: Sequence[VerticalResult]) -> None:
        """Require every sampled training run to observe the requested clock."""
        clocks = [
            int(row.metrics["sm_clock_mhz"])
            for result in results
            if result.vertical == "train_speed"
            for row in result.engines
            if int(row.metrics.get("sm_clock_mhz", 0)) > 0
        ]
        if not clocks:
            raise RuntimeError("GPU clock verification failed: no under-load samples")
        mismatches = [clock for clock in clocks if abs(clock - self.clock_mhz) > 15]
        if mismatches:
            raise RuntimeError(
                f"GPU clock verification failed: requested {self.clock_mhz} MHz, "
                f"sampled means {clocks}"
            )
        print(
            f"GPU clock verified under load: {min(clocks)}–{max(clocks)} MHz "
            f"across {len(clocks)} runs",
            flush=True,
        )

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            completed = self._run_sudo(["-rgc"])
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("GPU clock reset timed out") from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"failed to reset GPU clock: {detail}")
        print("GPU clock reset to its default policy.", flush=True)


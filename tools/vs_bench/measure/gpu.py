"""Engine-independent GPU process sampling through ``nvidia-smi``."""

from dataclasses import dataclass
import statistics
import subprocess
import threading
import time


@dataclass(frozen=True)
class GpuSampleSummary:
    """Engine-independent process memory and clock samples."""

    process_peak_mib: int
    clock_samples_mhz: tuple[int, ...]

    @property
    def mean_clock_mhz(self) -> int:
        """Rounded mean SM clock, or zero when no sample was available."""
        return round(statistics.mean(self.clock_samples_mhz)) if self.clock_samples_mhz else 0


class GpuSampler:
    """Collect process memory and clock samples while an engine runs."""

    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_seconds = interval_ms / 1000
        self.process_id: int | None = None
        self._memory: list[int] = []
        self._clocks: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    @staticmethod
    def _query(arguments: list[str]) -> list[list[str]]:
        completed = subprocess.run(
            ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        return [line.split(", ") for line in completed.stdout.strip().splitlines() if line]

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            if self.process_id is not None:
                for process_id, used_memory in self._query(
                    ["--query-compute-apps=pid,used_memory"]
                ):
                    if int(process_id) == self.process_id:
                        self._memory.append(int(used_memory))
                for (clock,) in self._query(["--query-gpu=clocks.sm"]):
                    self._clocks.append(int(clock))
            time.sleep(self.interval_seconds)

    def __enter__(self) -> "GpuSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def result(self) -> GpuSampleSummary:
        """Summarize samples collected for the assigned process."""
        return GpuSampleSummary(
            process_peak_mib=max(self._memory, default=0),
            clock_samples_mhz=tuple(self._clocks),
        )


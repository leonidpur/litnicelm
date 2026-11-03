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

    @property
    def min_clock_mhz(self) -> int:
        """Minimum sampled SM clock, or zero without samples."""
        return min(self.clock_samples_mhz, default=0)

    @property
    def max_clock_mhz(self) -> int:
        """Maximum sampled SM clock, or zero without samples."""
        return max(self.clock_samples_mhz, default=0)

    def verify_clock(self, requested_mhz: int, context: str) -> None:
        """Require a stable lock while tolerating rare downward transitions."""
        if not self.clock_samples_mhz:
            raise RuntimeError(f"{context}: no under-load GPU clock samples")
        mismatches = [
            value
            for value in self.clock_samples_mhz
            if abs(value - requested_mhz) > 15
        ]
        mean = statistics.mean(self.clock_samples_mhz)
        outlier_fraction = len(mismatches) / len(self.clock_samples_mhz)
        boosted = self.max_clock_mhz > requested_mhz + 15
        deep_drop = self.min_clock_mhz < requested_mhz - 60
        unstable = outlier_fraction > 0.05 or abs(mean - requested_mhz) > 15
        if boosted or deep_drop or unstable:
            raise RuntimeError(
                f"{context}: GPU clock lock lost; requested {requested_mhz} MHz, "
                f"sampled range {self.min_clock_mhz}-{self.max_clock_mhz} MHz "
                f"with mean {mean:.1f} MHz "
                f"({len(mismatches)}/{len(self.clock_samples_mhz)} samples outside "
                "the ±15 MHz target band; maximum allowed fraction 5%)"
            )

    def clock_outlier_count(self, requested_mhz: int) -> int:
        """Count samples outside the narrow target band used in reports."""
        return sum(
            abs(value - requested_mhz) > 15
            for value in self.clock_samples_mhz
        )


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
                target_present = False
                for process_id, used_memory in self._query(
                    ["--query-compute-apps=pid,used_memory"]
                ):
                    if int(process_id) == self.process_id:
                        target_present = True
                        self._memory.append(int(used_memory))
                if target_present:
                    for clock, utilization in self._query(
                        ["--query-gpu=clocks.sm,utilization.gpu"]
                    ):
                        # Clock transitions while a CUDA context is merely
                        # resident are not under-load evidence. Verify only
                        # samples in which the GPU was doing work.
                        if int(utilization) > 0:
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


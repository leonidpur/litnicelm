"""GPU sampling boundary for later implementation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSampleSummary:
    """Engine-independent process memory and clock samples."""

    process_peak_mib: int
    clock_samples_mhz: tuple[int, ...]


class GpuSampler:
    """Collect process memory and clock samples while an engine runs."""

    def sample(self, process_id: int) -> GpuSampleSummary:
        raise NotImplementedError("GPU sampling is not part of skeleton step 1")


"""GPU clock locking boundary for later implementation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClockState:
    """Observed GPU clock configuration and measured clock under load."""

    requested_mhz: int
    observed_mhz: int | None = None


class LockedGpuClock:
    """Context manager that will lock, verify, and always reset GPU clocks."""

    def __init__(self, clock_mhz: int) -> None:
        self.clock_mhz = clock_mhz

    def __enter__(self) -> ClockState:
        raise NotImplementedError("GPU clock control is not part of skeleton step 1")

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        raise NotImplementedError("GPU clock control is not part of skeleton step 1")


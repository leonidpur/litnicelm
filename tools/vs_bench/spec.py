"""Engine-neutral specification for a comparison run."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonSpec:
    """All workload conditions shared by every engine and vertical."""

    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    corpus_id: str
    seq_len: int
    batch_size: int
    accumulation_steps: int
    steps: int
    precisions: tuple[str, ...]
    repeats: int
    clock_mhz: int

    @property
    def tokens_per_step(self) -> int:
        """Number of training tokens consumed by one optimizer step."""
        return self.seq_len * self.batch_size * self.accumulation_steps


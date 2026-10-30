"""Engine-neutral specification for a comparison run."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComparisonSpec:
    """All workload conditions shared by every engine and vertical.

    Precision names are engine-neutral. ``fp32`` means FP32 throughout;
    ``bf16`` means BF16 activations and activation gradients with FP32 master
    weights. Each engine adapter translates these names to its native setup.
    """

    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    position_encoding: str
    corpus_id: str
    tokenized_corpus_path: Path
    export_dir: Path
    parity_precision: str
    parity_tolerance: float
    seq_len: int
    batch_size: int
    validation_fraction: float
    validation_chunk_tokens: int
    window_stride: int
    accumulation_steps: int
    steps: int
    seed: int
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    grad_clip: float
    lr_schedule: str
    warmup_steps: int
    min_lr_ratio: float
    precisions: tuple[str, ...]
    repeats: int
    clock_mhz: int

    @property
    def tokens_per_step(self) -> int:
        """Number of training tokens consumed by one optimizer step."""
        return self.seq_len * self.batch_size * self.accumulation_steps


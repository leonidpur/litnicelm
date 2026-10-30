"""Engine-neutral specification for a comparison run."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkloadSpec:
    """Model, data, optimizer, and precision conditions shared by engines.

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
    seq_len: int
    batch_size: int
    validation_fraction: float
    validation_chunk_tokens: int
    window_stride: int
    accumulation_steps: int
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

    @property
    def tokens_per_step(self) -> int:
        """Number of training tokens consumed by one optimizer step."""
        return self.seq_len * self.batch_size * self.accumulation_steps


@dataclass(frozen=True)
class EnvironmentSpec:
    """Hardware and process environment shared by all verticals."""

    clock_mhz: int


@dataclass(frozen=True)
class ParitySettings:
    """Settings used only by the exported-model parity vertical."""

    export_dir: Path
    precision: str
    tolerance: float


@dataclass(frozen=True)
class TrainSpeedSettings:
    """Settings used only by the training-speed vertical."""

    steps: int
    repeats: int


@dataclass(frozen=True)
class ComparisonSpec:
    """Complete comparison grouped by scope rather than by engine."""

    workload: WorkloadSpec
    environment: EnvironmentSpec
    parity: ParitySettings
    train_speed: TrainSpeedSettings


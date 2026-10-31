"""LitniceLM configuration, subprocess, and profiler adapter."""

import re
import tempfile
from pathlib import Path
from types import MappingProxyType

from ..spec import ComparisonSpec
from ..measure.process import run_sampled_process
from .base import Engine, EngineResult, EngineVariant


REPOSITORY = Path(__file__).resolve().parents[3]

# Every LitniceLM YAML choice not supplied by ComparisonSpec lives here.
NATIVE_CONFIG = MappingProxyType(
    {
        "backend.library": "./build/liblitnice_backend_cublas_plugin.so",
        "model_algo.attention": "fused_inplace_multistream",
        "model_algo.ffn": "inplace_fused_bias_relu",
        "transformer_layers": 0,
        "parameter_bytes": 0,
        "optimizer_bytes": 0,
        "arena_alignment": 64,
        "max_steps": 1,
        "memory.alignment_bytes": 64,
        "training.incremental": False,
        "training.diagnostics": "all disabled",
        "reporting.verbose_init": True,
    }
)

# Adapter mechanics that do not belong in LitniceLM's YAML.
ADAPTER_PARAMS = MappingProxyType(
    {
        "binary": "build/litnicelm",
        "base_config": (
            "conf/local/doyle/"
            "doyle_clean_spm512_seq1024_l4_d256_b1_rope_val_cosine_e12_accum4.yaml"
        ),
        "benchmark.epochs": 2,
        "profiler.batch_step_pattern": r"^- batch_step total=(.+?) count=(\d+)",
    }
)


def _yaml_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        rendered = str(value)
        if "e" in rendered and "." not in rendered.split("e", 1)[0]:
            rendered = rendered.replace("e", ".0e", 1)
        return rendered
    if isinstance(value, Path):
        return f'"{value}"'
    return str(value)


def _set_key(text: str, section: str | None, key: str, value: object) -> str:
    """Set one key while preserving the simple YAML form LitniceLM consumes."""
    rendered = _yaml_value(value)
    if section is None:
        pattern = rf"^{re.escape(key)}:.*$"
        if re.search(pattern, text, re.MULTILINE):
            return re.sub(pattern, f"{key}: {rendered}", text, count=1, flags=re.MULTILINE)
        return f"{key}: {rendered}\n" + text
    block = re.search(
        rf"^{re.escape(section)}:\n((?:[ \t]+.*\n|\n)*)", text, re.MULTILINE
    )
    if block is None:
        return text + f"\n{section}:\n  {key}: {rendered}\n"
    body = block.group(1)
    pattern = rf"^([ \t]+){re.escape(key)}:.*$"
    if re.search(pattern, body, re.MULTILINE):
        body = re.sub(
            pattern, rf"\g<1>{key}: {rendered}", body, count=1, flags=re.MULTILINE
        )
    else:
        body = f"  {key}: {rendered}\n" + body
    return text[: block.start(1)] + body + text[block.end(1) :]


def _seconds(duration: str) -> float:
    """Parse the duration formats emitted by the LitniceLM profiler."""
    match = re.fullmatch(r"(?:(\d+)m )?([0-9.]+) ?(s|ms|us)", duration.strip())
    if not match:
        raise ValueError(f"unknown duration {duration!r}")
    value = float(match.group(2)) * {"s": 1.0, "ms": 1e-3, "us": 1e-6}[
        match.group(3)
    ]
    return value + 60 * int(match.group(1) or 0)


def _disable_diagnostics(text: str) -> str:
    """Set every key in the nested training.diagnostics block to false."""
    block = re.search(r"^  diagnostics:\n((?:    .*\n|\n)*)", text, re.MULTILINE)
    if block is None:
        raise ValueError("base config has no training.diagnostics block")
    body = re.sub(r"^(\s+[^:]+):.*$", r"\1: false", block.group(1), flags=re.MULTILINE)
    return text[: block.start(1)] + body + text[block.end(1) :]


class LitniceLMEngine(Engine):
    """Translate the comparison spec into YAML and drive the LitniceLM binary."""

    @staticmethod
    def _native_precision(precision: str) -> str:
        """Translate a shared precision name to the LitniceLM config value."""
        return {"fp32": "fp32", "bf16": "bf16_mixed"}[precision]

    @property
    def name(self) -> str:
        return "LitniceLM"

    @property
    def determines_step_count(self) -> bool:
        return True

    def variants(
        self, spec: ComparisonSpec, vertical: str, precision: str
    ) -> tuple[EngineVariant, ...]:
        del spec
        if vertical != "train_speed":
            return ()
        return (EngineVariant(precision, "per_batch", "none"),)

    def run_speed(
        self, spec: ComparisonSpec, variant: EngineVariant, steps: int
    ) -> EngineResult:
        del steps  # The unchanged LitniceLM path is epoch-driven.
        return self.train(spec, variant.precision)

    def run_differential(
        self,
        spec: ComparisonSpec,
        variant: EngineVariant,
        step_counts: tuple[int, int] | None,
    ) -> tuple[EngineResult, EngineResult]:
        """Measure two checkpoint-free, validation-free whole-epoch runs."""
        del step_counts
        low, high = spec.train_speed.differential_epochs
        return (
            self.train(
                spec,
                variant.precision,
                epochs=low,
                operation="dry_run",
                validation_fraction=0.0,
            ),
            self.train(
                spec,
                variant.precision,
                epochs=high,
                operation="dry_run",
                validation_fraction=0.0,
            ),
        )

    def describe_setup(self, spec: ComparisonSpec) -> tuple[str, ...]:
        precisions = (self._native_precision(value) for value in spec.workload.precisions)
        native = tuple(f"native config {key}: {value}" for key, value in NATIVE_CONFIG.items())
        adapter = tuple(
            f"adapter parameter {key}: {value}" for key, value in ADAPTER_PARAMS.items()
        )
        return (
            "generate a temporary LitniceLM YAML from the shared spec",
            "drive the LitniceLM binary as a subprocess",
            f"request precisions: {', '.join(precisions)}",
        ) + native + adapter

    @property
    def binary(self) -> Path:
        return REPOSITORY / str(ADAPTER_PARAMS["binary"])

    @property
    def base_config(self) -> Path:
        return REPOSITORY / str(ADAPTER_PARAMS["base_config"])

    def config_text(
        self,
        spec: ComparisonSpec,
        precision: str,
        workdir: Path,
        validation_fraction: float | None = None,
    ) -> str:
        """Translate the shared spec and native settings into LitniceLM YAML."""
        text = self.base_config.read_text()
        workload = spec.workload
        shared_values = (
            ("model", "max_seq_len", workload.seq_len),
            ("model", "n_layers", workload.n_layers),
            ("model", "n_heads", workload.n_heads),
            ("model", "d_model", workload.d_model),
            ("model", "d_ff", workload.d_ff),
            ("model_algo", "position_encoding", workload.position_encoding),
            ("tokenization", "output_binary", workload.tokenized_corpus_path),
            ("training", "train_seq_len", workload.seq_len),
            ("training", "window_stride", workload.window_stride),
            ("training", "batch_size", workload.batch_size),
            ("training", "batches_per_step", workload.accumulation_steps),
            ("training", "learning_rate", workload.learning_rate),
            ("training", "beta1", workload.beta1),
            ("training", "beta2", workload.beta2),
            ("training", "eps", workload.epsilon),
            ("training", "weight_decay", workload.weight_decay),
            ("training", "grad_clip", workload.grad_clip),
            ("training", "lr_schedule", workload.lr_schedule),
            ("training", "warmup_steps", workload.warmup_steps),
            ("training", "min_lr_ratio", workload.min_lr_ratio),
            ("training", "validation_fraction", workload.validation_fraction),
            ("training", "validation_chunk_tokens", workload.validation_chunk_tokens),
        )
        text = _set_key(text, None, "precision", self._native_precision(precision))
        for section, key, value in shared_values:
            text = _set_key(text, section, key, value)
        if validation_fraction is not None:
            text = _set_key(
                text, "training", "validation_fraction", validation_fraction
            )
        for dotted, value in NATIVE_CONFIG.items():
            if dotted == "training.diagnostics":
                text = _disable_diagnostics(text)
                continue
            section, separator, key = dotted.partition(".")
            text = _set_key(text, section if separator else None, key or section, value)
        paths = {
            "model_file_latest": workdir / "latest.ckpt",
            "model_file_best": workdir / "best.ckpt",
            "journal_file": workdir / "journal.txt",
        }
        for key, value in paths.items():
            text = _set_key(text, "paths", key, value)
        return text

    def write_config(
        self,
        spec: ComparisonSpec,
        precision: str,
        workdir: Path,
        validation_fraction: float | None = None,
    ) -> Path:
        """Write one temporary native configuration and return its path."""
        path = workdir / f"litnicelm_{self._native_precision(precision)}.yaml"
        path.write_text(
            self.config_text(spec, precision, workdir, validation_fraction)
        )
        return path

    def parse_training_output(
        self, spec: ComparisonSpec, precision: str, output: str
    ) -> EngineResult:
        """Parse the same profiler and managed-memory fields as the old harness."""
        batch_lines = re.findall(
            str(ADAPTER_PARAMS["profiler.batch_step_pattern"]), output, re.MULTILINE
        )
        if len(batch_lines) != 1:
            raise ValueError(
                "expected exactly one top-level batch_step profiler line, "
                f"found {len(batch_lines)}"
            )
        total_seconds = _seconds(batch_lines[0][0])
        steps = int(batch_lines[0][1])
        managed = re.search(r"\| Total managed\s*\|\s*([0-9.]+)", output)
        losses = re.findall(
            r"Validation epoch \d+:.*?\| train ([0-9.eE+-]+),", output
        )
        tokens = steps * spec.workload.tokens_per_step
        return EngineResult(
            engine=self.name,
            operation="train_speed",
            metrics={
                "precision": precision,
                "compile_mode": "none",
                "sync": "per batch",
                "steps": steps,
                "tokens_per_step": spec.workload.tokens_per_step,
                "tokens_per_s": tokens / total_seconds,
                "step_ms_mean": 1000 * total_seconds / steps,
                "engine_memory_mib": float(managed.group(1)) if managed else "unknown",
                "n_layers": spec.workload.n_layers,
                "n_heads": spec.workload.n_heads,
                "d_model": spec.workload.d_model,
                "d_ff": spec.workload.d_ff,
                **({"final_train_loss": float(losses[-1])} if losses else {}),
            },
        )

    def train(
        self,
        spec: ComparisonSpec,
        precision: str,
        epochs: int | None = None,
        operation: str = "train",
        validation_fraction: float | None = None,
    ) -> EngineResult:
        """Run the old epoch-driven LitniceLM benchmark path once."""
        with tempfile.TemporaryDirectory(prefix="litnicelm_vs_bench_") as temporary:
            config = self.write_config(
                spec, precision, Path(temporary), validation_fraction
            )
            command = [
                str(self.binary),
                operation,
                "--config",
                str(config),
                "--epochs",
                str(epochs or ADAPTER_PARAMS["benchmark.epochs"]),
            ]
            measured = run_sampled_process(command, REPOSITORY)
            result = self.parse_training_output(spec, precision, measured.output)
            return EngineResult(
                engine=result.engine,
                operation=result.operation,
                metrics={
                    **result.metrics,
                    "process_wall_seconds": measured.wall_seconds,
                    "process_peak_mib": measured.gpu.process_peak_mib,
                    "sm_clock_mhz": measured.gpu.mean_clock_mhz,
                    "epochs": epochs or ADAPTER_PARAMS["benchmark.epochs"],
                    "validation_enabled": validation_fraction != 0.0,
                    "checkpointing_enabled": operation != "dry_run",
                "compile_warmup_seconds": 0.0,
                },
            )

    def run(
        self, spec: ComparisonSpec, operation: str, variant: EngineVariant | None = None
    ) -> EngineResult:
        del variant
        if operation != "train_speed":
            raise NotImplementedError(f"LitniceLM operation is not implemented: {operation}")
        return self.train(spec, spec.workload.precisions[0])


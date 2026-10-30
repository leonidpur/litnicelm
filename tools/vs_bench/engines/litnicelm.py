"""LitniceLM configuration, subprocess, and profiler adapter."""

import re
import subprocess
import tempfile
from pathlib import Path
from types import MappingProxyType

from ..spec import ComparisonSpec
from .base import Engine, EngineResult


REPOSITORY = Path(__file__).resolve().parents[3]

# Every LitniceLM-specific choice not supplied by ComparisonSpec lives here.
NATIVE_SETTINGS = MappingProxyType(
    {
        "binary": "build/litnicelm",
        "base_config": (
            "conf/local/doyle/"
            "doyle_clean_spm512_seq1024_l4_d256_b1_rope_val_cosine_e12_accum4.yaml"
        ),
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
        "benchmark.epochs": 2,
        "profiler.batch_step": "top-level batch_step total/count",
    }
)


def _yaml_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
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


class LitniceLMEngine(Engine):
    """Translate the comparison spec into YAML and drive the LitniceLM binary."""

    @staticmethod
    def _native_precision(precision: str) -> str:
        """Translate a shared precision name to the LitniceLM config value."""
        return {"fp32": "fp32", "bf16": "bf16_mixed"}[precision]

    @property
    def name(self) -> str:
        return "LitniceLM"

    def describe_setup(self, spec: ComparisonSpec) -> tuple[str, ...]:
        precisions = (self._native_precision(value) for value in spec.precisions)
        settings = tuple(f"native setting {key}: {value}" for key, value in NATIVE_SETTINGS.items())
        return (
            "generate a temporary LitniceLM YAML from the shared spec",
            "drive the LitniceLM binary as a subprocess",
            f"request precisions: {', '.join(precisions)}",
        ) + settings

    @property
    def binary(self) -> Path:
        return REPOSITORY / str(NATIVE_SETTINGS["binary"])

    @property
    def base_config(self) -> Path:
        return REPOSITORY / str(NATIVE_SETTINGS["base_config"])

    def config_text(self, spec: ComparisonSpec, precision: str, workdir: Path) -> str:
        """Translate the shared spec and native settings into LitniceLM YAML."""
        text = self.base_config.read_text()
        shared_values = (
            ("model", "max_seq_len", spec.seq_len),
            ("model", "n_layers", spec.n_layers),
            ("model", "n_heads", spec.n_heads),
            ("model", "d_model", spec.d_model),
            ("model", "d_ff", spec.d_ff),
            ("model_algo", "position_encoding", spec.position_encoding),
            ("tokenization", "output_binary", spec.tokenized_corpus_path),
            ("training", "train_seq_len", spec.seq_len),
            ("training", "batch_size", spec.batch_size),
            ("training", "batches_per_step", spec.accumulation_steps),
            ("training", "validation_fraction", spec.validation_fraction),
            ("training", "validation_chunk_tokens", spec.validation_chunk_tokens),
        )
        text = _set_key(text, None, "precision", self._native_precision(precision))
        for section, key, value in shared_values:
            text = _set_key(text, section, key, value)
        for dotted, value in NATIVE_SETTINGS.items():
            if dotted in {"binary", "base_config", "benchmark.epochs", "profiler.batch_step"}:
                continue
            if dotted == "training.diagnostics":
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

    def write_config(self, spec: ComparisonSpec, precision: str, workdir: Path) -> Path:
        """Write one temporary native configuration and return its path."""
        path = workdir / f"litnicelm_{self._native_precision(precision)}.yaml"
        path.write_text(self.config_text(spec, precision, workdir))
        return path

    def parse_training_output(
        self, spec: ComparisonSpec, precision: str, output: str
    ) -> EngineResult:
        """Parse the same profiler and managed-memory fields as the old harness."""
        batch_lines = re.findall(r"^\s*- batch_step total=(.+?) count=(\d+)", output, re.MULTILINE)
        total_seconds = _seconds(batch_lines[0][0])
        batches = sum(int(count) for _, count in batch_lines)
        managed = re.search(r"\| Total managed\s*\|\s*([0-9.]+)", output)
        tokens = batches * spec.batch_size * spec.seq_len
        steps = batches // spec.accumulation_steps
        return EngineResult(
            engine=self.name,
            operation="train_speed",
            metrics={
                "precision": precision,
                "sync": "per batch",
                "steps": steps,
                "tokens_per_step": spec.tokens_per_step,
                "tokens_per_s": tokens / total_seconds,
                "step_ms_mean": 1000 * total_seconds / steps,
                "engine_memory_mib": float(managed.group(1)) if managed else "unknown",
            },
        )

    def train(self, spec: ComparisonSpec, precision: str) -> EngineResult:
        """Run the old epoch-driven LitniceLM benchmark path once."""
        with tempfile.TemporaryDirectory(prefix="litnicelm_vs_bench_") as temporary:
            config = self.write_config(spec, precision, Path(temporary))
            command = [
                str(self.binary),
                "train",
                "--config",
                str(config),
                "--epochs",
                str(NATIVE_SETTINGS["benchmark.epochs"]),
            ]
            completed = subprocess.run(
                command,
                cwd=REPOSITORY,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"LitniceLM failed ({completed.returncode}): {' '.join(command)}\n"
                    f"{completed.stdout[-2000:]}"
                )
            return self.parse_training_output(spec, precision, completed.stdout)

    def run(self, spec: ComparisonSpec, operation: str) -> EngineResult:
        if operation != "train_speed":
            raise NotImplementedError(f"LitniceLM operation is not implemented: {operation}")
        return self.train(spec, spec.precisions[0])


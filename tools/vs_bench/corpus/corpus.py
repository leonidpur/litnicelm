"""LitniceLM tokenized-corpus discovery and validation windows."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..spec import ComparisonSpec


DATASET_MAGIC = 0x4C4E4750  # "LNGP"
DATASET_HEADER_WORDS = 3  # magic, version, vocabulary size


@dataclass(frozen=True)
class Corpus:
    """A resolved corpus artifact with a stable identifier and checksum."""

    corpus_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class TokenizedCorpus:
    """A verified corpus artifact and its vocabulary and token IDs."""

    artifact: Corpus
    vocab_size: int
    tokens: NDArray[np.int64]


class CorpusManager:
    """Locate and verify corpora; later download and encode missing artifacts."""

    def locate(self, spec: ComparisonSpec) -> Corpus:
        """Resolve the spec's local corpus and calculate its content checksum."""
        path = spec.tokenized_corpus_path
        if not path.is_file():
            raise FileNotFoundError(f"corpus {spec.corpus_id!r} not found: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return Corpus(corpus_id=spec.corpus_id, path=path, sha256=digest)

    def load(self, spec: ComparisonSpec) -> TokenizedCorpus:
        """Read the LitniceLM u32 dataset named by the shared spec."""
        artifact = self.locate(spec)
        raw = np.fromfile(artifact.path, dtype=np.uint32)
        if raw.size < DATASET_HEADER_WORDS or raw[0] != DATASET_MAGIC:
            raise ValueError(f"{artifact.path}: not a LitniceLM dataset (magic LNGP)")
        return TokenizedCorpus(
            artifact=artifact,
            vocab_size=int(raw[2]),
            tokens=raw[DATASET_HEADER_WORDS:].astype(np.int64),
        )


def validation_windows(num_tokens: int, spec: ComparisonSpec) -> list[int]:
    """Reproduce LitniceLM's held-out chunk and validation-window selection."""
    group = int(round(1.0 / spec.validation_fraction))
    num_chunks = (num_tokens + spec.validation_chunk_tokens - 1) // spec.validation_chunk_tokens
    starts = []
    for chunk_index in range(group - 1, num_chunks, group):
        begin = chunk_index * spec.validation_chunk_tokens
        end = min(begin + spec.validation_chunk_tokens, num_tokens)
        start = begin
        while start + spec.seq_len + 1 <= end:
            starts.append(start)
            start += spec.seq_len
    return starts[: len(starts) // spec.batch_size * spec.batch_size]


def window_batch(
    tokens: NDArray[np.int64], starts: list[int], spec: ComparisonSpec
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Materialize input IDs and next-token targets for the selected windows."""
    indices = np.asarray(starts)[:, None] + np.arange(spec.seq_len + 1)[None, :]
    windows = tokens[indices]
    return windows[:, :-1], windows[:, 1:]


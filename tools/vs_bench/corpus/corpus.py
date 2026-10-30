"""Corpus management boundary for later implementation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Corpus:
    """A resolved corpus artifact with a stable identifier and checksum."""

    corpus_id: str
    path: Path
    sha256: str


class CorpusManager:
    """Locate and verify corpora; later download and encode missing artifacts."""

    def locate(self, corpus_id: str) -> Corpus:
        raise NotImplementedError("corpus resolution is not part of skeleton step 1")


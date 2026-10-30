"""Compact JSON result reporting boundary for later implementation."""

from pathlib import Path
from typing import Sequence

from ..verticals.base import VerticalResult


def write_json(path: Path, results: Sequence[VerticalResult]) -> None:
    """Write compact structured results once execution is implemented."""
    raise NotImplementedError("JSON result writing is not part of skeleton step 1")


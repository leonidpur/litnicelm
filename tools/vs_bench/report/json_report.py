"""Compact, reproducible JSON result reporting."""

from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence

from ..spec import ComparisonSpec
from ..verticals.base import VerticalResult


def _command_output(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def write_json(
    path: Path,
    spec: ComparisonSpec,
    results: Sequence[VerticalResult],
    raw_log_directory: Path,
) -> None:
    """Write the spec, environment metadata, and every structured result row."""
    payload = {
        "spec": _json_value(asdict(spec)),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
            "gpu": _command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ]
            ),
            "raw_logs": str(raw_log_directory),
        },
        "verticals": [
            {
                "name": result.vertical,
                "rows": [
                    {
                        "engine": row.engine,
                        "operation": row.operation,
                        **row.metrics,
                    }
                    for row in result.engines
                ],
            }
            for result in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


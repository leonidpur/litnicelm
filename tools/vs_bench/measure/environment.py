# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Software versions and host/GPU power state recorded with every result."""

from functools import lru_cache
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import subprocess

REPOSITORY = Path(__file__).resolve().parents[3]
UNAVAILABLE = "unavailable"


def _command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return UNAVAILABLE
    return completed.stdout.strip() if completed.returncode == 0 else UNAVAILABLE


def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return UNAVAILABLE


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return UNAVAILABLE


def _header_version(header: Path, prefix: str) -> str:
    """MAJOR.MINOR.PATCH from ``#define <prefix>_{MAJOR,MINOR,PATCH}...`` lines."""
    try:
        text = header.read_text()
    except OSError:
        return UNAVAILABLE
    parts = []
    for part in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"#define {prefix}{part}(?:LEVEL|_VERSION)?\s+(\d+)", text)
        if match is None:
            return UNAVAILABLE
        parts.append(match.group(1))
    return ".".join(parts)


def _cmake_cache(build_directory: Path) -> dict[str, str]:
    try:
        lines = (build_directory / "CMakeCache.txt").read_text().splitlines()
    except OSError:
        return {}
    entries = {}
    for line in lines:
        match = re.match(r"([A-Za-z0-9_]+):[A-Z]+=(.*)", line)
        if match:
            entries[match.group(1)] = match.group(2)
    return entries


def _torch_versions() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"torch": UNAVAILABLE}
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "triton": _package_version("triton"),
        "nvidia_cudnn_wheel": _package_version("nvidia-cudnn-cu13"),
        "nvidia_cublas_wheel": _package_version("nvidia-cublas"),
    }


def _litnicelm_build_versions() -> dict[str, object]:
    cache = _cmake_cache(REPOSITORY / "build")
    nvcc = cache.get("CMAKE_CUDA_COMPILER", "nvcc")
    nvcc_release = re.search(r"release ([\d.]+)", _command_output([nvcc, "--version"]))
    cudnn_include = Path(cache.get("CUDNN_INCLUDE_DIR", "/nonexistent"))
    frontend_include = Path(cache.get("CUDNN_FRONTEND_INCLUDE_DIR", "/nonexistent"))
    return {
        "build_type": cache.get("CMAKE_BUILD_TYPE", UNAVAILABLE),
        "nvcc": nvcc_release.group(1) if nvcc_release else UNAVAILABLE,
        "cudnn": _header_version(cudnn_include / "cudnn_version.h", "CUDNN_"),
        "cudnn_frontend": _header_version(
            frontend_include / "cudnn_frontend_version.h", "CUDNN_FRONTEND_"
        ),
    }


@lru_cache(maxsize=1)
def software_versions() -> dict[str, object]:
    """Versions that can change a measurement without a code change."""
    cpu_model = next(
        (
            line.split(":", 1)[1].strip()
            for line in _read("/proc/cpuinfo").splitlines()
            if line.startswith("model name")
        ),
        UNAVAILABLE,
    )
    return {
        "kernel": platform.release(),
        "cpu": cpu_model,
        "nvidia_driver": _command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "pytorch": _torch_versions(),
        "litnicelm_build": _litnicelm_build_versions(),
    }


def inductor_cache_state() -> dict[str, object]:
    """Location, entry count and newest entry of the TorchInductor cache."""
    directory = Path(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        or f"/tmp/torchinductor_{os.environ.get('USER', 'unknown')}"
    )
    try:
        entries = list(directory.iterdir())
    except OSError:
        return {"directory": str(directory), "entries": 0}
    newest = max((entry.stat().st_mtime for entry in entries), default=None)
    return {
        "directory": str(directory),
        "entries": len(entries),
        "newest_entry_unix": round(newest) if newest is not None else None,
    }


def host_power_state() -> dict[str, object]:
    """AC/battery, platform profile and CPU frequency policy, right now."""
    supplies = Path("/sys/class/power_supply")
    ac_online = UNAVAILABLE
    for supply in sorted(supplies.glob("*")) if supplies.exists() else ():
        if _read(str(supply / "type")) == "Mains":
            ac_online = _read(str(supply / "online")) == "1"
            break
    frequencies = [
        float(line.split(":", 1)[1])
        for line in _read("/proc/cpuinfo").splitlines()
        if line.startswith("cpu MHz")
    ]
    cpufreq = "/sys/devices/system/cpu/cpu0/cpufreq"
    return {
        "ac_online": ac_online,
        "platform_profile": _read("/sys/firmware/acpi/platform_profile"),
        "cpu_governor": _read(f"{cpufreq}/scaling_governor"),
        "cpu_energy_performance_preference": _read(
            f"{cpufreq}/energy_performance_preference"
        ),
        "cpu_mhz_mean": round(sum(frequencies) / len(frequencies)) if frequencies else 0,
        "cpu_mhz_max": round(max(frequencies, default=0)),
    }


GPU_STATE_FIELDS = (
    "pstate",
    "clocks.sm",
    "clocks.mem",
    "power.draw",
    "enforced.power.limit",
    "power.default_limit",
    "temperature.gpu",
    "clocks_event_reasons.active",
)


def gpu_power_state() -> dict[str, str]:
    """``nvidia-smi`` clocks, power limit and performance state, right now."""
    output = _command_output(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_STATE_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if output == UNAVAILABLE:
        return {}
    values = output.splitlines()[0].split(", ")
    return dict(zip(GPU_STATE_FIELDS, values))


def power_state() -> dict[str, object]:
    """Host and GPU power state plus Inductor cache state, right now."""
    return {
        "host": host_power_state(),
        "gpu": gpu_power_state(),
        "inductor_cache": inductor_cache_state(),
    }

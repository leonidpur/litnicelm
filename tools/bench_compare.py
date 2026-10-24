#!/usr/bin/env python3
"""LitniceLM vs PyTorch training speed and memory on one LitniceLM config.

Runs, interleaved and repeated:
  litnicelm  fp32 and bf16_mixed: `litnicelm train` for --epochs epochs on a
             copy of the config (checkpoints and journal in a temporary
             directory); throughput from the run's profiler (all training
             batches, without validation and checkpoint saving);
  pytorch    fp32 and bf16 autocast, with per-batch loss reads like
             LitniceLM, and again with --no-sync (tools/bench_torch.py), for
             the same number of optimizer steps.
During every run nvidia-smi samples the process's GPU memory and the SM
clock (clocks are not locked: the clock column shows the state each run
ran in). Reports the median over repeats, writes all runs as JSON and
prints a Markdown table.

Example:
  tools/bench_compare.py --config conf/local/doyle/run.yaml --repeats 3 \\
      --json docs/comparison_results.json
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import yaml

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent


class GpuSampler:
    """Samples a process's GPU memory (MiB) and the SM clock (MHz)."""

    def __init__(self, interval_ms=100):
        self.interval = interval_ms / 1000
        self.pid = None
        self.memory = []
        self.clocks = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _query(self, args):
        out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return [line.split(", ") for line in out.strip().splitlines() if line]

    def _run(self):
        while not self._stop.is_set():
            if self.pid is not None:
                for pid, used in self._query(["--query-compute-apps=pid,used_memory"]):
                    if int(pid) == self.pid:
                        self.memory.append(int(used))
                for (clock,) in self._query(["--query-gpu=clocks.sm"]):
                    self.clocks.append(int(clock))
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def result(self):
        return {"process_peak_mib": max(self.memory, default=0),
                "sm_clock_mhz": round(statistics.mean(self.clocks)) if self.clocks else 0}


def run_sampled(cmd, cwd=REPO):
    """Runs cmd while sampling its GPU use; returns (stdout+stderr, samples)."""
    with GpuSampler() as sampler:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        sampler.pid = proc.pid
        out, _ = proc.communicate()
    if proc.returncode != 0:
        sys.exit(f"failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{out[-2000:]}")
    return out, sampler.result()


def seconds(text):
    """'50.057 s', '845.2 ms' or '1m 2.345s' -> seconds."""
    m = re.fullmatch(r"(?:(\d+)m )?([0-9.]+) ?(s|ms|us)", text.strip())
    if not m:
        raise ValueError(f"unknown duration {text!r}")
    value = float(m.group(2)) * {"s": 1.0, "ms": 1e-3, "us": 1e-6}[m.group(3)]
    return value + 60 * int(m.group(1) or 0)


def set_key(text, section, key, value):
    """The config text with section.key (or top-level key when section is
    None) set to value. LitniceLM's YAML reader takes only the flat form its
    configs use, so the text is edited instead of re-dumped."""
    if section is None:
        pattern = rf"^{re.escape(key)}:.*$"
        if re.search(pattern, text, re.M):
            return re.sub(pattern, f"{key}: {value}", text, count=1, flags=re.M)
        return f"{key}: {value}\n" + text
    block = re.search(rf"^{re.escape(section)}:\n((?:[ \t]+.*\n|\n)*)", text, re.M)
    if block is None:
        return text + f"\n{section}:\n  {key}: {value}\n"
    body = block.group(1)
    pattern = rf"^([ \t]+){re.escape(key)}:.*$"
    if re.search(pattern, body, re.M):
        body = re.sub(pattern, rf"\g<1>{key}: {value}", body, count=1, flags=re.M)
    else:
        body = f"  {key}: {value}\n" + body
    return text[:block.start(1)] + body + text[block.end(1):]


def litnicelm_run(binary, config, cfg_yaml, precision, epochs, workdir):
    text = config.read_text()
    text = set_key(text, None, "precision", precision)
    text = set_key(text, "training", "incremental", "false")
    text = set_key(text, "reporting", "verbose_init", "true")
    text = set_key(text, "paths", "model_file_latest", f'"{workdir / "latest.ckpt"}"')
    text = set_key(text, "paths", "model_file_best", f'"{workdir / "best.ckpt"}"')
    text = set_key(text, "paths", "journal_file", f'"{workdir / "journal.txt"}"')
    path = workdir / f"litnicelm_{precision}.yaml"
    path.write_text(text)
    out, gpu = run_sampled([binary, "train", "--config", str(path), "--epochs", str(epochs)])
    # Top-level batch_step: every training batch (nested lines are its parts).
    batch_lines = re.findall(r"^\s*- batch_step total=(.+?) count=(\d+)", out, re.M)
    total_s = seconds(batch_lines[0][0])
    batches = sum(int(count) for _, count in batch_lines)
    managed = re.search(r"\| Total managed\s*\|\s*([0-9.]+)", out)
    t = cfg_yaml["training"]
    tokens = batches * t["batch_size"] * t["train_seq_len"]
    steps = batches // t.get("batches_per_step", 1)
    return {"engine": "litnicelm", "precision": precision, "sync": "per batch",
            "steps": steps, "tokens_per_s": tokens / total_s,
            "step_ms_mean": 1000 * total_s / steps,
            "engine_memory_mib": float(managed.group(1)) if managed else None, **gpu}


def torch_run(python, config, precision, steps, no_sync):
    cmd = [python, str(TOOLS / "bench_torch.py"), "--config", str(config),
           "--precision", precision, "--steps", str(steps)]
    if no_sync:
        cmd.append("--no-sync")
    out, gpu = run_sampled(cmd)
    r = json.loads(out.strip().splitlines()[-1])
    r["step_ms_mean"] = 1000 * r["tokens_per_step"] / r["tokens_per_s"]
    r["engine_memory_mib"] = r["peak_allocated_mib"]
    return {**r, **gpu}


def table(runs):
    groups = {}
    for r in runs:
        groups.setdefault((r["engine"], r["precision"], r["sync"]), []).append(r)
    lines = ["| engine | precision | loss read | tokens/s | step (ms) | process peak (MiB) | "
             "engine memory (MiB) | SM clock (MHz) |",
             "|---|---|---|---|---|---|---|---|"]
    for (engine, precision, sync), rs in groups.items():
        med = lambda k: statistics.median(r[k] for r in rs)  # noqa: E731
        tps = [r["tokens_per_s"] for r in rs]
        lines.append(f"| {engine} | {precision} | {sync} | {med('tokens_per_s'):,.0f} "
                     f"({min(tps):,.0f}–{max(tps):,.0f}) | {med('step_ms_mean'):.1f} | "
                     f"{med('process_peak_mib'):.0f} | {med('engine_memory_mib'):.0f} | "
                     f"{med('sm_clock_mhz'):.0f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="LitniceLM training config")
    parser.add_argument("--litnicelm", default=str(REPO / "build" / "litnicelm"))
    parser.add_argument("--python", default=sys.executable, help="python with torch")
    parser.add_argument("--epochs", type=int, default=2, help="LitniceLM epochs per run")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path, help="write every run here")
    args = parser.parse_args()

    cfg_yaml = yaml.safe_load(args.config.read_text())
    runs = []
    with tempfile.TemporaryDirectory(prefix="litnicelm_bench_") as tmp:
        workdir = Path(tmp)
        steps = None
        for rep in range(args.repeats):
            for precision in ("fp32", "bf16_mixed"):
                r = litnicelm_run(args.litnicelm, args.config, cfg_yaml, precision, args.epochs, workdir)
                steps = r["steps"]
                runs.append(r)
                print(f"[{rep + 1}/{args.repeats}] litnicelm {precision}: "
                      f"{r['tokens_per_s']:,.0f} tokens/s", file=sys.stderr)
            for precision in ("fp32", "bf16"):
                for no_sync in (False, True):
                    r = torch_run(args.python, args.config, precision, steps, no_sync)
                    runs.append(r)
                    print(f"[{rep + 1}/{args.repeats}] pytorch {precision} "
                          f"({r['sync']}): {r['tokens_per_s']:,.0f} tokens/s", file=sys.stderr)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"config": str(args.config), "runs": runs}, indent=1))
    print(table(runs))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def run_command(command, cwd, log_path):
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command}\n\n")
        log.flush()
        proc = subprocess.run(
            shlex.split(command),
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.monotonic() - started
    return proc.returncode, elapsed


def parse_training_log(log_path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    best = None
    final_epoch = None
    final_loss = None
    for match in re.finditer(r"New best loss\s+([0-9.]+)\s+at epoch\s+(\d+)", text):
        best = (int(match.group(2)), float(match.group(1)))
    for match in re.finditer(r"Epoch\s+(\d+)\s+mean_loss=([0-9.eE+-]+)", text):
        final_epoch = int(match.group(1))
        final_loss = float(match.group(2))
    return best, final_epoch, final_loss


def parse_inference_log(log_path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"\[INFERENCE\]\[TOKEN_GEN\] step=0 val=0\.000000 (.*)", text)
    return matches[-1] if matches else ""


def main():
    parser = argparse.ArgumentParser(description="Run all syntetic1 staged experiments.")
    parser.add_argument("--manifest", default="/mnt/ext_ssd/litnicegpt/syntetic1_stages/manifest.tsv")
    parser.add_argument("--repo", default="/home/leonastu/projects/litnicegpt")
    parser.add_argument("--runs-dir", default="/mnt/ext_ssd/litnicegpt/syntetic1_stages/runs")
    parser.add_argument("--prompt", default="m n ")
    parser.add_argument("--limit", type=int, default=0, help="0 means all stages")
    args = parser.parse_args()

    repo = Path(args.repo)
    manifest = Path(args.manifest)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = runs_dir / "summary.tsv"
    with manifest.open("r", encoding="utf-8", newline="") as mf, \
            summary_path.open("w", encoding="utf-8", newline="") as sf:
        reader = csv.DictReader(mf, delimiter="\t")
        writer = csv.writer(sf, delimiter="\t")
        writer.writerow([
            "stage", "tokens", "tokenizer_s", "tokenize_s", "train_s",
            "best_epoch", "best_loss", "final_epoch", "final_loss",
            "inference"
        ])
        for idx, row in enumerate(reader, start=1):
            if args.limit and idx > args.limit:
                break

            stage = row["stage"]
            config = row["config"]
            print(f"[{idx}] {stage}: tokenizer", flush=True)
            tokenizer_log = runs_dir / f"{stage}_tokenizer.log"
            rc, tokenizer_s = run_command(row["tokenizer_cmd"], repo, tokenizer_log)
            if rc != 0:
                raise RuntimeError(f"{stage}: tokenizer failed, see {tokenizer_log}")

            print(f"[{idx}] {stage}: tokenize", flush=True)
            tokenize_log = runs_dir / f"{stage}_tokenize.log"
            rc, tokenize_s = run_command(row["tokenize_cmd"], repo, tokenize_log)
            if rc != 0:
                raise RuntimeError(f"{stage}: tokenize failed, see {tokenize_log}")

            print(f"[{idx}] {stage}: train", flush=True)
            train_log = runs_dir / f"{stage}_train.log"
            rc, train_s = run_command(row["train_cmd"], repo, train_log)
            if rc != 0:
                raise RuntimeError(f"{stage}: train failed, see {train_log}")

            print(f"[{idx}] {stage}: infer", flush=True)
            infer_log = runs_dir / f"{stage}_infer.log"
            infer_cmd = f'./build/litnicegpt infer --config {config} --temp 0 "{args.prompt}"'
            rc, _ = run_command(infer_cmd, repo, infer_log)
            if rc != 0:
                raise RuntimeError(f"{stage}: infer failed, see {infer_log}")

            best, final_epoch, final_loss = parse_training_log(train_log)
            best_epoch = best[0] if best else ""
            best_loss = best[1] if best else ""
            inference = parse_inference_log(infer_log)
            writer.writerow([
                stage,
                row["tokens"],
                f"{tokenizer_s:.3f}",
                f"{tokenize_s:.3f}",
                f"{train_s:.3f}",
                best_epoch,
                best_loss,
                final_epoch if final_epoch is not None else "",
                final_loss if final_loss is not None else "",
                inference,
            ])
            sf.flush()
            print(f"[{idx}] {stage}: done", flush=True)

    print(f"summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        sys.exit(1)

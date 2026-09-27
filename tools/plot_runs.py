#!/usr/bin/env python3
"""Compare training runs written by the "runs" experiment-tracking plugin.

Resumed runs are stitched to their parents (meta.json parent_run_id), so a
run trained in several sessions shows as one curve, named after its latest
run. Prints a table of the metric per epoch; with --plot, also draws it
(needs matplotlib).

Examples:
  tools/plot_runs.py runs                          # every run under runs/
  tools/plot_runs.py runs/2026..._learned runs/2026..._rope
  tools/plot_runs.py runs --metric train_loss --plot loss.png
  tools/plot_runs.py runs --x step --plot steps.png   # per-step metrics
"""

import argparse
import json
import sys
from pathlib import Path


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_runs(paths):
    """run_id -> {"dir", "meta", "metrics"} for every training run found."""
    runs = {}
    for p in paths:
        p = Path(p)
        candidates = [p] if (p / "meta.json").exists() else sorted(p.glob("*/"))
        for d in candidates:
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            if meta.get("type") != "run_start":
                continue  # tokenizer operations have no metrics
            runs[meta["run_id"]] = {
                "dir": d,
                "meta": meta,
                "metrics": read_jsonl(d / "metrics.jsonl"),
            }
    return runs


def chains(runs):
    """Lists of runs from the first session to the latest, one per model."""
    children = {r["meta"].get("parent_run_id") for r in runs.values()}
    leaves = [rid for rid in runs if rid not in children]
    result = []
    for leaf in sorted(leaves):
        chain = []
        rid = leaf
        while rid in runs:
            chain.append(runs[rid])
            rid = runs[rid]["meta"].get("parent_run_id")
        result.append(list(reversed(chain)))
    return result


def series(chain, metric, x_axis):
    """[(x, value)] for one stitched chain; later sessions win on overlap."""
    name = "epoch" if x_axis == "epoch" else "step"
    points = {}
    for run in chain:
        for m in run["metrics"]:
            if m.get("name") == name and m.get(metric) is not None:
                points[m[x_axis]] = m[metric]
    return sorted(points.items())


def label(chain):
    meta = chain[-1]["meta"]
    pe = meta.get("position_encoding", "?")
    sessions = f", {len(chain)} sessions" if len(chain) > 1 else ""
    return f"{meta['run_id']} [{pe}{sessions}]"


def print_table(named, metric, x_axis):
    xs = sorted({x for _, pts in named for x, _ in pts})
    if not xs:
        print(f"No '{metric}' metrics by {x_axis} found.")
        return
    print(f"{metric} by {x_axis}")
    for i, (name, _) in enumerate(named):
        print(f"  [{i}] {name}")
    header = f"{x_axis:>8} " + " ".join(f"{'[' + str(i) + ']':>12}" for i in range(len(named)))
    print(header)
    lookup = [dict(pts) for _, pts in named]
    for x in xs:
        cells = []
        for d in lookup:
            v = d.get(x)
            cells.append(f"{v:12.6f}" if isinstance(v, (int, float)) else f"{'':>12}")
        print(f"{x:>8} " + " ".join(cells))


def plot(named, metric, x_axis, out):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("--plot needs matplotlib (pip install matplotlib)")
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, pts in named:
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o" if x_axis == "epoch" else None, label=name)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(metric)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="*", default=["runs"],
                        help="run directories or directories containing runs")
    parser.add_argument("--metric", default="train_loss")
    parser.add_argument("--x", dest="x_axis", choices=["epoch", "step"],
                        default="epoch")
    parser.add_argument("--plot", metavar="PNG", help="also save a plot")
    args = parser.parse_args()

    runs = load_runs(args.paths)
    if not runs:
        sys.exit(f"No training runs found under: {' '.join(args.paths)}")
    named = [(label(c), series(c, args.metric, args.x_axis)) for c in chains(runs)]
    print_table(named, args.metric, args.x_axis)
    if args.plot:
        plot(named, args.metric, args.x_axis, args.plot)


if __name__ == "__main__":
    main()

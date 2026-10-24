#!/usr/bin/env python3
"""Parity check: a `litnicelm export` evaluated by the PyTorch reference model.

Computes the validation loss of the exported checkpoint over exactly the
validation windows LitniceLM uses (the dataset and validation settings come
from the training config) and compares it with the val loss LitniceLM
recorded for that checkpoint (the model card's best val loss, unless
--litnicelm-loss is given).

FP32 by default. --bf16 evaluates under torch.autocast(bfloat16), for
checkpoints trained with precision: bf16_mixed (whose recorded val loss was
computed with BF16 activations, so only an approximate match is expected).

Examples:
  tools/reference_torch.py exports/run_fp32 --config conf/run.yaml --tolerance 1e-4
  tools/reference_torch.py exports/run_16mix --config conf/run_16mix.yaml --bf16
"""

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import litnicelm_torch as lt  # noqa: E402


def card_val_loss(export_dir):
    """The best val loss in the export's model card, or None."""
    card = (Path(export_dir) / "README.md").read_text()
    m = re.search(r"\| best val loss \| ([0-9.eE+-]+)", card)
    return float(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_dir", help="output of `litnicelm export --out`")
    parser.add_argument("--config", required=True,
                        help="the LitniceLM training config (dataset and validation settings)")
    parser.add_argument("--litnicelm-loss", type=float,
                        help="LitniceLM's val loss (default: the model card's best val loss)")
    parser.add_argument("--bf16", action="store_true", help="evaluate under autocast(bfloat16)")
    parser.add_argument("--tolerance", type=float,
                        help="fail (exit 1) when |torch - litnicelm| exceeds this")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, cfg = lt.load_export(args.export_dir, args.device)
    training_cfg = lt.read_config(args.config)
    dtype = torch.bfloat16 if args.bf16 else None
    loss, batches, tokens = lt.mean_val_loss(model, training_cfg, args.device, dtype)
    reference = args.litnicelm_loss if args.litnicelm_loss is not None else card_val_loss(
        args.export_dir)

    print(f"model: {cfg['n_layer']} layers, d_model {cfg['n_embd']}, "
          f"{cfg['position_encoding']} positions; {'bf16 autocast' if args.bf16 else 'fp32'}")
    print(f"validation: {batches} batches, {tokens} tokens")
    print(f"torch val loss:     {loss:.7f}")
    if reference is None:
        print("litnicelm val loss: unknown (no best val loss in the model card)")
        return
    diff = abs(loss - reference)
    print(f"litnicelm val loss: {reference:.7f}")
    print(f"|difference|:       {diff:.2e}")
    if args.tolerance is not None:
        ok = diff <= args.tolerance
        print(f"{'PASS' if ok else 'FAIL'} (tolerance {args.tolerance:g})")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

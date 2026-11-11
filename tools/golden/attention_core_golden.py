# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Write tests/golden/attention_core.json: causal attention cases from PyTorch.

Run once from the repository root: python3 tools/golden/attention_core_golden.py
Inputs are F32-representable; expected values are computed in float64.
"""

import json
from pathlib import Path

import torch

CASES = (  # batch, seq, model dim, heads
    (2, 5, 8, 2),
    (1, 7, 12, 3),
)


def attention_case(batch: int, seq: int, model_dim: int, heads: int, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    qkv = (torch.rand(batch, seq, 3 * model_dim, generator=generator) * 2 - 1).double()
    dcontext = (torch.rand(batch, seq, model_dim, generator=generator) * 2 - 1).double()
    qkv.requires_grad_(True)
    head_dim = model_dim // heads

    def split(block: torch.Tensor) -> torch.Tensor:  # [B, S, D] -> [B, H, S, dh]
        return block.reshape(batch, seq, heads, head_dim).transpose(1, 2)

    q, k, v = (split(part) for part in qkv.split(model_dim, dim=-1))
    context = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    context = context.transpose(1, 2).reshape(batch, seq, model_dim)
    context.backward(dcontext)
    return {
        "batch": batch,
        "seq": seq,
        "model_dim": model_dim,
        "heads": heads,
        "qkv": qkv.detach().flatten().tolist(),
        "dcontext": dcontext.flatten().tolist(),
        "context": context.detach().flatten().tolist(),
        "dqkv": qkv.grad.flatten().tolist(),
    }


def main() -> None:
    cases = [attention_case(*shape, seed=index) for index, shape in enumerate(CASES)]
    out = Path("tests/golden/attention_core.json")
    out.write_text(
        json.dumps(
            {"generator": f"tools/golden/attention_core_golden.py, torch {torch.__version__}",
             "cases": cases},
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

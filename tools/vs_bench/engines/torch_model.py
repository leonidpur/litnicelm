"""PyTorch model matching a LitniceLM export."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_tables(
    seq_len: int, head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build LitniceLM's FP32 half-split rotary tables."""
    half = head_dim // 2
    indices = torch.arange(half, dtype=torch.float32, device=device)
    inv_freq = torch.pow(
        torch.tensor(theta, dtype=torch.float32, device=device),
        -2.0 * indices / head_dim,
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = positions[:, None] * inv_freq[None, :]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    values: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    """Rotate half-split pairs in a [batch, heads, sequence, head] tensor."""
    half = values.shape[-1] // 2
    first, second = values[..., :half], values[..., half:]
    return torch.cat(
        (first * cosine - second * sine, first * sine + second * cosine), dim=-1
    )


class Block(nn.Module):
    """One pre-LayerNorm LitniceLM transformer block."""

    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        width = int(config["n_embd"])
        inner = int(config["n_inner"])
        epsilon = float(config["layer_norm_epsilon"])
        self.n_head = int(config["n_head"])
        self.ln_1 = nn.LayerNorm(width, eps=epsilon)
        self.ln_2 = nn.LayerNorm(width, eps=epsilon)
        self.c_attn_w = nn.Parameter(torch.empty(width, 3 * width))
        self.c_attn_b = nn.Parameter(torch.empty(3 * width))
        self.c_proj_w = nn.Parameter(torch.empty(width, width))
        self.c_proj_b = nn.Parameter(torch.empty(width))
        self.c_fc_w = nn.Parameter(torch.empty(width, inner))
        self.c_fc_b = nn.Parameter(torch.empty(inner))
        self.mlp_proj_w = nn.Parameter(torch.empty(inner, width))
        self.mlp_proj_b = nn.Parameter(torch.empty(width))

    def forward(
        self, inputs: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None
    ) -> torch.Tensor:
        batch, sequence, width = inputs.shape
        qkv = self.ln_1(inputs) @ self.c_attn_w + self.c_attn_b
        query, key, value = (
            part.view(batch, sequence, self.n_head, width // self.n_head).transpose(1, 2)
            for part in qkv.split(width, dim=-1)
        )
        if rope is not None:
            query, key = apply_rope(query, *rope), apply_rope(key, *rope)
        attention = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        inputs = (
            inputs
            + attention.transpose(1, 2).reshape(batch, sequence, width) @ self.c_proj_w
            + self.c_proj_b
        )
        hidden = F.relu(self.ln_2(inputs) @ self.c_fc_w + self.c_fc_b)
        return inputs + hidden @ self.mlp_proj_w + self.mlp_proj_b


class LitniceLM(nn.Module):
    """PyTorch implementation of the architecture described by an export."""

    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        self.config = config
        width = int(config["n_embd"])
        vocab = int(config["vocab_size"])
        self.position_encoding = str(config["position_encoding"])
        if config["activation_function"] != "relu":
            raise ValueError("only activation_function relu is supported")
        if self.position_encoding == "rope" and config.get("rope_layout") != "half_split":
            raise ValueError("only rope_layout half_split is supported")
        self.wte = nn.Parameter(torch.empty(vocab, width))
        self.wpe = (
            nn.Parameter(torch.empty(int(config["n_positions"]), width))
            if self.position_encoding == "learned"
            else None
        )
        self.h = nn.ModuleList(Block(config) for _ in range(int(config["n_layer"])))
        self.ln_f = nn.LayerNorm(width, eps=float(config["layer_norm_epsilon"]))
        self.lm_head = nn.Parameter(torch.empty(vocab, width))
        self._rope: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def rope(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return cached rotary tables when this model uses RoPE."""
        if self.position_encoding != "rope":
            return None
        key = (seq_len, str(device))
        if key not in self._rope:
            head_dim = int(self.config["n_embd"]) // int(self.config["n_head"])
            self._rope[key] = rope_tables(
                seq_len, head_dim, float(self.config["rope_theta"]), device
            )
        return self._rope[key]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.wte[token_ids]
        if self.wpe is not None:
            hidden = hidden + self.wpe[: token_ids.shape[1]]
        rope = self.rope(token_ids.shape[1], token_ids.device)
        for block in self.h:
            hidden = block(hidden, rope)
        return F.linear(self.ln_f(hidden), self.lm_head)

    def export_names(self) -> dict[str, str]:
        """Map Safetensors names to this module's parameter names."""
        names = {
            "transformer.wte.weight": "wte",
            "lm_head.weight": "lm_head",
            "transformer.ln_f.weight": "ln_f.weight",
            "transformer.ln_f.bias": "ln_f.bias",
        }
        if self.wpe is not None:
            names["transformer.wpe.weight"] = "wpe"
        for index in range(len(self.h)):
            source = f"transformer.h.{index}."
            target = f"h.{index}."
            names.update(
                {
                    source + "ln_1.weight": target + "ln_1.weight",
                    source + "ln_1.bias": target + "ln_1.bias",
                    source + "attn.c_attn.weight": target + "c_attn_w",
                    source + "attn.c_attn.bias": target + "c_attn_b",
                    source + "attn.c_proj.weight": target + "c_proj_w",
                    source + "attn.c_proj.bias": target + "c_proj_b",
                    source + "ln_2.weight": target + "ln_2.weight",
                    source + "ln_2.bias": target + "ln_2.bias",
                    source + "mlp.c_fc.weight": target + "c_fc_w",
                    source + "mlp.c_fc.bias": target + "c_fc_b",
                    source + "mlp.c_proj.weight": target + "mlp_proj_w",
                    source + "mlp.c_proj.bias": target + "mlp_proj_b",
                }
            )
        return names


def load_export(
    export_dir: Path, device: torch.device
) -> tuple[LitniceLM, dict[str, object]]:
    """Load an exported configuration and weights onto the selected device."""
    from safetensors.torch import load_file

    config = json.loads((export_dir / "config.json").read_text())
    model = LitniceLM(config)
    tensors = load_file(str(export_dir / "model.safetensors"))
    names = model.export_names()
    missing = set(names) - set(tensors)
    extra = set(tensors) - set(names)
    if missing or extra:
        raise ValueError(
            f"export does not match config.json: missing {sorted(missing)}, "
            f"unexpected {sorted(extra)}"
        )
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for export_name, parameter_name in names.items():
            source = tensors[export_name]
            parameter = parameters[parameter_name]
            if tuple(source.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"{export_name}: shape {tuple(source.shape)}, "
                    f"model expects {tuple(parameter.shape)}"
                )
            parameter.copy_(source)
    return model.to(device), config

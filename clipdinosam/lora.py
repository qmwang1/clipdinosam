from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 8, dropout: float = 0.0):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Base weight is frozen
        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)
        # LoRA factors
        self.A = nn.Parameter(torch.zeros(self.r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, self.r))
        # Scaling
        self.scaling = alpha / r
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)) if self.r > 0 else None
        nn.init.zeros_(self.B) if self.r > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_out = self.dropout(x) @ self.A.t()
            lora_out = lora_out @ self.B.t()
            base_out = base_out + self.scaling * lora_out
        return base_out


def _match_any(name: str, patterns: Iterable[str]) -> bool:
    return any(p in name for p in patterns)


def inject_lora_linear(
    module: nn.Module,
    target_substrings: List[str],
    rank: int = 8,
    alpha: int = 8,
    dropout: float = 0.0,
) -> List[str]:
    """Replace matching nn.Linear layers with LoRALinear. Returns list of replaced module names."""
    replaced = []
    for name, child in list(module.named_modules()):
        if isinstance(child, nn.Linear) and _match_any(name, target_substrings):
            # Find parent to replace attribute
            parent = module
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            last = parts[-1]
            base_linear = getattr(parent, last)
            setattr(parent, last, LoRALinear(base_linear, r=rank, alpha=alpha, dropout=dropout))
            replaced.append(name)
    return replaced


def set_trainable(module: nn.Module, trainable: bool):
    for p in module.parameters(recurse=True):
        p.requires_grad = trainable


def enable_only_lora(module: nn.Module):
    """Freeze all but LoRA parameters in module."""
    for n, p in module.named_parameters():
        p.requires_grad = (".A" in n or ".B" in n)

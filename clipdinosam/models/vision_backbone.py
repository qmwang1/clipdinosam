"""Shared utilities for vision backbones."""
from typing import List, Optional, Tuple

import warnings

import torch
import torch.nn as nn


class VisionBackbone(nn.Module):
    """Common interface for token-producing image encoders."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def embed_dim(self) -> int:
        raise NotImplementedError

    def forward_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        raise NotImplementedError

    def lora_target_pairs(self, last_k: int) -> List[Tuple[nn.Module, List[str]]]:
        """Return modules and target substrings for LoRA injection."""
        return []


def build_backbone(name: str, backbone_type: Optional[str] = None, pretrained: bool = True, **kwargs) -> VisionBackbone:
    inferred = (backbone_type or "").lower()
    if not inferred:
        inferred = "swin" if "swin" in name.lower() else "dino"

    if inferred in {"dino", "dinov2", "vit"}:
        if kwargs:
            warnings.warn("Ignoring extra kwargs for DINO backbone construction.")
        from .dino import DINOBackbone  # local import to avoid circular dependency

        return DINOBackbone(model_name=name, pretrained=pretrained)

    if inferred == "swin":
        from .swin import SwinBackbone  # local import to avoid circular dependency

        return SwinBackbone(model_name=name, pretrained=pretrained, **kwargs)

    raise ValueError(f"Unsupported backbone type '{backbone_type}' for model '{name}'.")


__all__ = [
    "VisionBackbone",
    "build_backbone",
]

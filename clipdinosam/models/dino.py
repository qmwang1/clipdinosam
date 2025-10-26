"""DINO/DINOv2 backbone wrapper."""
from typing import Optional, Tuple

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:  # pragma: no cover - optional dependency
    timm = None

from .vision_backbone import VisionBackbone


class DINOBackbone(VisionBackbone):
    """Wrapper around DINO/DINOv2 ViT backbones from timm or torch.hub."""

    def __init__(self, model_name: str = "vit_base_patch16_224.dino", pretrained: bool = True) -> None:
        super().__init__()
        self.model: Optional[nn.Module] = None
        # 1) Try timm first (supports many DINO and DINOv2 variants)
        if timm is not None:
            try:
                self.model = timm.create_model(model_name, pretrained=pretrained, features_only=False)
            except Exception as exc:  # pragma: no cover - best effort
                self.model = None
                warnings.warn(f"timm could not create model '{model_name}': {exc}")

        # 2) Fallback: try official DINOv2 via torch.hub if requested
        if self.model is None and ("dinov2" in model_name.lower() or model_name.lower().startswith("dinov2_")):
            try:
                hub_name = self._map_name_to_dinov2_hub(model_name)
                self.model = torch.hub.load("facebookresearch/dinov2", hub_name)  # type: ignore[arg-type]
                if not pretrained:
                    warnings.warn("Official DINOv2 hub models load with pretrained weights by default.")
            except Exception as exc:  # pragma: no cover - network/offline issues
                raise ImportError(
                    f"Failed to build DINOv2 model '{model_name}'. Install timm with DINOv2 support or ensure torch.hub can access facebookresearch/dinov2. Error: {exc}"
                ) from exc

        if self.model is None:
            raise ImportError(
                "Unable to construct DINO/DINOv2 model. Ensure 'timm' is installed and 'model.model_name' is valid.'"
            )

        self.model_name = model_name
        self.patch_embed = getattr(self.model, "patch_embed", None)
        embed_dim = getattr(self.model, "embed_dim", None) or getattr(self.model, "num_features", None)
        if embed_dim is None:
            raise AttributeError(f"DINO model '{model_name}' does not expose embed_dim/num_features.")
        self._embed_dim = int(embed_dim)

    def _map_name_to_dinov2_hub(self, name: str) -> str:
        lowered = name.lower()
        if any(key in lowered for key in ["vits14", "vit_small", "small"]):
            return "dinov2_vits14"
        if any(key in lowered for key in ["vitl14", "vit_large", "large", "l14"]):
            return "dinov2_vitl14"
        if any(key in lowered for key in ["vitg14", "giant", "g14"]):
            return "dinov2_vitg14"
        return "dinov2_vitb14"

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        batch = x.size(0)
        try:
            expected = None
            if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "img_size"):
                expected = self.model.patch_embed.img_size
            elif hasattr(self.model, "img_size"):
                expected = getattr(self.model, "img_size")
            if expected is not None:
                if isinstance(expected, int):
                    expected = (expected, expected)
                height, width = x.shape[-2], x.shape[-1]
                if (height, width) != tuple(expected):
                    x = F.interpolate(x, size=expected, mode="bilinear", align_corners=False)
        except Exception:  # pragma: no cover - resize best effort
            pass

        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
            if isinstance(feats, dict) and "x_norm_clstoken" in feats:
                tokens = feats["x_norm_patchtokens"]
            elif isinstance(feats, dict) and "tokens" in feats:
                tokens = feats["tokens"]
            else:
                if not hasattr(self.model, "patch_embed"):
                    raise RuntimeError("Unsupported DINO model without forward_features/patch_embed fallback")
                x = self.model.patch_embed(x)
                grid = int((x.shape[1]) ** 0.5)
                cls = self.model.cls_token.expand(batch, -1, -1)
                x = torch.cat((cls, x), dim=1)
                x = x + self.model.pos_embed
                x = self.model.pos_drop(x)
                for block in self.model.blocks:
                    x = block(x)
                x = self.model.norm(x)
                tokens = x[:, 1:, :]
                return tokens, (grid, grid)
        else:
            raise RuntimeError("Unsupported DINO model without forward_features")

        if self.patch_embed is not None and hasattr(self.patch_embed, "grid_size"):
            height, width = self.patch_embed.grid_size
        else:
            num_tokens = tokens.shape[1]
            grid = int(num_tokens ** 0.5)
            height = width = grid
        return tokens, (height, width)

    def lora_target_pairs(self, last_k: int):
        if not hasattr(self.model, "blocks"):
            return []
        blocks = getattr(self.model, "blocks")
        total = len(blocks)
        if total == 0 or last_k <= 0:
            return []
        last_k = min(last_k, total)
        start = total - last_k
        targets = []
        for idx in range(start, total):
            targets.extend([f"blocks.{idx}.attn", f"blocks.{idx}.mlp"])
        return [(self.model, targets)]


def build_dino(name: str, pretrained: bool = True) -> DINOBackbone:
    return DINOBackbone(model_name=name, pretrained=pretrained)


__all__ = [
    "DINOBackbone",
    "build_dino",
]

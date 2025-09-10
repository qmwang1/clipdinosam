from typing import Tuple

import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    timm = None


class DINOBackbone(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224.dino", pretrained: bool = True):
        super().__init__()
        self.model = None
        # 1) Try timm first (supports many DINO and DINOv2 variants)
        if timm is not None:
            try:
                self.model = timm.create_model(model_name, pretrained=pretrained, features_only=False)
            except Exception as e:
                self.model = None
                warnings.warn(f"timm could not create model '{model_name}': {e}")

        # 2) Fallback: try official DINOv2 via torch.hub if requested
        if self.model is None and ("dinov2" in model_name.lower() or model_name.lower().startswith("dinov2_")):
            try:
                hub_name = self._map_name_to_dinov2_hub(model_name)
                self.model = torch.hub.load("facebookresearch/dinov2", hub_name)  # type: ignore
                if not pretrained:
                    warnings.warn("Official DINOv2 hub models load with pretrained weights by default.")
            except Exception as e:
                raise ImportError(
                    f"Failed to build DINOv2 model '{model_name}'. Install timm with DINOv2 support or ensure torch.hub can access facebookresearch/dinov2. Error: {e}"
                )

        if self.model is None:
            raise ImportError(
                "Unable to construct DINO/DINOv2 model. Ensure 'timm' is installed and 'model.model_name' is valid."
            )
        self.patch_embed = getattr(self.model, "patch_embed", None)
        self.num_patches = None
        self.vit_resolution = None

    def _map_name_to_dinov2_hub(self, name: str) -> str:
        n = name.lower()
        # Heuristics to map various name styles to official hub names
        if any(k in n for k in ["vits14", "vit_small", "small"]):
            return "dinov2_vits14"
        if any(k in n for k in ["vitl14", "vit_large", "large", "l14"]):
            return "dinov2_vitl14"
        if any(k in n for k in ["vitg14", "giant", "g14"]):
            return "dinov2_vitg14"
        # default/base
        return "dinov2_vitb14"

    @property
    def embed_dim(self) -> int:
        return getattr(self.model, "embed_dim", None) or getattr(self.model, "num_features", None)

    def forward_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # Handle ViT variants in timm: get token embeddings without cls.
        B = x.size(0)
        # Ensure input size matches model expectation (e.g., DINOv2 often uses 518)
        try:
            exp = None
            if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "img_size"):
                exp = self.model.patch_embed.img_size
            elif hasattr(self.model, "img_size"):
                exp = getattr(self.model, "img_size")
            if exp is not None:
                if isinstance(exp, int):
                    exp = (exp, exp)
                H, W = x.shape[-2], x.shape[-1]
                if (H, W) != tuple(exp):
                    x = F.interpolate(x, size=exp, mode="bilinear", align_corners=False)
        except Exception:
            pass
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)  # some models return dict
            if isinstance(feats, dict) and "x_norm_clstoken" in feats:
                x_tokens = feats["x_norm_patchtokens"]  # (B, N, C)
            elif isinstance(feats, dict) and "tokens" in feats:
                x_tokens = feats["tokens"]
            else:
                # Fallback: try to access .blocks pipeline
                x = self.model.patch_embed(x)
                h = w = int((x.shape[1]) ** 0.5)
                cls_token = self.model.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_token, x), dim=1)
                x = x + self.model.pos_embed
                x = self.model.pos_drop(x)
                for blk in self.model.blocks:
                    x = blk(x)
                x = self.model.norm(x)
                x_tokens = x[:, 1:, :]
                return x_tokens, (h, w)
        else:
            raise RuntimeError("Unsupported DINO model without forward_features")

        # Try to infer grid from patch_embed
        if self.patch_embed is not None and hasattr(self.patch_embed, "grid_size"):
            h, w = self.patch_embed.grid_size
        else:
            N = x_tokens.shape[1]
            hw = int(N ** 0.5)
            h = w = hw
        return x_tokens, (h, w)


def build_dino(name: str, pretrained: bool = True) -> DINOBackbone:
    return DINOBackbone(model_name=name, pretrained=pretrained)

    

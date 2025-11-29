"""Supervised Swin Transformer backbone wrapper."""
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import timm
except ImportError:  # pragma: no cover - optional dependency
    timm = None

from .vision_backbone import VisionBackbone


class SwinBackbone(VisionBackbone):
    """Wrapper around supervised Swin Transformer backbones from timm."""

    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        out_index: Optional[int] = None,
        out_indices: Optional[Tuple[int, ...]] = None,
        **model_kwargs,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required to build Swin backbones.")
        if out_index is not None and out_indices is not None:
            raise ValueError("Specify either out_index or out_indices, not both.")
        if out_indices is None:
            if out_index is None:
                out_indices = (0, 1, 2, 3)
                preferred_slot = -1
            else:
                out_indices = (int(out_index),)
                preferred_slot = 0
        else:
            out_indices = tuple(int(idx) for idx in out_indices)
            preferred_slot = -1

        model_kwargs = dict(model_kwargs)
        allow_dynamic = bool(model_kwargs.pop("dynamic_img_size", False))
        strict_override = model_kwargs.pop("strict_img_size", None)
        model_kwargs.pop("features_only", None)
        model_kwargs.pop("out_indices", None)

        try:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
                **model_kwargs,
            )
        except Exception as exc:  # pragma: no cover - best effort
            raise ImportError(f"timm could not create Swin model '{model_name}': {exc}") from exc

        if allow_dynamic or strict_override is not None:
            for module in self.model.modules():
                if hasattr(module, "strict_img_size"):
                    if allow_dynamic:
                        module.strict_img_size = False
                    elif strict_override is not None:
                        module.strict_img_size = bool(strict_override)
                if allow_dynamic and hasattr(module, "dynamic_img_pad"):
                    module.dynamic_img_pad = True

        self._allow_dynamic = allow_dynamic
        self._last_size = None

        channels = self.model.feature_info.channels()
        if len(channels) == 0:
            raise RuntimeError(f"Swin model '{model_name}' returned no feature maps.")
        if preferred_slot == -1:
            self._output_slot = len(channels) - 1
        else:
            if preferred_slot >= len(channels):
                raise ValueError(f"Requested out_index not available for model '{model_name}'.")
            self._output_slot = preferred_slot
        self._embed_dim = int(channels[self._output_slot])
        self.model_name = model_name

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _maybe_update_input_size(self, h: int, w: int):
        if not getattr(self, "_allow_dynamic", False):
            return
        if self._last_size == (h, w):
            return
        self._last_size = (h, w)
        patch_embed = self.model["patch_embed"] if "patch_embed" in self.model else None
        grid_h = h
        grid_w = w
        if patch_embed is not None and hasattr(patch_embed, "set_input_size"):
            patch_embed.set_input_size(img_size=(h, w))
            grid_h, grid_w = patch_embed.grid_size
        elif patch_embed is not None and hasattr(patch_embed, "patch_size"):
            ph, pw = patch_embed.patch_size
            grid_h, grid_w = h // ph, w // pw
        stage_keys = [k for k in self.model.keys() if k.startswith("layers_")]
        stage_keys.sort()
        target_device = None
        try:
            target_device = next(self.model.parameters()).device
        except StopIteration:
            pass
        if target_device is not None and patch_embed is not None:
            patch_embed.to(target_device)
        for key in stage_keys:
            stage = self.model[key]
            if stage is None or not hasattr(stage, "set_input_size"):
                continue
            if getattr(stage, "dynamic_mask", False) is False:
                stage.dynamic_mask = True
            if hasattr(stage, "blocks"):
                for blk in stage.blocks:
                    if hasattr(blk, "dynamic_mask") and blk.dynamic_mask is False:
                        blk.dynamic_mask = True
                    if hasattr(blk, "attn_mask"):
                        blk.attn_mask = None
            try:
                idx = int(key.split("_")[1])
            except Exception:
                idx = 0
            stage_scale = 2 ** max(idx - 1, 0)
            feat_h = max(grid_h // stage_scale, 1)
            feat_w = max(grid_w // stage_scale, 1)
            window_size = None
            if hasattr(stage, "window_size"):
                window_size = stage.window_size
            elif hasattr(stage, "blocks") and len(stage.blocks) > 0 and hasattr(stage.blocks[0], "window_size"):
                window_size = stage.blocks[0].window_size
            stage.set_input_size(
                feat_size=(feat_h, feat_w),
                window_size=window_size if window_size is not None else (feat_h, feat_w),
                always_partition=None,
            )
            if target_device is not None:
                stage.to(target_device)

    def forward_tokens(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Tuple[int, int]]:
        self._maybe_update_input_size(x.shape[-2], x.shape[-1])
        features = self.model(x)
        feat = features[self._output_slot]
        if feat.shape[1] != self._embed_dim:
            # timm may return Swin features in NHWC layout; convert to NCHW
            feat = feat.permute(0, 3, 1, 2).contiguous()
        _, _, height, width = feat.shape
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        return tokens, (height, width)


__all__ = [
    "SwinBackbone",
]

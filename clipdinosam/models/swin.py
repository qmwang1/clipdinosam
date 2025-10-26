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

    def forward_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
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

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .dino import DINOBackbone
from .clip_text import CLIPTextEncoder
from .sam_decoder import SAMDecoderWrapper
from .projection import TokenProjection, TokenToMaskEmbedding


class CLIPDinoSam(nn.Module):
    def __init__(
        self,
        dino: DINOBackbone,
        clip_text: CLIPTextEncoder,
        sam: SAMDecoderWrapper,
        token_to_text: TokenProjection,
        token_to_mask: TokenToMaskEmbedding,
    ):
        super().__init__()
        self.dino = dino
        self.clip_text = clip_text
        self.sam = sam
        self.token_to_text = token_to_text
        self.token_to_mask = token_to_mask

    def forward(
        self,
        images: torch.Tensor,
        texts: Optional[List[str]] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
        skip_sam: bool = False,
    ) -> Dict[str, torch.Tensor]:
        tokens, hw = self.dino.forward_tokens(images)        # (B, N, C)
        clip_tokens = self.token_to_text(tokens)             # (B, N, clip_dim)
        mask_embeddings = self.token_to_mask(tokens, hw)     # (B, 256, h, w)

        text_feats = None
        if texts is not None:
            text_feats = self.clip_text.encode_texts(texts)  # (T, D)

        if skip_sam:
            low_res_masks = None
            iou_preds = None
        else:
            low_res_masks, iou_preds = self.sam(
                mask_embeddings,
                points=points,
                labels=point_labels,
                boxes=boxes,
                multimask_output=multimask_output,
            )

        device = mask_embeddings.device

        return {
            "low_res_masks": low_res_masks,  # (B, 1|3, h, w) or None when skip_sam=True
            "iou_preds": iou_preds,          # (B, 1|3) or None when skip_sam=True
            "clip_tokens": clip_tokens,      # (B, N, D)
            "text_feats": text_feats,        # (T, D) or None
            "grid_hw": torch.tensor(hw, device=device),
        }

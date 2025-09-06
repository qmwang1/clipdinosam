from typing import Optional, Tuple
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from segment_anything.modeling.prompt_encoder import PromptEncoder
    from segment_anything.modeling.mask_decoder import MaskDecoder
    from segment_anything import sam_model_registry
except Exception:
    PromptEncoder = None
    MaskDecoder = None
    sam_model_registry = None


class SAMDecoderWrapper(nn.Module):
    def __init__(
        self,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        image_embedding_channels: int = 256,
    ):
        super().__init__()
        if PromptEncoder is None or MaskDecoder is None:
            raise ImportError("segment-anything is required for SAM decoder")
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.C = image_embedding_channels

    def forward(
        self,
        img_embeddings: torch.Tensor,  # (B, C=256, H, W)
        points: Optional[torch.Tensor] = None,  # (B, P, 2)
        labels: Optional[torch.Tensor] = None,  # (B, P)
        boxes: Optional[torch.Tensor] = None,   # (B, 4)
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode prompts
        B = img_embeddings.size(0)
        device = img_embeddings.device
        if points is not None and labels is not None:
            point_coords = points
            point_labels = labels
        else:
            point_coords = torch.zeros(B, 0, 2, device=device)
            point_labels = torch.zeros(B, 0, dtype=torch.int64, device=device)

        # Option 1: operate at the DINO grid size (no forced upsample to 64x64)
        B, C, H, W = img_embeddings.shape
        # Build positional encoding at current grid if available; otherwise resize default PE
        if hasattr(self.prompt_encoder, "pe_layer"):
            pe = self.prompt_encoder.pe_layer((H, W))
        else:
            pe = self.prompt_encoder.get_dense_pe()
            if pe.shape[-2:] != (H, W):
                pe = F.interpolate(pe, size=(H, W), mode="bilinear", align_corners=False)
        # Ensure positional encoding has batch dimension and matches B
        if pe.dim() == 3:
            pe = pe.unsqueeze(0)
        if pe.shape[0] != B:
            pe = pe.expand(B, -1, -1, -1)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels), boxes=boxes, masks=None
        )
        # Align batch sizes if SAM's internals return a different batch for prompts
        if dense_embeddings.shape[0] != img_embeddings.shape[0]:
            if dense_embeddings.shape[0] == 1:
                dense_embeddings = dense_embeddings.repeat(img_embeddings.shape[0], 1, 1, 1)
                if sparse_embeddings is not None and sparse_embeddings.shape[0] == 1:
                    sparse_embeddings = sparse_embeddings.repeat(img_embeddings.shape[0], 1, 1)
            elif img_embeddings.shape[0] == 1:
                img_embeddings = img_embeddings.repeat(dense_embeddings.shape[0], 1, 1, 1)
                # sparse already matches dense batch size
            else:
                # Fallback to expand/repeat to the larger batch size
                B = max(dense_embeddings.shape[0], img_embeddings.shape[0])
                dense_embeddings = dense_embeddings.repeat(B // dense_embeddings.shape[0], 1, 1, 1)
                img_embeddings = img_embeddings.repeat(B // img_embeddings.shape[0], 1, 1, 1)
                if sparse_embeddings is not None and sparse_embeddings.shape[0] != B:
                    sparse_embeddings = sparse_embeddings.repeat(B // sparse_embeddings.shape[0], 1, 1)
        # The prompt encoder's dense embeddings may be at a default grid (e.g., 64x64). Resize to (H, W).
        if dense_embeddings.shape[-2:] != (H, W):
            dense_embeddings = F.interpolate(dense_embeddings, size=(H, W), mode="bilinear", align_corners=False)

        # Some SAM versions repeat image embeddings internally based on prompts (repeat_image=True).
        # That can create a batch mismatch (e.g., src B=16 vs dense B=4). To avoid this,
        # detect the "repeat_image" argument and force it to False for external embeddings.
        md_forward_params = inspect.signature(self.mask_decoder.forward).parameters
        if "repeat_image" in md_forward_params:
            low_res_masks, iou_preds = self.mask_decoder(
                image_embeddings=img_embeddings,
                image_pe=pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=False,
            )
        else:
            # Prefer a stable, shape-safe decode path using SAM modules directly.
            # This avoids buggy batch repetition seen in some SAM builds.
            try:
                md = self.mask_decoder
                # Build output tokens: [iou_token, mask_tokens]
                output_tokens = torch.cat([md.iou_token.weight, md.mask_tokens.weight], dim=0)  # (1+M, C)
                output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, 1+M, C)
                if sparse_embeddings is not None:
                    tokens = torch.cat((output_tokens, sparse_embeddings), dim=1)  # (B, 1+M+S, C)
                else:
                    tokens = output_tokens

                # Combine image embedding with dense prompt embedding at the current grid.
                src0 = img_embeddings + dense_embeddings  # (B, C, H, W)
                # Run the transformer
                hs, src_feats = md.transformer(src0, pe, tokens)  # hs: (B, Nq, C), src_feats: (B, Nimg, C)

                # Token outputs
                iou_token_out = hs[:, 0, :]
                mask_tokens_out = hs[:, 1 : (1 + md.num_mask_tokens), :]

                # Upscale image features
                # src_feats: (B, Nimg, C) -> (B, C, H, W)
                src_feats = src_feats.transpose(1, 2).view(B, self.C, H, W)
                upscaled_embedding = md.output_upscaling(src_feats)  # (B, C//8, H*4, W*4)

                # Hypernetworks for each mask token
                hyper_in_list = []
                for i in range(md.num_mask_tokens):
                    hyper_in_list.append(md.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
                hyper_in = torch.stack(hyper_in_list, dim=1)  # (B, M, C//8)

                b, c2, h2, w2 = upscaled_embedding.shape
                masks = (hyper_in @ upscaled_embedding.view(b, c2, h2 * w2)).view(b, -1, h2, w2)
                iou_preds = md.iou_prediction_head(iou_token_out)

                # Select multimask or single mask
                mask_slice = slice(1, None) if multimask_output else slice(0, 1)
                low_res_masks = masks[:, mask_slice, :, :]
                iou_preds = iou_preds[:, mask_slice]
            except Exception:
                # Fallback to the decoder's own forward as a safety net
                low_res_masks, iou_preds = self.mask_decoder(
                    image_embeddings=img_embeddings,
                    image_pe=pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
        return low_res_masks, iou_preds


def build_sam_decoder(
    sam_type: str = "vit_b",
    checkpoint: Optional[str] = None,
) -> SAMDecoderWrapper:
    if sam_model_registry is None:
        raise ImportError("segment-anything is required for SAM decoder")
    sam = sam_model_registry[sam_type](checkpoint=checkpoint) if checkpoint else sam_model_registry[sam_type]()
    # Keep only prompt encoder and mask decoder
    pe = sam.prompt_encoder
    md = sam.mask_decoder
    return SAMDecoderWrapper(pe, md, image_embedding_channels=256)

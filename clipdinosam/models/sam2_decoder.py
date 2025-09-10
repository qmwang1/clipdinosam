from typing import Optional, Tuple

import os
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM2LikeDecoderWrapper(nn.Module):
    """
    Wrapper for a SAM-like prompt encoder + mask decoder pair from the SAM2 stack.

    Notes
    - Does not enforce types from segment-anything; relies on duck-typing.
    - Expects `prompt_encoder(points=(coords, labels), boxes=..., masks=None)` to return
      (sparse_embeddings, dense_embeddings).
    - Expects `mask_decoder` to expose either a stable `.forward(...)` signature compatible
      with SAM v1, or the internal modules used here: `.iou_token`, `.mask_tokens`,
      `.transformer`, `.output_upscaling`, `.output_hypernetworks_mlps`, `.iou_prediction_head`,
      and `.num_mask_tokens`.
    """

    def __init__(
        self,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        image_embedding_channels: int = 256,
    ):
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.C = image_embedding_channels

    def _build_pe(self, B: int, H: int, W: int) -> torch.Tensor:
        # Try SAM-style APIs
        if hasattr(self.prompt_encoder, "pe_layer"):
            pe = self.prompt_encoder.pe_layer((H, W))
        elif hasattr(self.prompt_encoder, "get_dense_pe"):
            pe = self.prompt_encoder.get_dense_pe()
            if pe.shape[-2:] != (H, W):
                pe = F.interpolate(pe, size=(H, W), mode="bilinear", align_corners=False)
        else:
            # Fallback: zeros positional encoding
            pe = torch.zeros(1, self.C, H, W)
        if pe.dim() == 3:
            pe = pe.unsqueeze(0)
        if pe.shape[0] != B:
            pe = pe.to(next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else "cpu")
            pe = pe.expand(B, -1, -1, -1)
        return pe

    def forward(
        self,
        img_embeddings: torch.Tensor,  # (B, C=256, H, W)
        points: Optional[torch.Tensor] = None,  # (B, P, 2)
        labels: Optional[torch.Tensor] = None,  # (B, P)
        boxes: Optional[torch.Tensor] = None,   # (B, 4)
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = img_embeddings.shape
        device = img_embeddings.device

        if points is not None and labels is not None:
            point_coords = points
            point_labels = labels
        else:
            point_coords = torch.zeros(B, 0, 2, device=device)
            point_labels = torch.zeros(B, 0, dtype=torch.int64, device=device)

        pe = self._build_pe(B, H, W)

        # Encode prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels), boxes=boxes, masks=None
        )
        # Align/resize dense prompt embeddings to image grid
        if dense_embeddings.shape[0] != img_embeddings.shape[0]:
            if dense_embeddings.shape[0] == 1:
                dense_embeddings = dense_embeddings.repeat(img_embeddings.shape[0], 1, 1, 1)
                if sparse_embeddings is not None and sparse_embeddings.shape[0] == 1:
                    sparse_embeddings = sparse_embeddings.repeat(img_embeddings.shape[0], 1, 1)
            elif img_embeddings.shape[0] == 1:
                img_embeddings = img_embeddings.repeat(dense_embeddings.shape[0], 1, 1, 1)
            else:
                B2 = max(dense_embeddings.shape[0], img_embeddings.shape[0])
                dense_embeddings = dense_embeddings.repeat(B2 // dense_embeddings.shape[0], 1, 1, 1)
                img_embeddings = img_embeddings.repeat(B2 // img_embeddings.shape[0], 1, 1, 1)
                if sparse_embeddings is not None and sparse_embeddings.shape[0] != B2:
                    sparse_embeddings = sparse_embeddings.repeat(B2 // sparse_embeddings.shape[0], 1, 1)
        if dense_embeddings.shape[-2:] != (H, W):
            dense_embeddings = F.interpolate(dense_embeddings, size=(H, W), mode="bilinear", align_corners=False)

        # Try a stable forward path if the decoder exposes compatible internals
        md = self.mask_decoder
        try:
            output_tokens = torch.cat([md.iou_token.weight, md.mask_tokens.weight], dim=0)  # (1+M, C)
            output_tokens = output_tokens.unsqueeze(0).expand(img_embeddings.shape[0], -1, -1)
            tokens = output_tokens if sparse_embeddings is None else torch.cat((output_tokens, sparse_embeddings), dim=1)
            src0 = img_embeddings + dense_embeddings
            hs, src_feats = md.transformer(src0, pe, tokens)
            iou_token_out = hs[:, 0, :]
            mask_tokens_out = hs[:, 1 : (1 + md.num_mask_tokens), :]

            src_feats = src_feats.transpose(1, 2).view(img_embeddings.shape[0], self.C, H, W)
            upscaled_embedding = md.output_upscaling(src_feats)
            hyper_in = torch.stack(
                [md.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(md.num_mask_tokens)],
                dim=1,
            )
            b, c2, h2, w2 = upscaled_embedding.shape
            masks = (hyper_in @ upscaled_embedding.view(b, c2, h2 * w2)).view(b, -1, h2, w2)
            iou_preds = md.iou_prediction_head(iou_token_out)

            mask_slice = slice(1, None) if multimask_output else slice(0, 1)
            low_res_masks = masks[:, mask_slice, :, :]
            iou_preds = iou_preds[:, mask_slice]
            return low_res_masks, iou_preds
        except Exception:
            # Fall back to calling the decoder's forward() if signature is SAM-compatible
            sig = inspect.signature(md.forward)
            kwargs = {
                "image_embeddings": img_embeddings,
                "image_pe": pe,
                "sparse_prompt_embeddings": sparse_embeddings,
                "dense_prompt_embeddings": dense_embeddings,
                "multimask_output": multimask_output,
            }
            # Some decoders have an extra repeat flag
            if "repeat_image" in sig.parameters:
                kwargs["repeat_image"] = False
            return md(**kwargs)


def _find_submodule_by_name(root: nn.Module, target_name: str) -> Optional[nn.Module]:
    for name, m in root.named_modules():
        if name.endswith(target_name) or name.split(".")[-1] == target_name:
            return m
    return None


def build_sam2_decoder(
    sam_type: str,
    config: Optional[str] = None,
    checkpoint: Optional[str] = None,
) -> SAM2LikeDecoderWrapper:
    """
    Build a SAM2-based decoder wrapper from a Hydra config and checkpoint.

    Requirements
    - `sam2` installed and importable.
    - `omegaconf` and `hydra-core` for instantiation from YAML.

    The function attempts to locate a SAM-like prompt encoder and mask decoder
    inside the instantiated SAM2 model and wraps them to accept external
    image embeddings (e.g., from DINOv2) at 256 channels.
    """
    try:
        import sam2  # noqa: F401
    except Exception as e:
        raise ImportError(
            "SAM2 is not installed. Please install the official SAM2 package to use SAM2 decoder."
        ) from e

    if not config or not os.path.exists(config):
        raise FileNotFoundError(
            f"SAM2 config YAML not found: {config}. Provide model.sam.config in your YAML."
        )

    # Instantiate via Hydra if available
    try:
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
    except Exception as e:
        raise ImportError(
            "hydra-core and omegaconf are required to instantiate SAM2 from config. Install them and retry."
        ) from e

    cfg = OmegaConf.load(config)
    model_cfg = cfg.get("model") if isinstance(cfg, dict) else cfg.model
    model = instantiate(model_cfg)
    if checkpoint and os.path.exists(checkpoint):
        try:
            ckpt = torch.load(checkpoint, map_location="cpu")
            state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                # Best-effort warning (do not crash)
                print(f"[build_sam2_decoder] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"[build_sam2_decoder] Warning: failed to load checkpoint: {e}")

    # Heuristics to find prompt encoder and mask decoder from the SAM2 model graph
    pe = None
    md = None

    # Common attribute paths
    for candidate in [
        "prompt_encoder",
        "sam.prompt_encoder",
        "sam2.prompt_encoder",
        "sam_module.prompt_encoder",
    ]:
        cur = model
        try:
            for part in candidate.split("."):
                cur = getattr(cur, part)
            pe = cur
            if pe is not None:
                break
        except Exception:
            continue

    for candidate in [
        "mask_decoder",
        "sam.mask_decoder",
        "sam2.mask_decoder",
        "sam_module.mask_decoder",
    ]:
        cur = model
        try:
            for part in candidate.split("."):
                cur = getattr(cur, part)
            md = cur
            if md is not None:
                break
        except Exception:
            continue

    # Fallback: search by submodule names
    if pe is None:
        pe = _find_submodule_by_name(model, "prompt_encoder")
    if md is None:
        md = _find_submodule_by_name(model, "mask_decoder")

    if pe is None or md is None:
        raise RuntimeError(
            "Could not locate prompt_encoder and mask_decoder inside the SAM2 model."
            " Ensure your SAM2 package/version exposes SAM-like submodules."
        )

    return SAM2LikeDecoderWrapper(pe, md, image_embedding_channels=256)


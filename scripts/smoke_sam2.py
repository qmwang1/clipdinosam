import argparse
import sys
from typing import Tuple, Optional

import torch


def run_fake(multimask: bool, B: int, H: int, W: int):
    from clipdinosam.models.sam2_decoder import SAM2LikeDecoderWrapper
    import torch.nn as nn
    import torch.nn.functional as F

    C = 256
    M = 3   # number of mask tokens
    C2 = 16 # channels after upscaling used by hypernets

    class FakePE(nn.Module):
        def __init__(self):
            super().__init__()

        def pe_layer(self, grid_hw: Tuple[int, int]):
            h, w = grid_hw
            return torch.zeros(1, C, h, w)

        def forward(self, points=None, boxes=None, masks=None):
            # Return (sparse_embeddings, dense_embeddings)
            B = points[0].shape[0] if points is not None else 1
            dense = torch.zeros(B, C, H, W)
            # Make one sparse prompt embedding per point if provided; else zero
            if points is not None and points[0].numel() > 0:
                sparse = torch.zeros(B, points[0].shape[1], C)
            else:
                sparse = torch.zeros(B, 0, C)
            return sparse, dense

    class IdentityTransformer(nn.Module):
        def forward(self, src, pe, tokens):
            # hs: (B, Nq, C)
            B, _, H, W = src.shape
            Nq = tokens.shape[1]
            C_ = tokens.shape[2]
            hs = torch.zeros(B, Nq, C_)
            # src_feats: (B, Nimg, C)
            src_feats = src.view(B, C, H * W).transpose(1, 2)
            return hs, src_feats

    class FakeMD(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_mask_tokens = M
            self.iou_token = nn.Embedding(1, C)
            self.mask_tokens = nn.Embedding(M, C)
            self.transformer = IdentityTransformer()
            # Map (B, C=256, H, W) -> (B, C2=16, H, W)
            self.output_upscaling = nn.Conv2d(C, C2, kernel_size=1, bias=True)
            self.output_hypernetworks_mlps = nn.ModuleList([nn.Sequential(nn.Linear(C, C2)) for _ in range(M)])
            self.iou_prediction_head = nn.Sequential(nn.Linear(C, M))

    wrapper = SAM2LikeDecoderWrapper(FakePE(), FakeMD(), image_embedding_channels=C)
    img = torch.randn(B, C, H, W)
    masks, iou = wrapper(img, points=None, labels=None, boxes=None, multimask_output=multimask)
    print("[FAKE] low_res_masks:", tuple(masks.shape), "iou:", tuple(iou.shape))


def run_sam2(config: str, checkpoint: Optional[str], multimask: bool, B: int, H: int, W: int):
    from clipdinosam.models.sam2_decoder import build_sam2_decoder

    try:
        import sam2  # noqa: F401
    except Exception as e:
        print("SAM2 not installed; skipping real SAM2 smoke:", e)
        return

    sam = build_sam2_decoder(sam_type="sam2", config=config, checkpoint=checkpoint)
    img = torch.randn(B, 256, H, W)
    masks, iou = sam(img, points=None, labels=None, boxes=None, multimask_output=multimask)
    print("[SAM2] low_res_masks:", tuple(masks.shape), "iou:", tuple(iou.shape))


def main():
    p = argparse.ArgumentParser(description="SAM2 smoke test: fake wrapper path and/or real config path")
    p.add_argument("--mode", choices=["auto", "fake", "sam2"], default="auto")
    p.add_argument("--config", type=str, help="Path to SAM2 hydra YAML with _target_ entries")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to SAM2 checkpoint (optional)")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--multimask", action="store_true")
    args = p.parse_args()

    if args.mode in ("auto", "sam2") and args.config:
        try:
            run_sam2(args.config, args.checkpoint, args.multimask, args.batch, args.height, args.width)
        except Exception as e:
            print("[SAM2] Smoke failed:", e)
            if args.mode == "sam2":
                sys.exit(1)

    if args.mode in ("auto", "fake"):
        try:
            run_fake(args.multimask, args.batch, args.height, args.width)
        except Exception as e:
            print("[FAKE] Smoke failed:", e)
            if args.mode == "fake":
                sys.exit(1)


if __name__ == "__main__":
    main()

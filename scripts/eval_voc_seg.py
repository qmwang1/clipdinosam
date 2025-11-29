"""
Evaluate binary segmentation (Dice/IoU) on a VOC-style split using a trained checkpoint.

Example:
python scripts/eval_voc_seg.py \
  --config configs/ham10000_voc_stage1.yaml \
  --checkpoint experiments/ham10000_onecls_stage1/checkpoints/best.pt \
  --split /home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC/ImageSets/Segmentation/val_Melanoma.txt \
  --data_root /home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC \
  --text "melanoma lesion" \
  --output_csv experiments/ham10000_onecls_stage1/eval_val_melanoma.csv
"""

import argparse
import warnings
import pickle
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

from clipdinosam.config import load_config_with_overrides
from clipdinosam.data import VOCSegDataset, ImageMaskDataset
from clipdinosam.models import (
    build_backbone,
    build_clip_text,
    build_sam_decoder,
    TokenProjection,
    TokenToMaskEmbedding,
    CLIPDinoSam,
)
from clipdinosam.models.sam2_decoder import build_sam2_decoder
from clipdinosam.lora import inject_lora_linear, set_trainable, enable_only_lora


def build_model_from_cfg(cfg, device: torch.device) -> CLIPDinoSam:
    mcfg = cfg["model"]
    backbone_cfg = mcfg.get("backbone") or mcfg.get("dino")
    if backbone_cfg is None:
        raise KeyError("Model config must define either 'backbone' or 'dino' settings.")
    backbone_name = backbone_cfg.get("name")
    if backbone_name is None:
        raise KeyError("Backbone config requires a 'name'.")
    backbone_type = backbone_cfg.get("type")
    backbone_pretrained = backbone_cfg.get("pretrained", True)
    backbone_kwargs = {k: v for k, v in backbone_cfg.items() if k not in {"name", "type", "pretrained"}}
    dino = build_backbone(
        backbone_name,
        backbone_type=backbone_type,
        pretrained=backbone_pretrained,
        **backbone_kwargs,
    )

    clip_cfg = mcfg["clip"]
    clip_text = build_clip_text(clip_cfg["name"], pretrained=clip_cfg.get("pretrained", "openai"))
    sam_cfg = mcfg["sam"]
    sam_type = sam_cfg["type"]
    if isinstance(sam_type, str) and "sam2" in sam_type.lower():
        sam = build_sam2_decoder(
            sam_type=sam_type,
            config=sam_cfg.get("config"),
            checkpoint=sam_cfg.get("checkpoint"),
        )
    else:
        sam = build_sam_decoder(sam_type, checkpoint=sam_cfg.get("checkpoint"))

    token_to_text = TokenProjection(in_dim=dino.embed_dim, out_dim=clip_cfg.get("width", clip_text.width))
    token_to_mask = TokenToMaskEmbedding(in_dim=dino.embed_dim, embed_dim=sam_cfg.get("embed_dim", 256))
    model = CLIPDinoSam(dino, clip_text, sam, token_to_text, token_to_mask)

    # Mirror training-time LoRA and freezing so checkpoints load properly
    stage = cfg.get("stage", 1)
    lora_cfg = cfg.get("lora", {})

    # Freeze CLIP text; keep projections/sam adapters trainable
    set_trainable(model.clip_text, False)
    set_trainable(model.dino, False)
    set_trainable(model.token_to_text, True)
    set_trainable(model.token_to_mask, True)

    # SAM LoRA injection (if enabled in config)
    sam_lora = lora_cfg.get("sam", {})
    if sam_lora.get("enable", True):
        sam_targets = sam_lora.get("targets", ["attn", "mlp", "lin", "proj"])
        inject_lora_linear(
            model.sam,
            sam_targets,
            rank=sam_lora.get("rank", 8),
            alpha=sam_lora.get("alpha", 8),
            dropout=sam_lora.get("dropout", 0.0),
        )

    # DINO: LoRA stages vs full-unfreeze
    if stage in (2, 3):
        dino_lora = lora_cfg.get("dino", {})
        if dino_lora.get("enable", True):
            k = dino_lora.get("last_k_blocks", 2)
            target_pairs = model.dino.lora_target_pairs(k)
            total_replaced = 0
            for module, target_substrings in target_pairs:
                replaced = inject_lora_linear(
                    module,
                    target_substrings,
                    rank=dino_lora.get("rank", 8),
                    alpha=dino_lora.get("alpha", 8),
                    dropout=dino_lora.get("dropout", 0.0),
                )
                total_replaced += len(replaced)
            if total_replaced == 0 and target_pairs:
                warnings.warn(
                    "LoRA enabled for backbone but no target linear layers were matched; skipping injection."
                )
        enable_only_lora(model.dino)
    elif stage >= 4:
        set_trainable(model.dino, True)

    return model.to(device)


def compute_batch_metrics(pred: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    inter = (pred * mask).sum(dim=[1, 2, 3])
    union = (pred + mask).sum(dim=[1, 2, 3])
    dice = (2 * inter / union.clamp(min=eps))
    iou = inter / (pred + mask - pred * mask).sum(dim=[1, 2, 3]).clamp(min=eps)
    return dice, iou


def main():
    parser = argparse.ArgumentParser(description="Binary segmentation eval (Dice/IoU) on VOC-style split.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--split", type=str, required=False, help="Path to split txt (VOC only; otherwise ignored)")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root. For folders mode, point to images; set data.masks or override.")
    parser.add_argument("--text", type=str, default=None, help="Optional text prompt (defaults to cfg.data.text if unset)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_csv", type=str, default=None, help="Optional per-image metrics CSV")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="key=value pairs to override config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = load_config_with_overrides(args.config, args.overrides)
    data_cfg = cfg.get("data", {})
    text_prompt = args.text if args.text is not None else data_cfg.get("text", "lesion")
    input_size = int(data_cfg.get("crop", 224))
    resize_size = data_cfg.get("resize", 256)

    # Build model and load checkpoint
    print("Building model ...")
    model = build_model_from_cfg(cfg, device)

    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        state = torch.load(args.checkpoint, map_location=device)
    except pickle.UnpicklingError as err:
        msg = str(err).lower()
        if "weights only load failed" in msg or "unsupported global" in msg:
            warnings.warn(
                "Falling back to torch.load(..., weights_only=False) for this checkpoint; "
                "ensure the file comes from a trusted source."
            )
            state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        else:
            raise
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    if missing:
        print(f"Warning: missing keys: {len(missing)} e.g. {missing[:5]}")
    if unexpected:
        print(f"Warning: unexpected keys: {len(unexpected)} e.g. {unexpected[:5]}")

    # Dataset + loader
    tfm = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    m_tfm = transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    data_format = str(data_cfg.get("format", "voc")).lower()
    if data_format == "voc":
        dataset = VOCSegDataset(
            root=args.data_root,
            split=args.split,
            transform=tfm,
            mask_transform=m_tfm,
            text_prompt=text_prompt,
            binary=True,
        )
    else:
        dataset = ImageMaskDataset(
            root=args.data_root,
            masks=data_cfg.get("masks"),
            transform=tfm,
            mask_transform=m_tfm,
            text_prompt=text_prompt,
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model.eval()
    dice_total = 0.0
    iou_total = 0.0
    count = 0
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"]
            if masks is None:
                continue
            masks = masks.to(device, non_blocking=True)
            out = model(images, texts=[text_prompt])
            logits = out["low_res_masks"]
            if logits is None:
                continue
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            prob = logits.sigmoid()
            pred = (prob > args.threshold).float()
            dice, iou = compute_batch_metrics(pred, masks)
            dice_total += dice.sum().item()
            iou_total += iou.sum().item()
            batch_size = masks.size(0)
            count += batch_size
            if args.output_csv:
                for i in range(batch_size):
                    rows.append({
                        "id": batch["id"][i],
                        "dice": float(dice[i].item()),
                        "iou": float(iou[i].item()),
                        "mask_coverage": float(masks[i].mean().item()),
                    })
    mean_dice = dice_total / max(count, 1)
    mean_iou = iou_total / max(count, 1)
    split_label = args.split or data_cfg.get("split") or data_format
    print(f"\nResults on split {split_label}:")
    print(f"Samples: {count}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU : {mean_iou:.4f}")

    if args.output_csv and rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.output_csv, index=False)
        print(f"Wrote per-image metrics to {args.output_csv}")


if __name__ == "__main__":
    main()

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from clipdinosam.config import load_config_with_overrides
from clipdinosam.models import (
    build_dino,
    build_clip_text,
    build_sam_decoder,
    TokenProjection,
    TokenToMaskEmbedding,
    CLIPDinoSam,
)
from clipdinosam.lora import inject_lora_linear, set_trainable, enable_only_lora
from clipdinosam.data import VOCSegDataset


def build_model_from_cfg(cfg: Dict, device: torch.device) -> CLIPDinoSam:
    mcfg = cfg["model"]
    dino = build_dino(mcfg["dino"]["name"], pretrained=True)
    clip_text = build_clip_text(mcfg["clip"]["name"], pretrained=mcfg["clip"].get("pretrained", "openai"))
    sam = build_sam_decoder(mcfg["sam"]["type"], checkpoint=mcfg["sam"].get("checkpoint"))

    token_to_text = TokenProjection(in_dim=dino.embed_dim, out_dim=mcfg["clip"].get("width", clip_text.width))
    token_to_mask = TokenToMaskEmbedding(in_dim=dino.embed_dim, embed_dim=mcfg["sam"].get("embed_dim", 256))
    model = CLIPDinoSam(dino, clip_text, sam, token_to_text, token_to_mask)

    # Mirror training-time LoRA / freezing so checkpoints load properly
    stage = cfg.get("stage", 1)
    lora_cfg = cfg.get("lora", {})

    set_trainable(model.clip_text, False)
    set_trainable(model.dino, False)
    set_trainable(model.token_to_text, True)
    set_trainable(model.token_to_mask, True)

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

    if stage in (2, 3):
        dino_lora = lora_cfg.get("dino", {})
        if dino_lora.get("enable", True):
            k = dino_lora.get("last_k_blocks", 2)
            block_targets = []
            if hasattr(model.dino.model, "blocks"):
                total = len(model.dino.model.blocks)
                for i in range(total - k, total):
                    block_targets += [f"blocks.{i}.attn", f"blocks.{i}.mlp"]
            inject_lora_linear(
                model.dino.model,
                block_targets,
                rank=dino_lora.get("rank", 8),
                alpha=dino_lora.get("alpha", 8),
                dropout=dino_lora.get("dropout", 0.0),
            )
        enable_only_lora(model.dino)
    elif stage >= 4:
        set_trainable(model.dino, True)

    return model.to(device)


@torch.no_grad()
def evaluate_voc_dataset(
    model: CLIPDinoSam,
    root: str,
    split: str,
    device: torch.device,
    text_prompt: Optional[str],
    batch_size: int,
    num_workers: int,
    threshold: float,
    image_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
    output_csv: Optional[str] = None,
    save_preds_dir: Optional[str] = None,
    resize: Optional[int] = None,
    crop: Optional[int] = None,
):
    # Transforms aligned with training defaults
    tfm = transforms.Compose([
        transforms.Resize(resize or 256),
        transforms.CenterCrop(crop or 224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    m_tfm = transforms.Compose([
        transforms.Resize(crop or 224, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    ds = VOCSegDataset(
        root=root,
        split=split,
        transform=tfm,
        mask_transform=m_tfm,
        text_prompt=text_prompt,
        binary=True,
        image_dir=image_dir or "JPEGImages",
        mask_dir=mask_dir or "SegmentationClass",
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model.eval()

    # Aggregates
    total_tp = total_fp = total_fn = total_tn = 0
    per_image_rows = []

    if save_preds_dir:
        Path(save_preds_dir).mkdir(parents=True, exist_ok=True)

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch.get("mask")  # can be None
        ids = batch.get("id")

        texts = None
        if batch.get("text"):
            texts = [batch["text"]] if isinstance(batch["text"], str) else batch["text"]

        out = model(images, texts=texts)
        logits = out["low_res_masks"]  # (B, 1|3, h, w)
        # Ensure a single-channel tensor with an explicit channel dim for interpolation
        if logits.dim() == 4 and logits.size(1) > 1:
            logits = logits[:, :1, :, :]
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)

        if masks is not None:
            gt = (masks.to(device) > 0.5).float()  # (B,1,H,W)
            if logits.shape[-2:] != gt.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()  # (B,1,H,W)

        # Save predictions if requested
        if save_preds_dir is not None:
            for i, pid in enumerate(ids):
                pmask = (preds[i, 0] * 255).byte().cpu()
                from PIL import Image

                Image.fromarray(pmask.numpy()).save(os.path.join(save_preds_dir, f"{pid}.png"))

        if masks is None:
            continue

        # Compute metrics per-image and aggregate
        B = preds.size(0)
        for i in range(B):
            p = preds[i, 0].view(-1)
            g = gt[i, 0].view(-1)
            tp = int(((p == 1) & (g == 1)).sum().item())
            tn = int(((p == 0) & (g == 0)).sum().item())
            fp = int(((p == 1) & (g == 0)).sum().item())
            fn = int(((p == 0) & (g == 1)).sum().item())

            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

            denom_iou = tp + fp + fn
            iou = tp / denom_iou if denom_iou > 0 else 1.0
            denom_dice = 2 * tp + fp + fn
            dice = (2 * tp) / denom_dice if denom_dice > 0 else 1.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 1.0

            per_image_rows.append({
                "id": ids[i],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "iou": iou,
                "dice": dice,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "accuracy": acc,
            })

    # Summaries
    total = total_tp + total_tn + total_fp + total_fn
    overall_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    overall_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0
    overall_acc = (total_tp + total_tn) / total if total > 0 else 0.0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0

    if output_csv:
        import csv

        header = list(per_image_rows[0].keys()) if per_image_rows else [
            "id", "tp", "fp", "fn", "tn", "iou", "dice", "precision", "recall", "f1", "accuracy"
        ]
        with open(output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for row in per_image_rows:
                w.writerow(row)

    results = {
        "num_images": len(per_image_rows),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_tn": total_tn,
        "overall_iou": overall_iou,
        "overall_dice": overall_dice,
        "overall_accuracy": overall_acc,
        "overall_precision": overall_prec,
        "overall_recall": overall_rec,
        "overall_f1": overall_f1,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on VOC-style dataset (e.g., HAM10000) test split")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config used for training")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--root", type=str, required=True, help="Root of VOC-style dataset (contains JPEGImages/SegmentationClass/ImageSets)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split name or path to .txt list (default: test)")
    parser.add_argument("--image_dir", type=str, default=None, help="Override image dir name (default: JPEGImages)")
    parser.add_argument("--mask_dir", type=str, default=None, help="Override mask dir name (default: SegmentationClass)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary mask")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional CSV path for per-image metrics")
    parser.add_argument("--save_preds", type=str, default=None, help="Optional directory to save predicted masks (.png)")
    parser.add_argument("--text", type=str, default=None, help="Optional text prompt override (defaults to cfg.data.text)")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="key=value pairs to override config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = load_config_with_overrides(args.config, args.overrides)
    data_cfg = cfg.get("data", {})
    text_prompt = args.text if args.text is not None else data_cfg.get("text")
    resize = int(data_cfg.get("resize", 256))
    crop = int(data_cfg.get("crop", 224))

    print("Building model ...")
    model = build_model_from_cfg(cfg, device)

    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    if missing:
        print(f"Warning: missing keys: {len(missing)} e.g. {missing[:5]}")
    if unexpected:
        print(f"Warning: unexpected keys: {len(unexpected)} e.g. {unexpected[:5]}")

    print("Starting VOC-style evaluation ...")
    print(f"Dataset root: {args.root}")
    print(f"Split: {args.split}")

    results = evaluate_voc_dataset(
        model=model,
        root=args.root,
        split=args.split,
        device=device,
        text_prompt=text_prompt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_csv=args.output_csv,
        save_preds_dir=args.save_preds,
        resize=resize,
        crop=crop,
    )

    print("\n" + "=" * 60)
    print("VOC-STYLE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Images evaluated:   {results['num_images']}")
    print(f"Overall IOU:        {results['overall_iou']:.4f}")
    print(f"Overall Dice:       {results['overall_dice']:.4f}")
    print(f"Overall Accuracy:   {results['overall_accuracy']:.4f}")
    print(f"Overall Precision:  {results['overall_precision']:.4f}")
    print(f"Overall Recall:     {results['overall_recall']:.4f}")
    print(f"Overall F1:         {results['overall_f1']:.4f}")


if __name__ == "__main__":
    main()

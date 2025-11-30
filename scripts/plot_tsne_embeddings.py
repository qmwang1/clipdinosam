#!/usr/bin/env python
"""
Offline t-SNE of token embeddings (CLIP-projected or raw ViT/DINO patch tokens).

Examples:
1) Pooled CLIP token embeddings:
python scripts/plot_tsne_embeddings.py \
  --config configs/ham10000_voc_stage4_sam2_swinv2.yaml \
  --checkpoint experiments/ham10000_onecls_stage4_sam2_swinv2/checkpoints/best.pt \
  --data_root /home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC \
  --split /home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC/ImageSets/Segmentation/val_Melanoma.txt \
  --text "melanoma lesion" \
  --output experiments/ham10000_onecls_stage4_sam2_swinv2/tsne_offline.png

2) ViT feature map (per-patch) t-SNE:
python scripts/plot_tsne_embeddings.py \
  --config configs/ham10000_voc_stage4_sam2_swinv2.yaml \
  --checkpoint experiments/ham10000_onecls_stage4_sam2_swinv2/checkpoints/best.pt \
  --data_root /home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC \
  --split /home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC/ImageSets/Segmentation/val_Melanoma.txt \
  --feature_source vit --patch_level \
  --output experiments/ham10000_onecls_stage4_sam2_swinv2/tsne_vit_patches.png

3) CLS token embeddings from ViT/DINO backbone:
python scripts/plot_tsne_embeddings.py \
  --config configs/ham10000_voc_stage4_sam2_swinv2.yaml \
  --checkpoint experiments/ham10000_onecls_stage4_sam2_swinv2/checkpoints/best.pt \
  --data_root /home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC \
  --split /home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC/ImageSets/Segmentation/val_Melanoma.txt \
  --representation vit_cls \
  --output experiments/ham10000_onecls_stage4_sam2_swinv2/tsne_vit_cls.png

4) CLS-to-patch attention maps (t-SNE over attention vectors):
python scripts/plot_tsne_embeddings.py \
  --config configs/ham10000_voc_stage4_sam2_swinv2.yaml \
  --checkpoint experiments/ham10000_onecls_stage4_sam2_swinv2/checkpoints/best.pt \
  --data_root /home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC \
  --split /home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC/ImageSets/Segmentation/val_Melanoma.txt \
  --representation vit_attn --attn_layer -1 \
  --output experiments/ham10000_onecls_stage4_sam2_swinv2/tsne_vit_attn.png
"""
import argparse
import os
import warnings
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402

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
    backbone_cfg = mcfg.get("backbone") or mcfg["dino"]
    dino = build_backbone(
        backbone_cfg["name"],
        backbone_type=backbone_cfg.get("type"),
        pretrained=backbone_cfg.get("pretrained", True),
        **{k: v for k, v in backbone_cfg.items() if k not in {"name", "type", "pretrained"}},
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
        inject_lora_linear(
            model.sam,
            sam_lora.get("targets", ["attn", "mlp", "lin", "proj"]),
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
                warnings.warn("LoRA enabled for backbone but no target linear layers were matched; skipping injection.")
        enable_only_lora(model.dino)
    elif stage >= 4:
        set_trainable(model.dino, True)

    return model.to(device)


def _safe_collate(batch):
    """Allow batches without masks by leaving mask as None instead of collating Nones."""
    images = default_collate([b["image"] for b in batch])
    masks_list = [b.get("mask") for b in batch]
    masks = None
    if masks_list and all(m is not None for m in masks_list):
        masks = default_collate(masks_list)
    texts = [b.get("text") for b in batch]
    text = texts[0] if texts and all(t == texts[0] for t in texts) else texts
    ids = [b.get("id") for b in batch]
    return {"image": images, "mask": masks, "text": text, "id": ids}


def compute_semantic_labels(mask_map: torch.Tensor) -> torch.Tensor:
    """
    Convert a mask map (H, W) into semantic labels:
      0 = flat/background, 1 = boundary, 2 = lesion/bruise interior.
    Boundary is detected via 3x3 max/min pooling.
    """
    if mask_map.ndim != 2:
        mask_map = mask_map.squeeze()
    mask_bin = (mask_map > 0.5).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    padding = 1  # 3x3 window
    max_pool = F.max_pool2d(mask_bin, kernel_size=3, stride=1, padding=padding)
    # min pooling via negation because older torch lacks min_pool2d
    min_pool = -F.max_pool2d(-mask_bin, kernel_size=3, stride=1, padding=padding)
    boundary = (max_pool - min_pool) > 0.0

    labels = torch.zeros_like(mask_map, dtype=torch.int64)
    labels[mask_bin[0, 0] > 0.5] = 2
    labels[boundary[0, 0]] = 1  # boundary overrides interior
    return labels


def main():
    parser = argparse.ArgumentParser(description="Offline t-SNE of token embeddings.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--data_root", required=True, help="Dataset root")
    parser.add_argument("--split", required=False, help="Split txt (VOC) or omitted for full dataset")
    parser.add_argument("--text", default=None, help="Text prompt (defaults to cfg.data.text)")
    parser.add_argument(
        "--feature_source",
        choices=["clip", "vit"],
        default="clip",
        help="Use CLIP-projected tokens (default) or raw ViT/DINO patch tokens.",
    )
    parser.add_argument(
        "--patch_level",
        action="store_true",
        help="Run t-SNE over individual patch tokens (feature map) instead of per-image pooled vectors.",
    )
    parser.add_argument(
        "--representation",
        choices=["clip_pooled", "clip_patch", "vit_pooled", "vit_patch", "vit_cls", "vit_attn"],
        default=None,
        help="Explicit representation to t-SNE. Overrides feature_source/patch_level if set.",
    )
    parser.add_argument(
        "--attn_layer",
        type=int,
        default=-1,
        help="ViT block index for attention maps (only used with representation=vit_attn).",
    )
    parser.add_argument("--max_samples", type=int, default=256, help="Max points to plot")
    parser.add_argument("--max_batches", type=int, default=4, help="Max batches to sample")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--learning_rate", type=float, default=200.0)
    parser.add_argument("--n_iter", type=int, default=750)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_dir", default=None, help="Optional images subdir (defaults to JPEGImages or cfg)")
    parser.add_argument("--mask_dir", default=None, help="Optional masks subdir (defaults to SegmentationClass or cfg)")
    parser.add_argument("--output", default="tsne_offline.png", help="Output PNG path")
    parser.add_argument(
        "--color_mode",
        choices=["mask", "semantic", "dataset"],
        default="mask",
        help="Color points by mask coverage (default), semantic labels (boundary/interior/skin), or dataset label.",
    )
    parser.add_argument(
        "--dataset_label",
        type=int,
        default=0,
        help="Integer label for dataset coloring (used when --color_mode=dataset).",
    )
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="key=value config overrides")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config_with_overrides(args.config, args.overrides)
    data_cfg = cfg.get("data", {})
    text_prompt = args.text if args.text is not None else data_cfg.get("text", "lesion")
    resize_size = data_cfg.get("resize", 256)
    crop_size = int(data_cfg.get("crop", 224))

    print(f"Using device: {device}")
    print("Building model ...")
    model = build_model_from_cfg(cfg, device)

    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        state = torch.load(args.checkpoint, map_location=device)
    except pickle.UnpicklingError as err:
        msg = str(err).lower()
        if "weights only load failed" in msg or "unsupported global" in msg:
            warnings.warn("Retrying torch.load(..., weights_only=False); ensure checkpoint is trusted.")
            state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        else:
            raise
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    if missing:
        print(f"Warning: missing keys: {len(missing)} e.g. {missing[:5]}")
    if unexpected:
        print(f"Warning: unexpected keys: {len(unexpected)} e.g. {unexpected[:5]}")
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    m_tfm = transforms.Compose([
        transforms.Resize(crop_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    data_format = str(data_cfg.get("format", "voc")).lower()
    if data_format == "voc":
        image_dir = args.image_dir or data_cfg.get("image_dir", "JPEGImages")
        mask_dir = args.mask_dir or data_cfg.get("mask_dir", "SegmentationClass")
        dataset = VOCSegDataset(
            root=args.data_root,
            split=args.split or data_cfg.get("split"),
            transform=tfm,
            mask_transform=m_tfm,
            text_prompt=text_prompt,
            binary=data_cfg.get("binary", True),
            image_dir=image_dir,
            mask_dir=mask_dir,
        )
    else:
        dataset = ImageMaskDataset(
            root=args.data_root,
            masks=data_cfg.get("masks"),
            transform=tfm,
            mask_transform=m_tfm,
            text_prompt=text_prompt,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_safe_collate,
        pin_memory=(device.type == "cuda"),
    )

    if args.representation is None:
        if args.feature_source == "clip":
            representation = "clip_patch" if args.patch_level else "clip_pooled"
        else:
            representation = "vit_patch" if args.patch_level else "vit_pooled"
    else:
        representation = args.representation

    print(f"t-SNE representation: {representation}")
    want_patch_tokens = representation in {"clip_patch", "vit_patch"}
    want_cls_token = representation == "vit_cls"
    want_attn_map = representation == "vit_attn"
    use_clip = representation.startswith("clip")

    feats = []
    colors = []
    ids = []
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if args.max_batches is not None and b_idx >= int(args.max_batches):
                break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch.get("mask")
            batch_ids = batch.get("id")
            if use_clip:
                out = model(images, texts=[text_prompt], skip_sam=True)
                tokens = out["clip_tokens"]  # (B, N, D)
                grid_hw = out.get("grid_hw")
                cls_tokens = None
                attn_map = None
                if isinstance(grid_hw, torch.Tensor):
                    grid_hw = tuple(int(x) for x in grid_hw.detach().cpu().tolist())
            else:
                vt_out = model.dino.forward_tokens(
                    images,
                    return_cls=(want_cls_token or want_attn_map),
                    return_attn=want_attn_map,
                    attn_layer=args.attn_layer,
                )
                tokens = None
                grid_hw = None
                cls_tokens = None
                attn_map = None
                if want_cls_token or want_attn_map:
                    if isinstance(vt_out, (list, tuple)) and len(vt_out) == 4:
                        tokens, grid_hw, cls_tokens, attn_map = vt_out  # type: ignore[misc]
                    elif isinstance(vt_out, (list, tuple)) and len(vt_out) == 2:
                        tokens, grid_hw = vt_out  # type: ignore[misc]
                    else:
                        tokens = vt_out  # type: ignore[assignment]
                else:
                    if isinstance(vt_out, (list, tuple)) and len(vt_out) >= 2:
                        tokens, grid_hw = vt_out[0], vt_out[1]  # type: ignore[misc]
                    else:
                        tokens = vt_out  # type: ignore[assignment]
                if want_cls_token and cls_tokens is None:
                    raise RuntimeError("CLS token requested but backbone did not return one (is your backbone ViT/DINO?).")
                if want_attn_map and attn_map is None:
                    raise RuntimeError(
                        "Attention map requested but backbone does not expose attention weights (requires ViT/DINO backbone)."
                    )
            bsz, num_tokens, _ = tokens.shape
            grid_hw = tuple(grid_hw) if grid_hw is not None else None
            if want_patch_tokens:
                if grid_hw is None or grid_hw[0] * grid_hw[1] != num_tokens:
                    raise ValueError("Patch-level t-SNE requested but token grid size is unavailable.")
                h, w = int(grid_hw[0]), int(grid_hw[1])
                token_map = tokens.view(bsz, h, w, -1).detach().cpu()
                mask_map = None
                if masks is not None:
                    mask_tensor = masks.to(device=device, dtype=torch.float32)
                    mask_map = F.interpolate(mask_tensor, size=(h, w), mode="area").detach().cpu()
                for i in range(bsz):
                    coverage_map = mask_map[i, 0] if mask_map is not None else None
                    semantic_map = compute_semantic_labels(coverage_map) if (coverage_map is not None and args.color_mode == "semantic") else None
                    bid = batch_ids[i] if isinstance(batch_ids, (list, tuple)) else batch_ids
                    bid = str(bid) if bid is not None else f"{b_idx}_{i}"
                    for y in range(h):
                        for x in range(w):
                            feats.append(token_map[i, y, x])
                            if args.color_mode == "dataset":
                                colors.append(args.dataset_label)
                            elif semantic_map is not None:
                                colors.append(int(semantic_map[y, x].item()))
                            elif coverage_map is not None:
                                colors.append(float(coverage_map[y, x].item()))
                            else:
                                colors.append(0.0)
                            ids.append(f"{bid}_p{y}x{x}")
                            if len(feats) >= args.max_samples:
                                break
                        if len(feats) >= args.max_samples:
                            break
                    if len(feats) >= args.max_samples:
                        break
            elif want_cls_token:
                if cls_tokens is None:
                    raise RuntimeError("CLS token requested but backbone did not return one (is the backbone ViT/DINO?).")
                if cls_tokens.dim() == 3 and cls_tokens.shape[1] == 1:
                    cls_tokens = cls_tokens[:, 0, :]
                for i in range(bsz):
                    feats.append(cls_tokens[i].detach().cpu())
                    if args.color_mode == "dataset":
                        colors.append(args.dataset_label)
                    elif masks is not None:
                        colors.append(float(masks[i].mean().item()))
                    else:
                        colors.append(0.0)
                    bid = batch_ids[i] if isinstance(batch_ids, (list, tuple)) else batch_ids
                    ids.append(str(bid) if bid is not None else f"{b_idx}_{i}")
                    if len(feats) >= args.max_samples:
                        break
            elif want_attn_map:
                if attn_map is None:
                    raise RuntimeError(
                        "Attention map requested but backbone did not expose attn weights (requires ViT/DINO with accessible blocks)."
                    )
                cls_attn = attn_map.mean(dim=1)  # (B, N, N)
                cls_attn = cls_attn[:, 0, 1:]     # drop CLS->CLS, keep CLS->patch
                for i in range(bsz):
                    feats.append(cls_attn[i].detach().cpu())
                    if args.color_mode == "dataset":
                        colors.append(args.dataset_label)
                    elif masks is not None:
                        colors.append(float(masks[i].mean().item()))
                    else:
                        colors.append(0.0)
                    bid = batch_ids[i] if isinstance(batch_ids, (list, tuple)) else batch_ids
                    ids.append(str(bid) if bid is not None else f"{b_idx}_{i}")
                    if len(feats) >= args.max_samples:
                        break
            else:
                pooled = tokens.mean(dim=1).cpu()
                for i in range(pooled.size(0)):
                    feats.append(pooled[i])
                    if args.color_mode == "dataset":
                        colors.append(args.dataset_label)
                    elif masks is not None:
                        colors.append(float(masks[i].mean().item()))
                    else:
                        colors.append(0.0)
                    bid = batch_ids[i] if isinstance(batch_ids, (list, tuple)) else batch_ids
                    ids.append(str(bid) if bid is not None else f"{b_idx}_{i}")
                    if len(feats) >= args.max_samples:
                        break
            if len(feats) >= args.max_samples:
                break

    if len(feats) < 2:
        print(f"Collected {len(feats)} samples; need at least 2. Aborting.")
        return

    feat_mat = torch.stack(feats, dim=0).numpy()
    perplexity = max(5, min(args.perplexity, len(feat_mat) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        init="random",
        metric="euclidean",
        verbose=False,
    )
    coords = tsne.fit_transform(feat_mat)

    out_path = args.output if os.path.isabs(args.output) else os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = "viridis"
    norm = None
    cbar_ticks = None
    cbar_ticklabels = None
    cbar_label = "Mask coverage (mean)"
    if args.color_mode == "semantic":
        cmap = ListedColormap(["royalblue", "crimson", "limegreen"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
        cbar_ticks = [0, 1, 2]
        cbar_ticklabels = ["flat/skin", "boundary", "lesion/bruise"]
        cbar_label = "Semantic label"
    elif args.color_mode == "dataset":
        cmap = plt.cm.get_cmap("tab10")
        cbar_label = "Dataset label"

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=colors, cmap=cmap, norm=norm, s=16, alpha=0.85, edgecolors="none")
    desc_map = {
        "clip_patch": "CLIP patch tokens",
        "clip_pooled": "CLIP pooled tokens",
        "vit_patch": "ViT patch tokens",
        "vit_pooled": "ViT pooled tokens",
        "vit_cls": "ViT CLS tokens",
        "vit_attn": "ViT CLS attention maps",
    }
    feature_desc = desc_map.get(representation, representation)
    ax.set_title(f"t-SNE of {feature_desc} (offline)")
    ax.set_xlabel("TSNE-1")
    ax.set_ylabel("TSNE-2")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks)
    if cbar_ticklabels is not None:
        cbar.ax.set_yticklabels(cbar_ticklabels)
    cbar.set_label(cbar_label)
    if len(ids) <= 30:
        for (x, y), label in zip(coords, ids):
            ax.text(x, y, str(label), fontsize=6, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"t-SNE saved to {out_path}")


if __name__ == "__main__":
    main()

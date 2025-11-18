import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from clipdinosam.config import load_config_with_overrides
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


def build_model_from_cfg(cfg: Dict, device: torch.device) -> CLIPDinoSam:
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
            target_pairs = model.dino.lora_target_pairs(k)
            for module, target_substrings in target_pairs:
                inject_lora_linear(
                    module,
                    target_substrings,
                    rank=dino_lora.get("rank", 8),
                    alpha=dino_lora.get("alpha", 8),
                    dropout=dino_lora.get("dropout", 0.0),
                )
        enable_only_lora(model.dino)
    elif stage >= 4:
        set_trainable(model.dino, True)

    return model.to(device)


def list_images(folder: Path, exts: Optional[Sequence[str]] = None) -> List[Path]:
    extensions = {e.lower() for e in (exts or [".jpg", ".jpeg", ".png", ".bmp"])}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in extensions])


def overlay_mask(base_img: Image.Image, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> Image.Image:
    """Blend mask probabilities onto the resized/cropped image for visualization."""
    mask = np.clip(mask, 0.0, 1.0)
    base = np.array(base_img).astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    overlay = base * (1.0 - alpha * mask[..., None]) + color_arr * (alpha * mask[..., None])
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Run inference on an unlabeled folder and save mask visualizations.")
    parser.add_argument("--config", required=True, help="Config used for training.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt).")
    parser.add_argument("--input_dir", required=True, help="Directory containing images.")
    parser.add_argument("--output_dir", required=True, help="Directory to save overlays/masks.")
    parser.add_argument("--text", default=None, help="Optional text override (defaults to cfg.data.text).")
    parser.add_argument("--resize", type=int, default=None, help="Resize for inference (defaults to cfg.data.resize or 256).")
    parser.add_argument("--crop", type=int, default=None, help="Center crop size (defaults to cfg.data.crop or 224).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Optional threshold for binary masks.")
    parser.add_argument("--overlay_color", type=int, nargs=3, default=(255, 0, 0), help="RGB color for overlay tint.")
    parser.add_argument("--overlay_alpha", type=float, default=0.45, help="Alpha strength for overlay tint.")
    parser.add_argument("--save_binary", action="store_true", help="Also save binarized masks.")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="key=value overrides for the config.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = load_config_with_overrides(args.config, args.overrides)
    data_cfg = cfg.get("data", {})
    resize = args.resize or int(data_cfg.get("resize", 256))
    crop = args.crop or int(data_cfg.get("crop", 224))
    text_prompt = args.text if args.text is not None else data_cfg.get("text")

    print("Building model ...")
    model = build_model_from_cfg(cfg, device)

    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        state = torch.load(args.checkpoint, map_location=device)
    except pickle.UnpicklingError as err:
        print("Encountered pickle error, retrying with weights_only=False. Ensure the checkpoint is trusted.")
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    if missing:
        print(f"Warning: missing keys ({len(missing)}). Example: {missing[:5]}")
    if unexpected:
        print(f"Warning: unexpected keys ({len(unexpected)}). Example: {unexpected[:5]}")

    model.eval()

    tfm = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    vis_tfm = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
    ])

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    bin_dir = output_dir / "binary_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_binary:
        bin_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No images found under {input_dir}.")

    print(f"Found {len(images)} images. Starting inference ...")
    texts = [text_prompt] if text_prompt else None

    for idx, img_path in enumerate(images, 1):
        img = Image.open(img_path).convert("RGB")
        vis_img = vis_tfm(img)
        tensor = tfm(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor, texts=texts)

        logits = out["low_res_masks"]
        if logits.dim() == 4 and logits.size(1) > 1:
            logits = logits[:, :1, :, :]
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        probs = torch.sigmoid(logits)
        probs = F.interpolate(probs, size=(vis_img.height, vis_img.width), mode="bilinear", align_corners=False)
        prob_map = probs[0, 0].cpu().numpy()

        mask_img = Image.fromarray(np.clip(prob_map * 255.0, 0, 255).astype(np.uint8))
        mask_img.save(mask_dir / f"{img_path.stem}.png")

        overlay = overlay_mask(vis_img, prob_map, tuple(args.overlay_color), args.overlay_alpha)
        overlay.save(overlay_dir / f"{img_path.stem}.png")

        if args.save_binary:
            binary = (prob_map >= args.threshold).astype(np.uint8) * 255
            Image.fromarray(binary).save(bin_dir / f"{img_path.stem}.png")

        if idx % 20 == 0 or idx == len(images):
            print(f"[{idx}/{len(images)}] processed {img_path.name}")

    print(f"Done. Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()

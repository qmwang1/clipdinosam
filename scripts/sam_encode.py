import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image


def list_images(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])


def load_sam(sam_type: str, checkpoint: str, device: torch.device):
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except Exception as e:
        raise ImportError("segment-anything is required. Install from the official repo.") from e

    sam = sam_model_registry[sam_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


def main():
    parser = argparse.ArgumentParser(description="Export SAM image embeddings for a folder or a list of images")
    parser.add_argument("--sam_type", type=str, required=True, help="SAM variant: vit_t | vit_b | vit_l | vit_h")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM checkpoint (.pth)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", type=str, help="Directory of images (recursively searched)")
    group.add_argument("--list_file", type=str, help="Text file with absolute/relative image paths, one per line")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save embeddings")
    parser.add_argument("--format", type=str, default="pt", choices=["pt", "npy"], help="Save format")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Collect image paths
    if args.input_dir:
        images = list_images(Path(args.input_dir))
    else:
        with open(args.list_file, "r") as f:
            images = [Path(line.strip()) for line in f if line.strip()]
    if not images:
        print("No images found")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = load_sam(args.sam_type, args.checkpoint, device)

    for img_path in images:
        try:
            image = Image.open(img_path).convert("RGB")
            image_np = np.asarray(image)
            predictor.set_image(image_np)
            # Most SAM versions expose get_image_embedding(); fall back to internal features otherwise
            with torch.no_grad():
                try:
                    embedding = predictor.get_image_embedding()  # (1, 256, 64, 64)
                except Exception:
                    # Fallback: use internal cached features if available
                    if hasattr(predictor, "features") and predictor.features is not None:
                        embedding = predictor.features
                    elif hasattr(predictor, "_features") and predictor._features is not None:
                        embedding = predictor._features
                    else:
                        # As a last resort, recompute via model.image_encoder on transformed image
                        transformed = predictor.transform.apply_image(image_np)
                        input_t = torch.as_tensor(transformed, device=device).permute(2, 0, 1).unsqueeze(0)
                        embedding = predictor.model.image_encoder(input_t)

            stem = img_path.stem
            if args.format == "pt":
                torch.save({
                    "embedding": embedding.detach().cpu(),
                    "orig_size": image.size,  # (W,H)
                    "path": str(img_path),
                }, out_dir / f"{stem}.pt")
            else:
                np.save(out_dir / f"{stem}.npy", embedding.detach().cpu().numpy())
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")


if __name__ == "__main__":
    main()


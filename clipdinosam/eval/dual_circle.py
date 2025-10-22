import os
import glob
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
import pandas as pd


def extract_dual_circle_regions_from_mask(mask_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract dual-circle regions directly from the colored mask.

    Expected mask format:
    - Red circle: Inner circle (pure lesion region)
    - Cyan/Blue ring: Outer ring (pure background region)
    - Black area: Ignored region (between circles)

    Returns:
        (inner_region, annulus_region, outer_region) as boolean masks
    """
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(mask_rgb)

    red_mask = (r > 200) & (g < 100) & (b < 100)
    cyan_mask = (b > 200) & (g > 200) & (r < 100)

    h, w = mask_rgb.shape[:2]
    boundary_mask = cyan_mask
    flood_src = np.where(boundary_mask, 255, 0).astype(np.uint8)
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood_src, flood_mask, (0, 0), 128)
    cv2.floodFill(flood_src, flood_mask, (w - 1, 0), 128)
    cv2.floodFill(flood_src, flood_mask, (0, h - 1), 128)
    cv2.floodFill(flood_src, flood_mask, (w - 1, h - 1), 128)
    exterior_region = (flood_src == 128)

    annulus_mask = ~(exterior_region | red_mask | boundary_mask)

    return red_mask, annulus_mask, cyan_mask


def extract_prediction_mask(pred_tensor: torch.Tensor | np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if isinstance(pred_tensor, torch.Tensor):
        pred_tensor = pred_tensor.detach().cpu().numpy()
    return (pred_tensor > threshold).astype(np.uint8)


def evaluate_dual_circle_single_image(
    pred_mask: np.ndarray,
    annotation_image_bgr: np.ndarray,
    additional_ignore_mask: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    try:
        inner_region, annulus_region, outer_region = extract_dual_circle_regions_from_mask(annotation_image_bgr)
        if inner_region.shape != pred_mask.shape:
            inner_region = cv2.resize(inner_region.astype(np.uint8), pred_mask.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            annulus_region = cv2.resize(annulus_region.astype(np.uint8), pred_mask.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            outer_region = cv2.resize(outer_region.astype(np.uint8), pred_mask.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)

        if additional_ignore_mask is not None:
            mask = additional_ignore_mask
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            if mask.shape != pred_mask.shape:
                mask = cv2.resize(mask.astype(np.uint8), pred_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
            extra_ignore_region = (mask > 0)
        else:
            extra_ignore_region = np.zeros_like(inner_region, dtype=bool)

        if extra_ignore_region.any():
            # Ensure ignore regions do not overlap with evaluation regions
            inner_region = inner_region & ~extra_ignore_region
            annulus_region = annulus_region & ~extra_ignore_region
            outer_region = outer_region & ~extra_ignore_region

        everything_else = ~(inner_region | annulus_region | outer_region | extra_ignore_region)
        ignore_region = annulus_region | extra_ignore_region
        background_region = (outer_region | everything_else) & ~extra_ignore_region

        tp = int(pred_mask[inner_region].sum())
        ignored = int(pred_mask[ignore_region].sum())
        fp = int(pred_mask[background_region].sum())

        total_inner = int(inner_region.sum())
        total_back = int(background_region.sum())
        total_ignore = int(ignore_region.sum())
        total_additional_ignore = int(extra_ignore_region.sum())

        fn = total_inner - tp
        tn = total_back - fp

        tp_pct = (tp / total_inner * 100) if total_inner > 0 else 0.0
        fp_pct = (fp / total_back * 100) if total_back > 0 else 0.0
        fn_pct = (fn / total_inner * 100) if total_inner > 0 else 0.0
        tn_pct = (tn / total_back * 100) if total_back > 0 else 0.0

        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        return {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "ignored": ignored,
            "total_inner_pixels": total_inner,
            "total_background_pixels": total_back,
            "total_ignore_pixels": total_ignore,
            "total_additional_ignore_pixels": total_additional_ignore,
            "tp_percentage": tp_pct,
            "fp_percentage": fp_pct,
            "fn_percentage": fn_pct,
            "tn_percentage": tn_pct,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "inner_region": inner_region,
            "annulus_region": annulus_region,
            "outer_region": outer_region,
            "background_region": background_region,
            "additional_ignore_region": extra_ignore_region,
        }
    except Exception as e:
        print(f"Error during dual-circle evaluation: {e}")
        return None


def _expand_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(path))


def _list_files_recursive(root: str, exts: List[str]) -> List[str]:
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True))
    return files


def _match_circle_mask(
    circle_dir: str,
    img_name: str,
    mask_files: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Try to find a corresponding mask for an image name using several heuristics:
    - Exact basename match after replacing '-noCircles' with '-mask'
    - Same stem (basename without extension)
    - Stem with '-mask' replacement
    - Filenames starting with stem and containing 'mask' or 'circle'
    Searches recursively and accepts PNG/JPG/JPEG (any case).
    """
    circle_dir = _expand_path(circle_dir)
    exts = ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]
    if mask_files is None:
        mask_files = _list_files_recursive(circle_dir, exts)

    base = os.path.basename(img_name)
    stem, _ = os.path.splitext(base)
    cand_exact = base.replace("-noCircles", "-mask")
    stem_mask = stem.replace("-noCircles", "-mask")
    stem_l = stem.lower()

    # 1) Exact basename match
    for p in mask_files:
        if os.path.basename(p) == cand_exact:
            return p

    # 2) Same stem (any extension)
    for p in mask_files:
        if os.path.splitext(os.path.basename(p))[0] == stem:
            return p

    # 3) Stem with '-mask' replacement
    for p in mask_files:
        if os.path.splitext(os.path.basename(p))[0] == stem_mask:
            return p

    # 4) Startswith stem and contains mask/circle
    for p in mask_files:
        bn = os.path.basename(p)
        bn_l = bn.lower()
        if bn_l.startswith(stem_l) and ("mask" in bn_l or "circle" in bn_l):
            return p

    # 5) Fallback: contains stem and mask/circle anywhere
    for p in mask_files:
        bn_l = os.path.basename(p).lower()
        if stem_l in bn_l and ("mask" in bn_l or "circle" in bn_l):
            return p

    return None


def _match_rectangle_ignore_mask(
    rect_dir: Optional[str],
    img_name: str,
    mask_files: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Try to find an ignore-rectangle mask corresponding to the image.
    Accepts masks whose names contain the image stem and 'rect'/'ignore'.
    """
    if not rect_dir:
        return None
    rect_dir = _expand_path(rect_dir)

    exts = ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]
    if mask_files is None:
        mask_files = _list_files_recursive(rect_dir, exts)

    base = os.path.basename(img_name)
    stem, _ = os.path.splitext(base)
    stem_variants = {
        stem,
        stem.replace("-noCircles", ""),
        stem.replace("-nocircles", ""),
    }
    candidates: List[str] = []
    for s in stem_variants:
        if not s:
            continue
        candidates.extend(
            [
                s,
                f"{s}_blue_rect_mask",
                f"{s}_blue_rect",
                f"{s}_ignore_rect",
                f"{s}_rect_ignore",
                f"{s}_ignore",
            ]
        )

    seen = set()
    ordered_candidates = []
    for c in candidates:
        if c not in seen:
            ordered_candidates.append(c)
            seen.add(c)

    for p in mask_files:
        name, _ = os.path.splitext(os.path.basename(p))
        if name in ordered_candidates:
            return p

    stem_l = stem.lower()
    for p in mask_files:
        bn_l = os.path.basename(p).lower()
        if stem_l in bn_l and ("rect" in bn_l or "ignore" in bn_l):
            return p

    return None


def get_image_pairs(image_dir: str, circle_dir: str) -> List[Tuple[str, str]]:
    image_dir = _expand_path(image_dir)
    circle_dir = _expand_path(circle_dir)

    exts = ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]
    images = _list_files_recursive(image_dir, exts)
    mask_files = _list_files_recursive(circle_dir, exts)

    pairs: List[Tuple[str, str]] = []
    for img_path in sorted(images):
        img_name = os.path.basename(img_path)
        circle_path = _match_circle_mask(circle_dir, img_name, mask_files=mask_files)
        if circle_path and os.path.exists(circle_path):
            pairs.append((img_path, circle_path))
        else:
            print(f"Warning: No circle annotation found for {img_name}")
    return pairs


def visualize_dual_circle_evaluation_from_regions(
    image: np.ndarray | torch.Tensor,
    pred_mask: np.ndarray | torch.Tensor,
    inner_region: np.ndarray,
    annulus_region: np.ndarray,
    outer_region: np.ndarray,
    additional_ignore_region: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    alpha: float = 0.6,
):
    import matplotlib.pyplot as plt

    if additional_ignore_region is None:
        additional_ignore_region = np.zeros_like(inner_region, dtype=bool)
    else:
        additional_ignore_region = additional_ignore_region.astype(bool)

    background_region = ~(inner_region | annulus_region | additional_ignore_region)

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()

    if image.max() <= 1.0:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original Image with Evaluation Regions")
    region_overlay = np.zeros((*inner_region.shape, 4))
    region_overlay[inner_region] = [1, 0, 0, 0.3]
    region_overlay[annulus_region] = [1, 1, 0, 0.3]
    region_overlay[background_region] = [0, 0, 1, 0.3]
    region_overlay[additional_ignore_region] = [0.6, 0, 0.6, 0.3]
    axes[0].imshow(region_overlay)
    axes[0].axis("off")

    axes[1].imshow(image)
    pred_overlay = np.zeros((*pred_mask.shape, 4))
    pred_overlay[pred_mask > 0.5] = [1, 0, 0, alpha]
    axes[1].imshow(pred_overlay)
    axes[1].set_title("Prediction Overlay")
    axes[1].axis("off")

    axes[2].imshow(image)
    region_masks = np.zeros((*inner_region.shape, 4))
    region_masks[inner_region] = [1, 0, 0, 0.5]
    region_masks[annulus_region] = [1, 1, 0, 0.5]
    region_masks[background_region] = [0, 0, 1, 0.5]
    region_masks[additional_ignore_region] = [0.6, 0, 0.6, 0.5]
    axes[2].imshow(region_masks)
    axes[2].set_title("Region Definitions\n(Red=Inner, Yellow=Annulus, Blue=Outer, Purple=Extra Ignore)")
    axes[2].axis("off")

    axes[3].imshow(image)
    evaluation_overlay = np.zeros(pred_mask.shape + (4,))
    tp_mask = (pred_mask > 0.5) & inner_region
    fp_mask = (pred_mask > 0.5) & background_region
    ignored_mask = (pred_mask > 0.5) & (annulus_region | additional_ignore_region)
    evaluation_overlay[tp_mask] = [0, 1, 0, alpha]
    evaluation_overlay[fp_mask] = [1, 0, 0, alpha]
    evaluation_overlay[ignored_mask] = [1, 1, 0, alpha]
    axes[3].imshow(evaluation_overlay)
    axes[3].set_title("Evaluation Results\n(Green=TP, Red=FP, Yellow=Ignored)")
    axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
        plt.close()
    else:
        plt.show()


def evaluate_dual_circle_dataset(
    model,
    image_dir: str,
    circle_dir: str,
    device: torch.device,
    rectangle_dir: Optional[str] = None,
    text_prompt: Optional[str] = None,
    output_csv: Optional[str] = None,
    visualize_samples: bool = False,
    vis_output_dir: Optional[str] = None,
    visualize_specific: Optional[str] = None,
    input_size: int = 224,
):
    from tqdm import tqdm

    model.eval()
    results: List[Dict] = []
    pairs = get_image_pairs(image_dir, circle_dir)
    print(f"Found {len(pairs)} image pairs to evaluate")

    rect_dir_expanded = _expand_path(rectangle_dir) if rectangle_dir else None
    rect_mask_files: Optional[List[str]] = None
    if rect_dir_expanded:
        rect_mask_files = _list_files_recursive(
            rect_dir_expanded,
            ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"],
        )

    if visualize_samples and vis_output_dir:
        os.makedirs(vis_output_dir, exist_ok=True)

    prompt_list = [text_prompt] if text_prompt else None

    with torch.no_grad():
        for i, (img_path, circle_path) in enumerate(tqdm(pairs, desc="Evaluating")):
            img_name = os.path.basename(img_path)
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f"Warning: could not load image {img_path}")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            img_resized = cv2.resize(image_rgb, (input_size, input_size))
            img_chw = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_norm = (img_chw - mean) / std
            batch = img_norm.unsqueeze(0).to(device)

            outputs = model(batch, texts=prompt_list)
            # Avoid Python truthiness on tensors (ambiguous for multi-element tensors)
            logits = outputs.get("low_res_masks", None)
            if logits is None:
                logits = outputs.get("masks", None)
            if logits is None:
                print("Warning: model output did not contain masks; skipping")
                continue

            if logits.dim() == 4:
                mask_logits = logits[0, 0]
            elif logits.dim() == 3:
                mask_logits = logits[0]
            else:
                mask_logits = logits

            prob = torch.sigmoid(mask_logits)
            H, W = image_bgr.shape[:2]
            m = prob
            while m.dim() < 4:
                m = m.unsqueeze(0)
            m_up = torch.nn.functional.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
            while m_up.dim() > 2:
                m_up = m_up.squeeze(0)

            pred_mask = extract_prediction_mask(m_up)

            circle_bgr = cv2.imread(circle_path)
            if circle_bgr is None:
                print(f"Warning: could not load circle annotation {circle_path}")
                continue

            rect_mask = None
            rect_path = None
            if rect_dir_expanded:
                rect_path = _match_rectangle_ignore_mask(rect_dir_expanded, img_name, mask_files=rect_mask_files)
                if rect_path and os.path.exists(rect_path):
                    rect_mask = cv2.imread(rect_path, cv2.IMREAD_GRAYSCALE)
                    if rect_mask is None:
                        print(f"Warning: could not load rectangle mask {rect_path}")
                elif rect_path:
                    print(f"Warning: rectangle mask path not found {rect_path}")

            res = evaluate_dual_circle_single_image(
                pred_mask=pred_mask,
                annotation_image_bgr=circle_bgr,
                additional_ignore_mask=rect_mask,
            )
            if res is None:
                continue

            res["image_name"] = os.path.basename(img_path)
            res["image_path"] = img_path
            res["circle_path"] = circle_path
            res["ignore_rectangle_path"] = rect_path
            results.append(res)

            if visualize_samples and vis_output_dir:
                do_vis = False
                if visualize_specific is None:
                    do_vis = i < 5
                else:
                    nm = os.path.basename(img_path)
                    do_vis = nm == visualize_specific or str(i) == visualize_specific or str(i + 1) == visualize_specific
                if do_vis:
                    img_for_vis = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
                    out_path = os.path.join(vis_output_dir, f"eval_{res['image_name']}")
                    try:
                        visualize_dual_circle_evaluation_from_regions(
                            image=img_for_vis,
                            pred_mask=pred_mask,
                            inner_region=res["inner_region"],
                            annulus_region=res["annulus_region"],
                            outer_region=res["outer_region"],
                            additional_ignore_region=res["additional_ignore_region"],
                            save_path=out_path,
                        )
                    except Exception as e:
                        print(f"Visualization failed for {res['image_name']}: {e}")

    if not results:
        print("No valid results computed")
        return {}

    df = pd.DataFrame(results)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Detailed results saved to: {output_csv}")

    total_tp = df["true_positives"].sum()
    total_fp = df["false_positives"].sum()
    total_tn = df["true_negatives"].sum()
    total_fn = df["false_negatives"].sum()
    total_ig = df["ignored"].sum()
    total_ignore_pixels = df["total_ignore_pixels"].sum()
    total_additional_ignore_pixels = df["total_additional_ignore_pixels"].sum()

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    total_eval = total_tp + total_fp + total_tn + total_fn
    overall_acc = (total_tp + total_tn) / total_eval if total_eval > 0 else 0.0

    return {
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "overall_accuracy": overall_acc,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_tn": total_tn,
        "total_fn": total_fn,
        "total_ignored": total_ig,
        "total_ignore_pixels": total_ignore_pixels,
        "total_additional_ignore_pixels": total_additional_ignore_pixels,
        "num_images": len(results),
        "mean_tp_percentage": df["tp_percentage"].mean(),
        "mean_fp_percentage": df["fp_percentage"].mean(),
        "mean_f1_score": df["f1_score"].mean(),
        "std_f1_score": df["f1_score"].std(),
        "detailed_results": df,
    }

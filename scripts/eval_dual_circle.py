###python scripts/eval_dual_circle.py --config configs/ham10000_voc_stage3.yaml --checkpoint experiments/runA-stage3/checkpoints/latest.pt --image_dir ~/Documents/AllBruiseTrainingData/Unlabelled/uni_dataset/no_circle_data --circle_dir ~/Documents/AllBruiseTrainingData/Unlabelled/uni_dataset/circle_mask --output_csv experiments/runA-stage3/dual_circle_results.csv

import argparse
import torch

from clipdinosam.config import load_config_with_overrides
from clipdinosam.models import (
    build_dino,
    build_clip_text,
    build_sam_decoder,
    TokenProjection,
    TokenToMaskEmbedding,
    CLIPDinoSam,
)
from clipdinosam.eval import evaluate_dual_circle_dataset
from clipdinosam.lora import inject_lora_linear, set_trainable, enable_only_lora


def build_model_from_cfg(cfg, device: torch.device) -> CLIPDinoSam:
    mcfg = cfg["model"]
    dino = build_dino(mcfg["dino"]["name"], pretrained=True)
    clip_text = build_clip_text(mcfg["clip"]["name"], pretrained=mcfg["clip"].get("pretrained", "openai"))
    sam = build_sam_decoder(mcfg["sam"]["type"], checkpoint=mcfg["sam"].get("checkpoint"))

    token_to_text = TokenProjection(in_dim=dino.embed_dim, out_dim=mcfg["clip"].get("width", clip_text.width))
    token_to_mask = TokenToMaskEmbedding(in_dim=dino.embed_dim, embed_dim=mcfg["sam"].get("embed_dim", 256))
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


def main():
    parser = argparse.ArgumentParser(description="Dual-circle evaluation for CLIP-DINO-SAM")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of images without circles")
    parser.add_argument("--circle_dir", type=str, required=True, help="Directory of dual-circle annotation masks")
    parser.add_argument("--output_csv", type=str, default="dual_circle_results.csv", help="Where to write per-image results CSV")
    parser.add_argument("--text", type=str, default=None, help="Optional text prompt (defaults to cfg.data.text if unset)")
    parser.add_argument("--visualize", action="store_true", help="Save sample visualizations")
    parser.add_argument("--visualize_specific", type=str, default=None, help="Filename or index to visualize")
    parser.add_argument("--vis_output_dir", type=str, default="visualizations", help="Visualization output directory")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="key=value pairs to override config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = load_config_with_overrides(args.config, args.overrides)
    data_cfg = cfg.get("data", {})
    text_prompt = args.text if args.text is not None else data_cfg.get("text", "bruise")
    input_size = int(data_cfg.get("crop", 224))

    print("Building model ...")
    model = build_model_from_cfg(cfg, device)

    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    if missing:
        print(f"Warning: missing keys: {len(missing)} e.g. {missing[:5]}")
    if unexpected:
        print(f"Warning: unexpected keys: {len(unexpected)} e.g. {unexpected[:5]}")

    print("Starting dual-circle evaluation ...")
    print(f"Image dir: {args.image_dir}")
    print(f"Circle dir: {args.circle_dir}")

    metrics = evaluate_dual_circle_dataset(
        model=model,
        image_dir=args.image_dir,
        circle_dir=args.circle_dir,
        device=device,
        text_prompt=text_prompt,
        output_csv=args.output_csv,
        visualize_samples=args.visualize,
        vis_output_dir=args.vis_output_dir if args.visualize else None,
        visualize_specific=args.visualize_specific,
        input_size=input_size,
    )

    if metrics:
        print("\n" + "=" * 60)
        print("DUAL-CIRCLE EVALUATION RESULTS")
        print("=" * 60)
        print(f"Images processed: {metrics['num_images']}")
        print(f"Overall Precision: {metrics['overall_precision']:.4f}")
        print(f"Overall Recall:    {metrics['overall_recall']:.4f}")
        print(f"Overall F1 Score:  {metrics['overall_f1']:.4f}")
        print(f"Overall Accuracy:  {metrics['overall_accuracy']:.4f}")
        print()
        print("Confusion Matrix (aggregated):")
        print(f"  True Positives:  {metrics['total_tp']:,}")
        print(f"  False Positives: {metrics['total_fp']:,}")
        print(f"  True Negatives:  {metrics['total_tn']:,}")
        print(f"  False Negatives: {metrics['total_fn']:,}")
        print(f"  Ignored Pixels:  {metrics['total_ignored']:,}")
        print()
        print("Per-Image Averages:")
        print(f"  Mean TP %: {metrics['mean_tp_percentage']:.2f}%")
        print(f"  Mean FP %: {metrics['mean_fp_percentage']:.2f}%")
        print(f"  Mean F1  : {metrics['mean_f1_score']:.4f} (±{metrics['std_f1_score']:.4f})")


if __name__ == "__main__":
    main()

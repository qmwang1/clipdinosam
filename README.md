CLIP-DINO/Swin-SAM with LoRA (staged fine-tuning)

This project wires together an image backbone (DINO/DINOv2 or supervised Swin Transformer), a frozen CLIP text encoder, and the SAM mask decoder (using its prompt encoder) to produce text-aware segmentation masks. It includes LoRA adapters and a staged training plan to minimize drift while adapting to your domain.

Stages

- Stage 0 — Initialise
  - Backbone: DINO/DINOv2 ViT (e.g., ViT-L/14, ViT-B/16) or a supervised Swin Transformer (e.g., Swin-B) from timm.
  - Text: CLIP text encoder (ViT-L/14, frozen).
  - Seg head: SAM mask decoder + prompt encoder (pretrained), using backbone tokens as image embeddings via a light projection.

- Stage 1 — Get it working
  - Freeze the vision backbone & CLIP.
  - Train: thin image projection head (backbone tokens → CLIP dim) and SAM mask decoder adapters.
  - Losses: Dice + BCE for masks; optional point/box robustness.

- Stage 2 — Light domain adaptation (recommended)
  - Add LoRA to the backbone’s last 2–4 blocks (available for DINO out of the box; add Swin targets if desired).
  - Keep CLIP text frozen; keep SAM adapters trainable.
  - Data: mix unlabelled with labelled/pseudo-labelled. Optionally enable self-distillation/MIM (small weight) and CLIP alignment loss (small weight) to stabilise semantics.

- Stage 3 — More labels (≥2–3k masks)
  - Unfreeze more of the backbone with low LR while keeping LoRA on; optionally tiny LoRA on CLIP text cross-attn inside the decoder.
  - Keep CLIP alignment loss on to reduce degradation.

Quick Start

1) Prepare environment (PyTorch + CUDA recommended) and install deps:

```
# minimal set
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm open-clip-torch pyyaml
# For SAM (mask decoder + prompt encoder)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

2) Place your SAM weights if needed (optional). If you want prebuilt decoders, set `sam.checkpoint` in configs.

3) Run Stage 1 training:

```
python scripts/train.py --config configs/stage1.yaml data.root=/path/to/images data.masks=/path/to/masks
```

4) Stage 2 training:

```
python scripts/train.py --config configs/stage2.yaml data.root=/path/to/images data.masks=/path/to/masks
```

5) Stage 3 (only if enough labels):

```
python scripts/train.py --config configs/stage3.yaml data.root=/path/to/images data.masks=/path/to/masks
```

Need Swin instead of DINO? Swap in the paired configs `configs/stage{1,2,3,4}_swin.yaml` which already set `model.backbone.type: swin` and disable backbone LoRA by default:

```
python scripts/train.py --config configs/stage1_swin.yaml data.root=/path/to/images data.masks=/path/to/masks
```

Stage 4 follows the same pattern (`stage4.yaml` or `stage4_swin.yaml`) when you are ready to fully unfreeze the backbone.

Notes

- CLIP text encoder is kept frozen by default. You can enable a tiny LoRA rank via config if you must adapt prompts.
- The vision backbone starts frozen; LoRA can be injected in the last K DINO blocks via config (extend `SwinBackbone.lora_target_pairs` if you need Swin LoRA).
- We bypass SAM’s heavy image encoder and instead feed projected DINO tokens to the mask decoder.
- Losses available: Dice + BCE, optional CLIP contrastive alignment, optional SSL/MIM with small weight.

Repo Layout

- `clipdinosam/`
  - `models/`: vision backbones (`dino.py`, `swin.py`), shared interface (`vision_backbone.py`), CLIP text wrapper, SAM decoder wrappers, projection heads, model assembly.
  - `lora.py`: LoRA modules and injection helpers.
  - `losses.py`: Dice, BCE, optional contrastive and SSL stubs.
  - `data/`: simple image(+mask) dataset.
- `eval/`: dual-circle evaluation utilities (inner/annulus/outer regions).
- `trainer.py`: staged trainer to match the plan.
- `config.py`: lightweight config loader with overrides.
- `configs/`: example YAML configs per stage.
- `scripts/train.py`: CLI entrypoint.
- `scripts/eval_dual_circle.py`: dual-circle evaluation CLI.
- `scripts/eval_voc.py`: VOC-style evaluation CLI (e.g., HAM10000 test split).
- `scripts/sam_encode.py`: export SAM image embeddings for a folder or list.

Backbones

- The legacy configs keep a `model.dino` block; newer configs use `model.backbone` so you can swap encoders without touching code.
- **DINO / DINOv2 (timm)**: set `model.dino.name` (or `model.backbone.name` with `type: dino`) to values such as `vit_base_patch16_224.dino` or `vit_large_patch14_dinov2.lvd142m`. dinov2 names are resolved through timm first, then torch.hub (`dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`).
- **Swin Transformer (timm, supervised)**: use the `model.backbone` block with `type: swin`, `name: swin_base_patch4_window7_224` (or any other timm Swin variant). Optional keys: `out_index`/`out_indices` to select which feature map to project (defaults to the deepest stage).
- Backbone LoRA is currently wired for DINO. When you switch to Swin, the configs disable backbone LoRA by default; add layer targets in `SwinBackbone.lora_target_pairs` if you want LoRA there.

SAM Variants

- SAM: set `model.sam.type` in {`vit_t`, `vit_b`, `vit_l`, `vit_h`} and optionally `model.sam.checkpoint` to a `.pth`.

SAM Encoder Export

- Compute and save SAM image embeddings for a directory (recursively) or a text file list.

Example:
```
python scripts/sam_encode.py \
  --sam_type vit_b \
  --checkpoint /path/to/sam_vit_b.pth \
  --input_dir /data/images \
  --output_dir /data/sam_embeddings \
  --format pt
```
Or from a list of image paths:
```
python scripts/sam_encode.py \
  --sam_type vit_b \
  --checkpoint /path/to/sam_vit_b.pth \
  --list_file paths.txt \
  --output_dir /data/sam_embeddings
```
Outputs are one file per input image (same stem) storing the embedding tensor.

Stages Overview

- Stage 1: Train projection + SAM adapters (LoRA on SAM), backbone frozen
- Stage 2: Add LoRA to last K backbone blocks (DINO supported out of the box)
- Stage 3: Same as Stage 2 with lower LR and longer training
- Stage 4: Fully unfreeze backbone (disable backbone LoRA); SAM LoRA may remain enabled

HAM10000 in VOC Style

- Expected structure under `data.root`:
  - `JPEGImages/{id}.jpg` — original lesions (e.g., from HAM10000)
  - `SegmentationClass/{id}.png` — binary masks (lesion=255 or >0, background=0)
  - `ImageSets/Segmentation/train.txt` — optional list of ids for training (one per line)
- Use the provided config: `configs/ham10000_voc_stage1.yaml` and set `data.root` to your prepared folder. Example:

Stage 1:
```
python scripts/train.py --config configs/ham10000_voc_stage1.yaml data.root=/data/HAM10000_VOC
```

Stage 2:
```
python scripts/train.py --config configs/ham10000_voc_stage2.yaml data.root=/data/HAM10000_VOC
```

Stage 3:
```
python scripts/train.py --config configs/ham10000_voc_stage3.yaml data.root=/data/HAM10000_VOC
```

Stage 4 (full DINO unfreeze):
```
python scripts/train.py --config configs/ham10000_voc_stage4.yaml data.root=/data/HAM10000_VOC
```

Swin backbone configs mirror each stage (`configs/ham10000_voc_stage{1-4}_swin.yaml`). Example for Stage 4 with Swin and a writable experiment directory:
```
python scripts/train.py --config configs/ham10000_voc_stage4_swin.yaml \
  data.root=/data/HAM10000_VOC \
  output.dir=experiments/runB-stage4-swin
```

VOC-Style Evaluation (Test Split)

- Evaluate a trained checkpoint on a VOC-style dataset split (e.g., HAM10000 `test.txt`). Computes IoU, Dice, accuracy, precision, recall, and F1. Optionally writes per-image metrics CSV and saves predicted masks.

Basic usage:
```
python scripts/eval_voc.py \
  --config configs/ham10000_voc_stage4.yaml \
  --checkpoint experiments/runA/checkpoints/best.pt \
  --root /data/HAM10000_VOC \
  --split test \
  --output_csv experiments/runA/test_metrics.csv
```

Save predicted masks (PNG):
```
python scripts/eval_voc.py \
  --config configs/ham10000_voc_stage4.yaml \
  --checkpoint experiments/runA/checkpoints/best.pt \
  --root /data/HAM10000_VOC \
  --split test \
  --save_preds experiments/runA/test_preds
```

Options and notes:
- `--split` can be `train`, `val`, `test`, or a path to a custom list file. Expects `ImageSets/Segmentation/{split}.txt` by default.
- Directory layout under `--root`:
  - `JPEGImages/{id}.jpg` or `.png`
  - `SegmentationClass/{id}.png` (binary mask; background=0, lesion>0)
  - `ImageSets/Segmentation/test.txt` (one id per line)
- Threshold defaults to `0.5` (override with `--threshold 0.45`).
- Uses the same Resize+CenterCrop+Normalize transforms as training (`cfg.data.resize`, `cfg.data.crop`).
- Prompt defaults to `cfg.data.text` (override with `--text "skin lesion"`).
- If your folder names differ, use `--image_dir`/`--mask_dir` to override.

Checkpointing and Epoch Evaluation

- Checkpoint options under `checkpoint`:
  - `save_best: true` — keep `best.pt` (lowest training loss)
  - `save_latest: true` — keep `latest.pt` after each epoch
  - `keep_per_epoch: false` — set `true` to also keep `epoch_{N}.pt`
- Optional dual-circle eval after each epoch:
  - In config, set (dual-circle evaluation runs by default whenever this block is present; set `enable: false` to skip):
    ```yaml
    eval:
      dual_circle:
        # enable: false  # optional, defaults to true
        image_dir: /path/to/no_circle_images
        circle_dir: /path/to/circle_masks
        # output_dir: /path/to/save/per-epoch/csv  # optional
        # text: "bruise"                           # optional, defaults to data.text
    ```
  - The trainer will print metrics (precision/recall/F1/accuracy) each epoch and write a CSV per epoch if `output_dir` is set.

- Notes on preparation:
  - Convert lesion masks to PNG with values 0 (background) and 255 (lesion). The loader converts them to binary tensors.
  - If you don’t provide `ImageSets/Segmentation/train.txt`, the loader uses all images in `JPEGImages`.
  - You can set `data.split: path/to/list.txt` to a custom list file.

This is a minimal scaffold intended for fast iteration. Extend as needed for your data and evaluation.

Dual‑Circle Evaluation

- Evaluate trained checkpoints on images with dual-circle annotations (inner lesion, ignored annulus, outer/background ring):

```
python scripts/eval_dual_circle.py \
  --config configs/stage2.yaml \
  --checkpoint experiments/runA-stage3/checkpoints/latest.pt \
  --image_dir /path/to/no_circle_images \
  --circle_dir /path/to/circle_masks \
  --ignore_rect_dir /path/to/ignore_rectangles \
  --output_csv results_dual_circle.csv \
  --visualize --vis_output_dir visualizations
```

- The script pairs files by replacing "-noCircles" with "-mask" and falls back to matching by stem (PNG). It computes TP in inner region, ignores the annulus (plus any optional rectangular ignore mask), and counts FP in background (outer ring + beyond). Optional `--text` overrides the prompt from config.

Multi‑GPU Training

- DataParallel (single process): enable by config and run normally.
  - In your YAML, set `distributed.use_data_parallel: true` to wrap the model in `nn.DataParallel` when multiple GPUs are visible.
  - Example run: `CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --config configs/stage2.yaml data.root=/data/HAM10000_VOC`
  - Note: DP splits the batch across GPUs but does not reduce per‑sample memory.

- DistributedDataParallel (recommended): launch one process per GPU via torchrun.
  - Example: `torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/stage2.yaml data.root=/data/HAM10000_VOC`
  - The trainer automatically initializes NCCL, uses a `DistributedSampler`, synchronizes epochs, and saves checkpoints only on rank 0.
  - Continue to pass overrides the same way; each rank receives the same config.

Mixed Precision

- AMP is enabled by default on CUDA. It uses bf16 when supported, otherwise fp16 with GradScaler.
- You can control it via config:
  ```yaml
  optim:
    amp: true        # or false to disable
    amp_dtype: bf16  # or fp16
  ```

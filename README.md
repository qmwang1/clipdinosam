CLIP-DINO-SAM with LoRA (staged fine-tuning)

This project wires together a DINO/DINOv2 vision backbone, a frozen CLIP text encoder, and the SAM mask decoder (using its prompt encoder) to produce text-aware segmentation masks. It includes LoRA adapters and a staged training plan to minimize drift while adapting to your domain.

Stages

- Stage 0 — Initialise
  - Backbone: DINOv2 ViT-L/14 or B/16 (pretrained).
  - Text: CLIP text encoder (ViT-L/14, frozen).
  - Seg head: SAM mask decoder + prompt encoder (pretrained), using DINO tokens as image embeddings via a light projection.

- Stage 1 — Get it working
  - Freeze DINO & CLIP.
  - Train: thin image projection head (DINO tokens → CLIP dim) and SAM mask decoder adapters.
  - Losses: Dice + BCE for masks; optional point/box robustness.

- Stage 2 — Light domain adaptation (recommended)
  - Add LoRA to DINO’s last 2–4 blocks.
  - Keep CLIP text frozen; keep SAM adapters trainable.
  - Data: mix unlabelled with labelled/pseudo-labelled. Optionally enable self-distillation/MIM (small weight) and CLIP alignment loss (small weight) to stabilise semantics.

- Stage 3 — More labels (≥2–3k masks)
  - Unfreeze more of DINO with low LR while keeping LoRA on; optionally tiny LoRA on CLIP text cross-attn inside the decoder.
  - Keep CLIP alignment loss on to reduce degradation.

Quick Start

1) Prepare environment (PyTorch + CUDA recommended) and install deps:

```
# minimal set
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm open-clip-torch pyyaml
# For SAM (only mask decoder + prompt encoder used)
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

Notes

- CLIP text encoder is kept frozen by default. You can enable a tiny LoRA rank via config if you must adapt prompts.
- DINO image encoder starts frozen; LoRA can be injected in the last K blocks via config.
- We bypass SAM’s heavy image encoder and instead feed projected DINO tokens to the mask decoder.
- Losses available: Dice + BCE, optional CLIP contrastive alignment, optional SSL/MIM with small weight.

Repo Layout

- `clipdinosam/`
  - `models/`: DINO, CLIP text, SAM decoder wrappers, projection heads, model assembly.
  - `lora.py`: LoRA modules and injection helpers.
  - `losses.py`: Dice, BCE, optional contrastive and SSL stubs.
  - `data/`: simple image(+mask) dataset.
  - `eval/`: dual-circle evaluation utilities (inner/annulus/outer regions).
  - `trainer.py`: staged trainer to match the plan.
  - `config.py`: lightweight config loader with overrides.
- `configs/`: example YAML configs per stage.
- `scripts/train.py`: CLI entrypoint.
- `scripts/eval_dual_circle.py`: dual-circle evaluation CLI.

Stages Overview

- Stage 1: Train projection + SAM adapters (LoRA on SAM), DINO frozen
- Stage 2: Add LoRA to last K DINO blocks; train only LoRA in DINO
- Stage 3: Same as Stage 2 with lower LR and longer training
- Stage 4: Fully unfreeze DINO (no DINO LoRA); SAM LoRA may remain enabled

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

Checkpointing and Epoch Evaluation

- Checkpoint options under `checkpoint`:
  - `save_best: true` — keep `best.pt` (lowest training loss)
  - `save_latest: true` — keep `latest.pt` after each epoch
  - `keep_per_epoch: false` — set `true` to also keep `epoch_{N}.pt`
- Optional dual-circle eval after each epoch:
  - In config, set:
    ```yaml
    eval:
      dual_circle:
        enable: true
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
  --output_csv results_dual_circle.csv \
  --visualize --vis_output_dir visualizations
```

- The script pairs files by replacing "-noCircles" with "-mask" and falls back to matching by stem (PNG). It computes TP in inner region, ignores the annulus, and counts FP in background (outer ring + beyond). Optional `--text` overrides the prompt from config.

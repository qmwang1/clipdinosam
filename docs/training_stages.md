# Four-Stage Fine-Tuning Strategy for CLIP–DINO–SAM

## Abstract
We describe a four-stage fine-tuning curriculum for CLIP–DINO–SAM that progressively increases modeling capacity while controlling memory and stability. Stage 1 trains lightweight heads and SAM adapters with the image backbone (DINO) frozen. Stage 2 and Stage 3 introduce LoRA adapters on the last K DINO blocks while keeping the rest frozen. Stage 4 fully unfreezes DINO and disables DINO LoRA. We detail, for each stage, which modules are trainable, where LoRA is applied, optimization groupings, and the loss formulation.

## 1. Model Overview
Let x be an input image and t an optional text prompt.
- DINO image encoder: produces patch tokens X ∈ R^{B×N×C} and a spatial grid h×w.
- Token projection: `TokenProjection` maps image tokens to the CLIP text width D_clip for alignment; `TokenToMaskEmbedding` maps tokens to a dense feature map E ∈ R^{B×E×h×w} for SAM.
- CLIP text encoder: encodes prompt(s) t into text features T ∈ R^{T×D_clip}.
- SAM decoder: consumes E (and optional prompts) to predict low-resolution masks M_low ∈ R^{B×C_m×h×w} and IoU scores. Masks are upsampled to image resolution for supervision.

Implementation mapping:
- `clipdinosam/models/model.py`: combines DINO, CLIP text, projection heads, and SAM decoder.
- `clipdinosam/lora.py`: defines LoRA injection for `nn.Linear` layers and utilities to freeze/unfreeze parameters.
- `clipdinosam/trainer.py`: builds model per stage and applies freezing/LoRA.

## 2. Losses and Optimization
- Segmentation loss: L_seg = Dice(M̂, Y) + BCE(M̂, Y), where M̂ are logits upsampled to ground-truth size and Y is the binary mask.
- Optional CLIP alignment: L_clip aligns pooled image token features with text features via a contrastive objective.
- Optional SSL stub: placeholder returning zero by default.

Two optimizer groups (AdamW):
- Base group: all trainable parameters excluding LoRA A/B matrices, learning rate `optim.lr`, weight decay `optim.wd`.
- LoRA group: LoRA parameters only (names contain `.A` or `.B`), learning rate `optim.lora_lr`, weight decay 0.

## 3. LoRA Adapters
We apply LoRA to selected `nn.Linear` layers by replacing them with a `LoRALinear` wrapper parameterized by low-rank matrices A ∈ R^{r×in} and B ∈ R^{out×r}. The effective weight is W + (α/r)·(BA). Dropout before A is optional.

Targets by module:
- SAM: substrings {"attn", "mlp", "proj", "lin"} inside the SAM mask decoder (exact coverage depends on the underlying implementation).
- DINO: substrings for the last K transformer blocks, e.g., `blocks.i.attn` and `blocks.i.mlp` for i ∈ {L−K, …, L−1}.

## 4. Training Stages
Below, “trainable” means `requires_grad=True` and “frozen” means `requires_grad=False` unless part of LoRA.

### Stage 1 — Heads + SAM LoRA; DINO Frozen
- DINO: frozen (all parameters).
- CLIP text encoder: frozen.
- TokenProjection (image→text): trainable.
- TokenToMaskEmbedding (image→mask): trainable.
- SAM: base weights frozen; LoRA injected into linear layers; only LoRA A/B parameters trainable.
- DINO LoRA: none.
- Optimizer: base group trains token heads; LoRA group trains SAM LoRA.
- Purpose: establish heads and mask decoding with low memory footprint and stable training.

### Stage 2 — Add DINO LoRA (last K blocks)
- DINO: base weights frozen; LoRA injected into the last K transformer blocks; only LoRA A/B trainable.
- CLIP text encoder: frozen.
- TokenProjection / TokenToMaskEmbedding: trainable.
- SAM: as in Stage 1 (LoRA only).
- Optimizer: base group for heads; LoRA group for both SAM and DINO LoRA.
- Purpose: allow limited adaptation of image features while controlling capacity via K and rank r.

### Stage 3 — Same as Stage 2, Longer and Lower LR
- Identical parameterization to Stage 2 (DINO and SAM with LoRA; backbones otherwise frozen).
- Typically uses a lower base LR and longer training schedule to consolidate improvements.
- Purpose: refine with stable updates after LoRA injection.

### Stage 4 — Full DINO Unfreeze; No DINO LoRA
- DINO: fully unfrozen (all parameters trainable) with no DINO LoRA insertion.
- CLIP text encoder: frozen.
- TokenProjection / TokenToMaskEmbedding: trainable.
- SAM: usually keeps SAM LoRA enabled (configurable); base SAM weights remain frozen if LoRA is used; alternatively, SAM can be fully frozen or unfrozen depending on configuration.
- Optimizer: base group now includes all DINO parameters; LoRA group, if present, covers SAM LoRA only.
- Purpose: maximize capacity for final adaptation when compute allows.

## 5. Data Pipeline
- Resize → CenterCrop → ToTensor → Normalize (ImageNet mean/std) for images.
- Masks resized with nearest-neighbor to the crop size.
- Formats: “folders” (images + optional masks) or VOC-style datasets.

## 6. Practical Notes
- Memory: Stages 1–3 train few parameters (heads + LoRA) and typically fit into ~5–8 GB at 224² with small batches. Stage 4 requires substantially more memory; adjust batch size accordingly.
- Learning rate scaling: When increasing batch size, linearly scale LR as a first approximation; consider warmup.
- DINOv2 vs. DINO: Some DINOv2 variants expect non-224 inputs (e.g., 518²). Ensure `data.resize`/`data.crop` match the model’s `img_size`.
- Evaluation: The evaluation script mirrors training-time LoRA injection per stage so that LoRA weights in checkpoints are loaded and used during inference.

## 7. Reproducibility and Configuration
- Stage selection: `stage: {1|2|3|4}` in the YAML config controls the behavior above.
- LoRA parameters: `lora.sam` and `lora.dino` blocks set `enable`, `rank`, `alpha`, `dropout`, and (for DINO) `last_k_blocks`.
- Optimization: `optim.lr`, `optim.lora_lr`, `optim.wd`, and `optim.batch_size` govern training dynamics.
- Loss weights: `loss.clip_align_weight` and `loss.ssl_weight` toggle optional objectives.

## 8. Limitations and Extensions
- The current LoRA injection matches by substring and replaces `nn.Linear` layers; coverage depends on model internals.
- Stage 4 does not currently support partial unfreezing of DINO (e.g., block ranges) out of the box, but this is a straightforward extension by adjusting `set_trainable` selectively.
- Mixed precision, gradient accumulation, and resume features can further improve throughput and flexibility.

## 9. Summary
The four-stage schedule offers a principled path from low-memory adaptation (heads + SAM LoRA) to high-capacity fine-tuning (full DINO unfreeze). By clearly separating which parameters are trainable and where LoRA is applied, the approach enables stable, resource-aware training while preserving a smooth route to maximal performance when compute budgets allow.

## 10. How We Train (or Freeze) CLIP

- Baseline strategy (this repo): the CLIP text encoder remains frozen across all stages. We use CLIP to produce fixed text features and apply an optional alignment loss so that image token projections align with CLIP’s text space, without updating CLIP itself.
- Where gradients flow: `clip_align_loss` backpropagates into `TokenProjection` (image→text) and, depending on the stage, into DINO LoRA or DINO weights (Stage 4). Gradients do not flow into CLIP when frozen.
- Rationale for freezing: CLIP encoders are strong priors with broad language grounding. Freezing stabilizes training, reduces memory, and avoids catastrophic drift of the text space under small datasets.

Optional fine-tuning options (not enabled by default):
- Full CLIP unfreeze: set the CLIP text encoder to `requires_grad=True` and lower its LR vs. other heads. This increases memory/compute and risks overfitting prompt embeddings; typically useful only with ample data or narrow domains.
- LoRA on CLIP: inject LoRA into CLIP’s transformer linear layers (e.g., attention and MLP projections) to adapt with fewer trainable parameters. This preserves most of CLIP while allowing lightweight domain adaptation. The repository does not ship CLIP LoRA injection code, but it mirrors the existing DINO/SAM LoRA pattern.
- Prompt engineering vs. training: before unfreezing CLIP, try richer prompts or prompt ensembling; these often recover much of the benefit without extra parameters.

Practical guidance:
- Start frozen: keep CLIP frozen through Stages 1–3; enable `loss.clip_align_weight` if you want image–text alignment pressure on the image side.
- Consider CLIP tuning late: if after Stage 4 performance saturates and you have sufficient data, try a short run with CLIP LoRA or low‑LR unfreeze. Monitor validation closely for drift.
- Mind tokenizer and tags: the OpenCLIP “openai” tag uses QuickGELU; mismatches raise warnings but are benign if weights and config are consistent.

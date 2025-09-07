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

## 4. Training Stages (What Trains vs. Freezes)
“Trainable” means `requires_grad=True`. For LoRA-wrapped layers, only `.A` and `.B` are trainable; the base linear `weight`/`bias` stay frozen.

For quick verification, print names of trainables at runtime:
`for n,p in model.named_parameters():
    if p.requires_grad: print(n)`

### Stage 1 — Train Heads + SAM LoRA; DINO Frozen
- DINO (image encoder): frozen. All submodules under `model.dino` are non-trainable: `patch_embed`, positional embeddings `pos_embed`, transformer `blocks.*` (attention and MLP), final `norm`.
- CLIP text encoder: frozen. All of `model.clip_text` stays frozen; text features are computed under `torch.no_grad()`.
- Token heads: trainable.
  - `model.token_to_text.net.{0:LayerNorm,1:Linear}`
  - `model.token_to_mask.proj.{0:LayerNorm,1:Linear}`
- SAM decoder stack: trainable parameters by default include both base weights and LoRA A/B if injected. Concretely, LoRA is injected into `nn.Linear` whose path contains any of `{"attn","mlp","proj","lin"}` under `model.sam` (covers SAM’s Two-Way Transformer attention projections `qkv/proj`, MLPs, and other linear projections). The LoRA params are named like `...linear_name.A` and `...linear_name.B`.
  - By default in `trainer.py`, base SAM weights remain trainable (there is no explicit freeze). To freeze SAM base weights and update only LoRA: call `enable_only_lora(model.sam)` after injection (not currently done in code).
- DINO LoRA: none inserted in this stage.
- Optimizer groups:
  - Base group: token heads (+ any non-LoRA SAM parameters, given the default trainer code).
  - LoRA group: all parameters with names containing `.A` or `.B` (i.e., SAM LoRA).

Layer-level view (examples; exact names depend on SAM version):
- Frozen: `dino.model.patch_embed.*`, `dino.model.pos_embed`, `dino.model.blocks.[0..L-1].attn.*`, `dino.model.blocks.[0..L-1].mlp.*`, `dino.model.norm.*`, all of `clip_text.*`.
- Trainable (heads): `token_to_text.net.0.weight|bias`, `token_to_text.net.1.weight|bias`, `token_to_mask.proj.0.weight|bias`, `token_to_mask.proj.1.weight|bias`.
- Trainable (SAM LoRA): `sam.….A`, `sam.….B` under attention/MLP linear layers; base linear weights remain frozen inside each `LoRALinear`, but other non-linear SAM layers (norms, conv upscalers, hypernetworks MLPs) are trainable unless you explicitly freeze SAM.

### Stage 2 — Add DINO LoRA (last K blocks)
- DINO: base weights frozen; LoRA injected into the last `K` transformer blocks of the ViT under `model.dino.model.blocks`.
  - Targets: `blocks.{L-K}..{L-1}.attn.*` and `blocks.{L-K}..{L-1}.mlp.*`.
  - Which linear layers get LoRA inside each block:
    - Attention: `qkv` and `proj` linears → names like `blocks.i.attn.qkv`, `blocks.i.attn.proj`.
    - MLP: feed-forward `fc1`, `fc2` linears → names like `blocks.i.mlp.fc1`, `blocks.i.mlp.fc2`.
  - Trainable: only LoRA factors `.A` and `.B` inside those linears (enforced by `enable_only_lora(self.model.dino)`).
- CLIP text encoder: frozen.
- Token heads: trainable (same as Stage 1).
- SAM: same LoRA injection as Stage 1. Base SAM remains trainable by default unless you explicitly freeze it.
- Optimizer groups:
  - Base group: token heads (+ non-LoRA SAM if not frozen).
  - LoRA group: SAM LoRA and DINO LoRA `.A/.B` parameters.

### Stage 3 — Same as Stage 2, Longer and Lower LR
- Identical train/freeze pattern to Stage 2.
- Often uses reduced base LR and a longer schedule to consolidate improvements.

### Stage 4 — Full DINO Unfreeze; No DINO LoRA
- DINO: fully unfrozen; no DINO LoRA insertion. Trainable submodules now include:
  - `patch_embed.*`, `pos_embed`, `pos_drop`, all transformer blocks `blocks.[0..L-1].attn.*` and `blocks.[0..L-1].mlp.*`, and final `norm.*`.
- CLIP text encoder: frozen.
- Token heads: trainable (same as Stage 1).
- SAM: by default we keep SAM as in earlier stages (LoRA still present if configured). Base SAM weights are trainable unless you freeze them; you may keep training SAM LoRA only by calling `enable_only_lora(model.sam)` yourself.
- Optimizer groups:
  - Base group: all DINO parameters + token heads (+ non-LoRA SAM if not frozen).
  - LoRA group: SAM LoRA `.A/.B` only (if SAM LoRA is enabled).

## 5. Data Pipeline
- Resize → CenterCrop → ToTensor → Normalize (ImageNet mean/std) for images.
- Masks resized with nearest-neighbor to the crop size.
- Formats: “folders” (images + optional masks) or VOC-style datasets.

## 6. Practical Notes
- Memory: With the default trainer, Stage 1 also updates SAM base weights, so memory is higher than “LoRA-only SAM.” If you freeze SAM base (recommended for tight budgets), Stages 1–3 train relatively few parameters (heads + LoRA) and typically fit into ~5–8 GB at 224² with small batches. Stage 4 requires substantially more memory; adjust batch size accordingly.
- Learning rate scaling: When increasing batch size, linearly scale LR as a first approximation; consider warmup.
- DINOv2 vs. DINO: Some DINOv2 variants expect non-224 inputs (e.g., 518²). Ensure `data.resize`/`data.crop` match the model’s `img_size`.
- Evaluation: The evaluation script mirrors training-time LoRA injection per stage so that LoRA weights in checkpoints are loaded and used during inference.

## 7. Reproducibility and Configuration
- Stage selection: `stage: {1|2|3|4}` in the YAML config controls the behavior above.
- LoRA parameters: `lora.sam` and `lora.dino` blocks set `enable`, `rank`, `alpha`, `dropout`, and (for DINO) `last_k_blocks`.
- Optimization: `optim.lr`, `optim.lora_lr`, `optim.wd`, and `optim.batch_size` govern training dynamics.
- Loss weights: `loss.clip_align_weight` and `loss.ssl_weight` toggle optional objectives.

## 8. Limitations and Extensions
- LoRA matching: The current LoRA injection matches by substring and replaces `nn.Linear` layers; coverage depends on model internals. For timm ViTs, this reaches `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`. For SAM, it reaches Two‑Way Transformer attention and MLP projections, and other linears whose names include the listed substrings; it does not affect convs, norms, or token embeddings.
- SAM freezing: The trainer does not freeze SAM base weights by default. If you want LoRA‑only SAM across Stages 1–3, add `enable_only_lora(self.model.sam)` after SAM LoRA injection.
- Partial unfreeze: Stage 4 does not currently support partial unfreezing of DINO (e.g., block ranges) out of the box, but this is a straightforward extension by adjusting `set_trainable` selectively.
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

## 11. CLIP as Instruction Signal for DINO

Motivation. Text conveys high‑level semantic intent that may not be fully captured by mask supervision alone (e.g., ambiguous boundaries, class imbalance, or weak labels). Leveraging CLIP’s text space provides a compact, compositional prior that can steer visual features toward concepts expressed in natural language (open‑vocabulary behavior) while preserving sample efficiency.

Mechanism of guidance. During training, we map DINO patch tokens into CLIP’s text width via `TokenProjection` and apply a symmetric contrastive alignment objective against the CLIP text embedding(s) of the prompt. Let `z_i ∈ R^D` denote the pooled, ℓ2‑normalized image representation (mean over tokens after projection) for image i, and `t_j ∈ R^D` the ℓ2‑normalized text embedding for prompt j. The loss is

L_clip = 1/2 [ CE(softmax(z_i^T t_j / τ), y_i) + CE(softmax(t_j^T z_i / τ), y_j) ],

implemented in `clipdinosam/losses.py` (`clip_align_loss`, temperature τ=0.07). With CLIP frozen, gradients flow to `TokenProjection` and then upstream to DINO’s trainable pathway for the current stage: LoRA factors in Stages 2–3, and full DINO weights in Stage 4. Thus, CLIP acts as a fixed teacher that shapes the geometry of the visual representation toward the text‑anchored directions specified by the prompt(s).

Why involve CLIP during training?
- Semantic instruction: The text embedding defines a direction in a strong, language‑grounded space. Aligning image features to that direction guides attention toward concept‑relevant regions, improving mask quality for sparse or noisy labels.
- Sample efficiency and regularization: Cross‑modal alignment provides additional supervision at minimal annotation cost, acting as a regularizer that discourages degenerate solutions and improves generalization under distribution shift.
- Compositionality and open‑vocabulary: The learned mapping from image tokens to CLIP space encourages features that are sensitive to attributes and compositions present in prompts, helping zero‑/few‑shot generalization to new categories described by text.
- Stability: Keeping CLIP frozen preserves its broad language prior. The image side learns to meet a stable target, reducing the risk of catastrophic drift in the text manifold.

Where the signal enters the pipeline.
- Heads: `TokenProjection` always receives gradients from `L_clip` and adapts the image‑to‑text mapping.
- DINO: receives gradients indirectly through `TokenProjection` depending on the stage.
  - Stages 1: DINO is frozen; only heads (and SAM components) update; `L_clip` influences the mask head through the shared tokens but does not alter DINO.
  - Stages 2–3: gradients update only DINO LoRA factors in the last K blocks, enabling controlled adaptation toward the prompt semantics.
  - Stage 4: the entire DINO backbone updates, maximizing capacity to internalize the instruction signal.
- CLIP: remains frozen by default; gradients do not flow into the text encoder.

Practical usage.
- Weighting: set `loss.clip_align_weight` (e.g., 0.05–0.1 in the provided configs) to trade off segmentation supervision and language guidance. Too large a weight can over‑constrain features toward text similarity at the expense of precise boundaries.
- Prompting: use clear, domain‑appropriate prompts; consider prompt ensembling or templates (e.g., “a photo of a {class}”).
- When to tune CLIP: only consider LoRA/low‑LR unfreeze after Stage 4 if performance saturates and you have sufficient data; otherwise prefer keeping CLIP fixed to avoid drifting the language prior.

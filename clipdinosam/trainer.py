from typing import Dict, List, Optional
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .models import (
    build_dino,
    build_clip_text,
    build_sam_decoder,
    TokenProjection,
    TokenToMaskEmbedding,
    CLIPDinoSam,
)
from .lora import inject_lora_linear, set_trainable, enable_only_lora
from .data import ImageMaskDataset, VOCSegDataset
from .losses import seg_loss, clip_align_loss, ssl_stub_loss


class Trainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()
        self._build_data()
        self._build_optim()

    def _build_model(self):
        mcfg = self.cfg["model"]
        dino = build_dino(mcfg["dino"]["name"], pretrained=True)
        clip_text = build_clip_text(mcfg["clip"]["name"], pretrained=mcfg["clip"].get("pretrained", "openai"))
        sam = build_sam_decoder(mcfg["sam"]["type"], checkpoint=mcfg["sam"].get("checkpoint"))

        token_to_text = TokenProjection(in_dim=dino.embed_dim, out_dim=mcfg["clip"]["width"]) if "width" in mcfg["clip"] else TokenProjection(dino.embed_dim, clip_text.width)
        token_to_mask = TokenToMaskEmbedding(in_dim=dino.embed_dim, embed_dim=mcfg["sam"].get("embed_dim", 256))

        # Build the combined model (keep on CPU until after LoRA injection)
        self.model = CLIPDinoSam(dino, clip_text, sam, token_to_text, token_to_mask)

        # Freezing and LoRA injection per stage
        stage = self.cfg.get("stage", 1)
        lora_cfg = self.cfg.get("lora", {})

        # Freeze CLIP text
        set_trainable(self.model.clip_text, False)
        # DINO freeze by default
        set_trainable(self.model.dino, False)

        # Train projection and SAM adapters in Stage 1
        set_trainable(self.model.token_to_text, True)
        set_trainable(self.model.token_to_mask, True)
        # SAM: enable LoRA on decoder linear layers
        if lora_cfg.get("sam", {}).get("enable", True):
            sam_targets = lora_cfg["sam"].get("targets", ["attn", "mlp", "lin", "proj"])
            inject_lora_linear(self.model.sam, sam_targets, rank=lora_cfg["sam"].get("rank", 8), alpha=lora_cfg["sam"].get("alpha", 8), dropout=lora_cfg["sam"].get("dropout", 0.0))

        if stage >= 2:
            # Add LoRA to DINO last K blocks
            dino_lora = lora_cfg.get("dino", {})
            if dino_lora.get("enable", True):
                k = dino_lora.get("last_k_blocks", 2)
                # Build target substrings for last k blocks
                block_targets = []
                if hasattr(self.model.dino.model, "blocks"):
                    total = len(self.model.dino.model.blocks)
                    for i in range(total - k, total):
                        block_targets += [f"blocks.{i}.attn", f"blocks.{i}.mlp"]
                inject_lora_linear(self.model.dino.model, block_targets, rank=dino_lora.get("rank", 8), alpha=dino_lora.get("alpha", 8), dropout=dino_lora.get("dropout", 0.0))
            # Keep DINO otherwise frozen (only LoRA updates trainable)
            enable_only_lora(self.model.dino)

        if stage >= 3:
            # Optionally unfreeze more of DINO with low LR (handled by optimizer groups)
            pass

        # Move entire model to the desired device AFTER LoRA injection so LoRA params move too
        self.model = self.model.to(self.device)

    def _build_data(self):
        dcfg = self.cfg["data"]
        tfm = transforms.Compose([
            transforms.Resize(dcfg.get("resize", 256)),
            transforms.CenterCrop(dcfg.get("crop", 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        m_tfm = transforms.Compose([
            transforms.Resize(dcfg.get("crop", 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        data_format = dcfg.get("format", "folders").lower()
        if data_format == "voc":
            self.train_set = VOCSegDataset(
                root=dcfg["root"],
                split=dcfg.get("split", None),
                transform=tfm,
                mask_transform=m_tfm,
                text_prompt=dcfg.get("text"),
                binary=dcfg.get("binary", True),
                image_dir=dcfg.get("image_dir", "JPEGImages"),
                mask_dir=dcfg.get("mask_dir", "SegmentationClass"),
            )
        else:
            self.train_set = ImageMaskDataset(
                root=dcfg["root"], masks=dcfg.get("masks"), transform=tfm, mask_transform=m_tfm, text_prompt=dcfg.get("text")
            )
        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg["optim"].get("batch_size", 8), shuffle=True, num_workers=dcfg.get("num_workers", 4))

    def _build_optim(self):
        ocfg = self.cfg["optim"]
        params = []
        # Base LR group for standard trainables
        params.append({
            "params": [p for n, p in self.model.named_parameters() if p.requires_grad and (".A" not in n and ".B" not in n)],
            "lr": ocfg.get("lr", 1e-4),
            "weight_decay": ocfg.get("wd", 1e-4),
        })
        # LoRA params can use higher LR
        lora_lr = ocfg.get("lora_lr", ocfg.get("lr", 1e-4))
        params.append({
            "params": [p for n, p in self.model.named_parameters() if p.requires_grad and (".A" in n or ".B" in n)],
            "lr": lora_lr,
            "weight_decay": 0.0,
        })
        self.optimizer = torch.optim.AdamW(params)

    def _compute_losses(self, batch, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        losses = {}
        if batch["mask"] is not None:
            gt = (batch["mask"].to(self.device) > 0.5).float()
            # Upsample low-res mask to gt size if needed
            logits = outputs["low_res_masks"]
            if logits.shape[-2:] != gt.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            losses["seg"] = seg_loss(logits, gt)
        # Optional CLIP alignment
        if self.cfg.get("loss", {}).get("clip_align_weight", 0.0) > 0 and outputs["text_feats"] is not None:
            losses["clip"] = clip_align_loss(outputs["clip_tokens"], outputs["text_feats"]) * self.cfg["loss"]["clip_align_weight"]
        # Optional SSL stub
        if self.cfg.get("loss", {}).get("ssl_weight", 0.0) > 0:
            losses["ssl"] = ssl_stub_loss(outputs["clip_tokens"]) * self.cfg["loss"]["ssl_weight"]

        total = sum(losses.values()) if losses else torch.zeros((), device=self.device)
        return total

    def run(self):
        epochs = self.cfg.get("epochs", 1)
        print(f"Using device: {self.device}")
        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                texts = None
                if batch.get("text"):
                    texts = [batch["text"]] if isinstance(batch["text"], str) else batch["text"]
                out = self.model(images, texts=texts)
                loss = self._compute_losses(batch, out)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg = total_loss / max(1, len(self.train_loader))
            print(f"Epoch {epoch}: loss={avg:.4f}")
            self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch: int):
        ckpt_cfg = self.cfg.get("checkpoint", {})
        out_root = self.cfg.get("output", {}).get("dir")
        # Priority: explicit checkpoint.dir > output.dir/checkpoints > ./checkpoints
        out_dir = ckpt_cfg.get("dir") or (os.path.join(out_root, "checkpoints") if out_root else "checkpoints")
        os.makedirs(out_dir, exist_ok=True)
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg,
        }
        path = os.path.join(out_dir, f"epoch_{epoch}.pt")
        latest = os.path.join(out_dir, "latest.pt")
        try:
            import torch
            torch.save(state, path)
            torch.save(state, latest)
        except Exception as e:
            print(f"Warning: failed to save checkpoint: {e}")

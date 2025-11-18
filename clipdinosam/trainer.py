from typing import Dict, List, Optional
import json
import os
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from .models import (
    build_backbone,
    build_clip_text,
    build_sam_decoder,
    TokenProjection,
    TokenToMaskEmbedding,
    CLIPDinoSam,
)
from .models.sam2_decoder import build_sam2_decoder  # SAM2 support
from .lora import inject_lora_linear, set_trainable, enable_only_lora
from .data import ImageMaskDataset, VOCSegDataset
from .losses import seg_loss, clip_align_loss, ssl_stub_loss
from .eval import evaluate_dual_circle_dataset


class Trainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self._init_metric_tracking()
        # Distributed setup
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        if torch.cuda.is_available() and ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
            self.distributed = True
            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # AMP configuration
        ocfg = cfg.get("optim", {})
        self.amp_enabled = bool(ocfg.get("amp", True)) and self.device.type == "cuda"
        amp_dtype_cfg = ocfg.get("amp_dtype")
        if self.amp_enabled:
            if amp_dtype_cfg is None:
                # Default to bf16 if supported, otherwise fp16
                amp_dtype_cfg = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
            self.amp_dtype = torch.bfloat16 if str(amp_dtype_cfg).lower() == "bf16" else torch.float16
        else:
            self.amp_dtype = None
        self.scaler = torch.cuda.amp.GradScaler() if (self.amp_enabled and self.amp_dtype == torch.float16) else None

        ckpt_cfg = self.cfg.get("checkpoint", {})
        best_metric_raw = str(ckpt_cfg.get("best_metric", "train_loss")).lower()
        best_metric_aliases = {
            "loss": "train_loss",
            "train_loss": "train_loss",
            "dual_circle_f1": "dual_circle_overall_f1",
            "dual_circle_overall_f1": "dual_circle_overall_f1",
            "dual_circle_mean_f1": "dual_circle_mean_f1",
        }
        best_metric = best_metric_aliases.get(best_metric_raw)
        if best_metric is None:
            warnings.warn(f"Unknown checkpoint.best_metric '{best_metric_raw}', falling back to train_loss.")
            best_metric = "train_loss"
        self.best_metric_name = best_metric
        self.best_metric_lower_is_better = self.best_metric_name == "train_loss"
        self.best_metric_value: Optional[float] = None
        self.last_train_loss: Optional[float] = None

        self._build_model()
        self._build_data()
        self._build_optim()

    def _build_model(self):
        mcfg = self.cfg["model"]
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
        clip_text = build_clip_text(mcfg["clip"]["name"], pretrained=mcfg["clip"].get("pretrained", "openai"))
        # Build SAM or SAM2 decoder per config
        sam_type = mcfg["sam"]["type"]
        if isinstance(sam_type, str) and "sam2" in sam_type.lower():
            sam = build_sam2_decoder(
                sam_type=sam_type,
                config=mcfg["sam"].get("config"),
                checkpoint=mcfg["sam"].get("checkpoint"),
            )
        else:
            sam = build_sam_decoder(sam_type, checkpoint=mcfg["sam"].get("checkpoint"))

        token_to_text = TokenProjection(in_dim=dino.embed_dim, out_dim=mcfg["clip"]["width"]) if "width" in mcfg["clip"] else TokenProjection(dino.embed_dim, clip_text.width)
        token_to_mask = TokenToMaskEmbedding(in_dim=dino.embed_dim, embed_dim=mcfg["sam"].get("embed_dim", 256))

        # Build the combined model (keep on CPU until after LoRA injection)
        self.model = CLIPDinoSam(dino, clip_text, sam, token_to_text, token_to_mask)

        # Freezing and LoRA injection per stage
        stage = self.cfg.get("stage", 1)
        lora_cfg = self.cfg.get("lora", {})

        # Freeze CLIP text
        set_trainable(self.model.clip_text, False)
        # Backbone freeze by default
        set_trainable(self.model.dino, False)

        # Train projection and SAM adapters in Stage 1
        set_trainable(self.model.token_to_text, True)
        set_trainable(self.model.token_to_mask, True)
        # SAM: enable LoRA on decoder linear layers
        if lora_cfg.get("sam", {}).get("enable", True):
            sam_targets = lora_cfg["sam"].get("targets", ["attn", "mlp", "lin", "proj"])
            inject_lora_linear(self.model.sam, sam_targets, rank=lora_cfg["sam"].get("rank", 8), alpha=lora_cfg["sam"].get("alpha", 8), dropout=lora_cfg["sam"].get("dropout", 0.0))

        if stage in (2, 3):
            # Add LoRA adapters to the vision backbone when supported.
            dino_lora = lora_cfg.get("dino", {})
            if dino_lora.get("enable", True):
                k = dino_lora.get("last_k_blocks", 2)
                target_pairs = self.model.dino.lora_target_pairs(k)
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
                if total_replaced == 0:
                    warnings.warn(
                        "LoRA enabled for backbone but no target linear layers were matched; skipping injection."
                    )
            enable_only_lora(self.model.dino)
        elif stage >= 4:
            # Fully unfreeze backbone without adding LoRA adapters
            set_trainable(self.model.dino, True)

        # Move entire model to the desired device AFTER LoRA injection so LoRA params move too
        self.model = self.model.to(self.device)

        # Wrap with DDP or optionally DP
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=True,
            )
        else:
            dp_flag = self.cfg.get("distributed", {}).get("use_data_parallel", False)
            if dp_flag and torch.cuda.device_count() > 1 and self.device.type == "cuda":
                self.model = torch.nn.DataParallel(self.model)

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
        sampler = DistributedSampler(self.train_set, shuffle=True) if self.distributed else None
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.cfg["optim"].get("batch_size", 8),
            shuffle=(sampler is None),
            num_workers=dcfg.get("num_workers", 4),
            sampler=sampler,
            pin_memory=(self.device.type == "cuda"),
        )

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
        if (not self.distributed) or self.rank == 0:
            print(f"Using device: {self.device}; distributed={self.distributed}; world_size={self.world_size}")
        for epoch in range(1, epochs + 1):
            # Ensure distinct shuffling per epoch under DDP
            if self.distributed and isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            total_loss = 0.0
            for batch in self.train_loader:
                images = batch["image"].to(self.device, non_blocking=True)
                texts = None
                if batch.get("text"):
                    texts = [batch["text"]] if isinstance(batch["text"], str) else batch["text"]
                if self.amp_dtype is not None:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                        out = self.model(images, texts=texts)
                        loss = self._compute_losses(batch, out)
                else:
                    out = self.model(images, texts=texts)
                    loss = self._compute_losses(batch, out)

                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
            avg = total_loss / max(1, len(self.train_loader))
            if (not self.distributed) or self.rank == 0:
                print(f"Epoch {epoch}: loss={avg:.4f}")
                self._save_checkpoint(epoch, avg)
                self.last_train_loss = avg

            # Optional dual-circle evaluation each epoch
            eval_metrics = self._maybe_eval_epoch(epoch)
            if (not self.distributed) or self.rank == 0:
                self._record_epoch_metrics(epoch, avg, eval_metrics)
            # Restore train mode for next epoch after eval switches to eval() internally
            self.model.train()
        # Graceful DDP teardown
        if self.distributed:
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    def _save_checkpoint(
        self,
        epoch: int,
        avg_loss: float,
        *,
        best_metric_value: Optional[float] = None,
        best_metric_payload: Optional[Dict] = None,
        update_latest: bool = True,
    ):
        """Handle checkpoint persistence (latest/per-epoch) and optionally update best based on configured metric."""
        ckpt_cfg = self.cfg.get("checkpoint", {})
        out_root = self.cfg.get("output", {}).get("dir")
        # Priority: explicit checkpoint.dir > output.dir/checkpoints > ./checkpoints
        out_dir = ckpt_cfg.get("dir") or (os.path.join(out_root, "checkpoints") if out_root else "checkpoints")
        os.makedirs(out_dir, exist_ok=True)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        state = {
            "epoch": epoch,
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg,
            "train_loss": avg_loss,
        }
        state["best_metric_name"] = self.best_metric_name
        state["best_metric_value"] = best_metric_value if best_metric_value is not None else avg_loss if self.best_metric_name == "train_loss" else None
        if best_metric_payload and self.best_metric_name != "train_loss":
            state["dual_circle_metrics"] = best_metric_payload

        keep_per_epoch = ckpt_cfg.get("keep_per_epoch", False)
        save_best = ckpt_cfg.get("save_best", True)
        save_latest = ckpt_cfg.get("save_latest", True)

        if update_latest and keep_per_epoch:
            path = os.path.join(out_dir, f"epoch_{epoch}.pt")
            try:
                torch.save(state, path)
            except Exception as e:
                print(f"Warning: failed to save per-epoch checkpoint: {e}")

        if update_latest and save_latest:
            latest = os.path.join(out_dir, "latest.pt")
            try:
                torch.save(state, latest)
            except Exception as e:
                print(f"Warning: failed to save latest checkpoint: {e}")

        if save_best:
            metric_value: Optional[float]
            metric_name = self.best_metric_name
            if metric_name == "train_loss":
                metric_value = avg_loss
            else:
                metric_value = best_metric_value

            if metric_value is None:
                return

            better = (
                self.best_metric_value is None
                or (self.best_metric_lower_is_better and metric_value < self.best_metric_value)
                or (not self.best_metric_lower_is_better and metric_value > self.best_metric_value)
            )
            if better:
                self.best_metric_value = metric_value
                best = os.path.join(out_dir, "best.pt")
                try:
                    torch.save(state, best)
                    metric_str = f"{metric_name}={metric_value:.4f}"
                    print(f"New best ({metric_str}) saved to {best}")
                except Exception as e:
                    print(f"Warning: failed to save best checkpoint: {e}")

    def _maybe_eval_epoch(self, epoch: int) -> Optional[Dict]:
        """Optionally run dual-circle evaluation after each epoch and print summary metrics."""
        # Only perform evaluation on main process
        if self.distributed and self.rank != 0:
            return None
        ecfg = self.cfg.get("eval", {}).get("dual_circle", {})
        if not ecfg:
            return None
        if ecfg.get("enable") is False:
            return None
        image_dir = ecfg.get("image_dir")
        circle_dir = ecfg.get("circle_dir")
        rect_dir = ecfg.get("ignore_rect_dir") or ecfg.get("rectangle_dir")
        if not image_dir or not circle_dir:
            print("Eval skipped: eval.dual_circle.image_dir or circle_dir not set")
            return None
        # Determine output path for per-epoch CSV if requested
        out_root = self.cfg.get("output", {}).get("dir")
        eval_out_dir = ecfg.get("output_dir") or (os.path.join(out_root, "eval") if out_root else None)
        if eval_out_dir:
            os.makedirs(eval_out_dir, exist_ok=True)
            csv_path = os.path.join(eval_out_dir, f"epoch_{epoch}.csv")
        else:
            csv_path = None

        data_cfg = self.cfg.get("data", {})
        text_prompt = ecfg.get("text", data_cfg.get("text"))
        input_size = int(data_cfg.get("crop", 224))

        # Switch to eval inside the evaluation utility, then back to train.
        metrics = evaluate_dual_circle_dataset(
            model=self.model,
            image_dir=image_dir,
            circle_dir=circle_dir,
            rectangle_dir=rect_dir,
            device=self.device,
            text_prompt=text_prompt,
            output_csv=csv_path,
            visualize_samples=False,
            vis_output_dir=None,
            visualize_specific=None,
            input_size=input_size,
        )
        if not metrics:
            return None
        print("\n" + "=" * 60)
        print(f"EPOCH {epoch} — DUAL-CIRCLE EVALUATION")
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
        self._maybe_update_best_from_eval(epoch, metrics)
        return metrics

    def _maybe_update_best_from_eval(self, epoch: int, metrics: Dict):
        """If configured, update the best checkpoint based on dual-circle evaluation metrics."""
        if self.best_metric_name not in {"dual_circle_overall_f1", "dual_circle_mean_f1"}:
            return

        metric_key = "overall_f1" if self.best_metric_name == "dual_circle_overall_f1" else "mean_f1_score"
        metric_value_raw = metrics.get(metric_key)
        if metric_value_raw is None:
            return
        metric_value = float(metric_value_raw)

        # Drop heavy objects (e.g., DataFrame) before saving inside the checkpoint.
        metric_summary = {k: v for k, v in metrics.items() if k != "detailed_results"}

        train_loss = self.last_train_loss if self.last_train_loss is not None else 0.0
        self._save_checkpoint(
            epoch,
            train_loss,
            best_metric_value=metric_value,
            best_metric_payload=metric_summary,
            update_latest=False,
        )

    def _init_metric_tracking(self):
        """Prepare paths for metric history/plots and warm-start any prior records."""
        output_section = self.cfg.get("output") or {}
        checkpoint_section = self.cfg.get("checkpoint") or {}
        out_root = output_section.get("dir") if isinstance(output_section, dict) else None
        ckpt_dir = checkpoint_section.get("dir") if isinstance(checkpoint_section, dict) else None
        artifact_root = out_root or (os.path.dirname(ckpt_dir) or "." if ckpt_dir else ".")
        self.artifact_root = artifact_root
        self.training_history_path = os.path.join(self.artifact_root, "training_history.json")
        self.training_curve_path = os.path.join(self.artifact_root, "training_curve.png")
        self.training_history: List[Dict] = self._load_training_history()
        self._matplotlib_modules = None

    def _load_training_history(self) -> List[Dict]:
        if not os.path.isfile(self.training_history_path):
            return []
        try:
            with open(self.training_history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError, TypeError) as exc:
            print(f"Warning: failed to load training history at {self.training_history_path}: {exc}")
            return []

        cleaned: List[Dict] = []
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                epoch = entry.get("epoch")
                loss = entry.get("train_loss")
                if epoch is None or loss is None:
                    continue
                record: Dict = {"epoch": int(epoch), "train_loss": float(loss)}
                if entry.get("dual_circle_overall_f1") is not None:
                    record["dual_circle_overall_f1"] = float(entry["dual_circle_overall_f1"])
                if entry.get("dual_circle_mean_f1") is not None:
                    record["dual_circle_mean_f1"] = float(entry["dual_circle_mean_f1"])
                cleaned.append(record)
        cleaned.sort(key=lambda item: item["epoch"])
        return cleaned

    def _record_epoch_metrics(self, epoch: int, loss: float, eval_metrics: Optional[Dict]):
        """Persist per-epoch loss (and optional F1) and refresh the plot artifact."""
        entry: Dict = {"epoch": int(epoch), "train_loss": float(loss)}
        if eval_metrics:
            if eval_metrics.get("overall_f1") is not None:
                entry["dual_circle_overall_f1"] = float(eval_metrics["overall_f1"])
            if eval_metrics.get("mean_f1_score") is not None:
                entry["dual_circle_mean_f1"] = float(eval_metrics["mean_f1_score"])
        self.training_history = [e for e in self.training_history if e.get("epoch") != entry["epoch"]]
        self.training_history.append(entry)
        self.training_history.sort(key=lambda item: item["epoch"])
        os.makedirs(self.artifact_root, exist_ok=True)
        try:
            with open(self.training_history_path, "w", encoding="utf-8") as f:
                json.dump(self.training_history, f, indent=2)
        except OSError as exc:
            print(f"Warning: failed to write training history at {self.training_history_path}: {exc}")
            return
        self._plot_training_history()

    def _plot_training_history(self):
        if not self.training_history:
            return
        if self._matplotlib_modules is False:
            return
        if self._matplotlib_modules is None:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                self._matplotlib_modules = (matplotlib, plt)
            except ImportError:
                self._matplotlib_modules = False
                if (not self.distributed) or self.rank == 0:
                    print("Matplotlib not available; skipping training curve plot.")
                return
        _, plt = self._matplotlib_modules

        epochs = [entry["epoch"] for entry in self.training_history]
        losses = [entry["train_loss"] for entry in self.training_history]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(epochs, losses, marker="o", color="tab:blue", label="Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Progress")
        ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        f1_series = None
        f1_label = None
        for key, label in (
            ("dual_circle_overall_f1", "Dual-Circle Overall F1"),
            ("dual_circle_mean_f1", "Dual-Circle Mean F1"),
        ):
            values = [(entry["epoch"], entry.get(key)) for entry in self.training_history if entry.get(key) is not None]
            if values:
                f1_series = values
                f1_label = label
                break

        if f1_series:
            f1_epochs, f1_values = zip(*f1_series)
            ax2 = ax1.twinx()
            ax2.plot(f1_epochs, f1_values, marker="s", color="tab:green", label=f1_label)
            ax2.set_ylabel(f1_label)
            ax2.set_ylim(0.0, 1.0)
            lines = ax1.lines + ax2.lines
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc="best")
        else:
            ax1.legend(loc="best")

        fig.tight_layout()
        os.makedirs(self.artifact_root, exist_ok=True)
        fig.savefig(self.training_curve_path, dpi=150)
        plt.close(fig)

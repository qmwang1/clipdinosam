from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    probs = logits.sigmoid()
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = (probs.pow(2) + targets.pow(2)).sum(dim=(2, 3)) + eps
    loss = 1 - (num + eps) / (den + eps)
    return loss.mean()


def bce_loss(logits: torch.Tensor, targets: torch.Tensor):
    return F.binary_cross_entropy_with_logits(logits, targets)


def seg_loss(logits: torch.Tensor, targets: torch.Tensor):
    return dice_loss(logits, targets) + bce_loss(logits, targets)


def clip_align_loss(token_feats: torch.Tensor, text_feats: torch.Tensor, temp: float = 0.07):
    # token_feats: (B, N, D) -> pooled per image
    pooled = token_feats.mean(dim=1)
    pooled = F.normalize(pooled, dim=-1)
    text_feats = F.normalize(text_feats, dim=-1)
    logits = pooled @ text_feats.t() / temp
    labels = torch.arange(logits.size(0), device=logits.device) % text_feats.size(0)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels[: text_feats.size(0)])
    return 0.5 * (loss_i + loss_t)


def ssl_stub_loss(_img_tokens: torch.Tensor) -> torch.Tensor:
    # Placeholder for MIM/SSL or self-distillation; return zero by default
    return torch.zeros((), device=_img_tokens.device)


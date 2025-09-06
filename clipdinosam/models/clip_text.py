from typing import List

import torch
import torch.nn as nn


try:
    import open_clip
except ImportError:
    open_clip = None


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai"):
        super().__init__()
        if open_clip is None:
            raise ImportError("open_clip_torch is required for CLIP text encoder")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.clip = model
        self.tokenizer = open_clip.get_tokenizer(model_name)
        # Freeze by default; config can override via LoRA for tiny adaptation
        for p in self.clip.parameters():
            p.requires_grad = False

    @property
    def width(self) -> int:
        return self.clip.text_projection.shape[1]

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        tok = self.tokenizer(texts)
        tok = tok.to(next(self.clip.parameters()).device)
        with torch.no_grad():
            feats = self.clip.encode_text(tok)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats


def build_clip_text(model_name: str, pretrained: str = "openai") -> CLIPTextEncoder:
    return CLIPTextEncoder(model_name=model_name, pretrained=pretrained)


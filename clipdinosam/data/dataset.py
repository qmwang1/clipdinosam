from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageMaskDataset(Dataset):
    def __init__(
        self,
        root: str,
        masks: Optional[str] = None,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        text_prompt: Optional[str] = None,
    ):
        self.root = Path(root)
        self.mask_dir = Path(masks) if masks else None
        self.transform = transform
        self.mask_transform = mask_transform
        self.text_prompt = text_prompt
        self.images = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        mask = None
        if self.mask_dir is not None:
            # Expect mask with same stem
            m = self.mask_dir / (img_path.stem + ".png")
            if m.exists():
                mask = Image.open(m).convert("L")
        if self.transform:
            img = self.transform(img)
        if mask is not None and self.mask_transform:
            mask = self.mask_transform(mask)
        sample = {
            "image": img,
            "mask": mask,
            "text": self.text_prompt,
            "id": img_path.stem,
        }
        return sample


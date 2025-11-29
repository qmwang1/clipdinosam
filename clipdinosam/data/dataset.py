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
        mask_exts: Optional[tuple[str, ...]] = None,
    ):
        self.root = Path(root)
        self.mask_dir = Path(masks) if masks else None
        self.transform = transform
        self.mask_transform = mask_transform
        self.text_prompt = text_prompt
        # Accept common mask extensions so datasets with JPG masks work without copying.
        self.mask_exts = mask_exts or (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
        candidate_images = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if self.mask_dir is not None:
            # Keep only images that have a corresponding mask to avoid None masks in batches.
            self.images = [p for p in candidate_images if self._find_mask_path(p) is not None]
        else:
            self.images = candidate_images

    def _find_mask_path(self, img_path: Path) -> Optional[Path]:
        for ext in self.mask_exts:
            m = self.mask_dir / (img_path.stem + ext)
            if m.exists():
                return m
        return None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        mask = None
        if self.mask_dir is not None:
            # Expect mask with same stem
            m = self._find_mask_path(img_path)
            if m is not None:
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

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from PIL import Image
import torch
from torch.utils.data import Dataset


class VOCSegDataset(Dataset):
    """
    PASCAL VOC-style dataset reader for semantic segmentation.

    Expected structure (under `root`):
      - JPEGImages/{id}.jpg (or .png)
      - SegmentationClass/{id}.png
      - ImageSets/Segmentation/{split}.txt (optional) where each line is an id

    Behavior:
      - If `split` is a string in {train, val, test} and the list exists, it is used.
      - If `split` is a path to a .txt file, it is used.
      - Otherwise, all images in JPEGImages are used.
      - Masks are read as L mode and treated as binary by default (non-zero -> 1).
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        text_prompt: Optional[str] = None,
        binary: bool = True,
        image_dir: str = "JPEGImages",
        mask_dir: str = "SegmentationClass",
        images_exts: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.mask_transform = mask_transform
        self.text_prompt = text_prompt
        self.binary = binary
        self.image_dir = self.root / image_dir
        self.mask_dir = self.root / mask_dir
        self.images_exts = images_exts or [".jpg", ".jpeg", ".png"]

        ids = None
        if split is not None:
            split_path = None
            if split.endswith(".txt"):
                split_path = Path(split)
            else:
                candidate = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
                if candidate.exists():
                    split_path = candidate
            if split_path is not None and split_path.exists():
                ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]

        if ids is None:
            # All images in JPEGImages
            ids = [p.stem for p in sorted(self.image_dir.iterdir()) if p.suffix.lower() in self.images_exts]

        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def _find_image_path(self, stem: str) -> Path:
        for ext in self.images_exts:
            p = self.image_dir / f"{stem}{ext}"
            if p.exists():
                return p
        # Fallback: search
        matches = list(self.image_dir.glob(f"{stem}.*"))
        if not matches:
            raise FileNotFoundError(f"Image for id {stem} not found in {self.image_dir}")
        return matches[0]

    def __getitem__(self, idx: int) -> Dict:
        id_ = self.ids[idx]
        img_path = self._find_image_path(id_)
        mask_path = self.mask_dir / f"{id_}.png"

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") if mask_path.exists() else None

        if self.transform:
            img = self.transform(img)
        if mask is not None and self.mask_transform:
            mask = self.mask_transform(mask)
            if self.binary:
                mask = (mask > 0.5).float()

        return {
            "image": img,
            "mask": mask,
            "text": self.text_prompt,
            "id": id_,
        }


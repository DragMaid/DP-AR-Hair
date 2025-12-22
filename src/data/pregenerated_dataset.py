from functools import lru_cache
from typing import Dict, Optional
import random
import os
import torch
import json
import cv2
import numpy as np
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import Dataset
from dataclasses import dataclass
from pathlib import Path


class CelebVHQDataset(Dataset):
    """
    Optimized Dataset for:
      - driving_images/{id}_front.jpeg
      - driving_images/{id}_side.jpeg
      - generated_images/{id}.jpg

    Returns:
      {
          "front": Tensor[C,H,W],
          "side": Tensor[C,H,W],
          "generated": Tensor[C,H,W]
      }
    """

    def __init__(
        self,
        driving_dir: str,
        reference_dir: str,
        transform=None,
        preload: bool = False,
        cache_size: int = 64
    ):
        self.driving_dir = Path(driving_dir)
        self.reference_dir = Path(reference_dir)
        self.transform = transform
        self.preload = preload

        # ----------------------------------------------------------
        # 1. Scan driving folder for pairs
        # ----------------------------------------------------------
        self.samples = self._scan_driving_images()

        if len(self.samples) == 0:
            raise RuntimeError("No valid driving samples found!")

        # ----------------------------------------------------------
        # 2. Scan reference folder
        # ----------------------------------------------------------
        self.reference_paths = sorted([
            p for p in self.reference_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])

        if len(self.reference_paths) == 0:
            raise RuntimeError("No reference images found!")

    # ==========================================================
    # UTILS
    # ==========================================================

    def _load_image(self, path: Path):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # ==========================================================
    # MAIN API
    # ==========================================================

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        id_ = sample["id"]

        # ------------------------------------------------------
        # Load front & side
        # ------------------------------------------------------
        if self.preloaded_driving:
            front_img, side_img = self.preloaded_driving[id_]
            front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
            side_img = cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB)
        else:
            front_img = self._load_image(sample["front"])
            side_img = self._load_image(sample["side"])

        # ------------------------------------------------------
        # Random reference image
        # ------------------------------------------------------
        ref_idx = random.randrange(len(self.reference_paths))
        ref_path = self.reference_paths[ref_idx]

        # Fast LRU loading
        ref_img = self._cache_ref(str(ref_path))
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        # ------------------------------------------------------
        # Transform or default to tensor
        # ------------------------------------------------------
        if self.transform:
            front_img = self.transform(front_img)
            side_img = self.transform(side_img)
            ref_img = self.transform(ref_img)
        else:
            front_img = torch.from_numpy(
                front_img).permute(2, 0, 1).float() / 255.0
            side_img = torch.from_numpy(
                side_img).permute(2, 0, 1).float() / 255.0
            ref_img = torch.from_numpy(ref_img).permute(
                2, 0, 1).float() / 255.0

        return {
            "front": front_img,
            "side": side_img,
            "reference": ref_img
        }

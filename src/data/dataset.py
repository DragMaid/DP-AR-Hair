from functools import lru_cache
from typing import Dict
import random
import torch
import cv2
import numpy as np
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import Dataset
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoClip:
    """Data class for video clip information"""
    clip_id: str
    ytb_id: str
    start_sec: float
    end_sec: float
    # top, bottom, left, right (normalized)
    bbox: Tuple[float, float, float, float]


@dataclass
class FramePair:
    """Data class for selected frame pair"""
    frontal_frame: np.ndarray
    frontal_yaw: float
    frontal_pitch: float
    frontal_idx: int
    side_frame: np.ndarray
    side_yaw: float
    side_pitch: float
    side_idx: int


class CelebVHQDataset(Dataset):
    """
    Optimized Dataset for:
      - driving_images/{id}_front.jpeg
      - driving_images/{id}_side.jpeg
      - reference_images/<random>.jpg

    Returns:
      {
          "front": Tensor[C,H,W],
          "side": Tensor[C,H,W],
          "reference": Tensor[C,H,W]
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

        # ----------------------------------------------------------
        # 3. Optional preload
        # ----------------------------------------------------------
        self.preloaded_driving = None
        if preload:
            self.preloaded_driving = self._preload_driving_images()

        # ----------------------------------------------------------
        # 4. LRU cache for reference images
        # ----------------------------------------------------------
        @lru_cache(maxsize=cache_size)
        def _cache_ref(path_str):
            img = cv2.imread(path_str)
            if img is None:
                raise FileNotFoundError(
                    f"Failed to load reference image: {path_str}")
            return img

        self._cache_ref = _cache_ref

    # ==========================================================
    # UTILS
    # ==========================================================

    def _scan_driving_images(self):
        """
        Detects valid {id}_front and {id}_side pairs.
        """
        fronts = {}
        sides = {}

        for p in self.driving_dir.iterdir():
            name = p.name.lower()
            if name.endswith("_front.jpeg") or name.endswith("_front.jpg") or name.endswith("_front.png"):
                id_ = name.replace("_front.jpeg", "").replace(
                    "_front.jpg", "").replace("_front.png", "")
                fronts[id_] = p

            if name.endswith("_side.jpeg") or name.endswith("_side.jpg") or name.endswith("_side.png"):
                id_ = name.replace("_side.jpeg", "").replace(
                    "_side.jpg", "").replace("_side.png", "")
                sides[id_] = p

        # Keep only IDs that have both
        valid_ids = sorted(set(fronts.keys()) & set(sides.keys()))

        samples = [{
            "id": id_,
            "front": fronts[id_],
            "side": sides[id_]
        } for id_ in valid_ids]

        return samples

    def _preload_driving_images(self):
        cache = {}
        for s in self.samples:
            front = cv2.imread(str(s["front"]))
            side = cv2.imread(str(s["side"]))
            cache[s["id"]] = (front, side)
        return cache

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


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((512, 512)),
        T.ToTensor(),
    ])

    dataset = CelebVHQDataset(driving_dir="./assets/driving_images",
                              reference_dir="./assets/reference_images",
                              transform=transform,
                              preload=False)

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=2,
                            pin_memory=True, drop_last=True)

    epoch_iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch in epoch_iterator:
        print(batch["front"].dim())

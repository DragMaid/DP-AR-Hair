import torch
import cv2
from torch.utils.data import Dataset
from pathlib import Path


class _CelebVHQBase(Dataset):
    """Based datset setup for CelevVHQ"""

    def __init__(self, driving_dir: str, transform=None):
        self.driving_dir = Path(driving_dir)
        self.transform = transform
        self.samples = []

    def _load_image(self, path: Path):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _apply(self, img):
        if self.transform:
            return self.transform(img)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def __len__(self):
        return len(self.samples)

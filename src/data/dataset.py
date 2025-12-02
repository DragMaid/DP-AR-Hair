import torch
import json
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
    """PyTorch Dataset for CelebVHQ with dual-pose frames"""

    def __init__(self, json_path: str, processed_video_root: str,
                 transform=None, preload: bool = False):
        """
        Args:
            json_path: Path to celebvhq_info.json
            processed_video_root: Directory containing processed videos
            transform: Optional transform to apply to frames
            preload: If True, load all frame pairs into memory
        """
        self.processed_video_root = Path(processed_video_root)
        self.transform = transform

        # Load video clips info
        with open(json_path) as f:
            data = json.load(f)

        self.clips = []
        for clip_id, info in data['clips'].items():
            self.clips.append(VideoClip(
                clip_id=clip_id,
                ytb_id=info['ytb_id'],
                start_sec=info['duration']['start_sec'],
                end_sec=info['duration']['end_sec'],
                bbox=(info['bbox']['top'], info['bbox']['bottom'],
                      info['bbox']['left'], info['bbox']['right'])
            ))

        self.preloaded_data = None
        if preload:
            self._preload_frames()

    def _preload_frames(self):
        """Load all frame pairs into memory"""
        print("Preloading all frames into memory...")
        self.preloaded_data = []
        for idx in tqdm(range(len(self.clips))):
            try:
                frontal, side = self._load_frames(idx)
                self.preloaded_data.append((frontal, side))
            except Exception as e:
                print(f"Failed to load clip {idx}: {e}")
                self.preloaded_data.append(None)

    def _load_frames(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load frame pair for a given index"""
        clip = self.clips[idx]
        video_path = self.processed_video_root / f"{clip.clip_id}.mp4"

        if not video_path.exists():
            raise FileNotFoundError(f"Processed video not found: {video_path}")

        # Load cached frames if they exist
        frontal_cache = self.processed_video_root / \
            f"{clip.clip_id}_frontal.jpg"
        side_cache = self.processed_video_root / f"{clip.clip_id}_side.jpg"

        if frontal_cache.exists() and side_cache.exists():
            frontal = cv2.imread(str(frontal_cache))
            side = cv2.imread(str(side_cache))
        else:
            raise FileNotFoundError(
                f"Cached frames not found for {clip.clip_id}")

        return frontal, side

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (frontal_frame, side_frame) as tensors
        """
        if self.preloaded_data is not None:
            if self.preloaded_data[idx] is None:
                raise ValueError(f"Failed to load clip {idx}")
            frontal, side = self.preloaded_data[idx]
        else:
            frontal, side = self._load_frames(idx)

        # Convert BGR to RGB
        frontal = cv2.cvtColor(frontal, cv2.COLOR_BGR2RGB)
        side = cv2.cvtColor(side, cv2.COLOR_BGR2RGB)

        if self.transform:
            frontal = self.transform(frontal)
            side = self.transform(side)
        else:
            # Default: convert to tensor and normalize
            frontal = torch.from_numpy(frontal).permute(
                2, 0, 1).float() / 255.0
            side = torch.from_numpy(side).permute(2, 0, 1).float() / 255.0

        return frontal, side

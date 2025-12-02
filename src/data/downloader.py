import os
import json
import cv2
import yt_dlp
import numpy as np
from data.dataset import FramePair, VideoClip
from typing import Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from sixdrepnet import SixDRepNet
from tqdm import tqdm


class VideoDownloader:
    """Handles video downloading using yt-dlp"""

    def __init__(self, proxy: Optional[str] = None):
        self.proxy = proxy

    def download(self, video_path: str, ytb_id: str) -> bool:
        """Download video from YouTube using yt-dlp"""
        if os.path.exists(video_path):
            return True

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]',
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True,
            'external_downloader': 'aria2c',
            'external_downloader_args': ['-x', '16', '-k', '1M'],
        }

        if self.proxy:
            ydl_opts['proxy'] = self.proxy

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={ytb_id}'])
            return True
        except Exception as e:
            print(f"Failed to download {ytb_id}: {e}")
            return False


class VideoProcessor:
    """Handles video cropping and processing"""

    @staticmethod
    def expand_bbox(bbox: Tuple[float, float, float, float],
                    ratio: float = 0.02) -> Tuple[float, float, float, float]:
        """Expand bounding box by ratio"""
        top, bottom, left, right = bbox
        top = max(top - ratio, 0)
        bottom = min(bottom + ratio, 1)
        left = max(left - ratio, 0)
        right = min(right + ratio, 1)
        return top, bottom, left, right

    @staticmethod
    def to_square(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Convert bounding box to square"""
        top, bottom, left, right = bbox
        h = bottom - top
        w = right - left
        c = min(h, w) / 2
        c_h = (top + bottom) / 2
        c_w = (left + right) / 2

        top = c_h - c
        bottom = c_h + c
        left = c_w - c
        right = c_w + c
        return top, bottom, left, right

    @staticmethod
    def denormalize_bbox(bbox: Tuple[float, float, float, float],
                         height: int, width: int) -> Tuple[int, int, int, int]:
        """Convert normalized bbox to pixel coordinates"""
        top, bottom, left, right = bbox
        return (
            round(top * height),
            round(bottom * height),
            round(left * width),
            round(right * width)
        )

    def process_video(self, input_path: str, output_path: str,
                      bbox: Tuple[float, float, float, float],
                      start_sec: float, end_sec: float) -> str:
        """Crop and trim video using OpenCV"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = cap.get(cv2.CAP_PROP_FPS)

        # Process bbox
        expanded = self.expand_bbox(bbox)
        squared = self.to_square(expanded)
        top, bottom, left, right = self.denormalize_bbox(
            squared, height, width)

        # Calculate frame range
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        # Setup writer
        crop_width = right - left
        crop_height = bottom - top

        # Early exit if already preprocessed
        if crop_height == height and crop_width == width:
            return output_path

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (crop_width, crop_height))

        # Process frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            cropped = frame[top:bottom, left:right]
            out.write(cropped)
            current_frame += 1

        cap.release()
        out.release()
        return output_path


class PoseFrameSelector:
    """Selects frames with different poses from video"""

    def __init__(self, model: SixDRepNet, stride: int = 10,
                 yaw_diff_threshold: float = 20.0,
                 pitch_diff_threshold: float = 15.0,
                 laplacian_threshold: float = 40.0):
        self.model = model
        self.stride = stride
        self.yaw_diff_threshold = yaw_diff_threshold
        self.pitch_diff_threshold = pitch_diff_threshold
        self.laplacian_threshold = laplacian_threshold

    @staticmethod
    def compute_sharpness(frame: np.ndarray) -> float:
        """Compute Laplacian variance as sharpness metric"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def select_frames(self, video_path: str) -> FramePair:
        """Select two frames with different poses"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        sampled = []  # (frame_idx, yaw, pitch, sharpness)
        frame_idx = 0

        # Fast scan with stride
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.stride != 0:
                frame_idx += 1
                continue

            # Check sharpness
            sharpness = self.compute_sharpness(frame)
            if sharpness < self.laplacian_threshold:
                frame_idx += 1
                continue

            # Estimate pose
            pitch, yaw, _ = self.model.predict(frame)
            sampled.append((frame_idx, yaw, pitch, sharpness))
            frame_idx += 1

        cap.release()

        if len(sampled) == 0:
            raise ValueError("No usable frames detected")

        # Select frontal frame (yaw ≈ 0, pitch ≈ 0)
        frontal = min(sampled, key=lambda x: (abs(x[1]) + abs(x[2]), -x[3]))
        frontal_idx, frontal_yaw, frontal_pitch, _ = frontal

        # Select side frame (max difference from frontal)
        candidates = [
            (idx, yaw, pitch, sharp, abs(
                yaw - frontal_yaw) + abs(pitch - frontal_pitch))
            for idx, yaw, pitch, sharp in sampled
            if abs(yaw - frontal_yaw) + abs(pitch - frontal_pitch) > self.yaw_diff_threshold * 0.7
        ]

        if not candidates:
            raise ValueError("No frame with sufficient pose difference found")

        side = max(candidates, key=lambda x: (x[4], x[3]))
        side_idx, side_yaw, side_pitch, _, _ = side

        # Load the two selected frames
        cap = cv2.VideoCapture(video_path)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frontal_idx)
        _, frontal_frame = cap.read()

        cap.set(cv2.CAP_PROP_POS_FRAMES, side_idx)
        _, side_frame = cap.read()

        cap.release()

        return FramePair(
            frontal_frame=frontal_frame,
            frontal_yaw=frontal_yaw,
            frontal_pitch=frontal_pitch,
            frontal_idx=frontal_idx,
            side_frame=side_frame,
            side_yaw=side_yaw,
            side_pitch=side_pitch,
            side_idx=side_idx
        )


def process_single_clip(args) -> Tuple[str, bool]:
    """Process a single clip (for multiprocessing)"""
    clip, raw_root, processed_root, model_device = args

    try:
        downloader = VideoDownloader()
        processor = VideoProcessor()

        raw_path = os.path.join(raw_root, f"{clip.ytb_id}.mp4")
        processed_path = os.path.join(processed_root, f"{clip.clip_id}.mp4")

        # Download if needed
        if not downloader.download(raw_path, clip.ytb_id):
            return clip.clip_id, False

        # Process video
        processor.process_video(
            raw_path, processed_path,
            clip.bbox, clip.start_sec, clip.end_sec
        )

        # Select frames
        model = SixDRepNet(model_device)
        selector = PoseFrameSelector(model)
        frame_pair = selector.select_frames(processed_path)

        # Save frames
        frontal_path = os.path.join(
            processed_root, f"{clip.clip_id}_frontal.jpg")
        side_path = os.path.join(processed_root, f"{clip.clip_id}_side.jpg")
        cv2.imwrite(frontal_path, frame_pair.frontal_frame)
        cv2.imwrite(side_path, frame_pair.side_frame)

        return clip.clip_id, True

    except Exception as e:
        print(f"Error processing {clip.clip_id}: {e}")
        return clip.clip_id, False


def process_dataset(json_path: str, raw_root: str, processed_root: str,
                    num_workers: int = 4, model_device: int = -1):
    """Process entire dataset with multiprocessing"""
    os.makedirs(raw_root, exist_ok=True)
    os.makedirs(processed_root, exist_ok=True)

    # Load clips
    with open(json_path) as f:
        data = json.load(f)

    clips = []
    for clip_id, info in data['clips'].items():
        clips.append(VideoClip(
            clip_id=clip_id,
            ytb_id=info['ytb_id'],
            start_sec=info['duration']['start_sec'],
            end_sec=info['duration']['end_sec'],
            bbox=(info['bbox']['top'], info['bbox']['bottom'],
                  info['bbox']['left'], info['bbox']['right'])
        ))

    # Process with multiprocessing
    args_list = [(clip, raw_root, processed_root, model_device)
                 for clip in clips]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_clip, args): args[0].clip_id
                   for args in args_list}

        results = {}
        for future in tqdm(as_completed(futures), total=len(futures)):
            clip_id, success = future.result()
            results[clip_id] = success

    # Summary
    successful = sum(results.values())
    print(f"\nProcessed {successful}/{len(clips)} clips successfully")

    return results


if __name__ == '__main__':
    from configs.pipeline_config import pipeline_config as pco
    results = process_dataset(
        json_path=pco.dataset.json_path,
        raw_root=pco.dataset.raw_video_root,
        processed_root=pco.dataset.processed_video_root,
        num_workers=pco.dataset.num_workers,
        model_device=pco.dataset.device  # -1 for CPU, 0+ for GPU
    )

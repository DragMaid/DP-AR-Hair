# Preprocess module
Location: `src/data/preprocess.py`

## Overview
This document describes the main classes and functions provided by `src/data/preprocess.py`.

The module includes utilities for:
- downloading videos from YouTube via `yt_dlp` (optional)
- processing/cropping videos using OpenCV
- selecting two frames from a video that have pose variation or are simply the sharpest
- high-level functions to process a dataset of mp4 files in parallel

The implementation uses `SixDRepNet` (a pose estimation model) to compute yaw/pitch on sampled frames. If the model fails or download is disabled, the code includes fallbacks.

## Public classes and functions
- `VideoDownloader(proxy: Optional[str]=None, enable_download: bool=False)`
  - download(video_path, ytb_id) -> bool
  - Wraps `yt_dlp` and optionally uses `aria2c` as an external downloader. Returns True when the file exists or download succeeded.

- `VideoProcessor()`
  - `expand_bbox(bbox, ratio=0.02)` -> expanded normalized bbox
  - `to_square_px(bbox)` -> converts pixel bbox to a square bbox centered on the original
  - `denormalize_bbox(bbox, height, width)` -> normalized->pixel coords
  - `process_video(input_path, output_path, bbox, start_sec, end_sec)` -> Optional[str]
    - reads input video, crops to square bbox region, writes output mp4 and returns its path on success.

- `PoseFrameSelector(model: SixDRepNet, stride: int=3, yaw_diff_threshold: float=20.0, pitch_diff_threshold: float=15.0, laplacian_threshold: float=40.0)`
  - `compute_sharpness(frame)` -> float (Laplacian variance)
  - `select_frames(video_path)` -> Optional[FramePair]
    - scans the video frames, computes sharpness for fallback, runs pose prediction on subsampled frames, and returns a `FramePair` containing frontal and side frames (or uses sharpest-frame fallback if pose data is insufficient).

- `process_single_clip(args)`
  - Helper used by multiprocessing to process a single clip; extracts frames and saves JPEG images.

- `process_dataset(json_path, raw_root, processed_root, processed_img_root, num_workers=4, model_device=-1, enable_download=False, proxy=None)`
  - Orchestrates dataset processing across a folder of MP4 files with a ProcessPoolExecutor. Returns a mapping of clip_id -> success flag.

## How frame selection works
1. Scan all frames and compute Laplacian variance (sharpness) for every frame; store (idx, sharpness).
2. For every `stride`-th frame that is sharp enough, attempt to predict (pitch, yaw) using `SixDRepNet`.
3. If no pose predictions succeeded, fallback to selecting the two sharpest frames.
4. Otherwise, select a frontal frame (minimum absolute yaw+pitch) and look for candidate side frames whose pose differs enough; if none, fallback to two sharpest pose-sampled frames.

## Important implementation notes
- `VideoDownloader` respects `enable_download`. If disabled, it simply returns False for missing video files.
- `VideoProcessor.process_video` writes an mp4 using fourcc `mp4v` and preserves the original video fps.
- `PoseFrameSelector` relies on `SixDRepNet` having a `predict(frame)` method returning `(pitch, yaw, roll)` or similar.
- The module logs to both a file `video_processing.log` and stdout using Python's `logging` configuration at module import.

## Usage examples
Single file processing (programmatic):

    from src.data.preprocess import VideoProcessor, VideoDownloader
    processor = VideoProcessor()
    downloader = VideoDownloader(enable_download=False)
    success = downloader.download('local.mp4', 'YT_ID')  # False if not present and downloads disabled
    out = processor.process_video('local.mp4', 'cropped.mp4', bbox=(0.1,0.9,0.1,0.9), start_sec=0, end_sec=10)

Processing dataset directory:

    from src.data.preprocess import process_dataset
    results = process_dataset(json_path='data.json', raw_root='raw_videos', processed_root='proc_videos', processed_img_root='driving_images')

## Testing suggestions and edge cases
- Unit tests can mock `cv2.VideoCapture` to return a controlled stream of frames and validate `select_frames` returns expected indices for both pose-based and sharpness-based branches.
- Test `process_video` with a synthetic video file created by OpenCV to ensure correct cropping and writing behaviour.
- Because `process_dataset` uses multiprocessing, unit tests should validate behaviour with `num_workers=1` to avoid forking issues in CI.

## Dependencies and environment
- OpenCV (cv2) for video IO and image processing
- yt_dlp for YouTube downloads (optional)
- SixDRepNet for pose estimation (model must be importable and implement `predict`)
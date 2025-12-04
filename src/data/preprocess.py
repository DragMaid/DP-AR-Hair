import os
import json
import cv2
import yt_dlp
import numpy as np
import logging
from data.dataset import FramePair, VideoClip
from typing import Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from sixdrepnet import SixDRepNet
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VideoDownloader:
    """Handles video downloading using yt-dlp"""

    def __init__(self, proxy: Optional[str] = None, enable_download: bool = False):
        """
        Initialize VideoDownloader.

        Args:
            proxy: Optional proxy server URL
            enable_download: Whether to allow downloading videos
        """
        self.proxy = proxy
        self.enable_download = enable_download
        logger.info(
            f"VideoDownloader initialized (download_enabled={enable_download})")

    def download(self, video_path: str, ytb_id: str) -> bool:
        """
        Download video from YouTube using yt-dlp.

        Args:
            video_path: Path where video should be saved
            ytb_id: YouTube video ID

        Returns:
            True if video exists or was downloaded successfully, False otherwise
        """
        if os.path.exists(video_path):
            logger.debug(f"Video already exists: {video_path}")
            return True

        if not self.enable_download:
            logger.warning(f"Download disabled. Video not found: {video_path}")
            return False

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
            logger.debug(f"Using proxy: {self.proxy}")

        try:
            logger.info(f"Downloading video {ytb_id} to {video_path}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={ytb_id}'])
            logger.info(f"Successfully downloaded {ytb_id}")
            return True
        except Exception as e:
            logger.error(
                f"Failed to download {ytb_id}: {type(e).__name__}: {e}")
            return False


class VideoProcessor:
    """Handles video cropping and processing"""

    @staticmethod
    def expand_bbox(bbox: Tuple[float, float, float, float],
                    ratio: float = 0.02) -> Tuple[float, float, float, float]:
        """
        Expand bounding box by ratio.

        Args:
            bbox: Normalized bounding box (top, bottom, left, right)
            ratio: Expansion ratio

        Returns:
            Expanded bounding box
        """
        top, bottom, left, right = bbox
        top = max(top - ratio, 0)
        bottom = min(bottom + ratio, 1)
        left = max(left - ratio, 0)
        right = min(right + ratio, 1)
        return top, bottom, left, right

    @staticmethod
    def to_square_px(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Convert pixel bbox to the largest possible square while staying inside image.

        Args:
            bbox: Pixel bounding box (top, bottom, left, right)

        Returns:
            Square bounding box
        """
        top, bottom, left, right = bbox
        h = bottom - top
        w = right - left
        size = min(h, w)

        cy = (top + bottom) // 2
        cx = (left + right) // 2

        half = size // 2
        new_top = cy - half
        new_bottom = cy + half
        new_left = cx - half
        new_right = cx + half

        return new_top, new_bottom, new_left, new_right

    @staticmethod
    def denormalize_bbox(bbox: Tuple[float, float, float, float],
                         height: int, width: int) -> Tuple[int, int, int, int]:
        """
        Convert normalized bbox to pixel coordinates.

        Args:
            bbox: Normalized bounding box (0-1 range)
            height: Image height in pixels
            width: Image width in pixels

        Returns:
            Pixel bounding box
        """
        top, bottom, left, right = bbox
        return (
            round(top * height),
            round(bottom * height),
            round(left * width),
            round(right * width)
        )

    def process_video(self, input_path: str, output_path: str,
                      bbox: Tuple[float, float, float, float],
                      start_sec: float, end_sec: float) -> Optional[str]:
        """
        Crop and trim video using OpenCV.

        Args:
            input_path: Input video path
            output_path: Output video path
            bbox: Normalized bounding box
            start_sec: Start time in seconds
            end_sec: End time in seconds

        Returns:
            Output path if successful, None otherwise
        """
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {input_path}")
                return None

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            logger.debug(f"Processing video: {width}x{height} @ {fps}fps")

            # Process bbox
            expanded = self.expand_bbox(bbox)
            denormed = self.denormalize_bbox(expanded, height, width)
            top, bottom, left, right = self.to_square_px(denormed)

            # Calculate frame range
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)

            # Setup writer
            crop_width = right - left
            crop_height = bottom - top

            # Early exit if already preprocessed
            if crop_height == height and crop_width == width:
                logger.info(f"Video already preprocessed: {output_path}")
                cap.release()
                return output_path

            if crop_width <= 0 or crop_height <= 0:
                logger.error(
                    f"Invalid crop dimensions: {crop_width}x{crop_height}")
                cap.release()
                return None

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps,
                                  (crop_width, crop_height))

            if not out.isOpened():
                logger.error(f"Cannot create video writer: {output_path}")
                cap.release()
                return None

            # Process frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame
            frames_written = 0

            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {current_frame}")
                    break

                cropped = frame[top:bottom, left:right]
                out.write(cropped)
                frames_written += 1
                current_frame += 1

            cap.release()
            out.release()

            logger.info(f"Processed {frames_written} frames to {output_path}")
            return output_path

        except Exception as e:
            logger.error(
                f"Error processing video {input_path}: {type(e).__name__}: {e}")
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            return None


class PoseFrameSelector:
    """Selects frames with different poses from video"""

    def __init__(self, model: SixDRepNet, stride: int = 3,
                 yaw_diff_threshold: float = 20.0,
                 pitch_diff_threshold: float = 15.0,
                 laplacian_threshold: float = 40.0):
        """
        Initialize PoseFrameSelector.

        Args:
            model: SixDRepNet model for pose estimation
            stride: Frame sampling stride
            yaw_diff_threshold: Minimum yaw difference for pose variation
            pitch_diff_threshold: Minimum pitch difference for pose variation
            laplacian_threshold: Minimum sharpness threshold
        """
        self.model = model
        self.stride = stride
        self.yaw_diff_threshold = yaw_diff_threshold
        self.pitch_diff_threshold = pitch_diff_threshold
        self.laplacian_threshold = laplacian_threshold

    @staticmethod
    def compute_sharpness(frame: np.ndarray) -> float:
        """
        Compute Laplacian variance as sharpness metric.

        Args:
            frame: Input frame

        Returns:
            Sharpness score
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception as e:
            logger.error(f"Error computing sharpness: {type(e).__name__}: {e}")
            return 0.0

    def _load_frame_pair(self, video_path: str, idx1: int, idx2: int,
                         frontal_yaw: float, frontal_pitch: float,
                         side_yaw: float, side_pitch: float) -> Optional[FramePair]:
        """
        Load frame pair from video.

        Args:
            video_path: Path to video file
            idx1: First frame index
            idx2: Second frame index
            frontal_yaw: Yaw angle of first frame
            frontal_pitch: Pitch angle of first frame
            side_yaw: Yaw angle of second frame
            side_pitch: Pitch angle of second frame

        Returns:
            FramePair object or None if loading fails
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(
                    f"Cannot open video for frame loading: {video_path}")
                return None

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx1)
            ret1, frame1 = cap.read()

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx2)
            ret2, frame2 = cap.read()

            cap.release()

            if not ret1 or not ret2:
                logger.error(
                    f"Failed to read frames {idx1} or {idx2} from {video_path}")
                return None

            return FramePair(
                frontal_frame=frame1,
                side_frame=frame2,
                frontal_idx=idx1,
                side_idx=idx2,
                frontal_yaw=frontal_yaw,
                frontal_pitch=frontal_pitch,
                side_yaw=side_yaw,
                side_pitch=side_pitch
            )

        except Exception as e:
            logger.error(f"Error loading frame pair: {type(e).__name__}: {e}")
            if 'cap' in locals():
                cap.release()
            return None

    def select_frames(self, video_path: str) -> Optional[FramePair]:
        """
        Select two frames with pose preference but always fallback to 2 least blurry frames.

        Args:
            video_path: Path to video file

        Returns:
            FramePair object or None if selection fails
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return None

            sampled = []  # (idx, yaw, pitch, sharpness)
            all_frames = []  # (idx, sharpness)
            frame_idx = 0

            # Scan video
            logger.debug(f"Scanning frames from {video_path}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Track sharpness for fallback
                sharpness = self.compute_sharpness(frame)
                all_frames.append((frame_idx, sharpness))

                if frame_idx % self.stride == 0:
                    if sharpness >= self.laplacian_threshold:
                        try:
                            pitch, yaw, _ = self.model.predict(frame)
                            sampled.append((frame_idx, yaw, pitch, sharpness))
                        except Exception as e:
                            logger.debug(
                                f"Pose estimation failed for frame {frame_idx}: {e}")

                frame_idx += 1

            cap.release()
            logger.info(
                f"Scanned {frame_idx} frames, {len(sampled)} with valid poses")

            # Fallback A: No sampled frames at all
            if len(sampled) == 0:
                logger.warning(
                    "No frames with valid poses, using sharpness-based selection")
                if len(all_frames) < 2:
                    logger.error("Not enough frames in video")
                    return None

                best = sorted(all_frames, key=lambda x: x[1], reverse=True)[:2]
                best.sort(key=lambda x: x[0])
                idx1, idx2 = best[0][0], best[1][0]

                return self._load_frame_pair(
                    video_path, idx1, idx2,
                    frontal_yaw=0, frontal_pitch=0,
                    side_yaw=0, side_pitch=0
                )

            # Try pose-based selection
            frontal = min(sampled, key=lambda x: (
                abs(x[1]) + abs(x[2]), -x[3]))
            frontal_idx, frontal_yaw, frontal_pitch, frontal_sharp = frontal

            # Candidates for "side"
            candidates = []
            for idx, yaw, pitch, sharp in sampled:
                pose_diff = abs(yaw - frontal_yaw) + abs(pitch - frontal_pitch)
                if pose_diff > self.yaw_diff_threshold * 0.7:
                    candidates.append((idx, yaw, pitch, sharp, pose_diff))

            # Fallback B: Not enough pose difference
            if not candidates:
                logger.warning(
                    "Not enough pose variation, using sharpest frames")
                best = sorted(sampled, key=lambda x: x[3], reverse=True)[:2]
                best.sort(key=lambda x: x[0])

                f_idx, f_yaw, f_pitch, _ = best[0]
                s_idx, s_yaw, s_pitch, _ = best[1]

                return self._load_frame_pair(
                    video_path, f_idx, s_idx,
                    frontal_yaw=f_yaw, frontal_pitch=f_pitch,
                    side_yaw=s_yaw, side_pitch=s_pitch
                )

            # Pose-based side selection
            side = max(candidates, key=lambda x: (x[4], x[3]))
            side_idx, side_yaw, side_pitch, _, pose_diff = side

            logger.info(f"Selected frames {frontal_idx} (frontal) and {side_idx} (side), "
                        f"pose_diff={float(pose_diff):.1f}")

            return self._load_frame_pair(
                video_path, frontal_idx, side_idx,
                frontal_yaw=frontal_yaw, frontal_pitch=frontal_pitch,
                side_yaw=side_yaw, side_pitch=side_pitch
            )

        except Exception as e:
            logger.error(
                f"Error selecting frames from {video_path}: {type(e).__name__}: {e}")
            if 'cap' in locals():
                cap.release()
            return None


def process_single_clip(args) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single clip (for multiprocessing).

    Args:
        args: Tuple of (clip, raw_root, processed_root, driving_img_root, 
                       model_device, enable_download, proxy)

    Returns:
        Tuple of (clip_id, success, error_message)
    """
    clip, raw_root, processed_root, driving_img_root, model_device, enable_download, proxy = args

    try:
        # downloader = VideoDownloader(
        # proxy=proxy, enable_download=enable_download)
        # processor = VideoProcessor()

        raw_path = os.path.join(raw_root, f"{clip.clip_id}.mp4")
        # processed_path = os.path.join(processed_root, f"{clip.clip_id}.mp4")

        # Check if video exists
        # if not Path(raw_path).exists():
        # if not downloader.download(raw_path, clip.ytb_id):
        # return clip.clip_id, False, "Video not found and download failed/disabled"

        # Process video
        # result = processor.process_video(
        # raw_path, processed_path,
        # clip.bbox, clip.start_sec, clip.end_sec
        # )

        # if result is None:
        # return clip.clip_id, False, "Video processing failed"
        try:
            cap = cv2.VideoCapture(raw_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {raw_path}")
                return None

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if width / height != 1:
                return clip.clip_id, False, "Video not pre-processed"

        except:
            return clip.clip_id, False, "Can't open video"

        # Select frames
        model = SixDRepNet(model_device)
        selector = PoseFrameSelector(model)
        frame_pair = selector.select_frames(raw_path)

        if frame_pair is None:
            return clip.clip_id, False, "Frame selection failed"

        # Save frames
        frontal_path = os.path.join(
            driving_img_root, f"{clip.clip_id}_frontal.jpg")
        side_path = os.path.join(
            driving_img_root, f"{clip.clip_id}_side.jpg")

        cv2.imwrite(frontal_path, frame_pair.frontal_frame)
        cv2.imwrite(side_path, frame_pair.side_frame)

        logger.info(f"Successfully processed clip {clip.clip_id}")
        return clip.clip_id, True, None

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"Error processing clip {clip.clip_id}: {error_msg}")
        return clip.clip_id, False, error_msg


def process_dataset(json_path: str, raw_root: str, processed_root: str,
                    processed_img_root: str, num_workers: int = 4,
                    model_device: int = -1, enable_download: bool = False,
                    proxy: Optional[str] = None) -> Dict[str, bool]:
    """
    Process entire dataset with multiprocessing.

    Args:
        json_path: Path to dataset JSON file
        raw_root: Directory for raw videos
        processed_root: Directory for processed videos
        processed_img_root: Directory for extracted frames
        num_workers: Number of parallel workers
        model_device: Device for pose model (-1 for CPU, 0+ for GPU)
        enable_download: Whether to download missing videos
        proxy: Optional proxy server URL

    Returns:
        Dictionary mapping clip_id to success status
    """
    logger.info("Starting dataset processing")
    logger.info(f"Configuration: workers={num_workers}, device={model_device}, "
                f"download_enabled={enable_download}")

    # Create directories
    os.makedirs(raw_root, exist_ok=True)
    os.makedirs(processed_root, exist_ok=True)
    os.makedirs(processed_img_root, exist_ok=True)

    # Load clips
    # try:
    # with open(json_path) as f:
    # data = json.load(f)
    # logger.info(f"Loaded dataset from {json_path}")
    # except Exception as e:
    # logger.error(f"Failed to load dataset JSON: {type(e).__name__}: {e}")
    # return {}

    # clips = []
    # for clip_id, info in data['clips'].items():
    # try:
    # clips.append(VideoClip(
    # clip_id=clip_id,
    # ytb_id=info['ytb_id'],
    # start_sec=info['duration']['start_sec'],
    # end_sec=info['duration']['end_sec'],
    # bbox=(info['bbox']['top'], info['bbox']['bottom'],
    # info['bbox']['left'], info['bbox']['right'])
    # ))
    # except Exception as e:
    # logger.error(
    # f"Failed to parse clip {clip_id}: {type(e).__name__}: {e}")

    from pathlib import Path

    clips = []
    # replace with your folder path
    video_folder = Path(raw_root)

    for mp4_file in video_folder.glob("*.mp4"):
        try:
            clip_id = mp4_file.stem  # filename without extension
            clips.append(VideoClip(
                clip_id=clip_id,
                ytb_id=None,             # no YouTube ID in this case
                start_sec=0,             # start of video
                end_sec=None,            # full length, you can set manually if needed
                bbox=None                # no bounding box info
            ))
        except Exception as e:
            logger.error(
                f"Failed to parse clip {mp4_file.name}: {type(e).__name__}: {e}")

    logger.info(f"Processing {len(clips)} clips")

    # Process with multiprocessing
    args_list = [
        (clip, raw_root, processed_root, processed_img_root,
         model_device, enable_download, proxy)
        for clip in clips
    ]

    results = {}
    errors = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_clip, args): args[0].clip_id
            for args in args_list
        }

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Processing clips"):
            try:
                clip_id, success, error_msg = future.result()
                results[clip_id] = success
                if error_msg:
                    errors[clip_id] = error_msg
            except Exception as e:
                clip_id = futures[future]
                logger.error(
                    f"Unexpected error for clip {clip_id}: {type(e).__name__}: {e}")
                results[clip_id] = False
                errors[clip_id] = f"Unexpected error: {type(e).__name__}: {e}"

    # Summary
    successful = sum(results.values())
    failed = len(results) - successful

    logger.info(f"\n{'='*60}")
    logger.info(
        f"Processing complete: {successful}/{len(clips)} clips successful")
    logger.info(f"Failed: {failed} clips")

    if errors:
        logger.info(f"\nError summary:")
        error_counts = {}
        for error in errors.values():
            error_counts[error] = error_counts.get(error, 0) + 1
        for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {count}x: {error}")

    logger.info(f"{'='*60}\n")

    return results


if __name__ == '__main__':
    from configs.pipeline_config import pipeline_config as pco

    LOCAL_DIR = Path(__file__).resolve().parent

    # Configure download behavior
    ENABLE_DOWNLOAD = False  # Set to True to allow downloading missing videos
    PROXY = None  # Set to proxy URL if needed

    results = process_dataset(
        json_path=LOCAL_DIR / pco.dataset.json_path,
        raw_root=pco.dataset.raw_video_root,
        processed_root=pco.dataset.processed_video_root,
        processed_img_root=pco.dataset.processed_images_root,
        num_workers=pco.dataset.num_workers,
        model_device=pco.dataset.device,
        enable_download=ENABLE_DOWNLOAD,
        proxy=PROXY
    )

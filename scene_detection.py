"""
YouTube Thumbnail Generator - Scene Detection Module
=====================================================
Detects scene changes in videos using PySceneDetect.
Extracts representative frames from each scene.
"""

import cv2
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from scenedetect import detect, AdaptiveDetector, ContentDetector, HashDetector, ThresholdDetector
from scenedetect.frame_timecode import FrameTimecode

from config import (
    SCENE_DETECTOR,
    SCENE_THRESHOLD,
    MIN_SCENE_LENGTH,
    MIN_FRAMES_PER_SCENE,
    MAX_FRAMES_PER_SCENE,
    MAX_TOTAL_FRAMES,
    FRAME_SCALING_BY_VIDEO_DURATION,
    FRAME_SCALING_BY_SCENE_DURATION,
)
from utils import setup_logger, VideoOutput

logger = setup_logger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Scene:
    """Represents a detected scene"""
    index: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    frame_indices: list  # Frames to extract from this scene

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SceneDetectionResult:
    """Result of scene detection for a video"""
    video_path: str
    total_scenes: int
    total_frames_to_extract: int
    scenes: list[Scene]

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "total_scenes": self.total_scenes,
            "total_frames_to_extract": self.total_frames_to_extract,
            "scenes": [s.to_dict() for s in self.scenes]
        }


# =============================================================================
# SCENE DETECTION
# =============================================================================

def get_detector(detector_type: str, threshold: float):
    """Get the appropriate scene detector based on config"""

    detectors = {
        "adaptive": AdaptiveDetector(
            adaptive_threshold=threshold,
            min_scene_len=int(MIN_SCENE_LENGTH * 30)  # Assuming 30fps
        ),
        "content": ContentDetector(
            threshold=threshold if threshold > 10 else 27.0,
            min_scene_len=int(MIN_SCENE_LENGTH * 30)
        ),
        "hash": HashDetector(
            threshold=threshold if threshold < 1 else 0.395,
            min_scene_len=int(MIN_SCENE_LENGTH * 30)
        ),
        "threshold": ThresholdDetector(
            threshold=threshold if threshold < 50 else 12.0,
            min_scene_len=int(MIN_SCENE_LENGTH * 30)
        ),
    }

    return detectors.get(detector_type, detectors["adaptive"])


def calculate_frames_for_scene(scene_duration: float, video_duration: float) -> int:
    """
    Calculate how many frames to extract from a scene based on durations.

    Args:
        scene_duration: Duration of this scene in seconds
        video_duration: Total video duration in seconds

    Returns:
        Number of frames to extract from this scene
    """
    # Get base frames from video duration
    base_frames = MIN_FRAMES_PER_SCENE
    for threshold in sorted(FRAME_SCALING_BY_VIDEO_DURATION.keys()):
        if video_duration < threshold:
            base_frames = FRAME_SCALING_BY_VIDEO_DURATION[threshold]
            break

    # Get bonus frames from scene duration
    bonus_frames = 0
    for threshold in sorted(FRAME_SCALING_BY_SCENE_DURATION.keys()):
        if scene_duration < threshold:
            bonus_frames = FRAME_SCALING_BY_SCENE_DURATION[threshold]
            break

    total = base_frames + bonus_frames

    # Apply limits
    return max(MIN_FRAMES_PER_SCENE, min(MAX_FRAMES_PER_SCENE, total))


def calculate_frame_indices(start_frame: int, end_frame: int, num_frames: int = 10) -> list[int]:
    """Calculate frame indices to extract from a scene"""

    duration = end_frame - start_frame

    if duration <= 1:
        return [start_frame]

    if num_frames == 1:
        # Return middle frame
        return [start_frame + duration // 2]

    if num_frames == 2:
        # Return start and end (with small offset)
        offset = max(1, duration // 10)
        return [start_frame + offset, end_frame - offset]

    # For 3+ frames, distribute evenly
    indices = []
    for i in range(num_frames):
        # Calculate position (avoid exact boundaries)
        if i == 0:
            pos = start_frame + max(1, duration // 20)
        elif i == num_frames - 1:
            pos = end_frame - max(1, duration // 20)
        else:
            pos = start_frame + int(duration * (i / (num_frames - 1)))

        indices.append(min(pos, end_frame - 1))

    return sorted(set(indices))


def detect_scenes(video_path: Path, output: VideoOutput) -> Optional[SceneDetectionResult]:
    """
    Detect scenes in a video and determine which frames to extract.

    Args:
        video_path: Path to the video file
        output: VideoOutput instance for saving results

    Returns:
        SceneDetectionResult with detected scenes and frame indices
    """

    logger.info(f"Detecting scenes in: {video_path.name}")

    # Check if already processed
    if output.scenes_file.exists():
        logger.info("Loading cached scene detection results...")
        try:
            with open(output.scenes_file, 'r') as f:
                data = json.load(f)
                scenes = [Scene(**s) for s in data['scenes']]
                return SceneDetectionResult(
                    video_path=data['video_path'],
                    total_scenes=data['total_scenes'],
                    total_frames_to_extract=data['total_frames_to_extract'],
                    scenes=scenes
                )
        except Exception as e:
            logger.warning(f"Could not load cached results: {e}")

    try:
        # Get detector
        detector = get_detector(SCENE_DETECTOR, SCENE_THRESHOLD)
        logger.info(f"Using {SCENE_DETECTOR} detector with threshold {SCENE_THRESHOLD}")

        # Detect scenes
        scene_list = detect(str(video_path), detector, show_progress=True)

        if not scene_list:
            logger.warning("No scenes detected, treating entire video as one scene")
            # Get video info to create a single scene
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            cap.release()

            scene_list = [(
                FrameTimecode(0, fps=fps),
                FrameTimecode(total_frames - 1, fps=fps)
            )]

        logger.success(f"Detected {len(scene_list)} scenes")

        # Calculate total video duration for scaling
        video_duration = sum(
            end_tc.get_seconds() - start_tc.get_seconds()
            for start_tc, end_tc in scene_list
        )
        logger.info(f"Total video duration: {video_duration:.1f}s ({video_duration/60:.1f} min)")

        # Process scenes with dynamic frame calculation
        scenes = []
        total_frames_to_extract = 0

        for idx, (start_tc, end_tc) in enumerate(scene_list):
            start_frame = start_tc.get_frames()
            end_frame = end_tc.get_frames()
            start_time = start_tc.get_seconds()
            end_time = end_tc.get_seconds()
            duration = end_time - start_time

            # Calculate frames dynamically based on durations
            num_frames = calculate_frames_for_scene(duration, video_duration)

            frame_indices = calculate_frame_indices(
                start_frame,
                end_frame,
                num_frames
            )

            scene = Scene(
                index=idx,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                duration=round(duration, 2),
                frame_indices=frame_indices
            )

            scenes.append(scene)
            total_frames_to_extract += len(frame_indices)

        # Apply total frames cap if configured - drop shortest scenes to fit
        if MAX_TOTAL_FRAMES > 0 and total_frames_to_extract > MAX_TOTAL_FRAMES:
            logger.warning(
                f"Total frames ({total_frames_to_extract}) exceeds cap ({MAX_TOTAL_FRAMES}). "
                f"Dropping shortest scenes to fit..."
            )

            # Sort scenes by duration (shortest first = candidates for removal)
            scenes_by_duration = sorted(scenes, key=lambda s: s.duration)

            # Drop shortest scenes until we're under the cap
            dropped_count = 0
            for scene in scenes_by_duration:
                if total_frames_to_extract <= MAX_TOTAL_FRAMES:
                    break
                total_frames_to_extract -= len(scene.frame_indices)
                scenes.remove(scene)
                dropped_count += 1

            # Re-index remaining scenes (maintain chronological order)
            scenes.sort(key=lambda s: s.start_frame)
            for idx, scene in enumerate(scenes):
                scene.index = idx

            logger.info(f"Dropped {dropped_count} shortest scenes to fit frame cap")

        logger.info(f"Will extract {total_frames_to_extract} frames from {len(scenes)} scenes")

        result = SceneDetectionResult(
            video_path=str(video_path),
            total_scenes=len(scenes),
            total_frames_to_extract=total_frames_to_extract,
            scenes=scenes
        )

        # Save results
        with open(output.scenes_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.success(f"Scene detection complete: {len(scenes)} scenes, {total_frames_to_extract} frames to extract")

        return result

    except Exception as e:
        logger.error(f"Scene detection failed: {e}")
        return None


def extract_scene_frames(video_path: Path, result: SceneDetectionResult, output: VideoOutput) -> list[Path]:
    """
    Extract the selected frames from each scene.

    Args:
        video_path: Path to the video file
        result: SceneDetectionResult with frame indices to extract
        output: VideoOutput instance for saving frames

    Returns:
        List of paths to extracted frames
    """

    logger.info(f"Extracting {result.total_frames_to_extract} frames from {result.total_scenes} scenes...")

    extracted_frames = []

    try:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error("Could not open video file")
            return []

        # Collect all frame indices to extract
        all_frames = []
        for scene in result.scenes:
            for frame_idx in scene.frame_indices:
                all_frames.append((scene.index, frame_idx))

        # Sort by frame index for sequential reading efficiency
        all_frames.sort(key=lambda x: x[1])

        for scene_idx, frame_idx in all_frames:
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # Save frame
                frame_filename = f"scene_{scene_idx:03d}_frame_{frame_idx:06d}.jpg"
                frame_path = output.frames_dir / frame_filename

                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted_frames.append(frame_path)
            else:
                logger.warning(f"Could not read frame {frame_idx}")

        cap.release()

        logger.success(f"Extracted {len(extracted_frames)} frames")
        return extracted_frames

    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        return []


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def process_video_scenes(video_path: Path, output: VideoOutput) -> tuple[Optional[SceneDetectionResult], list[Path]]:
    """
    Full scene detection and frame extraction pipeline.

    Args:
        video_path: Path to the video file
        output: VideoOutput instance

    Returns:
        Tuple of (SceneDetectionResult, list of extracted frame paths)
    """

    # Detect scenes
    result = detect_scenes(video_path, output)

    if not result:
        return None, []

    # Extract frames
    frames = extract_scene_frames(video_path, result, output)

    return result, frames

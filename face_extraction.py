"""
YouTube Thumbnail Generator - Face Extraction Module
=====================================================
Detects faces in extracted frames using InsightFace.
Expression detection and embedding extraction for clustering.
"""

# CUDA setup must be imported FIRST before any ONNX/InsightFace imports
import cuda_setup  # noqa: F401

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
import math

from config import (
    FACE_DETECTOR_MODEL,
    FACE_CONFIDENCE_THRESHOLD,
    FACE_QUALITY_FILTERS,
    EXPRESSION_DETECTION,
    EXPRESSION_THRESHOLDS,
)
from utils import setup_logger, VideoOutput

logger = setup_logger(__name__)

# =============================================================================
# LANDMARK CONSTANTS FOR EXPRESSION DETECTION
# =============================================================================

# 68-point landmark indices for mouth (standard mapping)
MOUTH_LEFT_CORNER = 48
MOUTH_RIGHT_CORNER = 54
UPPER_LIP_TOP = [50, 51, 52]      # Top of upper lip (outer)
LOWER_LIP_BOTTOM = [56, 57, 58]   # Bottom of lower lip (outer)
UPPER_LIP_INNER = [61, 62, 63]    # Inner upper lip
LOWER_LIP_INNER = [65, 66, 67]    # Inner lower lip
LIP_CENTER_TOP = 51               # Center top of upper lip
LIP_CENTER_BOTTOM = 57            # Center bottom of lower lip

# Global face analysis app (lazy loaded)
_face_app = None


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FaceInfo:
    """Information about a detected face"""
    frame_path: str
    bbox: list  # [x1, y1, x2, y2]
    confidence: float
    quality_score: float
    pose: list  # [pitch, yaw, roll] if available
    expression: str = "neutral"  # "mouth_closed", "smiling", "neutral", "mouth_open"
    expression_scores: dict = None  # {"mar": float, "smile_ratio": float, "corner_angle": float}
    embedding: Optional[list] = None  # 512D face embedding for clustering
    # DEPRECATED: These fields are kept for backwards compatibility with existing data
    # Clustering now replaces the need for reference-based matching
    similarity: float = 1.0  # DEPRECATED - always 1.0
    is_match: bool = True    # DEPRECATED - always True

    def __post_init__(self):
        if self.expression_scores is None:
            self.expression_scores = {}

    def to_dict(self, include_embedding: bool = True) -> dict:
        """Convert to dictionary, optionally excluding embedding to save space."""
        result = asdict(self)
        if not include_embedding:
            result.pop('embedding', None)
        return result


@dataclass
class FaceExtractionResult:
    """Result of face extraction for a video"""
    total_frames_analyzed: int
    frames_with_faces: int
    all_faces: list[FaceInfo]

    def to_dict(self, include_embeddings: bool = True) -> dict:
        """
        Convert to dictionary for JSON serialization.

        Args:
            include_embeddings: If True, include 512D embeddings (large).
                               Set to False for lighter exports.
        """
        return {
            "total_frames_analyzed": self.total_frames_analyzed,
            "frames_with_faces": self.frames_with_faces,
            "all_faces": [f.to_dict(include_embedding=include_embeddings) for f in self.all_faces]
        }


# =============================================================================
# INSIGHTFACE INITIALIZATION
# =============================================================================

def get_face_app():
    """Get or initialize the InsightFace analysis app"""
    global _face_app

    if _face_app is None:
        from insightface.app import FaceAnalysis

        logger.info(f"Initializing InsightFace with model: {FACE_DETECTOR_MODEL}")

        _face_app = FaceAnalysis(
            name=FACE_DETECTOR_MODEL,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))

        logger.success("InsightFace initialized successfully")

    return _face_app


# =============================================================================
# FACE DETECTION & ANALYSIS
# =============================================================================

def compute_face_quality(face, frame: np.ndarray) -> float:
    """
    Compute a quality score for a detected face.

    Factors considered:
    - Face size (larger is better)
    - Pose (frontal is better)
    - Sharpness (less blur is better)
    - Brightness (well-lit is better)
    """

    score = 100.0

    try:
        # Extract face region
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        # Ensure valid coordinates
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_region = frame[y1:y2, x1:x2]

        if face_region.size == 0:
            return 0.0

        face_width = x2 - x1
        face_height = y2 - y1

        # 1. Size score (0-25 points)
        min_size = FACE_QUALITY_FILTERS.get('min_face_size', 100)
        size_score = min(25, (min(face_width, face_height) / min_size) * 12.5)
        score = size_score

        # 2. Pose score (0-25 points) - if available
        if hasattr(face, 'pose') and face.pose is not None:
            pitch, yaw, roll = face.pose
            max_angle = FACE_QUALITY_FILTERS.get('max_pose_angle', 30)

            # Penalize non-frontal poses
            pose_deviation = (abs(pitch) + abs(yaw) + abs(roll)) / 3
            pose_score = max(0, 25 - (pose_deviation / max_angle) * 25)
            score += pose_score
        else:
            score += 15  # Neutral score if pose not available

        # 3. Sharpness score (0-25 points)
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()

        blur_threshold = FACE_QUALITY_FILTERS.get('max_blur_threshold', 100)
        sharpness_score = min(25, (laplacian_var / blur_threshold) * 25)
        score += sharpness_score

        # 4. Brightness score (0-25 points)
        brightness = np.mean(gray_face)
        min_brightness = FACE_QUALITY_FILTERS.get('min_brightness', 40)

        if brightness < min_brightness:
            brightness_score = (brightness / min_brightness) * 12.5
        elif brightness > 220:
            brightness_score = 12.5  # Overexposed
        else:
            brightness_score = 25.0

        score += brightness_score

    except Exception as e:
        logger.warning(f"Error computing face quality: {e}")
        return 50.0  # Default middle score

    return float(min(100.0, max(0.0, score)))


# =============================================================================
# EXPRESSION DETECTION FUNCTIONS
# =============================================================================

def compute_mouth_aspect_ratio(landmarks_68: np.ndarray) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) using 68-point landmarks.

    MAR = (vertical_distance) / (horizontal_distance)
    Low MAR = closed mouth, High MAR = open mouth
    """
    # Get vertical distance using inner lip points
    upper_inner = np.mean([landmarks_68[i] for i in UPPER_LIP_INNER], axis=0)
    lower_inner = np.mean([landmarks_68[i] for i in LOWER_LIP_INNER], axis=0)
    vertical_dist = np.linalg.norm(upper_inner - lower_inner)

    # Get horizontal distance (mouth width)
    left_corner = landmarks_68[MOUTH_LEFT_CORNER]
    right_corner = landmarks_68[MOUTH_RIGHT_CORNER]
    horizontal_dist = np.linalg.norm(right_corner - left_corner)

    if horizontal_dist == 0:
        return 0.0

    return float(vertical_dist / horizontal_dist)


def compute_smile_ratio(landmarks_68: np.ndarray) -> float:
    """
    Calculate smile ratio based on mouth width vs height.

    Smiling typically increases mouth width and decreases height.
    Higher ratio = more likely smiling
    """
    # Mouth width
    left_corner = landmarks_68[MOUTH_LEFT_CORNER]
    right_corner = landmarks_68[MOUTH_RIGHT_CORNER]
    width = np.linalg.norm(right_corner - left_corner)

    # Mouth height (outer lip)
    upper_outer = np.mean([landmarks_68[i] for i in UPPER_LIP_TOP], axis=0)
    lower_outer = np.mean([landmarks_68[i] for i in LOWER_LIP_BOTTOM], axis=0)
    height = np.linalg.norm(lower_outer - upper_outer)

    if height == 0:
        return 0.0

    return float(width / height)


def compute_mouth_corner_angle(landmarks_68: np.ndarray) -> float:
    """
    Calculate the angle of mouth corners relative to center.

    Positive angle = corners raised (smiling)
    Negative angle = corners lowered (frowning)
    """
    left_corner = landmarks_68[MOUTH_LEFT_CORNER]
    right_corner = landmarks_68[MOUTH_RIGHT_CORNER]

    # Mouth center (average of top and bottom lip centers)
    mouth_center_y = (landmarks_68[LIP_CENTER_TOP][1] + landmarks_68[LIP_CENTER_BOTTOM][1]) / 2

    # Average corner height relative to center
    avg_corner_y = (left_corner[1] + right_corner[1]) / 2

    # Horizontal distance for angle calculation
    horizontal_span = right_corner[0] - left_corner[0]

    if horizontal_span == 0:
        return 0.0

    # Negative because Y increases downward in images
    # So if corners are higher (smaller Y), we want positive angle
    angle = math.degrees(math.atan2(mouth_center_y - avg_corner_y, horizontal_span / 2))

    return float(angle)


def classify_expression(landmarks_68: np.ndarray) -> tuple[str, dict]:
    """
    Classify facial expression based on mouth landmarks.

    Returns:
        tuple: (expression_label, scores_dict)
    """
    mar = compute_mouth_aspect_ratio(landmarks_68)
    smile_ratio = compute_smile_ratio(landmarks_68)
    corner_angle = compute_mouth_corner_angle(landmarks_68)

    scores = {
        "mar": round(mar, 3),
        "smile_ratio": round(smile_ratio, 3),
        "corner_angle": round(corner_angle, 2),
    }

    thresholds = EXPRESSION_THRESHOLDS

    # Classification logic
    is_smiling = (smile_ratio > thresholds["smile_ratio"] or
                  corner_angle > thresholds["smile_corner_angle"])

    if mar < thresholds["mar_closed"]:
        if is_smiling:
            expression = "smiling"  # Closed-mouth smile
        else:
            expression = "mouth_closed"
    elif mar > thresholds["mar_open"]:
        if is_smiling:
            expression = "smiling"  # Open-mouth smile/laugh
        else:
            expression = "mouth_open"
    else:
        # Medium MAR - check for smile
        if is_smiling:
            expression = "smiling"
        else:
            expression = "neutral"

    return expression, scores


def analyze_frame(frame_path: Path) -> list[FaceInfo]:
    """
    Analyze a single frame for faces, including expression detection.

    All detected faces are extracted for clustering. The clustering step
    will group faces by person automatically.
    """

    faces_info = []

    try:
        app = get_face_app()
        frame = cv2.imread(str(frame_path))

        if frame is None:
            logger.warning(f"Could not read frame: {frame_path}")
            return []

        # Use InsightFace
        faces = app.get(frame)

        if not faces:
            return []

        for face in faces:
            confidence = float(face.det_score)

            if confidence < FACE_CONFIDENCE_THRESHOLD:
                continue

            # Compute quality
            quality = compute_face_quality(face, frame)

            # Get pose if available
            pose = face.pose.tolist() if hasattr(face, 'pose') and face.pose is not None else [0, 0, 0]

            # Detect expression using landmarks
            expression = "neutral"
            expression_scores = {}

            if EXPRESSION_DETECTION.get("enabled", True):
                landmarks = None

                # Try to get 3D 68-point landmarks (preferred - has x, y, z)
                if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
                    landmarks = face.landmark_3d_68[:, :2]  # Use only x, y coordinates
                # Fallback to 2D 106-point landmarks if available
                elif hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                    # For 106-point model, we need to map to equivalent mouth points
                    # Points 84-95 are outer lip, 96-103 are inner lip in 106-point model
                    # This is approximate - 68-point is more reliable
                    logger.debug("Using 106-point landmarks (less accurate for expression)")
                    pass  # Skip for now, 3d_68 should be available with buffalo_l

                if landmarks is not None and len(landmarks) >= 68:
                    try:
                        expression, expression_scores = classify_expression(landmarks)
                    except Exception as e:
                        logger.debug(f"Expression detection failed: {e}")

            # Get embedding for clustering (512D vector)
            embedding_list = None
            if hasattr(face, 'embedding') and face.embedding is not None:
                embedding_list = face.embedding.tolist()

            face_info = FaceInfo(
                frame_path=str(frame_path),
                bbox=face.bbox.tolist(),
                confidence=confidence,
                quality_score=quality,
                pose=pose,
                expression=expression,
                expression_scores=expression_scores,
                embedding=embedding_list
                # similarity and is_match use defaults (1.0, True) - deprecated
            )
            faces_info.append(face_info)

    except Exception as e:
        logger.warning(f"Error analyzing frame {frame_path.name}: {e}")

    return faces_info


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def _validate_cache(data: dict, frame_paths: list[Path]) -> tuple[bool, str]:
    """
    Validate cached face extraction results.

    Args:
        data: Cached data from faces.json
        frame_paths: Current list of frame paths to process

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    all_faces = data.get('all_faces', [])

    # Check if all referenced frames still exist
    missing_frames = []
    for face in all_faces:
        frame_path = face.get('frame_path', '')
        if frame_path and not Path(frame_path).exists():
            missing_frames.append(frame_path)

    if missing_frames:
        return False, f"Cache invalid: {len(missing_frames)} frame files missing"

    # Check if frames match current input (detect new/removed frames)
    cached_frame_set = set(face.get('frame_path', '') for face in all_faces)
    current_frame_set = set(str(p) for p in frame_paths)

    # Frames in current but not in cache
    new_frames = current_frame_set - cached_frame_set

    if new_frames:
        return False, f"Cache incomplete: {len(new_frames)} new frames detected"

    # Check embeddings availability
    faces_without_embeddings = [
        f for f in all_faces
        if not f.get('embedding') or len(f.get('embedding', [])) != 512
    ]

    if len(faces_without_embeddings) > len(all_faces) * 0.5:
        return False, f"Cache invalid: {len(faces_without_embeddings)} faces missing embeddings"

    return True, "Cache valid"


def extract_faces(
    frame_paths: list[Path],
    output: VideoOutput,
    force_refresh: bool = False,
    incremental: bool = True
) -> Optional[FaceExtractionResult]:
    """
    Analyze all frames and extract faces with smart caching.

    Args:
        frame_paths: List of frame image paths
        output: VideoOutput instance
        force_refresh: If True, ignore cache and reprocess all frames
        incremental: If True, only process new frames not in cache

    Returns:
        FaceExtractionResult with analysis results
    """

    logger.info(f"Analyzing {len(frame_paths)} frames for faces...")

    cached_faces = []
    cached_data = None
    use_cache = False

    # Check if already processed (with validation)
    if output.faces_file.exists() and not force_refresh:
        logger.info("Checking cached face extraction results...")
        try:
            with open(output.faces_file, 'r') as f:
                cached_data = json.load(f)

            is_valid, reason = _validate_cache(cached_data, frame_paths)

            if is_valid:
                logger.success(f"Cache validated: {reason}")
                all_faces = [FaceInfo(**f) for f in cached_data['all_faces']]
                return FaceExtractionResult(
                    total_frames_analyzed=cached_data['total_frames_analyzed'],
                    frames_with_faces=cached_data.get('frames_with_faces', len(set(f.frame_path for f in all_faces))),
                    all_faces=all_faces
                )
            else:
                logger.warning(f"{reason}")
                if incremental and cached_data:
                    # Keep valid cached faces for incremental processing
                    cached_faces = [
                        FaceInfo(**f) for f in cached_data['all_faces']
                        if f.get('frame_path') and Path(f['frame_path']).exists()
                    ]
                    if cached_faces:
                        use_cache = True
                        logger.info(f"Incremental mode: keeping {len(cached_faces)} valid cached faces")

        except Exception as e:
            logger.warning(f"Could not load cached results: {e}")

    # Start with cached faces if using incremental mode
    all_faces = list(cached_faces) if use_cache else []
    frames_with_faces = set(f.frame_path for f in all_faces)
    # Note: frames_with_match kept for compatibility but all faces are valid now
    frames_with_match = set(f.frame_path for f in all_faces)

    # Determine which frames need processing
    cached_frame_paths = set(f.frame_path for f in cached_faces) if use_cache else set()
    frames_to_process = [
        p for p in frame_paths
        if str(p) not in cached_frame_paths
    ]

    if use_cache and frames_to_process:
        logger.info(f"Incremental processing: {len(frames_to_process)} new frames, {len(cached_faces)} cached")
    elif not frames_to_process and use_cache:
        logger.info("All frames already cached, using cached results")
        frames_to_process = []

    for i, frame_path in enumerate(frames_to_process):
        if (i + 1) % 10 == 0:
            logger.info(f"Processing frame {i + 1}/{len(frames_to_process)}...")

        faces = analyze_frame(frame_path)

        for face in faces:
            all_faces.append(face)
            frames_with_faces.add(face.frame_path)

    # Log expression statistics for all analyzed faces
    if EXPRESSION_DETECTION.get("enabled", True):
        expression_stats = {}
        for face in all_faces:
            expr = face.expression
            expression_stats[expr] = expression_stats.get(expr, 0) + 1
        logger.info(f"Expression distribution in all frames: {expression_stats}")

    result = FaceExtractionResult(
        total_frames_analyzed=len(frame_paths),
        frames_with_faces=len(frames_with_faces),
        all_faces=all_faces
    )

    # Save results (embeddings included for clustering)
    with open(output.faces_file, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.success(
        f"Face extraction complete: {len(frames_with_faces)} frames with faces, "
        f"{len(all_faces)} total faces detected"
    )

    return result


def process_faces(frame_paths: list[Path], output: VideoOutput) -> Optional[FaceExtractionResult]:
    """
    Full face extraction pipeline.

    Detects all faces in frames, extracts embeddings for clustering,
    and detects facial expressions.

    Args:
        frame_paths: List of frame paths to analyze
        output: VideoOutput instance

    Returns:
        FaceExtractionResult with all detected faces and embeddings
    """
    return extract_faces(frame_paths, output)

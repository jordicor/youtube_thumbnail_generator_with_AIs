"""
YouTube Thumbnail Generator - Utilities
=======================================
Common utilities, logging, and helper functions.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

# Configure stdout encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# =============================================================================
# COLORED LOGGER
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output"""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[34m',      # Blue
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'SUCCESS': '\033[32m',   # Green
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')

        # Custom format
        formatted = f"[{timestamp}] [{color}{record.levelname:^8}{reset}] {record.getMessage()}"

        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


def setup_logger(name: str, log_file: Optional[Path] = None, level: str = "INFO") -> logging.Logger:
    """Setup a colored logger with optional file output"""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    # File handler (plain text)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)

    return logger


# Add SUCCESS level
logging.SUCCESS = 25
logging.addLevelName(logging.SUCCESS, 'SUCCESS')

def success(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.SUCCESS):
        self._log(logging.SUCCESS, message, args, **kwargs)

logging.Logger.success = success


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class ProgressTracker:
    """Track progress of batch operations"""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
        self.successes = 0
        self.failures = 0

    def update(self, success: bool = True, message: str = ""):
        """Update progress counter"""
        self.current += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1

        pct = (self.current / self.total) * 100 if self.total > 0 else 0
        elapsed = datetime.now() - self.start_time

        status = f"[{self.current}/{self.total}] ({pct:.1f}%)"
        if message:
            status += f" {message}"

        print(f"\n{'='*60}")
        print(f"{status}")
        print(f"Elapsed: {elapsed} | Success: {self.successes} | Failed: {self.failures}")
        print(f"{'='*60}")

    def summary(self) -> dict:
        """Get final summary"""
        elapsed = datetime.now() - self.start_time
        return {
            "total": self.total,
            "processed": self.current,
            "successes": self.successes,
            "failures": self.failures,
            "elapsed": str(elapsed),
        }


# =============================================================================
# FILE UTILITIES
# =============================================================================

def find_videos(source_dir: Path, extensions: set, exclude_folders: list) -> list[Path]:
    """Find all video files recursively, excluding specified folders"""
    import os

    videos = []
    exclude_lower = [f.lower() for f in exclude_folders]

    if not source_dir.exists():
        return videos

    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)

        # Check if any parent folder should be excluded
        path_parts = [p.lower() for p in root_path.parts]
        if any(exc in path_parts for exc in exclude_lower):
            continue

        # Filter out excluded directories from traversal
        dirs[:] = [d for d in dirs if d.lower() not in exclude_lower]

        for file in files:
            file_path = root_path / file
            if file_path.suffix.lower() in extensions:
                videos.append(file_path)

    videos.sort()
    return videos


def sanitize_filename(name: str) -> str:
    """Remove invalid characters from filename"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name.strip()


def safe_parse_fps(fps_string: str) -> float:
    """
    Safely parse frame rate string like '30/1' or '29.97'.
    Replaces eval() to prevent code injection from malicious video metadata.
    """
    if not fps_string:
        return 0.0
    try:
        if '/' in fps_string:
            parts = fps_string.split('/')
            if len(parts) == 2:
                num, den = float(parts[0]), float(parts[1])
                return num / den if den != 0 else 0.0
        return float(fps_string)
    except (ValueError, ZeroDivisionError, TypeError):
        return 0.0


def get_video_info(video_path: Path) -> dict:
    """Get basic video information using ffprobe"""
    import subprocess
    import json

    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            data = json.loads(result.stdout)

            # Extract relevant info
            format_info = data.get('format', {})
            video_stream = next(
                (s for s in data.get('streams', []) if s.get('codec_type') == 'video'),
                {}
            )

            return {
                "duration": float(format_info.get('duration', 0)),
                "size_mb": int(format_info.get('size', 0)) / (1024 * 1024),
                "width": video_stream.get('width', 0),
                "height": video_stream.get('height', 0),
                "fps": safe_parse_fps(video_stream.get('r_frame_rate', '0')),
                "codec": video_stream.get('codec_name', 'unknown'),
            }
    except Exception:
        pass

    return {"duration": 0, "size_mb": 0, "width": 0, "height": 0, "fps": 0, "codec": "unknown"}


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if needed"""
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# FRAME EXTRACTION UTILITIES
# =============================================================================

def extract_first_frame(video_path: Path, output_path: Path) -> Optional[Path]:
    """
    Extract the first frame from a video file for preview purposes.

    Args:
        video_path: Path to the video file
        output_path: Path where to save the frame (including filename)

    Returns:
        Path to the saved frame, or None if extraction failed
    """
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return None

        # Read the first frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the frame
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return output_path if output_path.exists() else None

    except Exception:
        return None


# =============================================================================
# VIDEO OUTPUT STRUCTURE
# =============================================================================

class VideoOutput:
    """Manages output structure for a single video"""

    def __init__(self, video_path: Path, output_base: Path):
        self.video_path = video_path
        self.video_name = video_path.stem

        # Create sanitized folder name
        safe_name = sanitize_filename(self.video_name)
        self.output_dir = output_base / safe_name

        # Sub-directories
        self.frames_dir = self.output_dir / "frames"
        self.clusters_dir = self.output_dir / "clusters"

        # Output files
        self.transcription_file = self.output_dir / "transcription.txt"
        self.scenes_file = self.output_dir / "scenes.json"
        self.faces_file = self.output_dir / "faces.json"
        self.prompt_file = self.output_dir / "thumbnail_prompt.txt"
        self.metadata_file = self.output_dir / "metadata.json"

    def setup(self):
        """Create all necessary directories"""
        ensure_dir(self.output_dir)
        ensure_dir(self.frames_dir)

    def save_metadata(self, data: dict):
        """Save processing metadata"""
        data['video_path'] = str(self.video_path)
        data['processed_at'] = datetime.now().isoformat()

        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_metadata(self) -> dict:
        """Load existing metadata if available"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def is_complete(self) -> bool:
        """Check if processing is already complete"""
        thumbnails_dir = self.output_dir / "thumbnails"
        if not thumbnails_dir.exists():
            return False
        return any(thumbnails_dir.glob("*_p*_v*.png"))

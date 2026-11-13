"""
YouTube Thumbnail Generator - Utilities
=======================================
Common utilities, logging, and helper functions.
"""

import os
import sys
import logging
from pathlib import Path, PurePosixPath, PureWindowsPath
from datetime import datetime
from typing import Optional, Sequence
import orjson

import config

# Configure Windows streams in place without taking ownership of their buffers.
if sys.platform == 'win32':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')


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
    """Remove invalid characters from a file or folder name.

    A name left empty or made only of dots ("..") would point at the folder
    itself or its parent: it raises ValueError instead.
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    name = name.strip()
    if not name.strip('.'):
        raise ValueError(f"Not a usable file or folder name: {name!r}")
    return name


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
            data = orjson.loads(result.stdout)

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
# IMAGE FILES
# =============================================================================
# cv2.imread and cv2.imwrite cannot open a non-ASCII path on Windows, and an
# accent in a video's name reaches every path under its output folder: imwrite
# writes under a mangled name, or nowhere, and still returns True. These read
# and write the file with Python instead and let OpenCV only decode and encode.

def read_image(path: Path):
    """Read an image file as a BGR array. Raises if it is missing or not an image."""
    import cv2
    import numpy as np

    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Not a readable image: {path}")
    return image


def write_image(path: Path, image, params: Sequence[int] = ()) -> None:
    """Write a BGR array in the format of the path's suffix. Raises if it cannot."""
    import cv2

    ok, encoded = cv2.imencode(path.suffix, image, list(params))
    if not ok:
        raise ValueError(f"Could not encode an image as {path.suffix}: {path}")
    path.write_bytes(encoded.tobytes())


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
        write_image(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return output_path

    except Exception:
        return None


# =============================================================================
# OUTPUT PATHS
# =============================================================================
# Files under OUTPUT_DIR are persisted (database and JSON caches) relative to
# OUTPUT_DIR, so the project and its output folder can move freely. In memory
# and in API responses they are absolute paths.

def to_output_relative(path) -> str:
    """Return the stored form of an absolute path inside OUTPUT_DIR (POSIX, relative)."""
    if not Path(path).is_absolute():
        raise ValueError(f"Expected an absolute path inside OUTPUT_DIR: {path}")
    output_dir = Path(os.path.abspath(config.OUTPUT_DIR))
    try:
        return Path(os.path.abspath(path)).relative_to(output_dir).as_posix()
    except ValueError:
        raise ValueError(f"Path is outside OUTPUT_DIR ({output_dir}): {path}") from None


def resolve_output_path(stored: str) -> Path:
    """Return the absolute path of a value stored by to_output_relative()."""
    if not stored or '\\' in stored:
        raise ValueError(f"Invalid stored output path: {stored!r}")
    relative = PurePosixPath(stored)
    if relative.is_absolute() or PureWindowsPath(stored).drive or '..' in relative.parts:
        raise ValueError(f"Stored output path must be relative to OUTPUT_DIR: {stored!r}")
    return Path(config.OUTPUT_DIR).joinpath(*relative.parts)


def resolve_row_paths(row: dict, *keys: str) -> dict:
    """Replace stored output paths in a database row with absolute path strings."""
    for key in keys:
        if row.get(key):
            row[key] = str(resolve_output_path(row[key]))
    return row


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

        with open(self.metadata_file, 'wb') as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def load_metadata(self) -> dict:
        """Load existing metadata if available"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'rb') as f:
                return orjson.loads(f.read())
        return {}

    def is_complete(self) -> bool:
        """Require a successful run's metadata and all its thumbnail files."""
        metadata = self.load_metadata()
        paths = metadata.get("thumbnail_paths", [])
        return bool(paths) and len(paths) == metadata.get("total_thumbnails") and all(
            Path(path).is_file() for path in paths
        )


class ReferenceImageError(ValueError):
    """A required reference cannot be sent faithfully to an image provider."""


def reference_png(data: bytes, label: str) -> bytes:
    """Decode a reference fully and keep its native resolution in lossless PNG."""
    import io
    from PIL import Image, ImageOps

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.format == 'PNG' and image.mode in ('RGB', 'RGBA') and image.getexif().get(274, 1) == 1:
                return data
            oriented = ImageOps.exif_transpose(image)
            mode = 'RGBA' if 'A' in oriented.getbands() or 'transparency' in oriented.info else 'RGB'
            with io.BytesIO() as buffer:
                oriented.convert(mode).save(buffer, format='PNG')
                return buffer.getvalue()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ReferenceImageError(f"Cannot decode reference {label}: {exc}") from exc


def load_reference_images(paths: Sequence[Path], max_count: Optional[int] = None,
                          max_bytes: Optional[int] = None) -> list[bytes]:
    """Read every reference in order; never omit, resize, or substitute one."""
    if max_count is not None and len(paths) > max_count:
        raise ReferenceImageError(f"This model accepts {max_count} references, but {len(paths)} were provided")
    images = []
    for path in paths:
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            raise ReferenceImageError(f"Cannot read reference {path}: {exc}") from exc
        png = reference_png(data, str(path))
        if max_bytes is not None and len(png) >= max_bytes:
            raise ReferenceImageError(f"Reference {path} exceeds the provider's {max_bytes} byte limit")
        images.append(png)
    return images

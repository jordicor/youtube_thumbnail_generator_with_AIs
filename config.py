"""
YouTube Thumbnail Generator - Configuration
============================================
Central configuration for all pipeline modules.

Configuration is loaded from environment variables, which can be set in a .env file.
See .env.example for a template.
"""

from pathlib import Path
import os

# Load environment variables from .env file
from dotenv import load_dotenv

# Find the project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Load .env file. When running from a git worktree
# (<main_repo>/.claude/worktrees/<branch>/), the .env normally lives in the
# main repo, not in the worktree — fall back to the main repo's .env so API
# keys and Gran Sabio config keep working across worktrees.
env_path = PROJECT_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    _parts = PROJECT_DIR.parts
    if ".claude" in _parts:
        _idx = _parts.index(".claude")
        _main_repo = Path(*_parts[:_idx])
        _fallback_env = _main_repo / ".env"
        if _fallback_env.exists():
            load_dotenv(_fallback_env)

# =============================================================================
# DIRECTORIES
# =============================================================================


def _project_path(value: str) -> Path:
    """Resolve a configured path; relative values are relative to the project."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


# Input/Output directories
VIDEOS_DIR = _project_path(os.getenv("VIDEOS_DIR", "videos"))
OUTPUT_DIR = _project_path(os.getenv("OUTPUT_DIR", "output"))

# DEPRECATED: reference_face.jpg is no longer needed.
# The system now uses automatic clustering to group faces by person.
# Users select which person/character to use from the detected clusters.
# REFERENCE_FACE_PATH = PROJECT_DIR / "reference_face.jpg"  # REMOVED

# Native ComfyUI (no paid fallback). The UI can override this connection.
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_WORKFLOW = os.getenv("COMFYUI_WORKFLOW", "qwen-image-2.1")

# Database path
DATABASE_PATH = _project_path(os.getenv("DATABASE_PATH", "database/thumbnails.db"))

# =============================================================================
# API KEYS (loaded from .env - no defaults for security)
# =============================================================================

# ElevenLabs API for transcription
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# Google Gemini API for image generation
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Anthropic Claude API for prompt generation
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# OpenAI API for prompt generation / image generation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Poe API for FLUX/Ideogram models
POE_API_KEY = os.getenv("POE_API_KEY", "")

# =============================================================================
# GRANSABIO LLM SETTINGS
# =============================================================================

# Gran Sabio LLM server URL for advanced prompt generation
# See: https://github.com/jordicor/GranSabio_LLM
GRANSABIO_LLM_URL = os.getenv("GRANSABIO_LLM_URL", "http://localhost:8000")

# Path to Gran Sabio LLM client directory (required for web interface)
# Should point to the 'client' folder inside your Gran Sabio LLM installation
GRANSABIO_CLIENT_PATH = os.getenv("GRANSABIO_CLIENT_PATH", "")

# Optional administrator-registered endpoint identities; URLs stay in Gran Sabio.
GRANSABIO_LOCAL_ENDPOINT_IDS = {
    provider: os.getenv(f"GRANSABIO_{provider.upper()}_ENDPOINT_ID", "").strip()
    for provider in ("ollama", "llamacpp", "openai_compatible")
}

# Gran Sabio Vision Settings (for sending reference images to LLM)
GRANSABIO_USERNAME = os.getenv("GRANSABIO_USERNAME", "thumbnail_generator")
GRANSABIO_IMAGE_DETAIL = os.getenv("GRANSABIO_IMAGE_DETAIL", "auto")  # low, high, auto

# =============================================================================
# VIDEO PROCESSING SETTINGS
# =============================================================================

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v', '.wmv', '.flv', '.mpg', '.mpeg'}
EXCLUDE_FOLDERS = ["!Subidos", "uploaded", "done"]

# =============================================================================
# SCENE DETECTION SETTINGS
# =============================================================================

SCENE_DETECTOR = "adaptive"
MIN_SCENE_LENGTH = 3.0  # Minimum 3 seconds - shorter scenes not relevant for thumbnails
SCENE_THRESHOLD = 2.0   # Lower threshold to detect subtle transitions (collaborations, cameos)

# =============================================================================
# FACE DETECTION SETTINGS
# =============================================================================

FACE_DETECTOR_MODEL = os.getenv("FACE_DETECTOR_MODEL", "buffalo_l")
FACE_CONFIDENCE_THRESHOLD = float(os.getenv("FACE_CONFIDENCE_THRESHOLD", "0.5"))
# DEPRECATED: FACE_SIMILARITY_THRESHOLD no longer used - clustering replaces reference matching
# FACE_SIMILARITY_THRESHOLD = float(os.getenv("FACE_SIMILARITY_THRESHOLD", "0.4"))
MAX_BEST_FRAMES = int(os.getenv("MAX_BEST_FRAMES", "10"))

# =============================================================================
# FRAME EXTRACTION SCALING
# =============================================================================

# Minimum frames per scene (guaranteed)
MIN_FRAMES_PER_SCENE = 10

# Maximum frames per scene (prevent saturation)
MAX_FRAMES_PER_SCENE = 30

# Maximum total frames per video (0 = unlimited)
MAX_TOTAL_FRAMES = 3000

# Base frames according to total video duration (threshold_seconds: base_frames)
FRAME_SCALING_BY_VIDEO_DURATION = {
    180: 10,    # < 3 min  -> 10 base
    600: 12,    # < 10 min -> 12 base
    1200: 15,   # < 20 min -> 15 base
    2400: 18,   # < 40 min -> 18 base
    99999: 20,  # > 40 min -> 20 base
}

# Bonus frames according to scene duration (threshold_seconds: bonus_frames)
FRAME_SCALING_BY_SCENE_DURATION = {
    30: 0,      # < 30 seg -> +0
    60: 2,      # < 1 min  -> +2
    180: 5,     # < 3 min  -> +5
    300: 8,     # < 5 min  -> +8
    99999: 10,  # > 5 min  -> +10
}

FACE_QUALITY_FILTERS = {
    "min_face_size": 100,
    "max_pose_angle": 30,
    "min_brightness": 40,
    "max_blur_threshold": 100,
}

# =============================================================================
# EXPRESSION DETECTION SETTINGS
# =============================================================================

EXPRESSION_DETECTION = {
    "enabled": True,
    "use_landmark_model": "3d_68",  # "3d_68" or "2d_106"
}

# Distribution of frames by expression type (should sum to MAX_BEST_FRAMES)
EXPRESSION_DISTRIBUTION = {
    "mouth_closed": 3,    # Frames with closed mouth (good for serious thumbnails)
    "smiling": 3,         # Frames with smile (good for friendly thumbnails)
    "neutral": 2,         # Neutral expression
    "random": 2,          # Random high-quality frames for variety
}

# Thresholds for expression detection
EXPRESSION_THRESHOLDS = {
    "mar_closed": 0.12,       # MAR below this = mouth closed
    "mar_open": 0.30,         # MAR above this = mouth open
    "smile_ratio": 2.5,       # Width/Height ratio above this = smiling
    "smile_corner_angle": 12, # Mouth corner angle for smile detection (degrees)
}

# =============================================================================
# TRANSCRIPTION SETTINGS
# =============================================================================

USE_LOCAL_WHISPER = os.getenv("USE_LOCAL_WHISPER", "false").lower() == "true"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "turbo")
TRANSCRIPTION_LANGUAGE = os.getenv("TRANSCRIPTION_LANGUAGE", "es")
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/speech-to-text"

# ElevenLabs Scribe model: scribe_v1, scribe_v2
# scribe_v2 offers keyterm prompting, entity detection, and improved accuracy
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "scribe_v2")

# Keyterms to bias transcription (comma-separated, max 100 terms, each <50 chars)
# Useful for channel names, product names, technical jargon, proper nouns
# Example: "MiCanal,ProductoX,TérminoTécnico"
ELEVENLABS_KEYTERMS = os.getenv("ELEVENLABS_KEYTERMS", "")

# Entity detection: detect entities with exact timestamps (additional cost)
# Options: "" (disabled), "all", "pii", "phi", "pci", or array like "pii,pci"
ELEVENLABS_ENTITY_DETECTION = os.getenv("ELEVENLABS_ENTITY_DETECTION", "")

# =============================================================================
# THUMBNAIL GENERATION SETTINGS
# =============================================================================

THUMBNAIL_WIDTH = 1280
THUMBNAIL_HEIGHT = 720
NUM_PROMPT_VARIATIONS = 5
NUM_IMAGE_VARIATIONS = 1

# Keep original (raw) thumbnails before resizing
# When True, saves the original API output to thumbnails/raw/ folder with _raw_ prefix
# Useful for preserving high-resolution originals (e.g., 2K/4K from Gemini 3 Pro)
KEEP_RAW_THUMBNAILS = os.getenv("KEEP_RAW_THUMBNAILS", "true").lower() == "true"

# Image provider: "gemini", "openai", "replicate", "poe", "comfyui"
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "gemini")

# Gemini settings
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-pro-image")

# Output resolution of the Gemini image models, for 16:9:
# "1K" (1376x768), "2K" (2752x1536) or "4K" (5504x3072)
GEMINI_PRO_OUTPUT_RESOLUTION = os.getenv("GEMINI_PRO_OUTPUT_RESOLUTION", "2K")

# OpenAI GPT Image settings
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2.5-sunburst")
# gpt-image-2 and later accept any WIDTHxHEIGHT in multiples of 16: an exact 16:9
OPENAI_IMAGE_SIZE = "1536x864"
OPENAI_IMAGE_FORMAT = "png"         # "png", "jpeg", "webp"
OPENAI_IMAGE_QUALITY = "high"       # "low", "medium", "high"; gpt-image-2.5 adds "xhigh", "max"

# Poe API Image settings (FLUX models support reference images)
# Bot names verified on poe.com 2025-12-28: flux2pro, flux2flex, fluxkontextpro, seedream40, Ideogram-v3.
# Nano Banana bots as listed by api.poe.com/v1/models on 2026-10-24.
POE_IMAGE_MODEL = os.getenv("POE_IMAGE_MODEL", "flux2pro")
POE_IMAGE_QUALITY = "high"
POE_IMAGE_ASPECT = "16:9"

# =============================================================================
# MODEL CATALOGS (for CLI and Web interface)
# =============================================================================

# Gemini models for image generation (checked against the Gemini API on 2026-10-24)
GEMINI_MODELS = {
    "gemini-3-pro-image": "Best quality, 4K (Nano Banana Pro)",
    "gemini-nano-banana-2.1": "Fast, 4K, cheaper (Nano Banana 2.1, 2026-10-06)",
}

# OpenAI models for image generation (checked against the OpenAI API on 2026-10-24)
OPENAI_MODELS = {
    "gpt-image-2.5-sunburst": "Most capable, precise editing (2026-09-08)",
    "gpt-image-2.5-flare": "Fast, high quality (2026-09-08)",
    "gpt-image-2": "Previous generation (2026-04-21)",
}

# Poe models for image generation (CORRECT bot names - no hyphens in flux models)
POE_MODELS = {
    "flux2pro": "Best quality, up to 8 references",
    "flux2flex": "High resolution (14MP)",
    "fluxkontextpro": "Best prompt following (editing model)",
    "seedream40": "Good for combining references",
    "nano-banana-pro": "Gemini 3 Pro Image via Poe - up to 14 refs",
    "nano-banana-2.1": "Nano Banana 2.1 via Poe - up to 14 refs",
    "Ideogram-v3": "Best for text/logos",
}

# =============================================================================
# GLOBAL REFERENCE LIMITS
# =============================================================================

# Automatic facial references per person. This is not a manual or model limit.
DEFAULT_FACE_REFERENCE_COUNT = 3

# Maximum reference images per POE model
# nano-banana-pro / nano-banana-2.1: 14 refs, like the Gemini models they run
# fluxkontextpro: 1 image (editing model, not generation)
# Ideogram-v3: 3 style references
POE_MODEL_MAX_REFS = {
    "nano-banana-pro": 14,
    "nano-banana-2.1": 14,
    "flux2pro": 8,
    "flux2flex": 8,
    "fluxkontextpro": 1,
    "seedream40": 8,
    "Ideogram-v3": 3,
}

# Maximum reference images per GEMINI model: up to 14 mixed references
# gemini-3-pro-image: up to 6 objects + 5 people
# gemini-nano-banana-2.1: up to 10 objects + 4 people
GEMINI_MODEL_MAX_REFS = {
    "gemini-3-pro-image": 14,
    "gemini-nano-banana-2.1": 14,
}

# Maximum reference images per OPENAI model
# gpt-image-2: up to 16 images, always processed at high fidelity (input_fidelity param not supported)
# gpt-image-2.5: OpenAI documents no limit of its own; the GPT Image limit of 16 is assumed
OPENAI_MODEL_MAX_REFS = {
    "gpt-image-2.5-sunburst": 16,
    "gpt-image-2.5-flare": 16,
    "gpt-image-2": 16,
}

# Maximum reference images per REPLICATE model
# flux-1.1-pro: 1 image via image_prompt parameter
# face-to-many: 1 face image input
REPLICATE_MODEL_MAX_REFS = {
    "flux-1.1-pro": 1,
    "face-to-many": 1,
}

# Providers that support reference images for face consistency
MODELS_WITH_REFERENCE_SUPPORT = {
    "gemini": True,
    "poe": True,
    "replicate": True,
    "openai": True,  # Up to 16 images
}

THUMBNAIL_STYLE = """
YouTube thumbnail style:
- Bold, eye-catching design
- High contrast colors
- Professional YouTuber aesthetic
- Clean, modern look
- Expressive facial expression if person is included
- Large, readable text if text is added
- 16:9 aspect ratio (1280x720)
"""

# =============================================================================
# IDENTITY PRESERVATION SETTINGS (for face consistency in thumbnails)
# =============================================================================

CHARACTER_DNA = {
    "character_id": "Creator-YT",
    "traits": "Content creator, professional YouTuber appearance",
}

IDENTITY_PRESERVATION = {
    "lock_level": "CRITICAL",
    "face_match_accuracy": 100,
    "preserve_features": [
        "interpupillary_distance",
        "nose_bridge_shape",
        "jaw_structure",
        "cheekbone_position",
        "eye_shape_and_size",
        "facial_proportions",
    ],
    "negative_constraints": [
        "do not morph nose shape",
        "do not change eye size or spacing",
        "do not alter cheekbone structure",
        "do not modify jaw line",
        "no plastic surgery look",
        "do not beautify or idealize features",
        "do not change skin tone",
        "do not alter facial hair if present",
    ],
}

# =============================================================================
# FACE SWAP THRESHOLDS (Phase 3 — adaptive matching)
# =============================================================================

# Cosine similarity bands for the face_swap post-processing decision.
#   sim >= FACE_SWAP_THRESHOLD_SWAP   → swap (high confidence match)
#   sim >= FACE_SWAP_THRESHOLD_WARN   → no swap, log a warning (medium confidence)
#   sim <  FACE_SWAP_THRESHOLD_WARN   → no swap, recorded as low-similarity skip
#
# Adaptive behaviour: when the generation job has exactly ONE cluster and the
# generated thumbnail contains exactly ONE detected face, the assignment is
# unambiguous and these thresholds are BYPASSED — the swap is always applied.
# That covers the most common case (single-creator channel) without ever
# leaving a bad face in place "to be safe".
FACE_SWAP_THRESHOLD_SWAP = float(os.getenv("FACE_SWAP_THRESHOLD_SWAP", "0.55"))
FACE_SWAP_THRESHOLD_WARN = float(os.getenv("FACE_SWAP_THRESHOLD_WARN", "0.40"))

# Face match of generated thumbnails: cosine similarity between each person and
# the face drawn for them, scored before any face swap. With buffalo_l, faces
# of the right person score about 0.55-0.85 and those of a stranger about 0.
#   sim >= FACE_MATCH_GOOD_THRESHOLD  → same person
#   sim >= FACE_MATCH_BAD_THRESHOLD   → doubtful
#   sim <  FACE_MATCH_BAD_THRESHOLD   → another person (a failure: the gallery
#                                       can show these last)
FACE_MATCH_GOOD_THRESHOLD = float(os.getenv("FACE_MATCH_GOOD_THRESHOLD", "0.45"))
FACE_MATCH_BAD_THRESHOLD = float(os.getenv("FACE_MATCH_BAD_THRESHOLD", "0.25"))

CLOTHING_PRESERVATION = {
    "enabled": True,
    "preserve_details": [
        "exact colors",
        "fabric texture",
        "logos and graphics",
        "clothing style and fit",
        "accessories",
    ],
}

REFERENCE_IMAGE_TIPS = """
For best face consistency:
- Use high-resolution, well-lit photos
- Include frontal view as primary reference
- Add profile views if available
- Neutral background preferred
- Avoid group photos or blurry images
"""

# =============================================================================
# SERVER SETTINGS (FastAPI)
# =============================================================================

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8057"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

# =============================================================================
# REDIS SETTINGS (Job Queue)
# =============================================================================

# Redis connection URL
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))

# Worker settings
WORKER_MAX_JOBS = int(os.getenv("WORKER_MAX_JOBS", "2"))  # Limited by GPU memory
JOB_TIMEOUT_ANALYSIS = int(os.getenv("JOB_TIMEOUT_ANALYSIS", "1800"))  # 30 min
JOB_TIMEOUT_GENERATION = int(os.getenv("JOB_TIMEOUT_GENERATION", "900"))  # 15 min
JOB_MAX_RETRIES = int(os.getenv("JOB_MAX_RETRIES", "3"))

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
SAVE_LOGS = True
LOG_FILE = OUTPUT_DIR / "pipeline.log"

# =============================================================================
# NETWORK SECURITY SETTINGS
# =============================================================================

# Allowed IP ranges for API access (CIDR notation)
# Default: Private network ranges (RFC 1918) + localhost
# This protects against accidental internet exposure (DMZ, port forwarding)
ALLOWED_IP_RANGES = [
    "127.0.0.0/8",      # Localhost (127.0.0.1 - 127.255.255.255)
    "10.0.0.0/8",       # Class A private (10.x.x.x)
    "172.16.0.0/12",    # Class B private (172.16.x.x - 172.31.x.x)
    "192.168.0.0/16",   # Class C private (192.168.x.x) - most home routers
    "::1/128",          # IPv6 localhost
    "fe80::/10",        # IPv6 link-local
]

# Set to False to disable IP filtering (not recommended if exposed to internet)
ENABLE_IP_FILTERING = os.getenv("ENABLE_IP_FILTERING", "true").lower() == "true"

# Browser origins (other than the app's own pages) allowed to call the API:
# CORS reads them, and only they may send requests that change data
CORS_ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8057",
    "http://127.0.0.1:8057",
]
# Pages served from private network addresses (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
CORS_ORIGIN_REGEX = r"^https?://(192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?$"

# Host names the app answers to besides IP addresses, localhost and this
# machine's own name (comma-separated, e.g. "mypc.home,thumbs.lan"). Any other
# Host is refused, so a website cannot reach the app by pointing its domain at
# a LAN address (DNS rebinding).
ALLOWED_HOSTS = [host.strip().lower() for host in os.getenv("ALLOWED_HOSTS", "").split(",") if host.strip()]

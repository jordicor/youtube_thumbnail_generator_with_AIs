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
PROJECT_DIR = Path(__file__).parent

# Load .env file if it exists
env_path = PROJECT_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

# =============================================================================
# DIRECTORIES
# =============================================================================

# Input/Output directories
VIDEOS_DIR = Path(os.getenv("VIDEOS_DIR", "./videos"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(PROJECT_DIR / "output")))

# DEPRECATED: reference_face.jpg is no longer needed.
# The system now uses automatic clustering to group faces by person.
# Users select which person/character to use from the detected clusters.
# REFERENCE_FACE_PATH = PROJECT_DIR / "reference_face.jpg"  # REMOVED

# Database path
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(PROJECT_DIR / "database" / "thumbnails.db")))

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

# =============================================================================
# PROMPT SYSTEM VERSION (JSON Structured Prompts)
# =============================================================================

# V3 prompt system: Optimized JSON structure (~1,000 chars)
# - Clear face_groups/characters/scene separation
# - physical_description for facial features
# - identify_in_references for cross-references
# - character_directions only for ACTIVE characters in scene
# - No redundancies (single source of truth for identity)
#
# V1 (legacy fallback): ~3,500 chars text with 22+ negations
# Used when V3 is disabled or when structured fields are not available
#
# Set to "false" to use V1 legacy prompts
USE_PROMPT_SYSTEM_V3 = os.getenv("USE_PROMPT_SYSTEM_V3", "true").lower() == "true"

# Image provider: "gemini", "openai", "replicate", "poe"
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "gemini")

# Gemini settings
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")

# Gemini 3 Pro output resolution (only for gemini-3-pro-image-preview)
# Options: "1K" (1376x768), "2K" (2752x1536), "4K" (5504x3072) for 16:9
# Note: gemini-2.5-flash-image does NOT support this parameter
GEMINI_PRO_OUTPUT_RESOLUTION = os.getenv("GEMINI_PRO_OUTPUT_RESOLUTION", "2K")

# OpenAI GPT Image settings
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_IMAGE_SIZE = "1536x1024"     # "1024x1024", "1536x1024" (landscape), "1024x1536" (portrait)
OPENAI_IMAGE_FORMAT = "png"         # "png", "jpeg", "webp"
OPENAI_IMAGE_QUALITY = "high"       # "high", "medium", "low"

# Poe API Image settings (FLUX models support reference images)
# Bot names verified on poe.com 2025-12-28: flux2pro, flux2flex, fluxkontextpro, seedream40, nanobananapro, Ideogram-v3
POE_IMAGE_MODEL = os.getenv("POE_IMAGE_MODEL", "flux2pro")
POE_IMAGE_QUALITY = "high"
POE_IMAGE_ASPECT = "16:9"

# =============================================================================
# MODEL CATALOGS (for CLI and Web interface)
# =============================================================================

# Gemini models for image generation
GEMINI_MODELS = {
    "gemini-2.5-flash-image": "Fast, good quality",
    "gemini-3-pro-image-preview": "Best quality, 4K (Nano Banana Pro)",
}

# OpenAI models for image generation
OPENAI_MODELS = {
    "gpt-image-1.5": "Best quality",
    "gpt-image-1": "Standard",
    "gpt-image-1-mini": "Faster, lower cost",
    "dall-e-3": "Legacy (deprecated)",
}

# Poe models for image generation (CORRECT bot names - no hyphens in flux models)
POE_MODELS = {
    "flux2pro": "Best quality, up to 8 references",
    "flux2flex": "High resolution (14MP)",
    "fluxkontextpro": "Best prompt following (editing model)",
    "seedream40": "Good for combining references",
    "nanobananapro": "Gemini 3 Pro via Poe - up to 14 refs, 5 faces",
    "Ideogram-v3": "Best for text/logos",
}

# =============================================================================
# GLOBAL REFERENCE LIMITS
# =============================================================================

# Maximum reference images the UI allows selecting (highest among all models)
# Change this value to adjust the limit across the entire application
MAX_REFERENCE_FRAMES = 14

# Maximum reference images per POE model
# nanobananapro: 14 refs, 5 simultaneous faces, 2K native
# fluxkontextpro: 1 image (editing model, not generation)
# Ideogram-v3: 3 style references
POE_MODEL_MAX_REFS = {
    "nanobananapro": 14,
    "flux2pro": 8,
    "flux2flex": 8,
    "fluxkontextpro": 1,
    "seedream40": 8,
    "Ideogram-v3": 3,
}

# Maximum reference images per GEMINI model
# gemini-3-pro-image-preview: 14 refs (up to 6 objects + 5 humans)
# gemini-2.5-flash-image: 3 refs (recommended for best results per Google docs)
GEMINI_MODEL_MAX_REFS = {
    "gemini-3-pro-image-preview": 14,
    "gemini-2.5-flash-image": 3,
}

# Maximum reference images per OPENAI model
# gpt-image-1.5/1/mini: up to 16 images for editing/reference
# dall-e-3: does not support reference images
OPENAI_MODEL_MAX_REFS = {
    "gpt-image-1.5": 16,
    "gpt-image-1": 16,
    "gpt-image-1-mini": 16,
    "dall-e-3": 0,
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
    "openai": True,  # Supports up to 16 images (except dall-e-3)
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
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
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

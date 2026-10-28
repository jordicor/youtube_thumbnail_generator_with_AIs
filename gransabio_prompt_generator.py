"""
Gran Sabio LLM Prompt Generator
===============================

Generates thumbnail prompts using Gran Sabio LLM Engine for enhanced
AI provider selection, thinking mode support, and prompt history.
"""

import sys
import orjson
import time
import base64
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Literal
from dataclasses import dataclass

from ai_json_cleanroom import validate_ai_json

from config import (
    GRANSABIO_LLM_URL,
    GRANSABIO_CLIENT_PATH,
    GRANSABIO_LOCAL_ENDPOINT_IDS,
    GRANSABIO_USERNAME,
    GRANSABIO_IMAGE_DETAIL,
)

logger = logging.getLogger(__name__)

TextProvider = Literal["anthropic", "openai", "google", "xai", "openrouter", "ollama", "llamacpp", "openai_compatible"]
LOCAL_TEXT_PROVIDERS = frozenset({"ollama", "llamacpp", "openai_compatible"})
CATALOG_TEXT_PROVIDERS = LOCAL_TEXT_PROVIDERS | {"openrouter"}
TEXT_PROVIDER_NAMES = {
    "anthropic": "Anthropic", "openai": "OpenAI", "google": "Google (Gemini)", "xai": "xAI (Grok)", "openrouter": "OpenRouter",
    "ollama": "Ollama (local)", "llamacpp": "llama.cpp (local)",
    "openai_compatible": "OpenAI-compatible (local)",
}

# Add Gran Sabio LLM client to path if configured
# The client is a standalone HTTP client that only requires 'requests' and 'aiohttp'
# Get it from: https://github.com/jordicor/GranSabio_LLM (the 'client' folder)
if GRANSABIO_CLIENT_PATH:
    client_path = Path(GRANSABIO_CLIENT_PATH)
    if client_path.exists():
        # Add parent so 'from client import' works. Appended: that folder holds other
        # packages (e.g. its own "services") that must not shadow this project's.
        sys.path.append(str(client_path.parent))
    else:
        logger.warning(f"Gran Sabio client path not found: {GRANSABIO_CLIENT_PATH}")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PromptGenerationConfig:
    """Configuration for prompt generation via Gran Sabio LLM."""
    provider: TextProvider = "anthropic"
    model: Optional[str] = None  # None = use default for provider
    thinking_enabled: bool = False
    thinking_level: str = "high"  # low, medium, high (sent as reasoning_effort)
    custom_instructions: Optional[str] = None
    include_history: bool = False
    history_prompts: Optional[List[str]] = None
    # Expression every person must show (a FACIAL_EXPRESSIONS key); None lets each concept choose
    facial_expression: Optional[str] = None
    # Reuse the catalog entry checked before transcription for the whole batch.
    model_info: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "thinking_enabled": self.thinking_enabled,
            "thinking_level": self.thinking_level,
            "custom_instructions": self.custom_instructions,
            "include_history": self.include_history,
            "facial_expression": self.facial_expression,
        }


# Default models per provider
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-5.2",
    "google": "gemini-3-flash-preview",
    "xai": "grok-4-1-fast-non-reasoning",
}

# Maximum history characters to include
MAX_HISTORY_CHARS = 32000

# Output tokens asked for the app's JSON answers. The largest (a thumbnail
# skeleton, a concept, a set of titles) is about 3,000 characters, ~1,000
# tokens; reasoning models spend their default reasoning from the same budget,
# so it leaves room for that too. Without max_tokens Gran Sabio reserves 95% of
# the model's max output, and a model with a small window (grok-4.7 is inferred
# at 131,072) is left too little for the prompt and its photos.
ANSWER_MAX_TOKENS = 16000
# Extra output tokens when thinking is asked for, per level: OpenAI, xAI and
# effort-based Claude and Gemini models reason inside max_tokens. Gran Sabio
# adds the thinking budget on top by itself for budget-based Claude and Gemini.
THINKING_EXTRA_TOKENS = {"low": 4000, "medium": 8000, "high": 16000}


def answer_budget(thinking_enabled: bool, thinking_level: str) -> dict:
    """Gran Sabio arguments for the size and thinking of a JSON answer.

    Thinking is asked as reasoning_effort, which Gran Sabio turns into each
    provider's own setting (a thinking budget, a level or an effort). Generator
    tools are off: these answers need none, and some models reject tools
    together with reasoning (gpt-6.1-terra over chat completions).
    """
    budget = {"max_tokens": ANSWER_MAX_TOKENS, "generation_tools_mode": "never"}
    if thinking_enabled:
        budget["max_tokens"] += THINKING_EXTRA_TOKENS[thinking_level]
        budget["reasoning_effort"] = thinking_level
    return budget


# =============================================================================
# GRANSABIO CLIENT
# =============================================================================

class GranSabioError(RuntimeError):
    """Gran Sabio LLM gave no usable answer; the message says why."""


def resolve_text_model(provider: str, model: Optional[str]) -> str:
    """Resolve direct-provider defaults; local and OpenRouter models are explicit."""
    if provider not in TEXT_PROVIDER_NAMES:
        raise ValueError(f"Unknown text provider: {provider}")
    if model and model.strip():
        return model.strip()
    if provider in LOCAL_TEXT_PROVIDERS:
        raise ValueError("Choose an explicit local text model from Gran Sabio's catalog")
    if provider == "openrouter":
        raise ValueError("Choose an explicit OpenRouter model from Gran Sabio's catalog")
    return DEFAULT_MODELS[provider]


def _catalog_model(catalog: dict, provider: str, model: Optional[str], *, vision: bool = False) -> dict:
    key = resolve_text_model(provider, model)
    entry = next((item for item in catalog.get(provider, [])
                  if item.get("key", item.get("model_id")) == key), None)
    if entry is None:
        raise GranSabioError(f"Model {provider}/{key} is not in Gran Sabio's catalog")
    capabilities = entry.get("capabilities") or []
    if provider in CATALOG_TEXT_PROVIDERS and "text" not in capabilities:
        raise GranSabioError(f"Model {provider}/{key} does not support text generation")
    if vision and "vision" not in capabilities:
        from i18n import t
        raise GranSabioError(t('api.errors.ideas_model_no_vision', model=entry.get("name") or key))
    return entry


async def require_text_model(provider: str, model: Optional[str]) -> Optional[dict]:
    """Validate explicit catalog selection before transcription and reuse its metadata."""
    if provider in CATALOG_TEXT_PROVIDERS:
        return _catalog_model(await _model_catalog(), provider, model)
    return None


def text_generation_options(provider: str, model: Optional[str], thinking_enabled: bool,
                            thinking_level: str, model_info: Optional[dict] = None,
                            *, vision: bool = False) -> dict:
    """Keep explicit catalog selections for generation, JSON repair, and preflight."""
    selected = resolve_text_model(provider, model)
    options = {"generator_model": selected, **answer_budget(thinking_enabled, thinking_level)}
    if provider not in CATALOG_TEXT_PROVIDERS:
        return options

    if model_info is None:
        try:
            catalog = get_gransabio_client().get_models()
        except Exception as error:
            raise GranSabioError(f"Could not load Gran Sabio's model catalog: {error}") from error
    else:
        catalog = {provider: [model_info]}
    entry = _catalog_model(catalog, provider, selected, vision=vision)
    output_tokens = entry.get("output_tokens")
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens <= 0:
        raise GranSabioError(f"Model {provider}/{selected} has no valid output token limit")
    reasoning = "reasoning" in (entry.get("capabilities") or [])
    if thinking_enabled and not reasoning:
        raise GranSabioError(f"Model {provider}/{selected} does not support thinking")
    # Qualifying the model keeps duplicate names on the explicitly chosen provider.
    selected = f"{provider}::{selected}"
    options["generator_model"] = selected
    options["max_tokens"] = min(options["max_tokens"], output_tokens)
    if reasoning:
        options["reasoning_effort"] = thinking_level if thinking_enabled else "none"
    endpoint_id = GRANSABIO_LOCAL_ENDPOINT_IDS.get(provider)
    endpoint = {"endpoint_id": endpoint_id} if endpoint_id else {}
    route = {
        "model": selected,
        **endpoint,
        "params": {"max_tokens": options["max_tokens"],
                   "reasoning_effort": options.get("reasoning_effort")},
    }
    options.update({
        "feedback_memory_enabled": False,
        "gran_sabio_fallback": False,
        "long_text_mode": "off",
        "llm_routing": {"calls": {
            "generation.main": route,
            "generation.json_retry": route,
            "preflight.validate": {
                "model": selected,
                **endpoint,
                "params": {"max_tokens": min(2048, output_tokens),
                           "reasoning_effort": "none" if reasoning else None},
            },
        }},
    })
    if provider in LOCAL_TEXT_PROVIDERS:
        options["execution_policy"] = {"scope": "local_only"}
    return options


def verify_local_execution(result: dict, provider: str) -> None:
    """An older or misconfigured server must not silently ignore local-only mode."""
    if provider in LOCAL_TEXT_PROVIDERS:
        attestation = result.get("local_only_attestation") or {}
        if attestation.get("local_only_verified") is not True:
            raise GranSabioError("Gran Sabio did not verify local-only execution")


_gransabio_client = None


def get_gransabio_client():
    """The Gran Sabio LLM client, connected on first use.

    Raises GranSabioError when the client cannot be imported or the server does
    not answer. Nothing is remembered on failure: the next call tries again.
    """
    global _gransabio_client

    if _gransabio_client is not None:
        return _gransabio_client

    try:
        from client import GranSabioClient
    except ImportError as e:
        raise GranSabioError(
            f"Gran Sabio LLM client cannot be imported (GRANSABIO_CLIENT_PATH={GRANSABIO_CLIENT_PATH!r}): {e}"
        ) from e

    client = GranSabioClient(base_url=GRANSABIO_LLM_URL)
    if not client.is_available():
        raise GranSabioError(f"Gran Sabio LLM server is not available at {GRANSABIO_LLM_URL}")
    _gransabio_client = client
    logger.info(f"Gran Sabio LLM client connected successfully to {GRANSABIO_LLM_URL}")
    return client


def detect_image_type(image_base64: str) -> Tuple[str, str]:
    """
    Detect image type from base64 data by examining magic bytes.

    Args:
        image_base64: Base64-encoded image data (without data: prefix)

    Returns:
        Tuple of (mime_type, extension) e.g. ("image/png", ".png")
    """
    try:
        # Decode first few bytes to check magic numbers
        raw_bytes = base64.b64decode(image_base64[:32])

        # PNG: 89 50 4E 47 0D 0A 1A 0A
        if raw_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return ("image/png", ".png")

        # JPEG: FF D8 FF
        if raw_bytes[:3] == b'\xff\xd8\xff':
            return ("image/jpeg", ".jpg")

        # GIF: GIF87a or GIF89a
        if raw_bytes[:6] in (b'GIF87a', b'GIF89a'):
            return ("image/gif", ".gif")

        # WebP: RIFF....WEBP
        if raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WEBP':
            return ("image/webp", ".webp")

        # BMP: BM
        if raw_bytes[:2] == b'BM':
            return ("image/bmp", ".bmp")

    except Exception:
        pass

    # Default to JPEG if detection fails
    return ("image/jpeg", ".jpg")


def upload_image_to_gransabio(
    image_base64: str,
    filename: Optional[str] = None,
    username: Optional[str] = None
) -> Optional[str]:
    """
    Upload an image to Gran Sabio LLM and return the upload_id.

    Gran Sabio requires images to be uploaded first, then referenced by upload_id
    in the generate request. This two-step process allows the server to validate
    and process images before they are used in generation.

    Args:
        image_base64: Base64-encoded image data (without data: prefix)
        filename: Optional filename (auto-detected if not provided)
        username: User namespace for the attachment

    Returns:
        upload_id string

    Raises:
        GranSabioError: the upload failed or gave no upload_id
    """
    client = get_gransabio_client()

    # Use configured username if not specified
    upload_username = username or GRANSABIO_USERNAME

    # Detect image type from content
    content_type, extension = detect_image_type(image_base64)

    # Generate filename if not provided, ensuring correct extension
    if filename:
        # Replace extension if filename was provided with wrong one
        base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
        actual_filename = f"{base_name}{extension}"
    else:
        actual_filename = f"reference_frame{extension}"

    try:
        result = client.upload_attachment_base64(
            username=upload_username,
            content_base64=image_base64,
            filename=actual_filename,
            content_type=content_type
        )
    except Exception as e:
        raise GranSabioError(f"Uploading reference image {actual_filename} to Gran Sabio failed: {e}") from e
    upload_id = result.get("upload_id")
    if not upload_id:
        raise GranSabioError(f"Gran Sabio returned no upload_id for reference image {actual_filename}")
    logger.info(f"Image uploaded to Gran Sabio ({content_type}): {upload_id[:16]}...")
    return upload_id


# Status and model list requests are answered on the API's event loop: they
# must not block it, and a server that does not answer must not keep the page
# waiting for long.
QUICK_REQUEST_TIMEOUT = 5  # seconds


def quick_gransabio_client():
    """Async Gran Sabio client for status and model list requests (use with `async with`)."""
    import aiohttp
    from client import AsyncGranSabioClient

    return AsyncGranSabioClient(
        base_url=GRANSABIO_LLM_URL,
        timeout=aiohttp.ClientTimeout(total=QUICK_REQUEST_TIMEOUT),
    )


def _unavailable_status(status: str, message: str) -> dict:
    return {
        "available": False,
        "status": status,
        "providers": [],
        "url": GRANSABIO_LLM_URL,
        "message": message,
    }


async def check_gransabio_status() -> dict:
    """
    Check Gran Sabio LLM server status.

    Returns:
        dict with status info: available, active_sessions, models, etc.
    """
    try:
        from client import GranSabioClientError
        client = quick_gransabio_client()
    except ImportError:
        return _unavailable_status("not_installed", "Gran Sabio LLM client not installed")

    try:
        async with client:
            try:
                health = await client.health_check()
            except GranSabioClientError:
                health = {}
            if health.get("status") != "healthy":
                return _unavailable_status("offline", f"Gran Sabio LLM not responding ({GRANSABIO_LLM_URL})")
            models = await client.get_models()
    except Exception as e:
        return _unavailable_status("error", f"Connection error: {str(e)}")

    return {
        "available": True,
        "status": health["status"],
        "active_sessions": health.get("active_sessions", 0),
        "providers": list(models.keys()) if models else [],
        "url": GRANSABIO_LLM_URL,
        "message": f"Gran Sabio LLM connected ({GRANSABIO_LLM_URL})"
    }


# =============================================================================
# PROMPT GENERATION
# =============================================================================

def _build_context_section(cluster_description):
    if not cluster_description or not cluster_description.strip():
        return ""
    return f"""
CONTEXT FOR REFERENCE IMAGES:
The user has provided this note about the person/scene in the reference images:

### START USER CONTEXT ###
{cluster_description.strip()}
### END USER CONTEXT ###

Use this information when interpreting the reference images.
"""


def _build_custom_section(custom_instructions):
    if not custom_instructions or not custom_instructions.strip():
        return ""
    return f"""
═══════════════════════════════════════════════════════════════════════════════
FINAL USER INSTRUCTIONS (HIGHEST PRIORITY):
═══════════════════════════════════════════════════════════════════════════════
{custom_instructions.strip()}
═══════════════════════════════════════════════════════════════════════════════
"""


def _build_history_section(history_prompts, include_history):
    if not (include_history and history_prompts):
        return ""
    history_text = "\n---\n".join(history_prompts)
    if len(history_text) > MAX_HISTORY_CHARS:
        history_text = history_text[-MAX_HISTORY_CHARS:]
        first_separator = history_text.find("\n---\n")
        if first_separator > 0:
            history_text = history_text[first_separator + 5:]
    return f"""
═══════════════════════════════════════════════════════════════════════════════
PREVIOUS THUMBNAILS FROM OTHER GENERATIONS (DO NOT REPEAT THEIR CONCEPTS):
═══════════════════════════════════════════════════════════════════════════════
{history_text}
═══════════════════════════════════════════════════════════════════════════════
"""


# Facial expressions the user can ask for, as the concept prompt words them
FACIAL_EXPRESSIONS = {
    "smiling": "smiling",
    "mouth_closed": "with the mouth closed (lips together)",
    "neutral": "with a neutral, relaxed expression",
}


def _build_expression_section(facial_expression):
    if not facial_expression:
        return ""
    if facial_expression not in FACIAL_EXPRESSIONS:
        raise ValueError(f"Unknown facial expression: {facial_expression}")
    return f"""
═══════════════════════════════════════════════════════════════════════════════
FACIAL EXPRESSION (CHOSEN BY THE USER):
═══════════════════════════════════════════════════════════════════════════════
Every person in this thumbnail must be {FACIAL_EXPRESSIONS[facial_expression]}.
Write subject_expression accordingly. Make the concepts of the batch differ in
angle, scene, mood and pose, not in the expression.
═══════════════════════════════════════════════════════════════════════════════
"""


def _build_selected_titles_section(selected_titles):
    if not selected_titles:
        return ""
    titles_list = "\n".join(f"- {t}" for t in selected_titles)
    return f"""
═══════════════════════════════════════════════════════════════════════════════
SELECTED TITLES (GUIDE YOUR IMAGES TO MATCH THESE):
═══════════════════════════════════════════════════════════════════════════════
{titles_list}
═══════════════════════════════════════════════════════════════════════════════
"""


def _build_style_reference_section(has_style_reference):
    if not has_style_reference:
        return ""
    # Use string concatenation to avoid escaping JSON braces in an f-string.
    return """
═══════════════════════════════════════════════════════════════════════════════
STYLE REFERENCE ANALYSIS (FIRST IMAGE IS A STYLE GUIDE)
═══════════════════════════════════════════════════════════════════════════════
The FIRST reference image is NOT a person to clone - it is a VISUAL STYLE GUIDE.
Analyze it thoroughly and extract these characteristics into a "style_analysis" object:

{
  "art_type": "...",
  "art_type_details": "...",
  "color_palette": ["#...", "#..."],
  "color_mood": "...",
  "visual_effects": [...],
  "rendering_style": "...",
  "composition_style": "...",
  "line_work": "...",
  "overall_aesthetic": "..."
}

The remaining reference images (2nd onwards) show REAL PEOPLE to clone.
Apply the analyzed style TO these people when generating thumbnails.
═══════════════════════════════════════════════════════════════════════════════
"""


def _build_reference_mapping_section(reference_mapping):
    if not reference_mapping:
        return ""
    lines = []
    for entry in reference_mapping:
        idx = entry.get("image_index")
        person_id = entry.get("person_id")
        is_style = entry.get("is_style_reference", False)
        source = entry.get("source_frame", "")
        if is_style:
            lines.append(f"- Image {idx}: STYLE REFERENCE (analyze for style_analysis, NOT a person to clone)")
        elif entry.get("reference_role") == "environment":
            lines.append(f"- Image {idx}: ENVIRONMENT ONLY. {entry.get('environment_instruction', 'Preserve the setting.')} Derive people from their dedicated references.")
        elif entry.get("reference_role") == "body":
            lines.append(
                f"- Image {idx}: BODY REFERENCE of {person_id}; visible coverage: {entry['body_coverage']}; "
                f"view: {entry.get('view') or 'unspecified'}; estimated posture: {entry['body_posture']}. Describe only observed proportions. "
                "Do not idealize, normalize, slim, or inflate the build. Use the PRIMARY face reference for facial details, hair, and character appearance. "
                "Do not derive clothing, wigs, hairstyles, or accessories from a body-only reference."
            )
        elif person_id and entry.get("face_sheet_views"):
            lines.append(
                f"- Image {idx}: FACE SHEET of {person_id} — a grid of crops of that one face "
                f"(left to right, top to bottom: {entry['face_sheet_views']})"
            )
        elif person_id:
            detail = f" — from frame `{source}`" if source else ""
            label = "PRIMARY face reference" if entry.get("is_primary") else "face reference"
            observed = " / ".join(filter(None, (entry.get("view"), entry.get("expression"))))
            detail += f" (observed: {observed})" if observed else ""
            lines.append(f"- Image {idx}: {label} for {person_id}{detail}")
            if entry.get("body_reference_status") == "unavailable":
                lines.append(f"  No reliable body reference for {person_id}: do not invent a precise build from the head crop.")
        else:
            lines.append(f"- Image {idx}: (unlabelled reference)")

    seen_persons = []
    for entry in reference_mapping:
        pid = entry.get("person_id")
        if pid and pid not in seen_persons:
            seen_persons.append(pid)
    persons_summary = ", ".join(seen_persons) if seen_persons else "(none)"

    return f"""
═══════════════════════════════════════════════════════════════════════════════
REFERENCE IMAGES — DETERMINISTIC ROLE MAPPING (TRUST THIS, DO NOT RE-INFER):
═══════════════════════════════════════════════════════════════════════════════
The reference editor assigns images to people using verified detections or explicit user confirmation. The mapping below is the GROUND TRUTH — do not try
to re-identify who is who by visual similarity.

{chr(10).join(lines)}

DISTINCT PEOPLE IN THIS GENERATION: {persons_summary}

Rules:
- All images labelled with the SAME person_id show the SAME biological face.
  They MUST share a single face_group with a consistent physical_description.
- DIFFERENT person_ids = DIFFERENT people, must belong to different face_groups.
- The style reference (if any) is for visual style only — never a person.
═══════════════════════════════════════════════════════════════════════════════
"""


def environment_references(reference_mapping):
    """The setting photos of a reference mapping, in image order."""
    return [entry for entry in reference_mapping or [] if entry.get("reference_role") == "environment"]


def image_numbers(entries):
    """'Image 4' or 'Images 4, 5' for the given reference mapping entries."""
    numbers = [str(entry["image_index"]) for entry in entries]
    return f"Image {numbers[0]}" if len(numbers) == 1 else "Images " + ", ".join(numbers)


def split_sides(settings):
    """The setting photos of a split screen: (left images, right images) as text."""
    return tuple(image_numbers([entry for entry in settings if entry["environment_side"] == side])
                 for side in ("left", "right"))


def _build_skeleton_setting_section(reference_mapping):
    settings = environment_references(reference_mapping)
    if not settings:
        return ""
    per_image = (
        ""
        if settings[0]["environment_mode"] == "same"
        else f' They are different places: describe each one apart, starting with its number ("Image {settings[0]["image_index"]}: ...").'
    )
    return f"""
═══════════════════════════════════════════════════════════════════════════════
SETTING (CHOSEN BY THE USER):
═══════════════════════════════════════════════════════════════════════════════
{image_numbers(settings)} show the real place where every thumbnail of this
series happens. Describe it in "setting": the kind of place, walls and surfaces,
furniture and objects, colors, materials and light, exactly as seen.{per_image}
Describe only what is visible; do not embellish it or imagine another place.
The thumbnail concepts are written from this description without the photos.
═══════════════════════════════════════════════════════════════════════════════
"""


def _build_concept_setting_section(reference_mapping):
    settings = environment_references(reference_mapping)
    if not settings:
        return ""
    mode = settings[0]["environment_mode"]
    if mode == "same":
        rule = f"{image_numbers(settings)} show ONE place: every concept happens in that place."
    elif mode == "alternatives":
        rule = (
            f"{image_numbers(settings)} are ALTERNATIVE places: each concept happens in exactly ONE of them, "
            'never a mix. Start background with the image it uses ("Image N: ..."); concepts may use different ones.'
        )
    else:
        left, right = split_sides(settings)
        rule = (
            f"SPLIT SCREEN: the left panel happens in the place of {left} and the right panel in the place of "
            f"{right}. Every concept keeps that split; say in scene who is in each panel."
        )
    return f"""
═══════════════════════════════════════════════════════════════════════════════
SETTING (CHOSEN BY THE USER — DO NOT INVENT ANOTHER PLACE):
═══════════════════════════════════════════════════════════════════════════════
The user attached photos of the real setting; the skeleton's "setting" describes
them, and the image model gets those photos to keep that place as it is.
{rule}
- scene and background put the people in that setting. Do not move them to
  another place and do not add another location, landscape, sky, room or backdrop.
- background: say how the setting is shown in this concept: which part of it,
  camera distance and angle, depth of field, lighting and color grading. Only
  the framing and the lighting change between concepts, never the place.
- visual_elements: only what the people hold or graphic overlays (arrows,
  icons, shapes); no furniture or scenery that changes the place.
═══════════════════════════════════════════════════════════════════════════════
"""


_LANGUAGE_SECTION = """
═══════════════════════════════════════════════════════════════════════════════
LANGUAGE REQUIREMENT:
═══════════════════════════════════════════════════════════════════════════════
All user-facing text fields (text_in_image, suggested_title,
concept_name, key_topics, mood) MUST be written in the SAME LANGUAGE as the
transcription. Detect the language from the transcription above and use it
consistently. Only use a different language if the user explicitly requests
it in their instructions.
═══════════════════════════════════════════════════════════════════════════════
"""


# =============================================================================
# SKELETON PROMPT (Stage 1 — identity + base, generated once per batch)
# =============================================================================


def build_skeleton_prompt(
    transcription,
    video_title,
    cluster_description=None,
    custom_instructions=None,
    thumbnail_style="",
    has_reference_image=False,
    has_style_reference=False,
    reference_mapping=None,
    num_titles_in_pool=10,
):
    """Build a single LLM prompt that produces only the identity/style/base
    block shared by all thumbnails in a series. Output is meant to be
    cacheable and reused across many concept calls."""
    context_section = _build_context_section(cluster_description)
    custom_section = _build_custom_section(custom_instructions)
    style_reference_section = _build_style_reference_section(has_style_reference)
    reference_mapping_section = _build_reference_mapping_section(reference_mapping)
    setting_section = _build_skeleton_setting_section(reference_mapping)
    setting_field = '\n  "setting": "...",' if setting_section else ""
    setting_rule = "\n- setting: REQUIRED, the description of the attached setting photos." if setting_section else ""

    subject_analysis_section = ""
    if has_reference_image:
        people_image_note = (
            "\nNOTE: The FIRST reference image is a style guide (already analyzed above).\n"
            "The REMAINING reference images (2nd onwards) contain the REAL PEOPLE to clone.\n"
            if has_style_reference else ""
        )
        subject_analysis_section = (
            "\n═══════════════════════════════════════════════════════════════════════════════\n"
            "IDENTITY MAPPING (FACE GROUPS + CHARACTERS)\n"
            "═══════════════════════════════════════════════════════════════════════════════"
            + people_image_note
            + """
Produce a face_groups dict (by biological identity) and a characters dict
(by appearance/outfit). If the SAME PERSON appears in different outfits, the
characters share ONE face_group.

face_groups example:
{
  "group_A": {
    "physical_description": "Adult [gender]: [face shape], [hair], [eyes], [skin], [distinctive features], [build]",
    "characters_with_this_face": ["person_01", "person_02"]
  }
}

Describe body proportions only from BODY REFERENCE images. If only face portraits
are available, mark the build as unobserved instead of inventing it. A torso
reference does not establish the shape of hidden legs. Preserve the person's
observed proportions without idealization or normalization.

PHYSICAL DESCRIPTION must be specific:
"Adult woman: oval face, long dark straight hair with bangs reaching eyebrows, brown almond-shaped eyes, light skin tone, defined cheekbones, small nose, slim build"

characters example:
{
  "person_01": {
    "belongs_to_face": "group_A",
    "outfit": "<concrete outfit observed in the references>",
    "identify_in_references": "Look for: <distinctive trait + outfit>"
  }
}

style_source: which character\'s outfit becomes the default for thumbnails.
═══════════════════════════════════════════════════════════════════════════════
"""
        )

    return f"""You are setting up the IDENTITY + STYLE BASE for a YouTube thumbnail series.

VIDEO TITLE: {video_title}
{context_section}
TRANSCRIPTION (excerpt):
{transcription}
{style_reference_section}{reference_mapping_section}{subject_analysis_section}{setting_section}{_LANGUAGE_SECTION}{custom_section}
═══════════════════════════════════════════════════════════════════════════════
TASK: Produce the IDENTITY BASE shared by every thumbnail in this series.
═══════════════════════════════════════════════════════════════════════════════

Do NOT generate scene descriptions, poses, expressions, moods, or any per-thumbnail
concept. Those will be generated separately, one by one. You only define constants.

Required JSON (single object, no array, no markdown fence):

{{
  "face_groups": {{}},
  "characters": {{}},
  "style_source": "person_XX",
  "style_analysis": null,
  "camera_and_lighting_base": {{
    "framing": "...",
    "equipment": "...",
    "lighting": "...",
    "composition": "..."
  }},
  "colors_palette_base": ["#...", "#..."],{setting_field}
  "suggested_titles_pool": ["..."]
}}

Rules for the JSON:
- face_groups / characters: REQUIRED when reference people are provided; otherwise empty {{}}.
- style_analysis: REQUIRED when a style reference image was provided; otherwise null.
- camera_and_lighting_base: REQUIRED. Pick cinematic defaults that fit the video tone.
- colors_palette_base: 4-5 hex codes that match the video mood.
- suggested_titles_pool: {num_titles_in_pool} distinct, catchy title candidates.{setting_rule}

Output ONLY the JSON. No markdown. No extra prose.

{thumbnail_style}
"""


# =============================================================================
# CONCEPT PROMPT (Stage 2 — one thumbnail concept at a time)
# =============================================================================


def build_concept_prompt(
    transcription,
    video_title,
    image_index,
    total_images,
    skeleton,
    prior_concepts,
    custom_instructions=None,
    history_prompts=None,
    include_history=False,
    selected_titles=None,
    thumbnail_style="",
    facial_expression=None,
    reference_mapping=None,
):
    """Build a single LLM prompt that produces ONE thumbnail concept building
    on a pre-existing identity skeleton, avoiding repetition of concepts
    already generated in this batch. With setting photos in the reference
    mapping, every concept happens in that setting."""
    custom_section = _build_custom_section(custom_instructions)
    history_section = _build_history_section(history_prompts, include_history)
    selected_titles_section = _build_selected_titles_section(selected_titles)
    expression_section = _build_expression_section(facial_expression)
    setting_section = _build_concept_setting_section(reference_mapping)
    # A chosen expression is the same in every concept, so it is not varied
    varied = "angle, scene, mood or pose" if facial_expression else "angle, scene, mood, pose or expression"

    skeleton_summary = orjson.dumps(skeleton, option=orjson.OPT_INDENT_2).decode("utf-8")

    prior_summary = ""
    if prior_concepts:
        prior_lines = []
        for c in prior_concepts:
            scene_short = (c.get("scene") or "")[:100]
            prior_lines.append(
                f"- #{c.get('image_index')} {c.get('concept_name', '?')!r}: "
                f"mood={c.get('mood', '?')}, scene='{scene_short}...', "
                f"pose='{c.get('subject_pose', '?')}', expression='{c.get('subject_expression', '?')}'"
            )
        prior_summary = (
            "\n═══════════════════════════════════════════════════════════════════════════════\n"
            f"CONCEPTS ALREADY GENERATED IN THIS BATCH ({len(prior_concepts)}/{total_images}) — "
            f"DO NOT REPEAT their {varied}:\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
            + "\n".join(prior_lines)
            + "\n═══════════════════════════════════════════════════════════════════════════════\n"
        )

    return f"""You are generating thumbnail concept {image_index} of {total_images} for a YouTube video.

VIDEO TITLE: {video_title}
TRANSCRIPTION (excerpt):
{transcription}

═══════════════════════════════════════════════════════════════════════════════
IDENTITY SKELETON (already defined for this series — DO NOT redefine):
═══════════════════════════════════════════════════════════════════════════════
{skeleton_summary}
═══════════════════════════════════════════════════════════════════════════════
{prior_summary}{history_section}{selected_titles_section}{expression_section}{setting_section}{_LANGUAGE_SECTION}{custom_section}
═══════════════════════════════════════════════════════════════════════════════
TASK: Generate ONE distinct thumbnail concept (image_index={image_index}).
═══════════════════════════════════════════════════════════════════════════════

Reference identities by character IDs from the skeleton (person_01, person_02, ...).
You may pick a title from suggested_titles_pool, or refine one. Aim for each
concept in the batch to use a different title.

camera_and_lighting_override: leave as null to inherit the skeleton\'s base.
Only set a different object if THIS concept benefits from a unique camera or
lighting setup (dramatic low angle, hard rim light, etc.) that the base
doesn\'t capture.

TEXT TO RENDER:
- text_in_image is the ONLY field for words that must appear inside the generated image.
- When the user requests a headline, caption, or exact visible wording, put that
  wording in text_in_image verbatim, including punctuation and accents. Do not
  leave it null or put the requested wording only in suggested_title.
- Use null when the concept has no visible text. An explicit request for no text
  must produce null. Do not turn a video title into lettering unless wanted.
- Keep typography, color, placement, and reserved space in visual_elements or
  camera_and_lighting_override.composition; text_in_image holds the literal words.
- suggested_title is the video title shown in the interface, not rendered lettering.

OUTFITS:
- A character's outfit in the skeleton is what they wear in the video: their
  default clothing, not part of their identity.
- outfit_changes: null keeps every outfit. When this concept needs other clothing
  for someone (a costume, a uniform, swimwear, pajamas, a towel, no clothes...)
  because the idea or the user's instructions call for it, list
  {{"person_id": "person_XX", "outfit": "..."}} for each person whose clothing
  changes, with the complete look they wear in this thumbnail. Without clothes,
  say what covers the body instead.

Required JSON (single object, no array, no markdown fence):

{{
  "image_index": {image_index},
  "concept_name": "...",
  "thumbnail_concept": "...",
  "suggested_title": "...",
  "mood": "...",
  "colors": ["#...", "#..."],
  "key_topics": ["...", "...", "..."],

  "scene": "person_01 ...",
  "subject_pose": "...",
  "subject_expression": "...",
  "subject_action": null,
  "visual_elements": null,
  "text_in_image": "...",
  "background": "...",
  "environment_effects": null,
  "outfit_changes": null,

  "camera_and_lighting_override": null
}}

Output ONLY the JSON object. No markdown fence. No extra prose.

{thumbnail_style}
"""


# Camera and lighting of a concept or of the skeleton's base
_CAMERA_SCHEMA = {
    "type": "object",
    "properties": {
        "framing": {"type": "string"},
        "equipment": {"type": "string"},
        "lighting": {"type": "string"},
        "composition": {"type": "string"},
    },
    "required": ["framing", "equipment", "lighting", "composition"],
    "additionalProperties": False,
}

# Stage 1 answer. Its face_groups and characters are keyed by the people of the
# video, so the schema is only checked here: Gran Sabio's native structured
# output does not accept objects with free keys.
SKELETON_SCHEMA = {
    "type": "object",
    "properties": {
        "face_groups": {"type": "object"},
        "characters": {"type": "object"},
        "style_source": {"type": ["string", "null"]},
        "style_analysis": {"type": ["object", "null"]},
        "camera_and_lighting_base": _CAMERA_SCHEMA,
        "colors_palette_base": {"type": "array", "items": {"type": "string"}},
        "suggested_titles_pool": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "face_groups", "characters", "style_source", "style_analysis",
        "camera_and_lighting_base", "colors_palette_base", "suggested_titles_pool",
    ],
}


def skeleton_schema(reference_mapping):
    """SKELETON_SCHEMA, plus the setting's description when setting photos are sent."""
    if not environment_references(reference_mapping):
        return SKELETON_SCHEMA
    return {
        **SKELETON_SCHEMA,
        "properties": {**SKELETON_SCHEMA["properties"], "setting": {"type": "string", "minLength": 1}},
        "required": [*SKELETON_SCHEMA["required"], "setting"],
    }

# Stage 2 answer, the JSON build_concept_prompt asks for. Sent to Gran Sabio as
# the structured output schema and checked again here.
CONCEPT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_index": {"type": "integer"},
        "concept_name": {"type": "string"},
        "thumbnail_concept": {"type": "string"},
        "suggested_title": {"type": "string"},
        "mood": {"type": "string"},
        "colors": {"type": "array", "items": {"type": "string"}},
        "key_topics": {"type": "array", "items": {"type": "string"}},
        "scene": {"type": "string"},
        "subject_pose": {"type": "string"},
        "subject_expression": {"type": "string"},
        "subject_action": {"type": ["string", "null"]},
        "visual_elements": {"type": ["array", "null"], "items": {"type": "string"}},
        "text_in_image": {
            "type": ["string", "null"],
            "minLength": 1,
            "description": "Exact visible wording requested for the image, including punctuation and accents; null for no lettering.",
        },
        "background": {"type": "string"},
        "environment_effects": {"type": ["string", "null"]},
        "outfit_changes": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "properties": {"person_id": {"type": "string"}, "outfit": {"type": "string"}},
                "required": ["person_id", "outfit"],
                "additionalProperties": False,
            },
            "description": "Clothing that replaces the skeleton outfit of these people in this thumbnail; null keeps every outfit.",
        },
        "camera_and_lighting_override": {**_CAMERA_SCHEMA, "type": ["object", "null"]},
    },
    "required": [
        "image_index", "concept_name", "thumbnail_concept", "suggested_title", "mood",
        "colors", "key_topics", "scene", "subject_pose", "subject_expression",
        "subject_action", "visual_elements", "text_in_image", "background",
        "environment_effects", "outfit_changes", "camera_and_lighting_override",
    ],
    "additionalProperties": False,
}

# Gran Sabio accepts at most this many images in one generation request
MAX_GRANSABIO_IMAGES = 20


def generate_with_gransabio(
    prompt: str,
    config: PromptGenerationConfig,
    reference_image_base64: Optional[str] = None,
    reference_images_base64: Optional[List[str]] = None,
    json_schema: Optional[dict] = None,
):
    """
    Generate a JSON answer with Gran Sabio LLM Engine.

    Args:
        prompt: The prompt to send
        config: Generation configuration
        reference_image_base64: Optional single base64-encoded reference image
        reference_images_base64: Optional list of base64-encoded images
        json_schema: Structured output schema Gran Sabio makes the model follow

    Returns:
        The answer's content (text or already parsed JSON)

    Raises:
        GranSabioError: the request failed or came back empty
    """
    client = get_gransabio_client()

    # Get model (use default if not specified)
    model = resolve_text_model(config.provider, config.model)

    # Build generation kwargs
    # username is required when using images
    gen_kwargs = {
        "prompt": prompt,
        "username": GRANSABIO_USERNAME,
        "qa_layers": [],  # Bypass QA for speed
        "json_output": True,
        "temperature": 0.8,
        "max_iterations": 1,
        "verbose": False,
        **text_generation_options(config.provider, model, config.thinking_enabled,
                                  config.thinking_level, config.model_info,
                                  vision=bool(reference_image_base64 or reference_images_base64)),
    }
    if json_schema is not None:
        gen_kwargs["json_schema"] = json_schema
        gen_kwargs["json_retry_without_iteration"] = True
        gen_kwargs["content_type"] = "json"  # Avoid conflict with default "article"
    if config.thinking_enabled:
        logger.info(f"Using thinking mode: {config.thinking_level}")

    # Handle reference images - Gran Sabio requires upload first, then reference by upload_id
    images_to_upload = []
    if reference_image_base64:
        images_to_upload.append(reference_image_base64)
    if reference_images_base64:
        images_to_upload.extend(reference_images_base64)
    if len(images_to_upload) > MAX_GRANSABIO_IMAGES:
        raise GranSabioError(
            f"{len(images_to_upload)} reference images given; Gran Sabio accepts at most {MAX_GRANSABIO_IMAGES}"
        )

    if images_to_upload:
        timestamp = int(time.time())
        gen_kwargs["images"] = [
            {
                "upload_id": upload_image_to_gransabio(
                    image_base64=img_b64,
                    filename=f"ref_frame_{timestamp}_{i}.jpg",
                    username=GRANSABIO_USERNAME
                ),
                "username": GRANSABIO_USERNAME,
                "detail": GRANSABIO_IMAGE_DETAIL
            }
            for i, img_b64 in enumerate(images_to_upload)
        ]
        logger.info(f"Including {len(images_to_upload)} reference image(s) in prompt generation")

    logger.info(f"Generating prompts with Gran Sabio LLM ({config.provider}/{model})...")
    try:
        result = client.generate(**gen_kwargs)
    except Exception as e:
        raise GranSabioError(f"Gran Sabio LLM ({config.provider}/{model}) failed: {e}") from e

    if not result or not result.get("content"):
        raise GranSabioError(f"Gran Sabio LLM ({config.provider}/{model}) returned no content")
    verify_local_execution(result, config.provider)
    return result["content"]


def validated_json(content, schema: dict, what: str) -> dict:
    """The JSON object of an LLM answer, checked against its schema.

    ai_json_cleanroom pulls the JSON out of fences or surrounding prose and makes
    only safe repairs. Raises GranSabioError naming the fields that do not fit.
    """
    result = validate_ai_json(content, schema=schema)
    if not result.json_valid:
        problems = "; ".join(f"{issue.path}: {issue.message.rstrip('.')}" for issue in result.errors[:6])
        truncated = " (the answer looks truncated)" if result.likely_truncated else ""
        raise GranSabioError(f"Gran Sabio returned an invalid {what}{truncated}: {problems}")
    return result.data


def generate_skeleton_gransabio(
    transcription,
    video_title,
    config,
    cluster_description=None,
    thumbnail_style="",
    reference_image_base64=None,
    reference_images_base64=None,
    has_style_reference=False,
    reference_mapping=None,
):
    """Stage 1: generate the identity/style/base skeleton via Gran Sabio."""
    has_reference = bool(reference_image_base64) or bool(reference_images_base64)
    prompt = build_skeleton_prompt(
        transcription=transcription,
        video_title=video_title,
        cluster_description=cluster_description,
        custom_instructions=config.custom_instructions,
        thumbnail_style=thumbnail_style,
        has_reference_image=has_reference,
        has_style_reference=has_style_reference,
        reference_mapping=reference_mapping,
    )
    response = generate_with_gransabio(
        prompt=prompt,
        config=config,
        reference_image_base64=reference_image_base64,
        reference_images_base64=reference_images_base64,
    )
    return validated_json(response, skeleton_schema(reference_mapping), "skeleton")


def generate_concept_gransabio(
    transcription,
    video_title,
    image_index,
    total_images,
    skeleton,
    prior_concepts,
    config,
    thumbnail_style="",
    selected_titles=None,
    reference_mapping=None,
):
    """Stage 2: generate one thumbnail concept via Gran Sabio. No reference
    images are re-sent — the identity and the setting already live in the skeleton."""
    prompt = build_concept_prompt(
        transcription=transcription,
        video_title=video_title,
        image_index=image_index,
        total_images=total_images,
        skeleton=skeleton,
        prior_concepts=prior_concepts,
        custom_instructions=config.custom_instructions,
        history_prompts=config.history_prompts,
        include_history=config.include_history,
        selected_titles=selected_titles,
        thumbnail_style=thumbnail_style,
        facial_expression=config.facial_expression,
        reference_mapping=reference_mapping,
    )
    response = generate_with_gransabio(
        prompt=prompt,
        config=config,
        # No reference images: identity is in the skeleton.
        json_schema=CONCEPT_SCHEMA,
    )
    concept = validated_json(response, CONCEPT_SCHEMA, f"concept {image_index}")
    _check_outfit_changes(concept, skeleton, f"concept {image_index}")
    return concept


def _check_outfit_changes(concept: dict, skeleton: dict, what: str) -> None:
    """A concept may only dress characters of the skeleton, once each, with an actual outfit."""
    changes = concept["outfit_changes"] or []
    people = [change["person_id"] for change in changes]
    problems = [f"{person} is not a character of the skeleton"
                for person in people if person not in skeleton["characters"]]
    if len(people) != len(set(people)):
        problems.append("a person is listed twice")
    if any(not change["outfit"].strip() for change in changes):
        problems.append("an outfit is empty")
    if problems:
        raise GranSabioError(f"Gran Sabio returned an invalid {what}: outfit_changes: {'; '.join(problems)}")


def get_prompt_history_for_video(video_id: int, limit: int = 50) -> List[str]:
    """
    Get previous prompt concepts for a video to avoid repetition.

    Args:
        video_id: Video ID
        limit: Maximum number of previous concepts to retrieve

    Returns:
        List of previous concept summaries
    """
    try:
        import sqlite3
        from config import DATABASE_PATH

        conn = sqlite3.connect(str(DATABASE_PATH), timeout=10)
        try:
            cursor = conn.cursor()

            # Get previous prompts from thumbnails table
            cursor.execute("""
                SELECT DISTINCT t.prompt_text, t.suggested_title, t.text_overlay
                FROM thumbnails t
                JOIN generation_jobs gj ON t.job_id = gj.id
                WHERE gj.video_id = ?
                ORDER BY t.id DESC
                LIMIT ?
            """, [video_id, limit])

            rows = cursor.fetchall()
        finally:
            conn.close()

        history = []
        for row in rows:
            prompt_text, title, overlay = row
            if prompt_text:
                summary = f"Title: {title or 'N/A'}\nOverlay: {overlay or 'N/A'}\nPrompt: {prompt_text[:500]}"
                history.append(summary)

        return history

    except Exception as e:
        logger.warning(f"Could not load prompt history: {e}")
        return []


# =============================================================================
# API ENDPOINT HELPERS
# =============================================================================

async def _model_catalog() -> dict:
    """Gran Sabio's models by provider. Raises GranSabioError when it does not answer."""
    try:
        async with quick_gransabio_client() as client:
            return await client.get_models()
    except Exception as e:
        raise GranSabioError(f"Gran Sabio's model list could not be loaded ({GRANSABIO_LLM_URL}): {e}") from e


async def require_vision_model(provider: str, model: Optional[str]) -> dict:
    """Fail unless Gran Sabio lists the thumbnail ideas model as one that sees images.

    The skeleton request carries the face photos, so a text-only model would
    only fail there, after the transcription. Raises GranSabioError naming it.
    """
    return _catalog_model(await _model_catalog(), provider, model, vision=True)


async def get_available_models() -> dict:
    """
    Get available models directly from Gran Sabio LLM API.

    Fetches from GRANSABIO_LLM_URL/models endpoint.
    Returns dict mapping provider names to model lists.

    Raises:
        GranSabioError: Gran Sabio did not answer with its model list
    """
    data = await _model_catalog()

    filtered = {}
    for provider in TEXT_PROVIDER_NAMES:
        if provider in data:
            # Transform to simplified format for frontend
            filtered[provider] = [
                {
                    "key": model.get("key", model.get("model_id", "")),
                    "name": model.get("name", ""),
                    "description": model.get("description", "")[:60] if model.get("description") else "",
                    # Thumbnail ideas need it: their requests carry the face photos
                    "vision": "vision" in (model.get("capabilities") or []),
                    "reasoning": "reasoning" in (model.get("capabilities") or []),
                    **({"pricing": model.get("pricing")} if provider == "openrouter" else {}),
                }
                for model in data[provider]
                if provider not in CATALOG_TEXT_PROVIDERS or "text" in (model.get("capabilities") or [])
            ]

    logger.info(f"Loaded {sum(len(v) for v in filtered.values())} models from Gran Sabio LLM")
    return filtered

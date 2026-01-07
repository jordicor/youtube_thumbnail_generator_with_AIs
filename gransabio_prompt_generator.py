"""
Gran Sabio LLM Prompt Generator
===============================

Generates thumbnail prompts using Gran Sabio LLM Engine for enhanced
AI provider selection, thinking mode support, and prompt history.
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from config import (
    GRANSABIO_LLM_URL,
    GRANSABIO_CLIENT_PATH,
    GRANSABIO_USERNAME,
    GRANSABIO_IMAGE_DETAIL,
    GRANSABIO_MAX_REF_IMAGES
)

logger = logging.getLogger(__name__)

# Add Gran Sabio LLM client to path if configured
# The client is a standalone HTTP client that only requires 'requests' and 'aiohttp'
# Get it from: https://github.com/jordicor/GranSabio_LLM (the 'client' folder)
if GRANSABIO_CLIENT_PATH:
    client_path = Path(GRANSABIO_CLIENT_PATH)
    if client_path.exists():
        sys.path.insert(0, str(client_path.parent))  # Add parent so 'from client import' works
    else:
        logger.warning(f"Gran Sabio client path not found: {GRANSABIO_CLIENT_PATH}")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PromptGenerationConfig:
    """Configuration for prompt generation via Gran Sabio LLM."""
    provider: str = "anthropic"  # anthropic, openai, google, xai
    model: Optional[str] = None  # None = use default for provider
    thinking_enabled: bool = False
    thinking_level: str = "medium"  # low, medium, high
    custom_instructions: Optional[str] = None
    include_history: bool = False
    history_prompts: Optional[List[str]] = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "thinking_enabled": self.thinking_enabled,
            "thinking_level": self.thinking_level,
            "custom_instructions": self.custom_instructions,
            "include_history": self.include_history,
        }


# Default models per provider
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-5.2",
    "google": "gemini-3-flash-preview",
    "xai": "grok-4-1-fast-non-reasoning",
}

# Thinking budget tokens per level (for Claude)
THINKING_BUDGET_TOKENS = {
    "low": 4000,
    "medium": 8000,
    "high": 16000,
}

# Reasoning effort per level (for OpenAI)
REASONING_EFFORT = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}

# Maximum history characters to include
MAX_HISTORY_CHARS = 32000


# =============================================================================
# GRANSABIO CLIENT
# =============================================================================

_gransabio_client = None
_gransabio_available = None


def get_gransabio_client():
    """Get or create Gran Sabio LLM client singleton."""
    global _gransabio_client, _gransabio_available

    if _gransabio_available is False:
        return None

    if _gransabio_client is not None:
        return _gransabio_client

    try:
        from client import GranSabioClient
        client = GranSabioClient(base_url=GRANSABIO_LLM_URL)

        # Test connection
        if client.is_available():
            _gransabio_client = client
            _gransabio_available = True
            logger.info(f"Gran Sabio LLM client connected successfully to {GRANSABIO_LLM_URL}")
            return client
        else:
            _gransabio_available = False
            logger.warning(f"Gran Sabio LLM server not available at {GRANSABIO_LLM_URL}")
            return None

    except ImportError as e:
        _gransabio_available = False
        logger.warning(f"Gran Sabio LLM client not available: {e}")
        return None
    except Exception as e:
        _gransabio_available = False
        logger.warning(f"Could not connect to Gran Sabio LLM at {GRANSABIO_LLM_URL}: {e}")
        return None


def upload_image_to_gransabio(
    image_base64: str,
    filename: str = "reference_frame.jpg",
    username: Optional[str] = None
) -> Optional[str]:
    """
    Upload an image to Gran Sabio LLM and return the upload_id.

    Gran Sabio requires images to be uploaded first, then referenced by upload_id
    in the generate request. This two-step process allows the server to validate
    and process images before they are used in generation.

    Args:
        image_base64: Base64-encoded image data (without data: prefix)
        filename: Filename to use for the upload
        username: User namespace for the attachment

    Returns:
        upload_id string or None on failure
    """
    client = get_gransabio_client()
    if not client:
        logger.warning("Gran Sabio client not available for image upload")
        return None

    # Use configured username if not specified
    upload_username = username or GRANSABIO_USERNAME

    try:
        result = client.upload_attachment_base64(
            username=upload_username,
            content_base64=image_base64,
            filename=filename,
            content_type="image/jpeg"
        )
        upload_id = result.get("upload_id")
        if upload_id:
            logger.info(f"Image uploaded to Gran Sabio: {upload_id[:16]}...")
            return upload_id
        else:
            logger.warning("Gran Sabio returned empty upload_id")
            return None
    except Exception as e:
        logger.error(f"Failed to upload image to Gran Sabio: {e}")
        return None


def check_gransabio_status() -> dict:
    """
    Check Gran Sabio LLM server status.

    Returns:
        dict with status info: available, active_sessions, models, etc.
    """
    try:
        from client import GranSabioClient
        client = GranSabioClient(base_url=GRANSABIO_LLM_URL)

        if client.is_available():
            info = client.get_info()
            models = client.get_models()

            # Extract available providers from models
            providers = list(models.keys()) if models else []

            return {
                "available": True,
                "status": info.get("status", "healthy"),
                "active_sessions": info.get("active_sessions", 0),
                "providers": providers,
                "url": GRANSABIO_LLM_URL,
                "message": f"Gran Sabio LLM connected ({GRANSABIO_LLM_URL})"
            }
        else:
            return {
                "available": False,
                "status": "offline",
                "providers": [],
                "url": GRANSABIO_LLM_URL,
                "message": f"Gran Sabio LLM not responding ({GRANSABIO_LLM_URL})"
            }

    except ImportError:
        return {
            "available": False,
            "status": "not_installed",
            "providers": [],
            "url": GRANSABIO_LLM_URL,
            "message": "Gran Sabio LLM client not installed"
        }
    except Exception as e:
        return {
            "available": False,
            "status": "error",
            "providers": [],
            "url": GRANSABIO_LLM_URL,
            "message": f"Connection error: {str(e)}"
        }


# =============================================================================
# PROMPT GENERATION
# =============================================================================

def build_analysis_prompt(
    transcription: str,
    video_title: str,
    num_concepts: int,
    num_variations: int,
    cluster_description: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    history_prompts: Optional[List[str]] = None,
    include_history: bool = False,
    thumbnail_style: str = "",
    selected_titles: Optional[List[str]] = None,
    has_reference_image: bool = False
) -> str:
    """
    Build the analysis prompt for thumbnail concept generation.

    Args:
        transcription: Video transcription text
        video_title: Original video title
        num_concepts: Number of concepts to generate
        num_variations: Number of variations per concept
        cluster_description: Optional context about reference images
        custom_instructions: Optional user instructions
        history_prompts: Previous prompts to avoid repetition
        include_history: Whether to include history in prompt
        thumbnail_style: Style guidance string
        selected_titles: Optional list of user-selected titles to guide image generation
        has_reference_image: Whether a reference image is being provided for visual inspiration

    Returns:
        Complete prompt string for LLM
    """
    total_images = num_concepts * num_variations

    # Build context section
    context_section = ""
    if cluster_description and cluster_description.strip():
        context_section = f"""
CONTEXT FOR REFERENCE IMAGES:
The user has provided this note about the person/scene in the reference images:

### START USER CONTEXT ###
{cluster_description.strip()}
### END USER CONTEXT ###

Use this information when crafting your thumbnail concepts and image prompts.
"""

    # Build custom instructions section (placed last to indicate highest priority)
    custom_section = ""
    if custom_instructions and custom_instructions.strip():
        custom_section = f"""
═══════════════════════════════════════════════════════════════════════════════
FINAL USER INSTRUCTIONS (HIGHEST PRIORITY):
═══════════════════════════════════════════════════════════════════════════════
The user has provided the following instructions. These take PRECEDENCE over
any other guidance above, including style suggestions, reference image analysis,
and previous concepts to avoid.

CRITICAL: If there are contradictions between these instructions and anything
else in this prompt, ALWAYS follow the user's instructions.

While respecting these instructions, you should still:
- Generate the requested number of concepts and variations
- Maintain creative variety within the user's constraints
- If the user is very specific about certain aspects, use that as a creative
  anchor while varying other elements to provide options

USER INSTRUCTIONS:
{custom_instructions.strip()}
═══════════════════════════════════════════════════════════════════════════════
"""

    # Build history section (to avoid repetition)
    history_section = ""
    if include_history and history_prompts:
        # Combine history prompts
        history_text = "\n---\n".join(history_prompts)

        # Truncate if too long (keep most recent)
        if len(history_text) > MAX_HISTORY_CHARS:
            history_text = history_text[-MAX_HISTORY_CHARS:]
            # Find first complete entry after truncation
            first_separator = history_text.find("\n---\n")
            if first_separator > 0:
                history_text = history_text[first_separator + 5:]

        history_section = f"""
═══════════════════════════════════════════════════════════════════════════════
PREVIOUS CONCEPTS (DO NOT REPEAT THESE):
═══════════════════════════════════════════════════════════════════════════════
The following concepts have already been generated. Create NEW and DIFFERENT concepts.
Avoid similar ideas, moods, compositions, and text overlays.

{history_text}
═══════════════════════════════════════════════════════════════════════════════
"""

    # Build selected titles section (to guide image generation based on user-selected titles)
    selected_titles_section = ""
    if selected_titles and len(selected_titles) > 0:
        titles_list = "\n".join(f"- {title}" for title in selected_titles)
        selected_titles_section = f"""
═══════════════════════════════════════════════════════════════════════════════
SELECTED TITLES (GUIDE YOUR IMAGES TO MATCH THESE):
═══════════════════════════════════════════════════════════════════════════════
The user has selected these potential video titles. Your thumbnail concepts and
image prompts MUST be designed to visually complement and support these titles.

{titles_list}

IMPORTANT: Create thumbnails that would make viewers want to click when they see
these titles next to them. The visual and the title should work together as a unit.
═══════════════════════════════════════════════════════════════════════════════
"""

    # Build reference image section (for visual inspiration)
    reference_image_section = ""
    if has_reference_image:
        reference_image_section = """
═══════════════════════════════════════════════════════════════════════════════
REFERENCE IMAGE (FOR INSPIRATION - NOT TO COPY):
═══════════════════════════════════════════════════════════════════════════════
A reference image has been provided for visual INSPIRATION. This is a starting
point, NOT something to replicate exactly. The image prompts you generate are
the PRIMARY guide for image generation.

ANALYZE the reference image for:
- Visual style (colors, mood, aesthetic)
- Composition elements (layout, focus points)
- Atmosphere and feeling

USE THIS AS INSPIRATION to enrich your concepts, but:
- The scene description in image_prompt takes PRIORITY over the reference
- Your creative concepts should go BEYOND simply copying the reference
- Use it as a springboard for ideas, not a constraint
- If user provides specific instructions (see below), those override any
  reference image analysis

You may mention stylistic inspirations from the reference in image_prompt when
relevant (e.g., "warm color palette", "dramatic lighting", "centered composition").
═══════════════════════════════════════════════════════════════════════════════
"""

    prompt = f"""You are an expert YouTube thumbnail designer and video content analyst.

Analyze the following video transcription and generate {num_concepts} thumbnail concepts, each with {num_variations} variations.

VIDEO TITLE: {video_title}
{context_section}
TRANSCRIPTION (excerpt):
{transcription}
{history_section}{selected_titles_section}{reference_image_section}{custom_section}
═══════════════════════════════════════════════════════════════════════════════
TASK: Generate {num_concepts} concepts x {num_variations} variations = {total_images} unique image prompts
═══════════════════════════════════════════════════════════════════════════════

CONCEPTS are different creative ideas for the thumbnail. Examples of concept approaches:
- Emotion/Reaction, Problem/Solution, Intrigue/Curiosity, Achievement, Tutorial, Humor, etc.
- Choose what fits best with the video content - these are just examples, use your creativity.

VARIATIONS are different interpretations of the same concept, like A/B testing for engagement.
Each variation should have a complete, independent image_prompt. Think of variations as:
- Different ways to visually represent the same idea
- Subtle or more noticeable differences depending on what makes sense
- Examples: different expressions, poses, camera angles, compositions, lighting, visual elements

For EACH concept, provide:
- **concept_name**: Short name for this concept (2-4 words)
- **thumbnail_concept**: Brief description of the idea (1-2 sentences)
- **suggested_title**: Catchy video title (max 60 chars)
- **mood**: Emotional tone (excited, mysterious, professional, fun, etc.)
- **colors**: 2-3 dominant colors as hex codes
- **key_topics**: 3-5 main keywords from the video
- **variations**: Array of {num_variations} variations, each with:
  - **variation_index**: 1, 2, 3...
  - **variation_focus**: What this variation emphasizes (e.g., "intense expression", "dynamic angle")
  - **image_prompt**: COMPLETE prompt for AI image generation. Include subject, expression,
    pose, background, lighting, style. Optimized for 16:9 (1280x720). Each prompt must work independently.
  - **text_overlay**: Short text for the thumbnail (2-5 words, punchy)

Respond in JSON format:
[
    {{
        "concept_name": "...",
        "thumbnail_concept": "...",
        "suggested_title": "...",
        "mood": "...",
        "colors": ["#...", "#..."],
        "key_topics": ["...", "..."],
        "variations": [
            {{
                "variation_index": 1,
                "variation_focus": "...",
                "image_prompt": "...",
                "text_overlay": "..."
            }}
        ]
    }}
]

Generate exactly {num_concepts} concepts, each with exactly {num_variations} variations.

{thumbnail_style}
"""
    return prompt


def generate_with_gransabio(
    prompt: str,
    config: PromptGenerationConfig,
    reference_image_base64: Optional[str] = None,
    reference_images_base64: Optional[List[str]] = None
) -> Optional[str]:
    """
    Generate response using Gran Sabio LLM Engine.

    Args:
        prompt: The prompt to send
        config: Generation configuration
        reference_image_base64: Optional single base64-encoded reference image
        reference_images_base64: Optional list of base64-encoded images (max 20)

    Returns:
        Generated response text or None on failure
    """
    client = get_gransabio_client()
    if not client:
        logger.warning("Gran Sabio LLM not available, falling back to direct API")
        return None

    try:
        # Get model (use default if not specified)
        model = config.model or DEFAULT_MODELS.get(config.provider, "gpt-4o")

        # Build generation kwargs
        # username is required when using images
        gen_kwargs = {
            "prompt": prompt,
            "generator_model": model,
            "username": GRANSABIO_USERNAME,
            "qa_layers": [],  # Bypass QA for speed
            "json_output": True,
            "max_tokens": 16000,
            "temperature": 0.8,
            "max_iterations": 1,
            "verbose": False,
        }

        # Add thinking mode if enabled
        if config.thinking_enabled:
            # Check if model supports thinking
            model_lower = model.lower()

            if "claude" in model_lower:
                # Claude uses thinking_budget_tokens
                budget = THINKING_BUDGET_TOKENS.get(config.thinking_level, 8000)
                gen_kwargs["thinking_budget_tokens"] = budget
                logger.info(f"Using Claude thinking mode with {budget} tokens")

            elif any(x in model_lower for x in ["o1", "o3", "gpt-5"]):
                # OpenAI reasoning models use reasoning_effort
                effort = REASONING_EFFORT.get(config.thinking_level, "medium")
                gen_kwargs["reasoning_effort"] = effort
                logger.info(f"Using OpenAI reasoning mode with effort: {effort}")

        # Handle reference images - Gran Sabio requires upload first, then reference by upload_id
        images_to_upload = []

        # Single image (backward compatibility)
        if reference_image_base64:
            images_to_upload.append(reference_image_base64)

        # Multiple images
        if reference_images_base64:
            images_to_upload.extend(reference_images_base64)

        # Limit to configured max (default 5) or absolute max of 20 (Gran Sabio server limit)
        max_images = min(GRANSABIO_MAX_REF_IMAGES, 20)
        images_to_upload = images_to_upload[:max_images]

        # Upload images and build references
        if images_to_upload:
            image_refs = []
            timestamp = int(time.time())

            for i, img_b64 in enumerate(images_to_upload):
                upload_id = upload_image_to_gransabio(
                    image_base64=img_b64,
                    filename=f"ref_frame_{timestamp}_{i}.jpg",
                    username=GRANSABIO_USERNAME
                )
                if upload_id:
                    image_refs.append({
                        "upload_id": upload_id,
                        "username": GRANSABIO_USERNAME,
                        "detail": GRANSABIO_IMAGE_DETAIL
                    })

            if image_refs:
                gen_kwargs["images"] = image_refs
                logger.info(f"Including {len(image_refs)} reference image(s) in prompt generation")
            else:
                logger.warning("Could not upload any reference images, proceeding without them")

        logger.info(f"Generating prompts with Gran Sabio LLM ({config.provider}/{model})...")
        result = client.generate(**gen_kwargs)

        if result and result.get("content"):
            return result["content"]
        else:
            logger.warning("Gran Sabio LLM returned empty content")
            return None

    except Exception as e:
        logger.error(f"Gran Sabio LLM generation error: {e}")
        return None


def generate_thumbnail_concepts_gransabio(
    transcription: str,
    video_title: str,
    num_concepts: int = 5,
    num_variations: int = 1,
    cluster_description: Optional[str] = None,
    config: Optional[PromptGenerationConfig] = None,
    thumbnail_style: str = "",
    selected_titles: Optional[List[str]] = None,
    reference_image_base64: Optional[str] = None,
    reference_images_base64: Optional[List[str]] = None
) -> Optional[list]:
    """
    Generate thumbnail concepts using Gran Sabio LLM.

    Args:
        transcription: Video transcription text
        video_title: Original video title
        num_concepts: Number of concepts to generate
        num_variations: Number of variations per concept
        cluster_description: Optional context about reference images
        config: Optional PromptGenerationConfig
        thumbnail_style: Style guidance string
        selected_titles: Optional list of user-selected titles to guide image generation
        reference_image_base64: Optional single base64-encoded reference image
        reference_images_base64: Optional list of base64-encoded images (max 20)

    Returns:
        List of concept dictionaries or None on failure
    """
    if config is None:
        config = PromptGenerationConfig()

    # Determine if we have any reference images
    has_reference = bool(reference_image_base64) or bool(reference_images_base64)

    # Build the prompt
    prompt = build_analysis_prompt(
        transcription=transcription,
        video_title=video_title,
        num_concepts=num_concepts,
        num_variations=num_variations,
        cluster_description=cluster_description,
        custom_instructions=config.custom_instructions,
        history_prompts=config.history_prompts,
        include_history=config.include_history,
        thumbnail_style=thumbnail_style,
        selected_titles=selected_titles,
        has_reference_image=has_reference
    )

    # Generate with Gran Sabio LLM (passing reference images if provided)
    response = generate_with_gransabio(
        prompt=prompt,
        config=config,
        reference_image_base64=reference_image_base64,
        reference_images_base64=reference_images_base64
    )

    if not response:
        return None

    # Handle response - may be already parsed (list/dict) or string
    try:
        # If already a list (json_output=True returns parsed JSON), use directly
        if isinstance(response, list):
            logger.info(f"Gran Sabio LLM generated {len(response)} concepts successfully")
            return response

        # If already a dict (single concept), wrap in list
        # This handles cases where LLM returns a single object instead of array
        if isinstance(response, dict):
            logger.warning("Gran Sabio LLM returned single concept (dict), wrapping in list")
            return [response]

        # If string, parse it
        if isinstance(response, str):
            # Clean response if needed
            response = response.strip()

            # Handle markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()

            data = json.loads(response)

            if isinstance(data, list):
                logger.info(f"Gran Sabio LLM generated {len(data)} concepts successfully")
                return data
            elif isinstance(data, dict):
                # LLM returned single object instead of array - wrap it
                logger.warning("Gran Sabio LLM returned single concept (parsed dict), wrapping in list")
                return [data]
            else:
                logger.warning(f"Gran Sabio LLM returned unexpected type: {type(data)}")
                return None

        logger.warning(f"Unexpected response type: {type(response)}")
        return None

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gran Sabio LLM response: {e}")
        return None


# =============================================================================
# HISTORY MANAGEMENT
# =============================================================================

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

def get_available_models() -> dict:
    """
    Get available models directly from Gran Sabio LLM API.

    Fetches from GRANSABIO_LLM_URL/models endpoint.
    Returns dict mapping provider names to model lists.
    """
    import requests

    try:
        response = requests.get(f"{GRANSABIO_LLM_URL}/models", timeout=5)
        response.raise_for_status()
        data = response.json()

        # Filter to only include main providers (exclude openrouter, ollama)
        main_providers = ["anthropic", "openai", "google", "xai"]
        filtered = {}
        for provider in main_providers:
            if provider in data:
                # Transform to simplified format for frontend
                filtered[provider] = [
                    {
                        "key": model.get("key", model.get("model_id", "")),
                        "name": model.get("name", ""),
                        "description": model.get("description", "")[:60] if model.get("description") else ""
                    }
                    for model in data[provider]
                ]

        logger.info(f"Loaded {sum(len(v) for v in filtered.values())} models from Gran Sabio LLM")
        return filtered

    except Exception as e:
        logger.warning(f"Could not fetch models from Gran Sabio LLM: {e}")
        # Minimal fallback
        return {
            "anthropic": [{"key": "claude-sonnet-4-5", "name": "Claude Sonnet 4.5", "description": "Default"}],
            "openai": [{"key": "gpt-5.2", "name": "GPT-5.2", "description": "Default"}],
            "google": [{"key": "gemini-3-flash-preview", "name": "Gemini 3 Flash", "description": "Default"}],
            "xai": [{"key": "grok-4-1-fast-non-reasoning", "name": "Grok 4.1 Fast", "description": "Default"}],
        }

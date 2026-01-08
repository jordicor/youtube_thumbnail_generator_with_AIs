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
    GRANSABIO_MAX_REF_IMAGES,
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
    thinking_level: str = "high"  # low, medium, high (Gran Sabio handles token conversion)
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

    # Build subject analysis section (when references are provided)
    subject_analysis_section = ""
    if has_reference_image:
        subject_analysis_section = """
═══════════════════════════════════════════════════════════════════════════════
CRITICAL: ANALYZE REFERENCE IMAGES AND IDENTIFY ALL SUBJECTS
═══════════════════════════════════════════════════════════════════════════════
The SAME reference images you see will ALSO be sent to the image generation AI.
You MUST analyze them and create a clear identification system.

CREATE A "subjects" ARRAY identifying ALL distinct people in references:
{
  "id": "person_01",                    // Sequential: person_01, person_02...
  "visual_identifier": "...",           // DISTINGUISHING features only
  "inferred_role": "...",               // Role deduced from transcription
  "is_costume_variant": true/false,     // Same person in different costume?
  "variant_of": null or "person_XX"     // If variant, which person's ID
}

VISUAL IDENTIFIER: Describe ONLY distinguishing features (clothing, accessories,
hairstyle differences). DO NOT describe facial features - references show those.
Examples: "young woman, long hair, casual outfit" or "same as person_01 but in bathrobe"

INFER ROLES from transcription context. Names in examples are illustrative only;
deduce actual roles from YOUR analysis of the specific transcription and images.

COSTUME VARIANTS: Same person in different outfit = is_costume_variant: true,
variant_of: "person_XX". They must have IDENTICAL face in generated images.

USE THESE IDs in your image_prompt descriptions:
✅ "person_01 reacts with shock while person_02 points at calendar"
❌ "A young woman reacts..." (too generic)
═══════════════════════════════════════════════════════════════════════════════
"""

    prompt = f"""You are an expert YouTube thumbnail designer and video content analyst.

Analyze the following video transcription and generate {num_concepts} thumbnail concepts, each with {num_variations} variations.

VIDEO TITLE: {video_title}
{context_section}
TRANSCRIPTION (excerpt):
{transcription}
{history_section}{selected_titles_section}{subject_analysis_section}{custom_section}
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
- **subjects**: Array of identified subjects from reference images (REQUIRED if references provided)
- **concept_name**: Short name for this concept (2-4 words)
- **thumbnail_concept**: Brief description of the idea (1-2 sentences)
- **suggested_title**: Catchy video title (max 60 chars)
- **mood**: Emotional tone (excited, mysterious, professional, fun, etc.)
- **colors**: 2-3 dominant colors as hex codes
- **key_topics**: 3-5 main keywords from the video
- **variations**: Array of {num_variations} variations, each with:
  - **variation_index**: 1, 2, 3...
  - **variation_focus**: What this variation emphasizes (e.g., "intense expression", "dynamic angle")
  - **image_prompt**: COMPLETE prompt using subject IDs (e.g., "person_01 reacts..."). Include pose,
    expression, background, lighting. Optimized for 16:9 (1280x720). Each prompt must work independently.
  - **text_overlay**: Short text for the thumbnail (2-5 words, punchy)

Respond in JSON format:
[
    {{
        "subjects": [
            {{
                "id": "person_01",
                "visual_identifier": "young woman, long hair, casual outfit",
                "inferred_role": "main creator",
                "is_costume_variant": false,
                "variant_of": null
            }}
        ],
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
                "image_prompt": "person_01 does X with Y expression...",
                "text_overlay": "..."
            }}
        ]
    }}
]

NOTE: The subjects example above is illustrative. Adapt based on YOUR analysis of
the actual reference images and transcription. Use your own identifiers and roles.

Generate exactly {num_concepts} concepts, each with exactly {num_variations} variations.

{thumbnail_style}
"""
    return prompt


def build_analysis_prompt_images(
    transcription: str,
    video_title: str,
    num_images: int,
    cluster_description: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    history_prompts: Optional[List[str]] = None,
    include_history: bool = False,
    thumbnail_style: str = "",
    selected_titles: Optional[List[str]] = None,
    has_reference_image: bool = False
) -> str:
    """
    Build the analysis prompt for flat image generation (no variations).

    This is the new simplified format where each image is completely independent.
    No concept/variation hierarchy - just N unique images.
    """

    # Build context section
    context_section = ""
    if cluster_description and cluster_description.strip():
        context_section = f"""
CONTEXT FOR REFERENCE IMAGES:
The user has provided this note about the person/scene in the reference images:

### START USER CONTEXT ###
{cluster_description.strip()}
### END USER CONTEXT ###

Use this information when crafting your thumbnail images.
"""

    # Build custom instructions section
    custom_section = ""
    if custom_instructions and custom_instructions.strip():
        custom_section = f"""
═══════════════════════════════════════════════════════════════════════════════
FINAL USER INSTRUCTIONS (HIGHEST PRIORITY):
═══════════════════════════════════════════════════════════════════════════════
USER INSTRUCTIONS:
{custom_instructions.strip()}
═══════════════════════════════════════════════════════════════════════════════
"""

    # Build history section (to avoid repetition)
    history_section = ""
    if include_history and history_prompts:
        history_text = "\n---\n".join(history_prompts)
        if len(history_text) > MAX_HISTORY_CHARS:
            history_text = history_text[-MAX_HISTORY_CHARS:]
            first_separator = history_text.find("\n---\n")
            if first_separator > 0:
                history_text = history_text[first_separator + 5:]

        history_section = f"""
═══════════════════════════════════════════════════════════════════════════════
PREVIOUS IMAGES (DO NOT REPEAT THESE):
═══════════════════════════════════════════════════════════════════════════════
{history_text}
═══════════════════════════════════════════════════════════════════════════════
"""

    # Build selected titles section
    selected_titles_section = ""
    if selected_titles and len(selected_titles) > 0:
        titles_list = "\n".join(f"- {title}" for title in selected_titles)
        selected_titles_section = f"""
═══════════════════════════════════════════════════════════════════════════════
SELECTED TITLES (GUIDE YOUR IMAGES TO MATCH THESE):
═══════════════════════════════════════════════════════════════════════════════
{titles_list}
═══════════════════════════════════════════════════════════════════════════════
"""

    # Build subject analysis section - V3 format
    subject_analysis_section = ""
    if has_reference_image:
        subject_analysis_section = """
═══════════════════════════════════════════════════════════════════════════════
REFERENCE IMAGE ANALYSIS - CREATE IDENTITY MAPPING (V3 FORMAT)
═══════════════════════════════════════════════════════════════════════════════
The reference images contain REAL PEOPLE. Create a mapping so the image AI
knows which face to clone and how to identify each person.

STEP 1: FACE GROUPS - Group by biological identity
───────────────────────────────────────────────────
Create face_groups with DETAILED PHYSICAL DESCRIPTIONS of facial features.
If the SAME PERSON appears multiple times (different outfits), they share ONE group.

"face_groups": {
  "group_A": {
    "physical_description": "Adult [gender]: [face shape], [hair], [eyes], [skin], [distinctive features], [body build]",
    "characters_with_this_face": ["person_01", "person_02"]
  }
}

PHYSICAL DESCRIPTION FORMAT (be specific about facial features):
"Adult woman: oval face, long dark straight hair with bangs reaching eyebrows, brown almond-shaped eyes, light skin tone, defined cheekbones, small nose, slim build"
"Adult man: square jawline, short brown wavy hair, full brown beard, blue eyes, fair skin with light freckles, medium athletic build"

STEP 2: CHARACTERS - Each outfit/look as dict entry
────────────────────────────────────────────────────
Create a "characters" DICT (not array) mapping person_XX to their appearance:

"characters": {
  "person_01": {
    "belongs_to_face": "group_A",
    "outfit": "teal blanket wrap, holding bear mug, cozy sick appearance",
    "identify_in_references": "Look for: woman with dark hair and bangs, wrapped in teal blanket"
  },
  "person_02": {
    "belongs_to_face": "group_A",
    "outfit": "white cardigan over black top, casual relaxed look",
    "identify_in_references": "Look for: same woman as person_01, but wearing white cardigan"
  }
}

RULES for identify_in_references:
✅ First character of a face_group: "Look for: [brief physical trait from group], [outfit]"
✅ Same person different outfit: "Look for: same [person] as person_XX, but wearing [outfit]"
✅ Different people: "Look for: [unique physical trait], [outfit]"

STEP 3: STYLE SOURCE
─────────────────────
"style_source": "person_XX"  // Which character's outfit to use in final image

STEP 4: ENVIRONMENT EFFECTS (optional, for special visuals)
───────────────────────────────────────────────────────────
For thumbnails with visual effects (glows, holographic elements, split screens):
"environment_effects": "Holographic glow from floating charts, subtle blue particles, lens flare from sun"

Leave as null if no special effects needed.

USE character IDs in scene descriptions:
✅ "person_02 reacts with shock while looking at calendar"
❌ "A woman reacts..." (too generic, unclear which reference)
═══════════════════════════════════════════════════════════════════════════════
"""

    prompt = f"""You are an expert YouTube thumbnail designer.

Analyze the following video transcription and generate {num_images} unique thumbnail images.

VIDEO TITLE: {video_title}
{context_section}
TRANSCRIPTION (excerpt):
{transcription}
{history_section}{selected_titles_section}{subject_analysis_section}{custom_section}
═══════════════════════════════════════════════════════════════════════════════
TASK: Generate {num_images} unique thumbnail images
═══════════════════════════════════════════════════════════════════════════════

Each IMAGE is a completely independent creative idea. Provide maximum variety:
- Different scenes, poses, expressions, compositions, backgrounds
- Different approaches: reaction, tutorial, mystery, achievement, humor, etc.
- Each image should stand alone as a distinct concept

For EACH image, provide:
- **image_index**: 1, 2, 3... (sequential number)
- **face_groups**: Object with physical_description and characters_with_this_face (REQUIRED if references provided)
- **characters**: Dict mapping person_XX to outfit/identify_in_references (REQUIRED if references provided)
- **style_source**: Which character's outfit to use in final image
- **concept_name**: Short name (2-4 words)
- **thumbnail_concept**: Brief description (1-2 sentences)
- **suggested_title**: Catchy video title (max 60 chars)
- **mood**: Emotional tone (excited, mysterious, professional, fun, etc.)
- **colors**: 2-4 hex codes for color palette
- **key_topics**: 3-5 main keywords from the video
- **scene**: Narrative description using character IDs (person_01, person_02...)
- **subject_pose**: Posture/body language for each character
- **subject_expression**: Facial expressions
- **subject_action**: OPTIONAL - What characters are doing
- **visual_elements**: OPTIONAL - Additional elements as array
- **text_in_image**: OPTIONAL - Text to render IN the image (~3 words)
- **background**: Background/environment description
- **lighting**: Lighting setup
- **environment_effects**: OPTIONAL - Visual effects (glows, holographic elements, particles)
- **equipment**: Camera/lens (e.g., "DSLR 85mm f/1.8")
- **framing**: Shot type (e.g., "medium close-up")
- **composition**: Composition rules (e.g., "rule of thirds")
- **style**: Visual style (default: "photorealistic YouTube thumbnail")
- **quality**: Quality descriptor (default: "8K, photorealistic, sharp focus")
- **text_overlay**: Short text for UI overlay (2-5 words) - NOT rendered in image

Respond in JSON format:
[
    {{
        "image_index": 1,

        "face_groups": {{
            "group_A": {{
                "physical_description": "Adult woman: oval face, long dark hair with bangs, brown almond-shaped eyes, light skin, defined cheekbones, slim build",
                "characters_with_this_face": ["person_01", "person_02"]
            }}
        }},

        "characters": {{
            "person_01": {{
                "belongs_to_face": "group_A",
                "outfit": "casual blue pajamas, hair down",
                "identify_in_references": "Look for: woman with dark hair and bangs, in blue pajamas"
            }},
            "person_02": {{
                "belongs_to_face": "group_A",
                "outfit": "white bathrobe, hair in bun",
                "identify_in_references": "Look for: same woman as person_01, but in white bathrobe"
            }}
        }},

        "style_source": "person_02",
        "concept_name": "...",
        "thumbnail_concept": "...",
        "suggested_title": "...",
        "mood": "...",
        "colors": ["#...", "#...", "#..."],
        "key_topics": ["...", "...", "..."],
        "scene": "person_02 does X while...",
        "subject_pose": "leaning forward, hands on hips",
        "subject_expression": "shocked, wide eyes",
        "subject_action": null,
        "visual_elements": null,
        "text_in_image": null,
        "background": "modern studio, colorful lights",
        "lighting": "dramatic side lighting, warm tone",
        "environment_effects": null,
        "equipment": "DSLR 85mm f/1.8",
        "framing": "medium close-up",
        "composition": "subject left third, space for text",
        "style": "photorealistic YouTube thumbnail",
        "quality": "8K, photorealistic, sharp focus",
        "text_overlay": "..."
    }}
]

IMPORTANT: If only ONE person appears in references (no costume changes), still create:
- face_groups with group_A containing physical_description and characters_with_this_face: ["person_01"]
- characters with person_01 entry including outfit and identify_in_references
- style_source: "person_01"

Generate exactly {num_images} unique images with MAXIMUM VARIETY between them.

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
            "temperature": 0.8,
            "max_iterations": 1,
            "verbose": False,
        }

        # Add thinking mode if enabled (Gran Sabio handles token conversion)
        if config.thinking_enabled:
            level = config.thinking_level or "high"
            gen_kwargs["thinking_level"] = level
            logger.info(f"Using thinking mode: {level}")

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

    # Build the prompt (V1 legacy format for concept-based generation)
    # Note: For image-based generation, use generate_thumbnail_images_gransabio instead
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


def generate_thumbnail_images_gransabio(
    transcription: str,
    video_title: str,
    num_images: int = 5,
    cluster_description: Optional[str] = None,
    config: Optional[PromptGenerationConfig] = None,
    thumbnail_style: str = "",
    selected_titles: Optional[List[str]] = None,
    reference_image_base64: Optional[str] = None,
    reference_images_base64: Optional[List[str]] = None
) -> Optional[list]:
    """
    Generate thumbnail images using Gran Sabio LLM (flat structure, no variations).

    Args:
        transcription: Video transcription text
        video_title: Original video title
        num_images: Number of images to generate
        cluster_description: Optional context about reference images
        config: Optional PromptGenerationConfig
        thumbnail_style: Style guidance string
        selected_titles: Optional list of user-selected titles
        reference_image_base64: Optional single base64-encoded reference image
        reference_images_base64: Optional list of base64-encoded images (max 20)

    Returns:
        List of image dictionaries or None on failure
    """
    if config is None:
        config = PromptGenerationConfig()

    # Determine if we have any reference images
    has_reference = bool(reference_image_base64) or bool(reference_images_base64)

    # Build the prompt using the new flat structure
    prompt = build_analysis_prompt_images(
        transcription=transcription,
        video_title=video_title,
        num_images=num_images,
        cluster_description=cluster_description,
        custom_instructions=config.custom_instructions,
        history_prompts=config.history_prompts,
        include_history=config.include_history,
        thumbnail_style=thumbnail_style,
        selected_titles=selected_titles,
        has_reference_image=has_reference
    )

    # Generate with Gran Sabio LLM
    response = generate_with_gransabio(
        prompt=prompt,
        config=config,
        reference_image_base64=reference_image_base64,
        reference_images_base64=reference_images_base64
    )

    if not response:
        return None

    # Handle response
    try:
        if isinstance(response, list):
            logger.info(f"Gran Sabio LLM generated {len(response)} images successfully")
            return response

        if isinstance(response, dict):
            logger.warning("Gran Sabio LLM returned single image (dict), wrapping in list")
            return [response]

        if isinstance(response, str):
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
                logger.info(f"Gran Sabio LLM generated {len(data)} images successfully")
                return data
            elif isinstance(data, dict):
                logger.warning("Gran Sabio LLM returned single image (parsed dict), wrapping in list")
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

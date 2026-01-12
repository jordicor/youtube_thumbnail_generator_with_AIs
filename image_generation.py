"""
YouTube Thumbnail Generator - Image Generation Module
======================================================
Generates thumbnail images using Gemini (Nano Banana Pro),
OpenAI DALL-E, or Replicate APIs.
"""

import base64
import io
from pathlib import Path
from datetime import datetime
from typing import Optional
from PIL import Image

from config import (
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    POE_API_KEY,
    GEMINI_IMAGE_MODEL,
    GEMINI_PRO_OUTPUT_RESOLUTION,
    OPENAI_IMAGE_MODEL,
    OPENAI_IMAGE_SIZE,
    OPENAI_IMAGE_FORMAT,
    OPENAI_IMAGE_QUALITY,
    POE_IMAGE_MODEL,
    POE_IMAGE_QUALITY,
    POE_IMAGE_ASPECT,
    POE_MODEL_MAX_REFS,
    GEMINI_MODEL_MAX_REFS,
    OPENAI_MODEL_MAX_REFS,
    REPLICATE_MODEL_MAX_REFS,
    THUMBNAIL_WIDTH,
    THUMBNAIL_HEIGHT,
    IMAGE_PROVIDER,
    KEEP_RAW_THUMBNAILS,
    USE_PROMPT_SYSTEM_V3,
)
from utils import setup_logger, VideoOutput
from prompt_generation import (
    ThumbnailPrompt,
    ThumbnailConcept,
    ThumbnailVariation,
    ThumbnailImage,
    enhance_prompt_with_face_description,
    enhance_prompt_with_face_description_from_text,
    get_json_prompt_for_generation
)

logger = setup_logger(__name__)


# =============================================================================
# FILENAME HELPERS
# =============================================================================

def normalize_model_name_for_filename(model_name: str) -> str:
    """Convert model name to filename-safe format (dots to dashes)."""
    return model_name.replace(".", "-")


def save_raw_thumbnail(image_data: bytes, output_path: Path, logger) -> Optional[Path]:
    """
    Save the original (raw) image before resizing if KEEP_RAW_THUMBNAILS is enabled.

    Saves to: thumbnails/raw/{filename_with_raw_prefix}

    Args:
        image_data: Raw image bytes from API
        output_path: The final output path (used to derive raw path)
        logger: Logger instance

    Returns:
        Path to saved raw image or None if not saved
    """
    if not KEEP_RAW_THUMBNAILS:
        return None

    try:
        # Create raw directory
        raw_dir = output_path.parent / "raw"
        raw_dir.mkdir(exist_ok=True)

        # Build raw filename: add _raw_ before timestamp
        # Original: model_timestamp_cX_vY.png
        # Raw: model_raw_timestamp_cX_vY.png
        stem = output_path.stem
        parts = stem.split("_", 1)  # Split at first underscore
        if len(parts) >= 2:
            raw_filename = f"{parts[0]}_raw_{parts[1]}{output_path.suffix}"
        else:
            raw_filename = f"raw_{stem}{output_path.suffix}"

        raw_path = raw_dir / raw_filename

        # Save raw image
        with open(raw_path, 'wb') as f:
            f.write(image_data)

        # Get dimensions for logging
        img = Image.open(raw_path)
        logger.info(f"Raw image saved: {raw_path.name} ({img.width}x{img.height})")

        return raw_path

    except Exception as e:
        logger.warning(f"Could not save raw thumbnail: {e}")
        return None


def get_effective_model(provider: str, gemini_model: str = None,
                        openai_model: str = None, poe_model: str = None) -> str:
    """Get the model name that will be used for a given provider."""
    if provider == "gemini":
        return gemini_model or GEMINI_IMAGE_MODEL
    elif provider == "openai":
        return openai_model or OPENAI_IMAGE_MODEL
    elif provider == "poe":
        return poe_model or POE_IMAGE_MODEL
    elif provider == "replicate":
        return "flux-1-1-pro"
    return "unknown"


# =============================================================================
# V1 PROMPT HELPERS - SUBJECTS SECTION FOR TEXT PROMPTS
# =============================================================================

def build_subjects_section(
    subjects: list[dict],
    face_groups: dict = None,
    style_source: str = None
) -> str:
    """
    Build a subjects identification section for image generation prompts.

    This section tells the image generation AI how to identify each person
    in the reference images, which faces are the same person, and which
    outfit to use for the final image.

    Args:
        subjects: List of subject dictionaries with:
                  id, visual_identifier, inferred_role, face_group,
                  is_costume_variant, variant_of
        face_groups: Dict mapping face_group IDs to member lists, e.g.:
                     {"face_A": {"members": ["person_01", "person_02"], "description": "..."}}
        style_source: Subject ID whose outfit should be used in the final image

    Returns:
        Formatted string section for the image prompt, or empty string if no subjects
    """
    if not subjects:
        return ""

    lines = [
        "═══════════════════════════════════════════════════════════════════════════════",
        "REFERENCE IMAGE IDENTITY MAPPING - CLONE THESE EXACT FACES",
        "═══════════════════════════════════════════════════════════════════════════════",
        "The attached reference images contain REAL PEOPLE. You MUST clone their",
        "EXACT faces from those images. Each person_XX ID maps to specific references.",
        ""
    ]

    # Face groups section - critical for same-person-different-outfit scenarios
    if face_groups:
        lines.append("FACE GROUPS (same person, different outfits - use ALL images to learn face):")
        lines.append("─────────────────────────────────────────────────────────────────────────────")
        for group_id, group_data in face_groups.items():
            if isinstance(group_data, dict):
                members = group_data.get("members", [])
                description = group_data.get("description", "")
            else:
                # Handle case where group_data is just a list
                members = group_data if isinstance(group_data, list) else []
                description = ""

            members_str = ", ".join(members) if members else "unknown"
            lines.append(f"  • {group_id}: [{members_str}]")
            lines.append(f"    → Use ALL reference images of these IDs to learn this ONE face")
            if description:
                lines.append(f"    → {description}")
        lines.append("")

    # Subjects section
    lines.append("SUBJECTS FROM REFERENCE IMAGES:")
    lines.append("─────────────────────────────────────────────────────────────────────────────")

    for s in subjects:
        subject_id = s.get('id', 'unknown')
        visual_id = s.get('visual_identifier', 'no description')
        role = s.get('inferred_role', '')
        face_group = s.get('face_group', '')
        is_variant = s.get('is_costume_variant', False)
        variant_of = s.get('variant_of')
        is_style = (style_source == subject_id) if style_source else False

        # Build the subject line
        line = f"  • {subject_id}"
        if face_group:
            line += f" (face: {face_group})"
        line += f": {visual_id}"

        if role:
            line += f" → {role}"

        lines.append(line)

        # Add annotations
        if is_variant and variant_of:
            lines.append(f"    ↳ SAME FACE as {variant_of} - clone IDENTICAL facial features")
        if is_style:
            lines.append(f"    ↳ ★ USE THIS OUTFIT/STYLING FOR THE FINAL IMAGE ★")

    # Style source explicit mention
    lines.append("")
    if style_source:
        lines.append(f"OUTFIT/STYLING: Use {style_source}'s appearance (clothing, hair style, accessories)")
    else:
        lines.append("OUTFIT/STYLING: Use the first subject's appearance")

    lines.extend([
        "",
        "CRITICAL GENERATION RULES:",
        "─────────────────────────────────────────────────────────────────────────────",
        "1. FACE: Clone the EXACT face from ALL reference images in the same face_group",
        "2. OUTFIT: Use the outfit from the subject marked as style_source",
        "3. VERIFICATION: A friend of this person MUST recognize them INSTANTLY",
        "4. NO MODIFICATIONS: Same nose, eyes, lips, skin tone, body type as references",
        "═══════════════════════════════════════════════════════════════════════════════",
        ""
    ])

    return "\n".join(lines)


# =============================================================================
# V3 PROMPT SYSTEM - OPTIMIZED JSON STRUCTURE (~1,000 chars)
# =============================================================================
# V3 features:
# - Clear face_groups/characters/scene separation (single source of truth)
# - physical_description for facial features in face_groups
# - identify_in_references for cross-references in characters
# - character_directions only for ACTIVE characters in scene
# - No redundancies (face_groups, style_source defined once)
# - Generation constraints and clone rules at the top
# =============================================================================

import re


def _build_clone_rules() -> dict:
    """
    Build static clone rules section for V3 prompt.

    These rules instruct the image AI on how to use reference images
    for face cloning vs outfit cloning. Includes hard restrictions.
    """
    return {
        "CLONE_MODE": "EXACT_FACE_MATCH",
        "face": "Clone from ALL reference images showing this person (multiple angles = better accuracy)",
        "outfit": "Use outfit from style_source character",
        "FORBIDDEN": "generating new faces, using stock faces, modifying facial features",
        "test": "Would a friend recognize them instantly?"
    }


def _build_generation_constraints() -> dict:
    """
    Build hard constraints for face/character generation.

    These constraints MUST appear at the top of the JSON to establish
    the absolute rules before any creative content.
    """
    return {
        "RULE": "CLONE_FACES_FROM_REFERENCES",
        "faces": "ONLY from attached reference images - NO generated/stock faces allowed",
        "matching": "Use physical_description in face_groups to identify each person across ALL reference images",
        "enforcement": "STRICT"
    }


def _build_face_groups_v3(image: ThumbnailImage) -> dict:
    """
    Build face_groups section for V3 prompt.

    Handles both V3 native format (with physical_description) and
    legacy format (with description/members).

    Args:
        image: ThumbnailImage containing face_groups data

    Returns:
        dict mapping group IDs to group data with physical_description
    """
    if not image.face_groups:
        return {}

    result = {}
    for group_id, group_data in image.face_groups.items():
        if isinstance(group_data, dict):
            # V3 format: physical_description + characters_with_this_face
            physical_desc = group_data.get("physical_description", "")
            members = group_data.get("characters_with_this_face") or group_data.get("members", [])
            description = group_data.get("description", "")

            # Fallback: use description if no physical_description
            if not physical_desc and description:
                physical_desc = description
        else:
            # Handle legacy format where group_data is just a list
            members = group_data if isinstance(group_data, list) else []
            physical_desc = ""

        # Build members string for clone instruction
        members_str = ", ".join(members) if members else "unknown"

        result[group_id] = {
            "CLONE_FROM_REFERENCES": True,
            "physical_description": physical_desc or f"Person appearing in references as {members_str}",
            "characters_with_this_face": members,
            "clone_instruction": f"Find this person in attached reference images by matching: {physical_desc or 'visual appearance'}. Clone their EXACT face."
        }

    return result


def _extract_outfit_from_visual_identifier(visual_identifier: str) -> str:
    """
    Extract outfit description from visual_identifier.

    Handles both old format "FROM REFERENCES: outfit description"
    and new format with direct outfit text.
    """
    if not visual_identifier:
        return "outfit from reference"

    # Remove "FROM REFERENCES:" prefix if present
    if visual_identifier.upper().startswith("FROM REFERENCES:"):
        return visual_identifier[16:].strip()

    return visual_identifier


def _build_identify_in_references(subject: dict, face_groups: dict) -> str:
    """
    Build identify_in_references string for a subject.

    Creates a clear instruction on how to find this character in references,
    using cross-references to other characters when they share a face.

    Args:
        subject: Subject dict with id, face_group, visual_identifier, etc.
        face_groups: Dict of all face groups

    Returns:
        String like "Look for: woman matching group_A description, wearing blue dress"
    """
    subject_id = subject.get('id', 'unknown')
    face_group = subject.get('face_group', '')
    visual_id = subject.get('visual_identifier', '')
    is_variant = subject.get('is_costume_variant', False)
    variant_of = subject.get('variant_of')

    # Extract outfit description
    outfit = _extract_outfit_from_visual_identifier(visual_id)

    # If this is a costume variant, reference the original
    if is_variant and variant_of:
        return f"Look for: same person as {variant_of}, but wearing {outfit}"

    # If we have face group info, reference it
    if face_group and face_groups:
        group_data = face_groups.get(face_group, {})
        if isinstance(group_data, dict):
            physical = group_data.get('physical_description', '')
            if physical:
                # Shorten physical description for identify string
                short_physical = physical.split(',')[0] if ',' in physical else physical
                return f"Look for: {short_physical}, {outfit}"

    # Fallback: just describe the outfit
    return f"Look for: person with {outfit}"


def _build_characters_v3(image: ThumbnailImage) -> dict:
    """
    Build characters section for V3 prompt.

    Prefers native V3 format (image.characters dict) if available.
    Falls back to converting legacy subjects array.

    Args:
        image: ThumbnailImage containing characters or subjects data

    Returns:
        dict mapping character IDs to character data
    """
    # V3 native: use characters dict directly if available
    if image.characters:
        return image.characters

    # Fallback: convert legacy subjects array to characters dict
    if not image.subjects:
        return {}

    characters = {}
    for s in image.subjects:
        subject_id = s.get('id', 'unknown')

        # Get outfit from visual_identifier or dedicated outfit field
        outfit = s.get('outfit') or _extract_outfit_from_visual_identifier(
            s.get('visual_identifier', '')
        )

        # Build identify_in_references
        identify = s.get('identify_in_references') or _build_identify_in_references(
            s, image.face_groups or {}
        )

        characters[subject_id] = {
            "belongs_to_face": s.get('face_group', 'group_A'),
            "outfit": outfit,
            "identify_in_references": identify
        }

    return characters


def _extract_active_characters_from_scene(
    scene_text: str,
    subjects: list = None,
    characters: dict = None
) -> list:
    """
    Extract which characters are ACTIVE in the scene description.

    Searches for person_XX IDs in the scene text to determine which
    characters actually appear in this specific scene.

    Args:
        scene_text: Scene description text (e.g., "person_02 sits at desk...")
        subjects: Legacy list of subject dicts (V2 format)
        characters: V3 dict mapping person_XX to character data

    Returns:
        List of character IDs that appear in the scene
    """
    # Collect all character IDs from either format
    all_ids = []
    if characters:
        all_ids = list(characters.keys())
    elif subjects:
        all_ids = [s.get('id', '') for s in subjects if s.get('id')]

    if not scene_text or not all_ids:
        # Fallback: return first character if available
        if all_ids:
            return [all_ids[0]]
        return ['person_01']

    # Find which IDs appear in the scene text
    active = [char_id for char_id in all_ids if char_id in scene_text]

    # If no characters found in scene text, assume first character
    if not active and all_ids:
        active = [all_ids[0]]

    return active


def _build_scene_v3(image: ThumbnailImage) -> dict:
    """
    Build scene section for V3 prompt.

    Creates scene with:
    - description: what's happening (with clone prefixes for each active character)
    - character_directions: pose/expression for ACTIVE characters only (with MUST_CLONE_FACE)
    - environment: setting, props, lighting, effects

    Args:
        image: ThumbnailImage with scene data

    Returns:
        dict with scene structure
    """
    # Get active characters from scene description (supports both V3 and legacy formats)
    active_chars = _extract_active_characters_from_scene(
        scene_text=image.scene or image.image_prompt,
        subjects=image.subjects,
        characters=image.characters
    )

    # Build clone prefix for active characters
    clone_prefixes = []
    for char_id in active_chars:
        clone_prefixes.append(f"[CLONE {char_id} FACE FROM REFERENCES]")
    clone_prefix = " ".join(clone_prefixes) + " " if clone_prefixes else ""

    # Build character_directions only for active characters
    character_directions = {}
    for char_id in active_chars:
        # Find position info if available in scene text
        position = "center frame"  # Default
        scene_lower = (image.scene or "").lower()
        if f"{char_id.lower()} on left" in scene_lower or "left side" in scene_lower:
            position = "left side of frame"
        elif f"{char_id.lower()} on right" in scene_lower or "right side" in scene_lower:
            position = "right side of frame"
        elif "behind" in scene_lower:
            position = "behind desk/table"
        elif "seated" in scene_lower or "sitting" in scene_lower:
            position = "seated"

        character_directions[char_id] = {
            "MUST_CLONE_FACE": True,
            "face_source": "attached_reference_images",
            "position": position,
            "pose": image.subject_pose or "natural engaging pose",
            "expression": image.subject_expression or "engaging expression"
        }

    # Build environment
    environment = {
        "setting": image.background or "professional studio background",
        "props": image.visual_elements or [],
        "lighting": image.lighting or "professional three-point lighting"
    }

    # Add effects if present (V3 field)
    if image.environment_effects:
        environment["effects"] = image.environment_effects

    # Prepend clone prefix to scene description
    scene_description = clone_prefix + (image.scene or image.image_prompt)

    return {
        "description": scene_description,
        "character_directions": character_directions,
        "environment": environment
    }


def build_image_prompt_v3(
    image: ThumbnailImage,
    has_reference_images: bool = False
) -> dict:
    """
    Build a V3 optimized JSON prompt from ThumbnailImage.

    V3 structure with hard constraints for identity preservation:
    - style_directive: Mandatory style from style_analysis (if provided)
    - generation_constraints: HARD restrictions (face source, forbidden actions)
    - clone_rules: Instructions for face/outfit cloning with enforcement
    - face_groups: Real people with physical descriptions
    - characters: Each appearance/outfit with identification
    - scene: What happens, with character_directions for ACTIVE chars only
    - output: Quality and aspect ratio

    Args:
        image: ThumbnailImage with all fields
        has_reference_images: Whether reference images will be provided

    Returns:
        dict ready to serialize to JSON for image generation
    """
    prompt_dict = {}

    # Style directive FIRST - if style_analysis exists, this takes priority
    has_style_analysis = image.style_analysis and isinstance(image.style_analysis, dict)
    if has_style_analysis:
        style = image.style_analysis
        prompt_dict["style_directive"] = {
            "mandatory": True,
            "art_type": style.get("art_type", "photorealistic"),
            "art_type_details": style.get("art_type_details"),
            "color_mood": style.get("color_mood"),
            "visual_effects": style.get("visual_effects"),
            "rendering_style": style.get("rendering_style"),
            "composition_style": style.get("composition_style"),
            "line_work": style.get("line_work"),
            "overall_aesthetic": style.get("overall_aesthetic"),
            "instruction": "Generate the image in THIS style. The FIRST reference image shows the target style. Do NOT default to photorealistic if a different style is specified."
        }

    # Check if we have identity data (V3 characters or legacy subjects)
    has_identity_data = image.characters or image.subjects

    # Only include constraints and identity sections if we have references and identity data
    if has_reference_images and has_identity_data:
        # Generation constraints FIRST - hard restrictions before any creative content
        prompt_dict["generation_constraints"] = _build_generation_constraints()

        # Clone rules with enforcement
        prompt_dict["clone_rules"] = _build_clone_rules()

        # Face groups with physical descriptions
        face_groups = _build_face_groups_v3(image)
        if face_groups:
            prompt_dict["face_groups"] = face_groups

        # Characters (replaces subjects array)
        characters = _build_characters_v3(image)
        if characters:
            prompt_dict["characters"] = characters

    # Scene with character_directions for active characters only
    prompt_dict["scene"] = _build_scene_v3(image)

    # Output settings - adapt quality based on style_analysis if present
    if has_style_analysis:
        # Use style-appropriate quality descriptor
        art_type = image.style_analysis.get("art_type", "")
        overall_aesthetic = image.style_analysis.get("overall_aesthetic", "")
        quality = image.quality or overall_aesthetic or f"high quality {art_type} style"
        prompt_dict["output"] = {
            "quality": quality,
            "aspect_ratio": "16:9",
            "style": art_type  # Reinforce the style in output
        }
    else:
        prompt_dict["output"] = {
            "quality": image.quality or "8K photorealistic YouTube thumbnail",
            "aspect_ratio": "16:9"
        }

    # Optional: text_in_image (different from text_overlay which is for UI)
    if image.text_in_image:
        prompt_dict["text_in_image"] = image.text_in_image

    # Final reinforcement reminder when we have references and identity data
    if has_reference_images and has_identity_data:
        prompt_dict["REMINDER"] = "Every person_XX MUST be cloned from reference images. Search ALL attached images for matching faces."

    return prompt_dict


def _build_face_source_header() -> str:
    """
    Build a mandatory header that establishes face source restrictions.

    This header appears BEFORE the JSON to ensure the image AI reads
    the constraints before processing any creative content.
    """
    return """### FACE CLONING REQUIRED ###
Clone ALL faces from attached reference images ONLY.
NO generated/stock faces allowed - each person_XX must match a reference photo exactly.
A friend must recognize them INSTANTLY.

"""


def build_v3_prompt_string(
    image: ThumbnailImage,
    has_reference_images: bool = False
) -> str:
    """
    Build a V3 JSON prompt string from ThumbnailImage.

    When reference images are provided, includes a header with hard
    constraints before the JSON content.

    Returns:
        String with optional header + JSON content
    """
    import json
    prompt_dict = build_image_prompt_v3(image, has_reference_images)
    json_content = json.dumps(prompt_dict, indent=2, ensure_ascii=False)

    # Add header only when we have references and identity data
    has_identity_data = image.characters or image.subjects
    if has_reference_images and has_identity_data:
        return _build_face_source_header() + json_content

    return json_content


def get_prompt_system_version() -> str:
    """
    Get the currently active prompt system version.

    Returns:
        "V3" if USE_PROMPT_SYSTEM_V3 is True (default)
        "V1" otherwise (legacy fallback)
    """
    if USE_PROMPT_SYSTEM_V3:
        return "V3"
    return "V1"


# =============================================================================
# IMAGE GENERATION PROVIDERS
# =============================================================================

def generate_with_gemini(
    prompt: str,
    reference_images: list[Path] = None,
    output_path: Path = None,
    model: str = None,
    concept: ThumbnailConcept = None,
    variation: ThumbnailVariation = None,
    image: ThumbnailImage = None
) -> Optional[Path]:
    """
    Generate image using Google Gemini API (Nano Banana / Nano Banana Pro).

    Args:
        prompt: Text prompt for image generation (V1) or fallback (V2)
        reference_images: Optional list of reference image paths
        output_path: Where to save the generated image
        model: Gemini model to use
        concept: V2 - ThumbnailConcept with structured fields (legacy)
        variation: V2 - ThumbnailVariation with structured fields (legacy)
        image: New flat ThumbnailImage structure (preferred)

    Returns:
        Path to generated image or None if failed
    """

    if not GEMINI_API_KEY:
        logger.error("Gemini API key not set")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        has_refs = reference_images and len(reference_images) > 0

        # V3: Optimized JSON structure (default when ThumbnailImage available)
        if USE_PROMPT_SYSTEM_V3 and image and image.has_v2_fields():
            logger.info("Using V3 optimized JSON prompt from ThumbnailImage (~1,000 chars)")
            full_prompt = build_v3_prompt_string(image, has_refs)
            logger.debug(f"V3 prompt length: {len(full_prompt)} chars")

        # V1: Legacy text prompt with negations (~3,500 chars)
        elif has_refs:
            # Build subjects section if available (from image or legacy concept)
            subjects_text = ""
            subjects = (image.subjects if image else None) or (concept.subjects if concept else None)
            face_groups = (image.face_groups if image else None) or (getattr(concept, 'face_groups', None) if concept else None)
            style_source = (image.style_source if image else None) or (getattr(concept, 'style_source', None) if concept else None)
            if subjects:
                subjects_text = build_subjects_section(subjects, face_groups, style_source) + "\n"

            full_prompt = f"""{subjects_text}STRICT IDENTITY CLONING - THIS IS THE SAME PERSON, NOT A SIMILAR ONE:
The generated image MUST show an EXACT CLONE of the person(s) in the reference photos.
This is NOT "inspired by" or "similar to" - it must be THE EXACT SAME PERSON(S).

═══════════════════════════════════════════════════════════════
ABSOLUTELY FORBIDDEN - DO NOT DO ANY OF THESE:
═══════════════════════════════════════════════════════════════
❌ Add facial hair (beard, mustache, stubble) if NOT present in reference
❌ Remove facial hair if present in reference
❌ Change body weight or build (no thinner, no heavier, no more muscular)
❌ Modify face shape, jaw line, chin, or cheekbones
❌ Alter nose size, shape, or bridge
❌ Change eye size, shape, spacing, or color
❌ Modify lip size, shape, or fullness
❌ Add or remove glasses/sunglasses
❌ Change hairstyle, hair length, or hair color
❌ Alter skin tone, complexion, or texture
❌ Add wrinkles, scars, moles, or marks not in reference
❌ "Beautify", "idealize", or "enhance" any features
❌ Age the person (older or younger)
❌ Change clothing style or colors from reference

═══════════════════════════════════════════════════════════════
MANDATORY - COPY EXACTLY FROM REFERENCE:
═══════════════════════════════════════════════════════════════
✅ Exact facial proportions (interpupillary distance, nose-to-lip ratio)
✅ Same body type and physical build
✅ Identical skin texture, tone, and any visible features
✅ Same hair (exact style, color, length, texture)
✅ Same facial hair status (if clean-shaven in reference, MUST be clean-shaven)
✅ Same clothing from reference images
✅ Same accessories (glasses, earrings, etc.)

SCENE TO CREATE (uses subject IDs from above if provided):
{prompt}

IMPORTANT: The scene description uses subject IDs (person_01, person_02, etc.) to
indicate which person from the references does what. Match each ID to the
corresponding reference image based on the subject identification above.
Reference images provide identity to preserve; the scene describes actions/poses.

TECHNICAL:
- 16:9 aspect ratio (1280x720)
- Eye-catching and professional
- Bold, contrasting colors
- DO NOT include any text in the image

VERIFICATION: A friend of these people should recognize them INSTANTLY.
"""
        else:
            full_prompt = f"""Create a professional YouTube thumbnail.

{prompt}

Requirements:
- 16:9 aspect ratio (1280x720 pixels)
- Eye-catching and bold design
- Professional YouTube thumbnail style
- DO NOT include any text in the image
"""

        # Use provided model or default
        gemini_model = model or GEMINI_IMAGE_MODEL

        logger.info(f"Generating image with Gemini ({gemini_model})...")

        # Prepare content parts
        contents = []

        # Add reference images if provided
        # Limit depends on model: Gemini 3 Pro = 14, Gemini 2.5 Flash = 3
        if reference_images:
            max_refs = GEMINI_MODEL_MAX_REFS.get(gemini_model, 3)  # Default 3 for Flash
            logger.info(f"Using up to {max_refs} reference images for {gemini_model}")
            for img_path in reference_images[:max_refs]:
                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        if max(img.size) > 1024:
                            img.thumbnail((1024, 1024))

                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format='JPEG', quality=85)
                        img_bytes = img_buffer.getvalue()

                        contents.append(types.Part.from_bytes(
                            data=img_bytes,
                            mime_type="image/jpeg"
                        ))
                        logger.info(f"Added reference image: {img_path.name}")
                    except Exception as e:
                        logger.warning(f"Could not load reference image {img_path}: {e}")

        # Add text prompt
        contents.append(full_prompt)

        # Build generation config
        # Gemini 3 Pro supports image_size and aspect_ratio for higher resolution output
        is_gemini_3_pro = "gemini-3-pro" in gemini_model

        if is_gemini_3_pro:
            gen_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(
                    aspect_ratio="16:9",
                    image_size=GEMINI_PRO_OUTPUT_RESOLUTION,  # "1K", "2K", or "4K"
                ),
            )
            logger.info(f"Using Gemini 3 Pro with {GEMINI_PRO_OUTPUT_RESOLUTION} resolution (16:9)")
        else:
            # Standard config for other models (gemini-2.5-flash-image, etc.)
            gen_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            )

        # Generate with image output modality
        response = client.models.generate_content(
            model=gemini_model,
            contents=contents,
            config=gen_config,
        )

        # Extract image from response
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            if part.inline_data.mime_type.startswith('image/'):
                                image_data = part.inline_data.data

                                # Save image
                                if output_path:
                                    # Save raw image first (if enabled)
                                    save_raw_thumbnail(image_data, output_path, logger)

                                    # Load image for resizing
                                    img = Image.open(io.BytesIO(image_data))

                                    # Resize to exact thumbnail dimensions
                                    img = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)
                                    img.save(output_path, 'PNG', quality=95)

                                    logger.success(f"Image saved to: {output_path} ({THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT})")
                                    return output_path

        logger.warning("No image in Gemini response")
        return None

    except ImportError:
        logger.error("google-genai package not installed. Install with: pip install google-genai")
        return None
    except Exception as e:
        logger.error(f"Gemini image generation failed: {e}")
        return None


def generate_with_openai(
    prompt: str,
    reference_images: list[Path] = None,
    output_path: Path = None,
    model: str = None,
    concept: ThumbnailConcept = None,
    variation: ThumbnailVariation = None,
    image: ThumbnailImage = None
) -> Optional[Path]:
    """
    Generate image using OpenAI GPT Image or DALL-E.

    Supports:
    - gpt-image-1.5 (best quality, recommended) - up to 16 reference images
    - gpt-image-1 (standard) - up to 16 reference images
    - gpt-image-1-mini (faster, lower cost) - up to 16 reference images
    - dall-e-3 (legacy, deprecated) - NO reference support

    Args:
        prompt: Text prompt for image generation (V1) or fallback (V2)
        reference_images: Reference images are passed as base64 encoded images
        output_path: Where to save the generated image
        model: OpenAI model to use
        concept: V2 - ThumbnailConcept with structured fields (legacy)
        variation: V2 - ThumbnailVariation with structured fields (legacy)
        image: New flat ThumbnailImage structure (preferred)
    """

    if not OPENAI_API_KEY:
        logger.error("OpenAI API key not set")
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)

        # Use provided model or default from config
        selected_model = model or OPENAI_IMAGE_MODEL
        is_gpt_image = selected_model.startswith("gpt-image")

        # Check reference image support for this model
        max_refs = OPENAI_MODEL_MAX_REFS.get(selected_model, 0)
        has_refs = reference_images and len(reference_images) > 0 and max_refs > 0

        if reference_images and max_refs == 0:
            logger.warning(
                f"{selected_model} does NOT support reference images. "
                f"The {len(reference_images)} provided reference image(s) will be IGNORED."
            )

        logger.info(f"Generating image with OpenAI ({selected_model})...")
        if has_refs:
            logger.info(f"Using up to {max_refs} reference images")

        # V3: Optimized JSON structure (default when ThumbnailImage available)
        if USE_PROMPT_SYSTEM_V3 and image and image.has_v2_fields():
            logger.info("Using V3 optimized JSON prompt from ThumbnailImage (~1,000 chars)")
            full_prompt = build_v3_prompt_string(image, has_refs)
            logger.debug(f"V3 prompt length: {len(full_prompt)} chars")

        # V1: Legacy text prompt
        elif has_refs:
            # Build subjects section if available (from image or legacy concept)
            subjects_text = ""
            subjects = (image.subjects if image else None) or (concept.subjects if concept else None)
            face_groups = (image.face_groups if image else None) or (getattr(concept, 'face_groups', None) if concept else None)
            style_source = (image.style_source if image else None) or (getattr(concept, 'style_source', None) if concept else None)
            if subjects:
                subjects_text = build_subjects_section(subjects, face_groups, style_source) + "\n"

            # Build prompt with reference image instructions
            ref_count = min(len(reference_images), max_refs)
            full_prompt = f"""{subjects_text}STRICT IDENTITY CLONING from the {ref_count} reference image(s) provided.
The generated image MUST show the EXACT SAME PERSON(S) from the reference photos.

Professional YouTube video thumbnail.

SCENE (uses subject IDs if provided above):
{prompt}

MANDATORY - COPY EXACTLY FROM REFERENCE:
- Exact facial features and proportions for each person
- Same hair style, color, and length
- Same skin tone and texture
- Same body type
- Costume variants (same person in different outfit) = IDENTICAL face

Style: Bold, eye-catching YouTube thumbnail with high contrast colors.
Do NOT include any text or letters in the image.
Photorealistic style, professional quality.

VERIFICATION: A friend of these people should recognize them INSTANTLY.
"""
        else:
            full_prompt = f"""Professional YouTube video thumbnail.

{prompt}

Style: Bold, eye-catching YouTube thumbnail with high contrast colors.
Do NOT include any text or letters in the image.
Photorealistic style, professional quality.
"""

        if is_gpt_image:
            # GPT Image models (gpt-image-1, gpt-image-1.5, gpt-image-1-mini)
            response = None

            if has_refs:
                # Use images.edit endpoint with reference images
                # images.edit accepts an array of images (up to 16 for gpt-image models)
                ref_image_files = []
                for img_path in reference_images[:max_refs]:
                    if img_path.exists():
                        try:
                            # Load and resize image for efficiency
                            img = Image.open(img_path)
                            if max(img.size) > 1024:
                                img.thumbnail((1024, 1024))

                            # Convert to PNG bytes (required format)
                            img_buffer = io.BytesIO()
                            img.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            ref_image_files.append(img_buffer)
                            logger.info(f"Added reference image: {img_path.name}")
                        except Exception as e:
                            logger.warning(f"Could not load reference image {img_path}: {e}")

                if ref_image_files:
                    # Use images.edit with reference images
                    # input_fidelity="high" preserves facial features better (not supported on mini)
                    # Pass as tuples with filename and mime type for proper detection
                    image_tuples = [
                        (f"ref_{i}.png", img_buffer, "image/png")
                        for i, img_buffer in enumerate(ref_image_files)
                    ]

                    # Build edit params - input_fidelity only supported on gpt-image-1 and gpt-image-1.5
                    edit_params = {
                        "model": selected_model,
                        "image": image_tuples if len(image_tuples) > 1 else image_tuples[0],
                        "prompt": full_prompt,
                        "size": OPENAI_IMAGE_SIZE,
                    }
                    if selected_model in {"gpt-image-1", "gpt-image-1.5"}:
                        edit_params["input_fidelity"] = "high"

                    response = client.images.edit(**edit_params)

            # Fallback to standard generation if no refs or refs failed to load
            if response is None:
                response = client.images.generate(
                    model=selected_model,
                    prompt=full_prompt,
                    size=OPENAI_IMAGE_SIZE,
                    quality=OPENAI_IMAGE_QUALITY,
                    n=1,
                )

            # Extract image from response
            if response.data and len(response.data) > 0:
                image_data = response.data[0]

                # GPT Image returns b64_json by default
                if hasattr(image_data, 'b64_json') and image_data.b64_json:
                    img_bytes = base64.b64decode(image_data.b64_json)
                elif hasattr(image_data, 'url') and image_data.url:
                    # Fallback to URL if available
                    import requests
                    img_response = requests.get(image_data.url)
                    if img_response.status_code != 200:
                        logger.error("Failed to download image from URL")
                        return None
                    img_bytes = img_response.content
                else:
                    logger.error("No image data in response")
                    return None

                # Save raw image first (if enabled)
                save_raw_thumbnail(img_bytes, output_path, logger)

                # Load and resize to exact thumbnail dimensions
                img = Image.open(io.BytesIO(img_bytes))
                img = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)

                # Save with configured format
                save_format = OPENAI_IMAGE_FORMAT.upper()
                if save_format == "JPG":
                    save_format = "JPEG"

                img.save(output_path, save_format, quality=95)
                logger.success(f"Image saved to: {output_path} ({THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT})")
                return output_path

        else:
            # Legacy DALL-E 3 (deprecated but still supported until 2026)
            response = client.images.generate(
                model=selected_model,
                prompt=full_prompt[:4000],  # DALL-E 3 has a 4000 char limit
                size="1792x1024",  # Closest to 16:9
                quality="hd",
                n=1,
            )

            if response.data and len(response.data) > 0:
                image_url = response.data[0].url

                # Download image
                import requests
                img_response = requests.get(image_url)

                if img_response.status_code == 200:
                    img_bytes = img_response.content

                    # Save raw image first (if enabled)
                    save_raw_thumbnail(img_bytes, output_path, logger)

                    # Load and resize to exact thumbnail dimensions
                    img = Image.open(io.BytesIO(img_bytes))
                    img = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)

                    # Save
                    img.save(output_path, 'PNG', quality=95)
                    logger.success(f"Image saved to: {output_path} ({THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT})")
                    return output_path

        return None

    except Exception as e:
        logger.error(f"OpenAI image generation failed: {e}")
        return None


def generate_with_replicate(
    prompt: str,
    reference_images: list[Path] = None,
    output_path: Path = None,
    model: str = None
) -> Optional[Path]:
    """
    Generate image using Replicate API (FLUX, SDXL, etc.).

    Supports face reference via image_prompt (FLUX) or face-to-many.
    Reference limits per model:
    - flux-1.1-pro: 1 image (via image_prompt)
    - face-to-many: 1 face image
    """

    try:
        import replicate
        import requests

        # Use provided model or default
        selected_model = model or "black-forest-labs/flux-1.1-pro"
        model_short_name = selected_model.split("/")[-1] if "/" in selected_model else selected_model

        # Get max refs for this model
        max_refs = REPLICATE_MODEL_MAX_REFS.get(model_short_name, 1)

        logger.info(f"Generating image with Replicate ({selected_model})...")

        # Prepare input
        input_data = {
            "prompt": f"Professional YouTube thumbnail. {prompt}. Bold colors, eye-catching, 16:9 aspect ratio.",
            "aspect_ratio": "16:9",
            "output_format": "png",
            "output_quality": 95,
        }

        # If we have reference images, use them according to model limits
        if reference_images and len(reference_images) > 0:
            ref_img = reference_images[0]  # Most Replicate models only support 1 reference
            if ref_img.exists():
                try:
                    logger.info(f"Using {min(len(reference_images), max_refs)} reference image(s) for {model_short_name}")

                    # Resize image for efficiency
                    img = Image.open(ref_img)
                    if max(img.size) > 1024:
                        img.thumbnail((1024, 1024))
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='JPEG', quality=85)
                    img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

                    if "face-to-many" in selected_model:
                        # Use face-to-many for stylized face transformations
                        input_data = {
                            "image": f"data:image/jpeg;base64,{img_data}",
                            "prompt": prompt,
                            "style": "Video game",  # or "3D", "Emoji", etc.
                            "negative_prompt": "blurry, low quality",
                        }
                    else:
                        # FLUX 1.1 Pro uses image_prompt for reference
                        input_data["image_prompt"] = f"data:image/jpeg;base64,{img_data}"
                        logger.info(f"Added reference image: {ref_img.name}")

                except Exception as e:
                    logger.warning(f"Could not use face reference: {e}")

        output = replicate.run(selected_model, input=input_data)

        # Handle output (can be URL or list)
        if isinstance(output, list):
            image_url = output[0] if output else None
        else:
            image_url = output

        if image_url:
            # Download and save
            response = requests.get(image_url)
            if response.status_code == 200:
                img_bytes = response.content

                # Save raw image first (if enabled)
                save_raw_thumbnail(img_bytes, output_path, logger)

                # Load and resize
                img = Image.open(io.BytesIO(img_bytes))
                img = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)
                img.save(output_path, 'PNG', quality=95)
                logger.success(f"Image saved to: {output_path} ({THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT})")
                return output_path

        return None

    except ImportError:
        logger.error("replicate package not installed")
        return None
    except Exception as e:
        logger.error(f"Replicate image generation failed: {e}")
        return None


def generate_with_poe(
    prompt: str,
    reference_images: list[Path] = None,
    output_path: Path = None,
    model: str = None,
    concept: ThumbnailConcept = None,
    variation: ThumbnailVariation = None,
    image: ThumbnailImage = None
) -> Optional[Path]:
    """
    Generate image using Poe API with FLUX/Ideogram models.

    Supports reference images for character/face consistency.
    Available models (bot names verified 2025-12-28):
        - flux2pro: Best quality, up to 8 references
        - flux2flex: High resolution (14MP)
        - fluxkontextpro: Best prompt following
        - seedream40: Good for combining references
        - nanobananapro: Gemini 3 Pro via Poe
        - Ideogram-v3: Best for text/logos

    Args:
        prompt: Text prompt for image generation (V1) or fallback (V2)
        reference_images: Optional list of reference image paths (up to 3)
        output_path: Where to save the generated image
        model: Model to use (default from config: flux2pro)
        concept: V2 - ThumbnailConcept with structured fields (legacy)
        variation: V2 - ThumbnailVariation with structured fields (legacy)
        image: New flat ThumbnailImage structure (preferred)

    Returns:
        Path to generated image or None if failed
    """

    if not POE_API_KEY:
        logger.error("Poe API key not set")
        return None

    try:
        from openai import OpenAI
        import requests

        # Initialize Poe client (OpenAI-compatible)
        client = OpenAI(
            api_key=POE_API_KEY,
            base_url="https://api.poe.com/v1",
        )

        selected_model = model or POE_IMAGE_MODEL
        logger.info(f"Generating image with Poe ({selected_model})...")

        # Build message content
        content = []

        # Add reference images
        # Each model has different limits (nanobananapro: 14, fluxkontextpro: 1, etc.)
        has_refs = reference_images and len(reference_images) > 0
        if reference_images:
            max_images = POE_MODEL_MAX_REFS.get(selected_model, 8)
            for img_path in reference_images[:max_images]:
                if img_path.exists():
                    try:
                        # Load and resize image for efficiency
                        img = Image.open(img_path)
                        if max(img.size) > 1024:
                            img.thumbnail((1024, 1024))

                        # Convert to base64
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format='JPEG', quality=85)
                        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        })
                        logger.info(f"Added reference image: {img_path.name}")
                    except Exception as e:
                        logger.warning(f"Could not load reference image {img_path}: {e}")

        # V3: Optimized JSON structure (default when ThumbnailImage available)
        if USE_PROMPT_SYSTEM_V3 and image and image.has_v2_fields():
            logger.info("Using V3 optimized JSON prompt from ThumbnailImage (~1,000 chars)")
            full_prompt = build_v3_prompt_string(image, has_refs)
            logger.debug(f"V3 prompt length: {len(full_prompt)} chars")

        # V1: Legacy text prompt with negations (~3,500 chars)
        elif has_refs:
            # Build subjects section if available (from image or legacy concept)
            subjects_text = ""
            subjects = (image.subjects if image else None) or (concept.subjects if concept else None)
            face_groups = (image.face_groups if image else None) or (getattr(concept, 'face_groups', None) if concept else None)
            style_source = (image.style_source if image else None) or (getattr(concept, 'style_source', None) if concept else None)
            if subjects:
                subjects_text = build_subjects_section(subjects, face_groups, style_source) + "\n"

            full_prompt = f"""{subjects_text}STRICT IDENTITY CLONING - THIS IS THE SAME PERSON, NOT A SIMILAR ONE:
The generated image MUST show an EXACT CLONE of the person(s) in the reference photos.
This is NOT "inspired by" or "similar to" - it must be THE EXACT SAME PERSON(S).

═══════════════════════════════════════════════════════════════
ABSOLUTELY FORBIDDEN - DO NOT DO ANY OF THESE:
═══════════════════════════════════════════════════════════════
❌ Add facial hair (beard, mustache, stubble) if NOT present in reference
❌ Remove facial hair if present in reference
❌ Change body weight or build (no thinner, no heavier, no more muscular)
❌ Modify face shape, jaw line, chin, or cheekbones
❌ Alter nose size, shape, or bridge
❌ Change eye size, shape, spacing, or color
❌ Modify lip size, shape, or fullness
❌ Add or remove glasses/sunglasses
❌ Change hairstyle, hair length, or hair color
❌ Alter skin tone, complexion, or texture
❌ Add wrinkles, scars, moles, or marks not in reference
❌ "Beautify", "idealize", or "enhance" any features
❌ Age the person (older or younger)
❌ Change clothing style or colors from reference

═══════════════════════════════════════════════════════════════
MANDATORY - COPY EXACTLY FROM REFERENCE:
═══════════════════════════════════════════════════════════════
✅ Exact facial proportions (interpupillary distance, nose-to-lip ratio)
✅ Same body type and physical build
✅ Identical skin texture, tone, and any visible features
✅ Same hair (exact style, color, length, texture)
✅ Same facial hair status (if clean-shaven in reference, MUST be clean-shaven)
✅ Same clothing from reference images
✅ Same accessories (glasses, earrings, etc.)

SCENE TO CREATE (uses subject IDs from above if provided):
{prompt}

IMPORTANT: The scene description uses subject IDs (person_01, person_02, etc.) to
indicate which person from the references does what. Match each ID to the
corresponding reference image based on the subject identification above.

TECHNICAL:
- YouTube thumbnail style: bold, eye-catching, professional
- 16:9 aspect ratio (1280x720)
- High contrast colors
- DO NOT include any text in the image

VERIFICATION: A friend of these people should recognize them INSTANTLY."""
        else:
            full_prompt = f"""Create a professional YouTube thumbnail.

{prompt}

Requirements:
- 16:9 aspect ratio (1280x720)
- Bold, eye-catching design
- Professional YouTube thumbnail style
- DO NOT include any text in the image"""

        content.append({
            "type": "text",
            "text": full_prompt
        })

        # Make API call (stream=False is required for image models)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": content}],
            stream=False,
            extra_body={
                "aspect": POE_IMAGE_ASPECT,
                "quality": POE_IMAGE_QUALITY
            }
        )

        # Extract image from response
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message

            # Check for image in content
            if hasattr(message, 'content') and message.content:
                response_content = message.content

                # Handle different response formats
                image_url = None

                # If response is a list of content blocks
                if isinstance(response_content, list):
                    for block in response_content:
                        if isinstance(block, dict):
                            if block.get('type') == 'image_url':
                                image_url = block.get('image_url', {}).get('url')
                            elif block.get('type') == 'image':
                                image_url = block.get('url') or block.get('image_url')
                        elif hasattr(block, 'image_url'):
                            image_url = block.image_url.url if hasattr(block.image_url, 'url') else block.image_url

                # If response contains markdown image
                elif isinstance(response_content, str):
                    import re
                    # Look for markdown image: ![...](URL)
                    match = re.search(r'!\[.*?\]\((https?://[^\)]+)\)', response_content)
                    if match:
                        image_url = match.group(1)
                    # Look for direct URL
                    elif response_content.startswith('http'):
                        image_url = response_content.strip()

                # Download and save image
                if image_url:
                    if image_url.startswith('data:image'):
                        # Base64 encoded image
                        header, data = image_url.split(',', 1)
                        img_bytes = base64.b64decode(data)
                    else:
                        # URL - download image
                        img_response = requests.get(image_url, timeout=60)
                        if img_response.status_code != 200:
                            logger.error(f"Failed to download image: HTTP {img_response.status_code}")
                            return None
                        img_bytes = img_response.content

                    # Save raw image first (if enabled)
                    save_raw_thumbnail(img_bytes, output_path, logger)

                    # Load and resize to exact thumbnail dimensions
                    img = Image.open(io.BytesIO(img_bytes))
                    img = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)

                    # Save
                    img.save(output_path, 'PNG', quality=95)
                    logger.success(f"Image saved to: {output_path} ({THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT})")
                    return output_path

                else:
                    logger.warning(f"No image URL found in Poe response. Content: {response_content[:500] if response_content else 'None'}")

        logger.warning("No image in Poe response")
        return None

    except ImportError:
        logger.error("openai package not installed. Install with: pip install openai")
        return None
    except Exception as e:
        logger.error(f"Poe image generation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


# =============================================================================
# COMPOSITE THUMBNAIL CREATION
# =============================================================================

def create_composite_thumbnail(
    background_prompt: str,
    face_frame: Path,
    text_overlay: str,
    output_path: Path,
    colors: list[str] = None
) -> Optional[Path]:
    """
    Create a composite thumbnail with:
    - AI-generated or solid background
    - Real face/frame overlaid
    - Text overlay

    This is an alternative approach that keeps your real face.
    """

    try:
        from PIL import Image, ImageDraw, ImageFont

        logger.info("Creating composite thumbnail...")

        # Create base canvas
        thumbnail = Image.new('RGB', (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), '#1a1a2e')

        # Load face frame
        if face_frame and face_frame.exists():
            face_img = Image.open(face_frame)

            # Resize to fit nicely (about 60% of thumbnail height)
            face_height = int(THUMBNAIL_HEIGHT * 0.8)
            face_ratio = face_img.width / face_img.height
            face_width = int(face_height * face_ratio)

            face_img = face_img.resize((face_width, face_height), Image.Resampling.LANCZOS)

            # Position on right side
            x_pos = THUMBNAIL_WIDTH - face_width - 20
            y_pos = (THUMBNAIL_HEIGHT - face_height) // 2

            thumbnail.paste(face_img, (x_pos, y_pos))

        # Add gradient overlay on left for text area
        draw = ImageDraw.Draw(thumbnail)

        # Simple gradient effect
        primary_color = colors[0] if colors else '#FF0000'
        for i in range(THUMBNAIL_WIDTH // 2):
            alpha = int(255 * (1 - i / (THUMBNAIL_WIDTH // 2)))
            # This is simplified - real gradient would need alpha compositing

        # Add text overlay
        if text_overlay:
            try:
                # Try to use a bold font
                font_size = 80
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()

                # Text position (left side, vertically centered)
                text_x = 40
                text_y = THUMBNAIL_HEIGHT // 2 - font_size

                # Draw text with shadow for readability
                shadow_offset = 3
                draw.text((text_x + shadow_offset, text_y + shadow_offset),
                         text_overlay.upper(), fill='#000000', font=font)
                draw.text((text_x, text_y),
                         text_overlay.upper(), fill='#FFFFFF', font=font)

            except Exception as e:
                logger.warning(f"Could not add text overlay: {e}")

        # Save
        thumbnail.save(output_path, 'PNG', quality=95)
        logger.success(f"Composite thumbnail saved to: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Composite thumbnail creation failed: {e}")
        return None


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_thumbnails(
    concepts: list[ThumbnailConcept],
    best_frames: list[Path],
    output: VideoOutput,
    use_composite: bool = False,
    gemini_model: str = None,
    openai_model: str = None,
    poe_model: str = None,
    image_provider: str = None,
    progress_callback: callable = None,
    has_external_style_ref: bool = False
) -> list[Path]:
    """
    Generate thumbnails from concepts with variations.

    Args:
        concepts: List of ThumbnailConcept objects, each containing variations
        best_frames: List of best frame paths for reference
        output: VideoOutput instance
        use_composite: If True, create composite instead of full AI generation
        gemini_model: Override Gemini model (e.g., "gemini-3-pro-image-preview")
        openai_model: Override OpenAI model (e.g., "gpt-image-1.5")
        poe_model: Override Poe model (e.g., "flux2pro", "fluxkontextpro", "Ideogram-v3")
        image_provider: Override image provider ("gemini", "openai", "replicate", "poe")
        progress_callback: Optional callback function(current, total, thumbnail_info) to report progress
                          thumbnail_info is a dict with: path, concept_index, variation_index, concept, variation
        has_external_style_ref: If True, first reference image is an external style reference (not a person)

    Returns:
        List of paths to generated thumbnails
    """

    thumbnails_dir = output.output_dir / "thumbnails"
    thumbnails_dir.mkdir(exist_ok=True)

    # Calculate total images from concepts and their variations
    total_images = sum(len(concept.variations) for concept in concepts)
    num_concepts = len(concepts)
    logger.info(f"Generating {total_images} thumbnails ({num_concepts} concepts with variations)")

    generated_thumbnails = []
    image_counter = 0

    for concept in concepts:
        concept_idx = concept.concept_index
        logger.info(f"\n[Concept {concept_idx}/{num_concepts}] {concept.concept_name}: {concept.suggested_title}")

        for variation in concept.variations:
            variation_idx = variation.variation_index
            image_counter += 1
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

            # Determine model prefix for filename
            selected_provider = image_provider or IMAGE_PROVIDER
            effective_model = get_effective_model(
                selected_provider, gemini_model, openai_model, poe_model
            )
            model_prefix = normalize_model_name_for_filename(effective_model)

            output_path = thumbnails_dir / f"{model_prefix}_{timestamp}_c{concept_idx}_v{variation_idx}.png"

            logger.info(f"  Variation {variation_idx} [{variation.variation_focus}] ({image_counter}/{total_images})...")

            # Option 1: Composite thumbnail
            if use_composite and best_frames:
                composite_path = thumbnails_dir / f"composite_{timestamp}_c{concept_idx}_v{variation_idx}.png"
                result = create_composite_thumbnail(
                    background_prompt=variation.image_prompt,
                    face_frame=best_frames[0],
                    text_overlay=variation.text_overlay,
                    output_path=composite_path,
                    colors=concept.colors
                )
                if result:
                    generated_thumbnails.append(result)
                    if progress_callback:
                        progress_callback(image_counter, total_images, {
                            'path': result,
                            'concept_index': concept_idx,
                            'variation_index': variation_idx,
                            'concept': concept,
                            'variation': variation
                        })
                else:
                    if progress_callback:
                        progress_callback(image_counter, total_images, None)
                continue

            # Option 2: Full AI generation
            # Each variation has its own unique image_prompt from the LLM
            enhanced_prompt = enhance_prompt_with_face_description_from_text(
                variation.image_prompt,
                best_frames[0] if best_frames else None
            )

            # Add note about external style reference if present
            if has_external_style_ref and best_frames:
                style_ref_note = """
═══════════════════════════════════════════════════════════════════════════════
STYLE REFERENCE (INSPIRATION ONLY - DO NOT COPY EXACTLY):
═══════════════════════════════════════════════════════════════════════════════
The FIRST reference image is for STYLE INSPIRATION only:
- Use it as a loose guide for colors, mood, or aesthetic if relevant
- The SCENE DESCRIPTION BELOW takes PRIORITY over the style reference
- Do NOT replicate the style reference exactly - use it as inspiration only
- Do NOT try to clone any person from the style reference

PRIORITY ORDER:
1. HIGHEST: The scene description below (image prompt)
2. SECOND: Identity of the person from reference images 2 onwards
3. LOWEST: Style inspiration from first reference (use loosely)

The remaining reference images show the actual person who must appear in the
thumbnail. Clone their identity ONLY from reference images 2 onwards.
═══════════════════════════════════════════════════════════════════════════════

"""
                enhanced_prompt = style_ref_note + enhanced_prompt

            # Try configured provider
            selected_provider = image_provider or IMAGE_PROVIDER

            generators = {
                "gemini": generate_with_gemini,
                "openai": generate_with_openai,
                "replicate": generate_with_replicate,
                "poe": generate_with_poe,
            }

            success = False

            # Try primary provider
            primary_generator = generators.get(selected_provider)
            if primary_generator:
                kwargs = {
                    "prompt": enhanced_prompt,
                    "reference_images": best_frames[:14] if best_frames else None,
                    "output_path": output_path,
                    # V2: Pass concept and variation for structured prompts
                    "concept": concept,
                    "variation": variation
                }
                if selected_provider == "gemini" and gemini_model:
                    kwargs["model"] = gemini_model
                elif selected_provider == "openai" and openai_model:
                    kwargs["model"] = openai_model
                elif selected_provider == "poe" and poe_model:
                    kwargs["model"] = poe_model
                # Replicate doesn't support V2 yet, remove concept/variation
                if selected_provider == "replicate":
                    kwargs.pop("concept", None)
                    kwargs.pop("variation", None)

                result = primary_generator(**kwargs)
                if result:
                    generated_thumbnails.append(result)
                    success = True

            # Try fallbacks if primary failed
            if not success:
                for provider, generator in generators.items():
                    if provider != selected_provider:
                        logger.info(f"    Trying fallback provider: {provider}")

                        fallback_model = get_effective_model(
                            provider, gemini_model, openai_model, poe_model
                        )
                        fallback_prefix = normalize_model_name_for_filename(fallback_model)
                        fallback_path = thumbnails_dir / f"{fallback_prefix}_{timestamp}_c{concept_idx}_v{variation_idx}.png"

                        kwargs = {
                            "prompt": enhanced_prompt,
                            "reference_images": best_frames[:14] if best_frames else None,
                            "output_path": fallback_path,
                            # V2: Pass concept and variation for structured prompts
                            "concept": concept,
                            "variation": variation
                        }
                        if provider == "gemini" and gemini_model:
                            kwargs["model"] = gemini_model
                        elif provider == "openai" and openai_model:
                            kwargs["model"] = openai_model
                        elif provider == "poe" and poe_model:
                            kwargs["model"] = poe_model
                        # Replicate doesn't support V2 yet, remove concept/variation
                        if provider == "replicate":
                            kwargs.pop("concept", None)
                            kwargs.pop("variation", None)

                        result = generator(**kwargs)
                        if result:
                            generated_thumbnails.append(result)
                            success = True
                            break

            # Last resort: composite
            if not success and best_frames:
                logger.warning("    All AI providers failed, creating composite")
                composite_path = thumbnails_dir / f"composite_{timestamp}_c{concept_idx}_v{variation_idx}.png"
                result = create_composite_thumbnail(
                    background_prompt=variation.image_prompt,
                    face_frame=best_frames[0],
                    text_overlay=variation.text_overlay,
                    output_path=composite_path,
                    colors=concept.colors
                )
                if result:
                    generated_thumbnails.append(result)
                    success = True

            # Report progress after each image
            if progress_callback:
                last_thumbnail = generated_thumbnails[-1] if success and generated_thumbnails else None
                thumbnail_info = {
                    'path': last_thumbnail,
                    'concept_index': concept_idx,
                    'variation_index': variation_idx,
                    'concept': concept,
                    'variation': variation
                } if last_thumbnail else None
                progress_callback(image_counter, total_images, thumbnail_info)

    logger.success(f"Generated {len(generated_thumbnails)}/{total_images} thumbnails")
    logger.info(f"Thumbnails saved to: {thumbnails_dir}")

    return generated_thumbnails


# =============================================================================
# NEW IMAGE-BASED GENERATION (flat structure, no variations)
# =============================================================================

def generate_thumbnails_from_images(
    images: list[ThumbnailImage],
    best_frames: list[Path],
    output: VideoOutput,
    use_composite: bool = False,
    gemini_model: str = None,
    openai_model: str = None,
    poe_model: str = None,
    image_provider: str = None,
    progress_callback: callable = None,
    has_external_style_ref: bool = False,
    cancellation_check: callable = None
) -> list[Path]:
    """
    Generate thumbnails from a flat list of ThumbnailImage objects.

    This is the new simplified function that replaces generate_thumbnails().
    Each ThumbnailImage is independent - no concept/variation hierarchy.

    Args:
        images: List of ThumbnailImage objects
        best_frames: List of best frame paths for reference
        output: VideoOutput instance
        use_composite: If True, create composite instead of full AI generation
        gemini_model: Override Gemini model
        openai_model: Override OpenAI model
        poe_model: Override Poe model
        image_provider: Override image provider ("gemini", "openai", "replicate", "poe")
        progress_callback: Optional callback function(current, total, thumbnail_info)
                          thumbnail_info is a dict with: path, image_index, image
        has_external_style_ref: If True, first reference image is an external style reference
        cancellation_check: Optional callback that returns True if generation should be cancelled.
                           Checked before each image generation to allow early abort.

    Returns:
        List of paths to generated thumbnails (may be partial if cancelled)
    """

    thumbnails_dir = output.output_dir / "thumbnails"
    thumbnails_dir.mkdir(exist_ok=True)

    total_images = len(images)
    logger.info(f"Generating {total_images} thumbnails")

    generated_thumbnails = []

    for img in images:
        # Check for cancellation before starting each image
        if cancellation_check and cancellation_check():
            logger.info(f"Generation cancelled, returning {len(generated_thumbnails)} thumbnails generated so far")
            break

        image_idx = img.image_index
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Determine model prefix for filename
        selected_provider = image_provider or IMAGE_PROVIDER
        effective_model = get_effective_model(
            selected_provider, gemini_model, openai_model, poe_model
        )
        model_prefix = normalize_model_name_for_filename(effective_model)

        # Simplified filename: model_timestamp_N.png
        output_path = thumbnails_dir / f"{model_prefix}_{timestamp}_{image_idx}.png"

        logger.info(f"\n[Image {image_idx}/{total_images}] {img.concept_name}: {img.suggested_title}")

        # Option 1: Composite thumbnail
        if use_composite and best_frames:
            composite_path = thumbnails_dir / f"composite_{timestamp}_{image_idx}.png"
            result = create_composite_thumbnail(
                background_prompt=img.image_prompt,
                face_frame=best_frames[0],
                text_overlay=img.text_overlay,
                output_path=composite_path,
                colors=img.colors
            )
            if result:
                generated_thumbnails.append(result)
                if progress_callback:
                    progress_callback(image_idx, total_images, {
                        'path': result,
                        'image_index': image_idx,
                        'image': img
                    })
            else:
                if progress_callback:
                    progress_callback(image_idx, total_images, None)
            continue

        # Option 2: Full AI generation
        enhanced_prompt = enhance_prompt_with_face_description_from_text(
            img.image_prompt,
            best_frames[0] if best_frames else None
        )

        # Add note about external style reference if present
        if has_external_style_ref and best_frames:
            # Check if we have style_analysis from the ThumbnailImage
            if img.style_analysis and isinstance(img.style_analysis, dict):
                # Build a SPECIFIC style directive with analyzed characteristics
                sa = img.style_analysis
                art_type = sa.get("art_type", "unknown style")
                art_details = sa.get("art_type_details", "")
                color_mood = sa.get("color_mood", "")
                visual_effects = sa.get("visual_effects", [])
                rendering_style = sa.get("rendering_style", "")
                overall_aesthetic = sa.get("overall_aesthetic", "")

                effects_str = ", ".join(visual_effects) if isinstance(visual_effects, list) else str(visual_effects or "none")

                style_ref_note = f"""
═══════════════════════════════════════════════════════════════════════════════
MANDATORY STYLE DIRECTIVE - GENERATE IN THIS SPECIFIC STYLE:
═══════════════════════════════════════════════════════════════════════════════
The FIRST reference image defines the target visual style. You MUST generate
the image in this style:

  ART TYPE: {art_type}
  DETAILS: {art_details}
  COLOR MOOD: {color_mood}
  VISUAL EFFECTS: {effects_str}
  RENDERING: {rendering_style}
  OVERALL: {overall_aesthetic}

CRITICAL INSTRUCTIONS:
1. DO NOT generate photorealistic if the style is cartoon/anime/kawaii
2. The characters from reference images 2 onwards must appear IN THIS STYLE
3. Apply the color palette, visual effects, and rendering style specified above
4. The scene description below tells you WHAT happens, but THIS determines HOW it looks

The remaining reference images (2nd onwards) show REAL PEOPLE whose identity
you must preserve, but render them in the style specified above.
═══════════════════════════════════════════════════════════════════════════════

"""
            else:
                # Fallback to generic note if no style_analysis available
                style_ref_note = """
═══════════════════════════════════════════════════════════════════════════════
STYLE REFERENCE (FIRST IMAGE):
═══════════════════════════════════════════════════════════════════════════════
The FIRST reference image is a STYLE GUIDE. Analyze its visual characteristics
and apply them to the generated image:
- Art style (photorealistic, cartoon, anime, kawaii, illustration, etc.)
- Color palette and mood
- Visual effects (sparkles, glows, action lines, etc.)
- Rendering style (2D, 3D, cel-shaded, etc.)

Generate the image in the SAME STYLE as the first reference image.
The remaining reference images show people whose identity you must preserve.
═══════════════════════════════════════════════════════════════════════════════

"""
            enhanced_prompt = style_ref_note + enhanced_prompt

        # Try configured provider
        selected_provider = image_provider or IMAGE_PROVIDER

        generators = {
            "gemini": generate_with_gemini,
            "openai": generate_with_openai,
            "replicate": generate_with_replicate,
            "poe": generate_with_poe,
        }

        success = False

        # Try primary provider
        primary_generator = generators.get(selected_provider)
        if primary_generator:
            kwargs = {
                "prompt": enhanced_prompt,
                "reference_images": best_frames[:14] if best_frames else None,
                "output_path": output_path,
            }
            # For V2 prompt support, pass ThumbnailImage
            # Convert to concept/variation format for backward compatibility with providers
            if selected_provider in ["gemini", "openai", "poe"]:
                kwargs["image"] = img

            if selected_provider == "gemini" and gemini_model:
                kwargs["model"] = gemini_model
            elif selected_provider == "openai" and openai_model:
                kwargs["model"] = openai_model
            elif selected_provider == "poe" and poe_model:
                kwargs["model"] = poe_model
            # Replicate doesn't support image parameter
            if selected_provider == "replicate":
                kwargs.pop("image", None)

            result = primary_generator(**kwargs)
            if result:
                generated_thumbnails.append(result)
                success = True

        # Try fallbacks if primary failed
        if not success:
            for provider, generator in generators.items():
                if provider != selected_provider:
                    logger.info(f"    Trying fallback provider: {provider}")

                    fallback_model = get_effective_model(
                        provider, gemini_model, openai_model, poe_model
                    )
                    fallback_prefix = normalize_model_name_for_filename(fallback_model)
                    fallback_path = thumbnails_dir / f"{fallback_prefix}_{timestamp}_{image_idx}.png"

                    kwargs = {
                        "prompt": enhanced_prompt,
                        "reference_images": best_frames[:14] if best_frames else None,
                        "output_path": fallback_path,
                    }
                    if provider in ["gemini", "openai", "poe"]:
                        kwargs["image"] = img
                    if provider == "gemini" and gemini_model:
                        kwargs["model"] = gemini_model
                    elif provider == "openai" and openai_model:
                        kwargs["model"] = openai_model
                    elif provider == "poe" and poe_model:
                        kwargs["model"] = poe_model
                    if provider == "replicate":
                        kwargs.pop("image", None)

                    result = generator(**kwargs)
                    if result:
                        generated_thumbnails.append(result)
                        success = True
                        break

        # Last resort: composite
        if not success and best_frames:
            logger.warning("    All AI providers failed, creating composite")
            composite_path = thumbnails_dir / f"composite_{timestamp}_{image_idx}.png"
            result = create_composite_thumbnail(
                background_prompt=img.image_prompt,
                face_frame=best_frames[0],
                text_overlay=img.text_overlay,
                output_path=composite_path,
                colors=img.colors
            )
            if result:
                generated_thumbnails.append(result)
                success = True

        # Report progress after each image
        if progress_callback:
            last_thumbnail = generated_thumbnails[-1] if success and generated_thumbnails else None
            thumbnail_info = {
                'path': last_thumbnail,
                'image_index': image_idx,
                'image': img
            } if last_thumbnail else None
            progress_callback(image_idx, total_images, thumbnail_info)

    logger.success(f"Generated {len(generated_thumbnails)}/{total_images} thumbnails")
    logger.info(f"Thumbnails saved to: {thumbnails_dir}")

    return generated_thumbnails


if __name__ == "__main__":
    # Test thumbnail generation
    from prompt_generation import ThumbnailPrompt

    test_prompt = ThumbnailPrompt(
        video_title="Test Video",
        suggested_title="Amazing Gaming Tips",
        thumbnail_concept="A gamer looking excited at the camera",
        image_prompt="Professional YouTube thumbnail, excited gamer with headphones, gaming setup background, neon lights, bold colors",
        text_overlay="PRO TIPS",
        mood="excited",
        colors=["#FF0000", "#00FF00", "#0000FF"],
        key_topics=["gaming", "tips", "youtube"]
    )

    from config import OUTPUT_DIR

    output = VideoOutput(Path("test_video.mp4"), OUTPUT_DIR)
    output.setup()

    result = generate_thumbnail(test_prompt, [], output)

    if result:
        print(f"Thumbnail generated: {result}")
    else:
        print("Thumbnail generation failed")

"""
YouTube Thumbnail Generator - Image Generation Module
======================================================
Generates thumbnail images using Gemini (Nano Banana Pro),
OpenAI GPT Image, or Replicate APIs.
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
)
from utils import load_reference_images, setup_logger, VideoOutput
from prompt_generation import ThumbnailImage
from gransabio_prompt_generator import environment_references, image_numbers, split_sides

logger = setup_logger(__name__)


class ImageProviderError(RuntimeError):
    """An image provider gave no image; the message names it and says why, in its own words."""


def _api_error_detail(error: Exception) -> str:
    """What an API error says: the message and code of its body, not the SDK's dump of it."""
    body = getattr(error, "body", None)
    if isinstance(body, dict) and body.get("message"):
        return f"{body['message']} ({body['code']})" if body.get("code") else body["message"]
    return str(error) or type(error).__name__


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
        with Image.open(raw_path) as img:
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
        return "flux-1.1-pro"
    return "unknown"


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
    Edit Equation block: Change + Preserve + Physical Realism.

    This is the prompting pattern that empirically reduces face drift in
    Gemini Nano Banana and GPT-image-2: explicit list of what changes vs
    what must stay exactly as in the reference images, plus a realism
    anchor. Identity phrases ("same person", "returning character") are
    measurably more effective than long lists of negations.
    """
    return {
        "change": "Compose the scene described in 'scene.description' below.",
        "preserve": [
            "faces from reference images — exact pixel-level identity",
            "skin tone and texture from references",
            "hair color, length and style from references",
            "facial hair status from references (clean-shaven stays clean-shaven, beard stays beard)",
            "body type and build from references",
            "ear shape and any visible distinctive features"
        ],
        "physical_realism": "Natural lighting consistent with the scene. Realistic shadows and skin texture. No idealization, no plastic skin, no beauty filtering.",
        "identity_phrases": "same person from reference, consistent facial features, returning character, identical face",
        "verification": "A friend should recognize them INSTANTLY."
    }


def _build_generation_constraints() -> dict:
    """
    Build hard constraints for face/character generation.

    Kept short and declarative; the longer Edit Equation lives in
    clone_rules so the model can scan rules first and then act on the
    detailed preserve list.
    """
    return {
        "RULE": "CLONE_FACES_FROM_REFERENCES",
        "faces": "ONLY from attached reference images - NO generated/stock faces allowed",
        "identity_source": "Use the reference_mapping block to know which image shows which person_XX",
        "enforcement": "STRICT — if you cannot match a person to a reference, do not invent one"
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

    Prefers native V3 format (image.characters dict) if available, with the
    concept's outfit_changes replacing those characters' outfit.
    Falls back to converting legacy subjects array.

    Args:
        image: ThumbnailImage containing characters or subjects data

    Returns:
        dict mapping character IDs to character data
    """
    # V3 native: use characters dict directly if available
    if image.characters:
        characters = {person: dict(character) for person, character in image.characters.items()}
        for change in image.outfit_changes or []:
            # The photos show the video's clothes; this image asks for others
            characters[change["person_id"]].update(
                outfit=change["outfit"],
                outfit_note="Changed for this thumbnail: dress them as described in outfit, "
                            "not in the clothes of the reference photos.",
            )
        return characters

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


def _build_camera_and_lighting(image: ThumbnailImage) -> dict:
    """
    Standardised camera & lighting block.

    Locking camera and lighting across generations is what stops face
    "drift" when running the same prompt several times: when the angle
    and light vary, the model re-imagines features. Defaults are
    cinematic and identity-friendly (eye-level, 85mm-ish, soft daylight).
    """
    return {
        "framing": image.framing or "medium close-up, eye-level",
        "equipment": image.equipment or "85mm lens equivalent, f/1.8 shallow depth of field",
        "lighting": image.lighting or "soft natural daylight from camera-left, realistic skin shadows",
        "composition": image.composition or "rule of thirds, subject slightly off-centre with negative space for text overlay",
    }


def _attached_setting(settings: list) -> str:
    """The V3 setting when the user attached photos of it, by their relationship."""
    mode = settings[0]["environment_mode"]
    keep = "Keep it as it is: same walls, furniture, objects and colors; do not replace, relocate or redecorate it."
    if mode == "same":
        return f"The real place shown in {image_numbers(settings)} (ENVIRONMENT reference). {keep}"
    if mode == "alternatives":
        return (f"ONE of the real places shown in {image_numbers(settings)} (ENVIRONMENT references): the one "
                f"framing_and_lighting names; never a mix of them. {keep}")
    left, right = split_sides(settings)
    return (f"Split screen: the left panel is the real place shown in {left}, the right panel the one in "
            f"{right} (ENVIRONMENT references). {keep}")


def _build_scene_v3(image: ThumbnailImage) -> dict:
    """
    Build scene section for V3 prompt.

    Creates scene with:
    - description: what's happening (with clone prefixes for each active character)
    - character_directions: pose/expression for ACTIVE characters only (with MUST_CLONE_FACE)
    - environment: setting, props, and effects (lighting/camera live in camera_and_lighting)

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

    # Build environment (setting + props; lighting/camera moved to camera_and_lighting)
    settings = environment_references(image.reference_mapping)
    if settings:
        # The setting photos are the place: the concept only frames and lights it
        environment = {
            "setting": _attached_setting(settings),
            "framing_and_lighting": image.background,
            "props": image.visual_elements or [],
        }
    else:
        environment = {
            "setting": image.background or "professional studio background",
            "props": image.visual_elements or [],
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
            "instruction": (
                "Generate the image in THIS style. "
                + ("Use the attached style image identified in reference_mapping. " if any(
                    entry.get("is_style_reference") for entry in (image.reference_mapping or [])
                ) else "Use this style description; the identity photos do not define the style. ")
                + "Do NOT default to photorealistic if a different style is specified."
            )
        }

    # Check if we have identity data (V3 characters or legacy subjects)
    has_identity_data = image.characters or image.subjects

    # Only include constraints and identity sections if we have references and identity data
    if has_reference_images and has_identity_data:
        # Generation constraints FIRST - hard restrictions before any creative content
        prompt_dict["generation_constraints"] = _build_generation_constraints()

        # Clone rules with enforcement
        prompt_dict["clone_rules"] = _build_clone_rules()

        # Deterministic image_index → person_id mapping (from face clustering).
        # When present, this is the GROUND TRUTH for which reference image shows
        # which person. The generator MUST trust this mapping over visual inference.
        if image.reference_mapping:
            prompt_dict["reference_mapping"] = [
                {
                    "image_index": entry.get("image_index"),
                    "person_id": entry.get("person_id"),
                    "is_style_reference": entry.get("is_style_reference", False),
                    **({"face_sheet_views": entry["face_sheet_views"]} if entry.get("face_sheet_views") else {}),
                    "reference_role": entry.get("reference_role", "style" if entry.get("is_style_reference") else "face"),
                    **({key: entry[key] for key in ("body_coverage", "body_posture", "body_reference_status", "environment_mode", "environment_side", "environment_instruction", "source_type", "source_asset", "is_primary", "view", "expression") if entry.get(key)}),
                }
                for entry in image.reference_mapping
            ]

        # Face groups with physical descriptions
        face_groups = _build_face_groups_v3(image)
        if face_groups:
            prompt_dict["face_groups"] = face_groups

        # Characters (replaces subjects array)
        characters = _build_characters_v3(image)
        if characters:
            prompt_dict["characters"] = characters

    # Standardised camera & lighting block — locking these reduces face drift
    prompt_dict["camera_and_lighting"] = _build_camera_and_lighting(image)

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

    # The concept supplies the visible wording; its UI label is derived from it.
    if image.text_in_image:
        prompt_dict["text_in_image"] = image.text_in_image

    # Final reinforcement reminder when we have references and identity data
    if has_reference_images and has_identity_data:
        prompt_dict["REMINDER"] = "Every person_XX MUST be cloned from reference images. Search ALL attached images for matching faces."

    return prompt_dict


def _build_face_source_header(reference_mapping: Optional[list] = None) -> str:
    """
    Build a mandatory header that establishes face source restrictions.

    This header appears BEFORE the JSON to ensure the image AI reads
    the constraints before processing any creative content. When a
    reference_mapping is provided, an explicit "image N → person_XX"
    list is rendered so the AI does not have to infer identity by
    visual similarity.
    """
    base = """### FACE CLONING REQUIRED ###
Preserve the selected people's facial identities from their dedicated FACE references only.
BODY references describe visible proportions. ENVIRONMENT references describe the place, not additional people.
Each person_XX must match their own face references.
A friend must recognize them INSTANTLY.
"""

    if not reference_mapping:
        return base + "\n"

    # Group consecutive image indices that share the same person_id into ranges
    # so the header reads "Images 1-2: person_01" instead of one line per image.
    lines = []
    runs: list[tuple] = []  # (label, first_idx, last_idx)
    for entry in reference_mapping:
        idx = entry.get("image_index")
        if entry.get("is_style_reference"):
            label = "STYLE REFERENCE (visual style only, NOT a person to clone)"
        elif entry.get("reference_role") == "environment":
            label = "ENVIRONMENT: " + entry.get("environment_instruction", "Preserve the setting.") + " Use dedicated references for each person's identity and body."
        elif entry.get("reference_role") == "body":
            label = (
                f"BODY REFERENCE for {entry['person_id']}: {entry['body_coverage']}, "
                f"view {entry.get('view') or 'unspecified'}, estimated posture {entry['body_posture']}; preserve observed proportions, not the pose or clothing. "
                "Do not beautify, slim, inflate, or normalize the person's build. "
                "Observed body proportions take precedence over earlier textual build descriptions. "
                "Face identity, hair, and character appearance come from the PRIMARY face reference. "
                "Do not copy a costume, wig, hairstyle, or accessories from a body-only reference."
            )
        elif entry.get("person_id") and entry.get("face_sheet_views"):
            label = (
                f"face sheet for {entry['person_id']}: a grid of crops of that one face "
                f"({entry['face_sheet_views']}), not a layout to copy"
            )
        elif entry.get("person_id"):
            kind = "PRIMARY face reference" if entry.get("is_primary") else "face reference"
            label = f"{kind} for {entry['person_id']}"
            observed = " / ".join(filter(None, (entry.get("view"), entry.get("expression"))))
            if observed:
                label += f"; observed view/expression: {observed}"
            if entry.get("body_reference_status") == "unavailable":
                label += "; no reliable body reference available: do not claim full-body proportions are observed"
        else:
            label = "unlabelled reference"

        if runs and runs[-1][0] == label and runs[-1][2] == idx - 1:
            runs[-1] = (label, runs[-1][1], idx)
        else:
            runs.append((label, idx, idx))

    for label, first, last in runs:
        if first == last:
            lines.append(f"- Image {first}: {label}")
        else:
            lines.append(f"- Images {first}-{last}: {label}")

    return (
        base
        + "\nREFERENCE IMAGES — DETERMINISTIC ROLE MAPPING (trust this, do not re-infer):\n"
        + "\n".join(lines)
        + "\n\n"
    )


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
    import orjson
    prompt_dict = build_image_prompt_v3(image, has_reference_images)
    json_content = orjson.dumps(prompt_dict, option=orjson.OPT_INDENT_2).decode('utf-8')

    # Add header only when we have references and identity data
    has_identity_data = image.characters or image.subjects
    if has_reference_images and has_identity_data:
        return _build_face_source_header(image.reference_mapping) + json_content

    return json_content


# =============================================================================
# IMAGE GENERATION PROVIDERS
# =============================================================================

def generate_with_gemini(
    prompt: str,
    reference_images: list[Path] = None,
    output_path: Path = None,
    model: str = None,
    image: ThumbnailImage = None
) -> Path:
    """
    Generate image using Google Gemini API (Nano Banana / Nano Banana Pro / Nano Banana 2).

    Args:
        prompt: Text prompt for image generation (V1) or fallback (V2)
        reference_images: Optional list of reference image paths
        output_path: Where to save the generated image
        model: Gemini model to use
        image: ThumbnailImage with structured V2/V3 fields used to build the prompt

    Returns:
        Path to the generated image

    Raises:
        ImageProviderError: Gemini gave no image (the message says why)
    """

    # Use provided model or default
    gemini_model = model or GEMINI_IMAGE_MODEL
    if not GEMINI_API_KEY:
        raise ImageProviderError("Gemini: GEMINI_API_KEY is not set")
    if gemini_model not in GEMINI_MODEL_MAX_REFS:
        raise ValueError(f"Unknown Gemini image model: {gemini_model}")

    has_refs = bool(reference_images)

    # V3 optimized JSON structure - requires V2 fields populated by LLM
    if image and image.has_v2_fields():
        logger.info("Using V3 optimized JSON prompt from ThumbnailImage")
        full_prompt = build_v3_prompt_string(image, has_refs)
        logger.debug(f"V3 prompt length: {len(full_prompt)} chars")
    elif has_refs:
        # Fail-fast: references provided but no V2 fields means the LLM
        # didn't populate scene/subject_pose, or cached prompts are stale.
        raise ValueError(
            "Cannot build the Gemini prompt: reference images were given but the "
            "ThumbnailImage has no V2 fields (scene/subject_pose). "
            "Regenerate prompts with force_prompts=True."
        )
    else:
        full_prompt = f"""Create a professional YouTube thumbnail.

{prompt}

Requirements:
- 16:9 aspect ratio (1280x720 pixels)
- Eye-catching and bold design
- Professional YouTube thumbnail style
- DO NOT include any text in the image
"""

    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise ImageProviderError("Gemini: google-genai package not installed. Install with: pip install google-genai") from error

    logger.info(f"Generating image with Gemini ({gemini_model})...")

    # Prepare content parts
    contents = []

    # All references must be valid and within the model's count limit.
    if reference_images:
        for data in load_reference_images(reference_images, GEMINI_MODEL_MAX_REFS[gemini_model]):
            contents.append(types.Part.from_bytes(data=data, mime_type="image/png"))

    # Add text prompt
    contents.append(full_prompt)

    # Build generation config: every supported model takes image_size and aspect_ratio
    gen_config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_config=types.ImageConfig(
            aspect_ratio="16:9",
            image_size=GEMINI_PRO_OUTPUT_RESOLUTION,  # "1K", "2K", or "4K"
        ),
    )
    logger.info(f"Using {gemini_model} with {GEMINI_PRO_OUTPUT_RESOLUTION} resolution (16:9)")

    # Generate with image output modality
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=gemini_model,
            contents=contents,
            config=gen_config,
        )
    except Exception as error:
        raise ImageProviderError(f"Gemini {gemini_model}: {_api_error_detail(error)}") from error

    # Extract image from response
    image_data = next((
        part.inline_data.data
        for candidate in response.candidates or []
        if candidate.content and candidate.content.parts
        for part in candidate.content.parts
        if getattr(part, 'inline_data', None) and part.inline_data.mime_type.startswith('image/')
    ), None)
    if image_data is None:
        raise ImageProviderError(f"Gemini {gemini_model}: no image in the answer ({_gemini_no_image_reason(response)})")

    # Save raw image first (if enabled)
    save_raw_thumbnail(image_data, output_path, logger)

    # Resize to exact thumbnail dimensions
    with Image.open(io.BytesIO(image_data)) as img:
        resized = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)
        resized.save(output_path, 'PNG', quality=95)

    logger.success(f"Image saved to: {output_path} ({THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT})")
    return output_path


def _gemini_no_image_reason(response) -> str:
    """Why a Gemini answer holds no image: its block and finish reasons, and any text it wrote instead."""
    reasons = []
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        reasons.append(f"prompt blocked: {getattr(feedback.block_reason, 'name', feedback.block_reason)}")
    for candidate in response.candidates or []:
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason:
            reasons.append(f"finish reason: {getattr(finish_reason, 'name', finish_reason)}")
        parts = candidate.content.parts if candidate.content and candidate.content.parts else []
        reasons += [part.text.strip() for part in parts if getattr(part, "text", None)]
    return "; ".join(reasons) or "no reason given"


def generate_with_openai(
    prompt: str,
    reference_images: list[Path] = None,
    output_path: Path = None,
    model: str = None,
    image: ThumbnailImage = None
) -> Path:
    """
    Generate image using OpenAI GPT Image.

    Supports (all with up to 16 reference images, sent to images.edit):
    - gpt-image-2.6-sunburst (detailed compositions and precise editing)
    - gpt-image-2.6-flare (fast generation and editing)
    - gpt-image-2.5-sunburst (precise editing)
    - gpt-image-2.5-flare (fast, high quality)
    - gpt-image-2 (previous generation)

    Args:
        prompt: Text prompt for image generation (V1) or fallback (V2)
        reference_images: Reference images are passed as base64 encoded images
        output_path: Where to save the generated image
        model: OpenAI model to use
        image: ThumbnailImage with structured V2/V3 fields used to build the prompt

    Returns:
        Path to the generated image

    Raises:
        ImageProviderError: OpenAI gave no image (the message says why, e.g. its safety system)
    """

    # Use provided model or default from config
    selected_model = model or OPENAI_IMAGE_MODEL
    if not OPENAI_API_KEY:
        raise ImageProviderError("OpenAI: OPENAI_API_KEY is not set")
    if selected_model not in OPENAI_MODEL_MAX_REFS:
        raise ValueError(f"Unknown OpenAI image model: {selected_model}")

    max_refs = OPENAI_MODEL_MAX_REFS[selected_model]
    has_refs = bool(reference_images)

    logger.info(f"Generating image with OpenAI ({selected_model})...")
    if has_refs:
        logger.info(f"Using up to {max_refs} reference images")

    # V3 optimized JSON structure - requires V2 fields populated by LLM
    if image and image.has_v2_fields():
        logger.info("Using V3 optimized JSON prompt from ThumbnailImage")
        full_prompt = build_v3_prompt_string(image, has_refs)
        logger.debug(f"V3 prompt length: {len(full_prompt)} chars")
    elif has_refs:
        # Fail-fast: references provided but no V2 fields means the LLM
        # didn't populate scene/subject_pose, or cached prompts are stale.
        raise ValueError(
            "Cannot build the OpenAI prompt: reference images were given but the "
            "ThumbnailImage has no V2 fields (scene/subject_pose). "
            "Regenerate prompts with force_prompts=True."
        )
    else:
        full_prompt = f"""Professional YouTube video thumbnail.

{prompt}

Style: Bold, eye-catching YouTube thumbnail with high contrast colors.
Do NOT include any text or letters in the image.
Photorealistic style, professional quality.
"""

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    references = load_reference_images(reference_images, max_refs, max_bytes=50_000_000) if has_refs else []
    try:
        if has_refs:
            image_tuples = [(f"ref_{i}.png", data, "image/png") for i, data in enumerate(references)]
            response = client.images.edit(
                model=selected_model,
                image=image_tuples if len(image_tuples) > 1 else image_tuples[0],
                prompt=full_prompt,
                size=OPENAI_IMAGE_SIZE,
                quality=OPENAI_IMAGE_QUALITY,
            )
        else:
            response = client.images.generate(
                model=selected_model,
                prompt=full_prompt,
                size=OPENAI_IMAGE_SIZE,
                quality=OPENAI_IMAGE_QUALITY,
                n=1,
            )
    except Exception as error:
        raise ImageProviderError(f"OpenAI {selected_model}: {_api_error_detail(error)}") from error

    # Extract image from response
    if not response.data:
        raise ImageProviderError(f"OpenAI {selected_model}: no image in the answer")
    image_data = response.data[0]

    # GPT Image returns b64_json by default
    if hasattr(image_data, 'b64_json') and image_data.b64_json:
        img_bytes = base64.b64decode(image_data.b64_json)
    elif hasattr(image_data, 'url') and image_data.url:
        # Fallback to URL if available
        import requests
        img_response = requests.get(image_data.url)
        if img_response.status_code != 200:
            raise ImageProviderError(
                f"OpenAI {selected_model}: downloading the image failed (HTTP {img_response.status_code})"
            )
        img_bytes = img_response.content
    else:
        raise ImageProviderError(f"OpenAI {selected_model}: no image data in the answer")

    # Save raw image first (if enabled)
    save_raw_thumbnail(img_bytes, output_path, logger)

    # Load and resize to exact thumbnail dimensions
    with Image.open(io.BytesIO(img_bytes)) as img:
        resized = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)

        # Save with configured format
        save_format = OPENAI_IMAGE_FORMAT.upper()
        if save_format == "JPG":
            save_format = "JPEG"

        resized.save(output_path, save_format, quality=95)
    logger.success(f"Image saved to: {output_path} ({THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT})")
    return output_path


def generate_with_replicate(
    prompt: str,
    reference_images: list[Path] = None,
    output_path: Path = None,
    model: str = None
) -> Path:
    """
    Generate image using Replicate API (FLUX, SDXL, etc.).

    Supports face reference via image_prompt (FLUX) or face-to-many.
    Reference limits per model:
    - flux-1.1-pro: 1 image (via image_prompt)
    - face-to-many: 1 face image

    Raises:
        ImageProviderError: Replicate gave no image (the message says why)
    """

    try:
        import replicate
        import requests
    except ImportError as error:
        raise ImageProviderError("Replicate: replicate package not installed") from error

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

    if reference_images:
        references = load_reference_images(reference_images, max_refs)
        image_url = "data:image/png;base64," + base64.b64encode(references[0]).decode('ascii')
        if "face-to-many" in selected_model:
            input_data = {"image": image_url, "prompt": prompt, "style": "Video game",
                          "negative_prompt": "blurry, low quality"}
        else:
            input_data["image_prompt"] = image_url

    try:
        output = replicate.run(selected_model, input=input_data)

        # Handle output (can be URL or list)
        if isinstance(output, list):
            image_url = output[0] if output else None
        else:
            image_url = output

        # Download
        response = requests.get(image_url) if image_url else None
    except Exception as error:
        raise ImageProviderError(f"Replicate {selected_model}: {_api_error_detail(error)}") from error

    if response is None:
        raise ImageProviderError(f"Replicate {selected_model}: no image in the answer")
    if response.status_code != 200:
        raise ImageProviderError(f"Replicate {selected_model}: downloading the image failed (HTTP {response.status_code})")
    img_bytes = response.content

    # Save raw image first (if enabled)
    save_raw_thumbnail(img_bytes, output_path, logger)

    # Load and resize
    with Image.open(io.BytesIO(img_bytes)) as img:
        resized = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)
        resized.save(output_path, 'PNG', quality=95)
    logger.success(f"Image saved to: {output_path} ({THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT})")
    return output_path


def generate_with_poe(
    prompt: str,
    reference_images: list[Path] = None,
    output_path: Path = None,
    model: str = None,
    image: ThumbnailImage = None
) -> Path:
    """
    Generate image using Poe API with FLUX/Ideogram models.

    Supports reference images for character/face consistency.
    Available models (bot names verified 2025-12-28, Nano Banana 2026-10-24):
        - flux2pro: Best quality, up to 8 references
        - flux2flex: High resolution (14MP)
        - fluxkontextpro: Best prompt following
        - seedream40: Good for combining references
        - nano-banana-pro: Gemini 3 Pro Image via Poe
        - nano-banana-2.1: Nano Banana 2.1 via Poe
        - Ideogram-v3: Best for text/logos

    Args:
        prompt: Text prompt for image generation (V1) or fallback (V2)
        reference_images: Optional list of reference image paths (up to 3)
        output_path: Where to save the generated image
        model: Model to use (default from config: flux2pro)
        image: ThumbnailImage with structured V2/V3 fields used to build the prompt

    Returns:
        Path to the generated image

    Raises:
        ImageProviderError: Poe gave no image (the message says why, with the bot's answer)
    """

    selected_model = model or POE_IMAGE_MODEL
    if not POE_API_KEY:
        raise ImageProviderError("Poe: POE_API_KEY is not set")
    if selected_model not in POE_MODEL_MAX_REFS:
        raise ValueError(f"Unknown Poe image model: {selected_model}")

    from openai import OpenAI
    import requests

    # Initialize Poe client (OpenAI-compatible)
    client = OpenAI(
        api_key=POE_API_KEY,
        base_url="https://api.poe.com/v1",
    )
    logger.info(f"Generating image with Poe ({selected_model})...")

    # Build message content
    content = []

    has_refs = bool(reference_images)
    if reference_images:
        for data in load_reference_images(reference_images, POE_MODEL_MAX_REFS[selected_model]):
            encoded = base64.b64encode(data).decode('ascii')
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})

    # V3 optimized JSON structure - requires V2 fields populated by LLM
    if image and image.has_v2_fields():
        logger.info("Using V3 optimized JSON prompt from ThumbnailImage")
        full_prompt = build_v3_prompt_string(image, has_refs)
        logger.debug(f"V3 prompt length: {len(full_prompt)} chars")
    elif has_refs:
        # Fail-fast: references provided but no V2 fields means the LLM
        # didn't populate scene/subject_pose, or cached prompts are stale.
        raise ValueError(
            "Cannot build the Poe prompt: reference images were given but the "
            "ThumbnailImage has no V2 fields (scene/subject_pose). "
            "Regenerate prompts with force_prompts=True."
        )
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
    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": content}],
            stream=False,
            extra_body={
                "aspect": POE_IMAGE_ASPECT,
                "quality": POE_IMAGE_QUALITY
            }
        )
    except Exception as error:
        raise ImageProviderError(f"Poe {selected_model}: {_api_error_detail(error)}") from error

    # Extract image from response
    response_content = response.choices[0].message.content if response.choices else None

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
        # Look for markdown image: ![...](URL)
        match = re.search(r'!\[.*?\]\((https?://[^\)]+)\)', response_content)
        if match:
            image_url = match.group(1)
        # Look for direct URL
        elif response_content.startswith('http'):
            image_url = response_content.strip()

    if not image_url:
        # The bot answers in words when it will not draw: that answer is the reason
        answer = str(response_content)[:500] if response_content else "an empty answer"
        raise ImageProviderError(f"Poe {selected_model}: no image in the answer, it said: {answer}")

    if image_url.startswith('data:image'):
        # Base64 encoded image
        header, data = image_url.split(',', 1)
        img_bytes = base64.b64decode(data)
    else:
        # URL - download image
        try:
            img_response = requests.get(image_url, timeout=60)
        except requests.RequestException as error:
            raise ImageProviderError(f"Poe {selected_model}: downloading the image failed: {error}") from error
        if img_response.status_code != 200:
            raise ImageProviderError(f"Poe {selected_model}: downloading the image failed (HTTP {img_response.status_code})")
        img_bytes = img_response.content

    # Save raw image first (if enabled)
    save_raw_thumbnail(img_bytes, output_path, logger)

    # Load and resize to exact thumbnail dimensions
    with Image.open(io.BytesIO(img_bytes)) as img:
        resized = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)
        resized.save(output_path, 'PNG', quality=95)
    logger.success(f"Image saved to: {output_path} ({THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT})")
    return output_path


# =============================================================================
# COMPOSITE THUMBNAIL CREATION
# =============================================================================

def create_composite_thumbnail(
    background_prompt: str,
    face_frame: Path,
    text_overlay: str,
    output_path: Path,
    colors: list[str] = None
) -> Path:
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
            with Image.open(face_frame) as face_img:
                # Resize to fit nicely (about 60% of thumbnail height)
                face_height = int(THUMBNAIL_HEIGHT * 0.8)
                face_ratio = face_img.width / face_img.height
                face_width = int(face_height * face_ratio)

                face_resized = face_img.resize((face_width, face_height), Image.Resampling.LANCZOS)

            # Position on right side
            x_pos = THUMBNAIL_WIDTH - face_width - 20
            y_pos = (THUMBNAIL_HEIGHT - face_height) // 2

            thumbnail.paste(face_resized, (x_pos, y_pos))

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
        raise ImageProviderError(f"Composite thumbnail: {e}") from e


# =============================================================================
# THUMBNAIL GENERATION (ThumbnailImage-based)
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
    cancellation_check: callable = None,
    allow_provider_fallback: bool = False,
    comfyui_workflow: str = None,
    comfyui_config: dict = None,
    comfyui_profile: dict = None,
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
        allow_provider_fallback: When the chosen provider fails, try the other providers
                           and, if all fail, a composite of a frame. Off: that image fails.

    Returns:
        List of paths to generated thumbnails (may be partial if cancelled)

    Raises:
        ImageProviderError: an image could not be generated (after the other providers,
                           when fallback is allowed). The message names the image and
                           what each provider said; the images before it are kept.
    """

    if (image_provider or IMAGE_PROVIDER) == "comfyui":
        if allow_provider_fallback or use_composite:
            raise ValueError("ComfyUI does not allow provider fallback or composite generation")
        from services.comfyui_service import generate_batch
        return generate_batch(images, best_frames, output, comfyui_workflow, comfyui_config,
                              comfyui_profile, progress_callback, cancellation_check)

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
            generated_thumbnails.append(result)
            if progress_callback:
                progress_callback(image_idx, total_images, {
                    'path': result,
                    'image_index': image_idx,
                    'image': img
                })
            continue

        # Option 2: Full AI generation
        # V3 providers (Gemini/OpenAI/Poe) rebuild the prompt from `img` directly
        # via build_v3_prompt_string(). This text is only used by Replicate (no V3 support)
        # and by the plain fallback inside providers when no V2 fields are present.
        enhanced_prompt = img.image_prompt

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

        if selected_provider not in generators:
            raise ValueError(f"Unknown image provider: {selected_provider}")

        # What each provider said when it gave no image
        failures = []
        result = None

        # Try primary provider
        kwargs = {
            "prompt": enhanced_prompt,
            "reference_images": best_frames or None,  # Already within the model limit (generation service)
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

        try:
            result = generators[selected_provider](**kwargs)
        except ImageProviderError as error:
            if not allow_provider_fallback:
                raise ImageProviderError(f"Image {image_idx} of {total_images}: {error}") from error
            logger.warning(f"    {error}")
            failures.append(str(error))

        # Try fallbacks if primary failed (only when the user allows it)
        if result is None:
            for provider, generator in generators.items():
                if provider != selected_provider:
                    logger.info(f"    Trying fallback provider: {provider}")

                    fallback_model = get_effective_model(
                        provider, gemini_model, openai_model, poe_model
                    )
                    fallback_limits = {
                        "gemini": GEMINI_MODEL_MAX_REFS,
                        "openai": OPENAI_MODEL_MAX_REFS,
                        "poe": POE_MODEL_MAX_REFS,
                        "replicate": REPLICATE_MODEL_MAX_REFS,
                    }
                    if len(best_frames or []) > fallback_limits[provider][fallback_model]:
                        logger.warning(f"    Skipping {provider}: it cannot accept all identity references")
                        continue
                    fallback_prefix = normalize_model_name_for_filename(fallback_model)
                    fallback_path = thumbnails_dir / f"{fallback_prefix}_{timestamp}_{image_idx}.png"

                    kwargs = {
                        "prompt": enhanced_prompt,
                        "reference_images": best_frames or None,  # Already within the model limit (generation service)
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

                    try:
                        result = generator(**kwargs)
                        break
                    except ImageProviderError as error:
                        logger.warning(f"    {error}")
                        failures.append(str(error))

        # Last resort: composite
        if result is None and best_frames:
            logger.warning("    All AI providers failed, creating composite")
            composite_path = thumbnails_dir / f"composite_{timestamp}_{image_idx}.png"
            try:
                result = create_composite_thumbnail(
                    background_prompt=img.image_prompt,
                    face_frame=best_frames[0],
                    text_overlay=img.text_overlay,
                    output_path=composite_path,
                    colors=img.colors
                )
            except ImageProviderError as error:
                failures.append(str(error))

        if result is None:
            raise ImageProviderError(f"Image {image_idx} of {total_images}: {'; '.join(failures)}")
        generated_thumbnails.append(result)

        # Report progress after each image
        if progress_callback:
            progress_callback(image_idx, total_images, {
                'path': result,
                'image_index': image_idx,
                'image': img
            })

    logger.success(f"Generated {len(generated_thumbnails)}/{total_images} thumbnails")
    logger.info(f"Thumbnails saved to: {thumbnails_dir}")

    return generated_thumbnails

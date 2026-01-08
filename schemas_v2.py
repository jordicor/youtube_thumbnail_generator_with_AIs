"""
Prompt System V2 - Pydantic Schemas
===================================
Structured JSON prompts for improved thumbnail generation.

Based on PROMPT_SYSTEM_V2_PROPOSAL.md research:
- JSON prompts are 85-92% more precise than natural text (60-68%)
- 200-700ms faster processing
- Reduced "concept bleeding" between attributes
- Better identity preservation with positive instructions (no negations)
"""

from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# SCHEMA A: ConceptSchema (Output from LLM Concept Generator)
# =============================================================================

class VariationSchemaV2(BaseModel):
    """A variation of a thumbnail concept (V2 structured format)."""

    variation_index: int = Field(..., ge=1)
    variation_focus: str = Field(..., max_length=60,
        description="What this variation emphasizes")

    # UI fields
    text_overlay: str = Field(..., max_length=30,
        description="Short text for UI overlay (2-5 words)")

    # Generation fields
    scene: str = Field(..., max_length=300,
        description="Complete narrative description of the scene")

    subject_pose: str = Field(..., max_length=100,
        description="Subject's posture (free-form, creative)")

    subject_expression: str = Field(..., max_length=100,
        description="Facial expression (free-form, creative)")

    subject_action: Optional[str] = Field(None, max_length=120,
        description="What the subject is doing (optional)")

    visual_elements: Optional[list[str]] = Field(None,
        description="Additional visual elements (props, effects)")

    text_in_image: Optional[str] = Field(None, max_length=25,
        description="Text to render IN the image (prefer ~3 words, optional)")


class ConceptSchemaV2(BaseModel):
    """A complete thumbnail concept with V2 structured format."""

    concept_index: int = Field(..., ge=1)

    # UI fields
    concept_name: str = Field(..., max_length=50)
    suggested_title: str = Field(..., max_length=80)
    description: str = Field(..., max_length=200)
    key_topics: list[str] = Field(..., min_length=3, max_length=5)

    # Generation fields (shared across variations)
    mood: str = Field(..., max_length=60,
        description="Emotional tone (free-form)")

    background: str = Field(..., max_length=150,
        description="Background/environment description")

    lighting: str = Field(..., max_length=100,
        description="Lighting type (free-form)")

    equipment: str = Field(..., max_length=120,
        description="Camera equipment: camera, lens, aperture, film stock")

    framing: str = Field(..., max_length=80,
        description="Framing/shot type (free-form)")

    composition: str = Field(..., max_length=100,
        description="Composition (free-form)")

    color_palette: list[str] = Field(..., min_length=2, max_length=4,
        description="Color palette (hex codes)")

    # Style fields (flexible)
    style: str = Field(
        default="photorealistic YouTube thumbnail",
        max_length=100,
        description="Visual style (photorealistic, anime, illustration, etc.)"
    )

    quality: str = Field(
        default="high resolution, sharp focus, professional",
        max_length=100,
        description="Desired quality (can be blurry for effect, etc.)"
    )

    materials: Optional[dict] = Field(None,
        description="Specific textures (skin, clothing, props) - for realism")

    # Optional appearance override
    clothing_override: Optional[str] = Field(None, max_length=150,
        description="Different clothing from reference (costume, thematic)")

    # Variations
    variations: list[VariationSchemaV2] = Field(..., min_length=1)


# =============================================================================
# SCHEMA B: ImagePromptSchema (JSON for Gemini/Image Generator)
# =============================================================================

class ImagePromptSchemaV2(BaseModel):
    """
    Structured JSON prompt for Gemini/Nano Banana Pro.
    Target: ~1,000-1,400 characters serialized.
    All values are free-form - maximum flexibility.
    """

    # Identity (only if person references present)
    identity: Optional[str] = Field(None,
        description="Identity lock phrase (only with references)")

    # Scene (the heart of the prompt)
    scene: str = Field(...,
        description="Complete narrative description")

    # Subject
    subject: dict = Field(...,
        description="Pose, expression, action of the subject")

    # Materials/Textures (optional, for realism)
    materials: Optional[dict] = Field(None,
        description="Textures for realism (skin, clothing, props)")

    # Environment
    environment: dict = Field(...,
        description="Background, lighting")

    # Camera/Equipment
    camera: dict = Field(...,
        description="Equipment, framing, composition, depth_of_field")

    # Style (flexible)
    style: dict = Field(...,
        description="Aesthetic, colors, mood")

    # Text in image (optional)
    text_in_image: Optional[str] = Field(None,
        description="Text to render (prefer ~3 words)")

    # Quality (flexible)
    quality: str = Field(...,
        description="Desired quality (may vary by effect)")

    # Technical
    aspect_ratio: str = Field(default="16:9")


# =============================================================================
# BUILDER FUNCTION
# =============================================================================

def build_image_prompt_v2(
    concept: ConceptSchemaV2,
    variation: VariationSchemaV2,
    has_reference_images: bool = False
) -> ImagePromptSchemaV2:
    """
    Build a structured JSON prompt from concept and variation.

    Args:
        concept: The ConceptSchemaV2 with shared generation fields
        variation: The VariationSchemaV2 with specific scene details
        has_reference_images: Whether reference images will be provided

    Returns:
        ImagePromptSchemaV2 ready to serialize to JSON
    """
    # Identity lock (only with references, positive phrasing - no negations!)
    identity = None
    if has_reference_images:
        identity = (
            "Clone exact person from reference: face, body, hair, skin, outfit. "
            "Must be instantly recognizable."
        )

    # Build subject dict
    subject = {
        "pose": variation.subject_pose,
        "expression": variation.subject_expression,
    }
    if variation.subject_action:
        subject["action"] = variation.subject_action

    # Build environment dict
    environment = {
        "background": concept.background,
        "lighting": concept.lighting,
    }

    # Build camera dict
    camera = {
        "equipment": concept.equipment,
        "framing": concept.framing,
        "composition": concept.composition,
    }

    # Build style dict
    style = {
        "aesthetic": concept.style,
        "colors": concept.color_palette,
        "mood": concept.mood,
    }

    return ImagePromptSchemaV2(
        identity=identity,
        scene=variation.scene,
        subject=subject,
        materials=concept.materials,
        environment=environment,
        camera=camera,
        style=style,
        text_in_image=variation.text_in_image,
        quality=concept.quality,
        aspect_ratio="16:9"
    )


def image_prompt_to_json_string(prompt: ImagePromptSchemaV2) -> str:
    """
    Serialize ImagePromptSchemaV2 to JSON string for image generation.

    Returns:
        JSON string (~1,000-1,400 characters)
    """
    return prompt.model_dump_json(exclude_none=True, indent=2)


def image_prompt_to_dict(prompt: ImagePromptSchemaV2) -> dict:
    """
    Convert ImagePromptSchemaV2 to dict for image generation.

    Returns:
        Dict with all non-None fields
    """
    return prompt.model_dump(exclude_none=True)

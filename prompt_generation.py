"""
YouTube Thumbnail Generator - Prompt Generation Module
=======================================================
Holds the active ThumbnailImage dataclass plus the entry point that
delegates to Gran Sabio LLM (gransabio_prompt_generator) to produce
structured V3 thumbnail prompts.
"""

import base64
import hashlib
import orjson
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass, asdict

from config import THUMBNAIL_STYLE
from utils import setup_logger, VideoOutput
from transcription import get_transcription_summary

if TYPE_CHECKING:
    from gransabio_prompt_generator import PromptGenerationConfig

logger = setup_logger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ThumbnailImage:
    """A single thumbnail image to generate (flat, independent structure)."""
    image_index: int                               # Sequential index (1, 2, 3...)
    concept_name: str                              # Short name for this concept (2-4 words)
    thumbnail_concept: str                         # Brief description of the idea
    suggested_title: str                           # Suggested video title
    image_prompt: str                              # Complete prompt for image generation
    text_overlay: str                              # Text to overlay on the thumbnail
    mood: str                                      # Emotional tone
    colors: list[str]                              # Dominant colors as hex codes
    key_topics: list[str]                          # Main keywords from the video

    # Subject identification from reference image analysis
    subjects: Optional[list[dict]] = None

    # V2 fields for structured JSON prompts
    scene: Optional[str] = None                    # Complete narrative description
    subject_pose: Optional[str] = None             # Subject's posture/body language
    subject_expression: Optional[str] = None       # Facial expression
    subject_action: Optional[str] = None           # What the subject is doing
    visual_elements: Optional[list[str]] = None    # Additional visual elements
    text_in_image: Optional[str] = None            # Text to render IN the image
    background: Optional[str] = None               # Background/environment description
    lighting: Optional[str] = None                 # Lighting setup
    equipment: Optional[str] = None                # Camera/lens/film stock
    framing: Optional[str] = None                  # Shot type (close-up, medium, wide)
    composition: Optional[str] = None              # Composition rules
    color_palette: Optional[list[str]] = None      # Alias for colors in V2 format
    style: Optional[str] = None                    # Visual style
    quality: Optional[str] = None                  # Quality descriptor
    materials: Optional[dict] = None               # Textures for realism

    # Face identity grouping for same person with different outfits
    face_groups: Optional[dict] = None             # {"group_A": {"physical_description": "...", "characters_with_this_face": [...]}}
    style_source: Optional[str] = None             # Which character defines the outfit for final image

    # V3 fields - characters dict replaces subjects array
    characters: Optional[dict] = None              # {"person_01": {"belongs_to_face": "group_A", "outfit": "...", "identify_in_references": "..."}}
    outfit_changes: Optional[list[dict]] = None    # [{"person_id": "person_01", "outfit": "..."}]: this image's clothing instead of the characters' outfit
    environment_effects: Optional[str] = None      # Visual effects: glows, holographic elements, particles, etc.

    # Style reference analysis (extracted from user's style reference image)
    style_analysis: Optional[dict] = None          # {"art_type": "kawaii/chibi", "color_palette": [...], "visual_effects": [...], ...}

    # Deterministic mapping of reference image position → person_id derived
    # from the cluster the frame came from (set by the generation service).
    # Each entry: {"image_index": 1-based int, "person_id": "person_01" | None,
    #              "cluster_id": int | None, "source_frame": str,
    #              "is_style_reference": bool, "face_sheet_views": str | None}.
    # face_sheet_views is set on a person's face sheet (several crops of the
    # face in one image) and says which views it shows.
    # person_id is None and is_style_reference is True for the optional
    # user-uploaded style image that occupies position 1 when present.
    reference_mapping: Optional[list] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def has_v2_fields(self) -> bool:
        """Check if this image has V2 structured fields populated."""
        return self.scene is not None and self.subject_pose is not None

# Batches kept in prompts/images.json, newest first: going back and forth
# between a few choices (background, count, starred titles...) reuses each
# one's ideas instead of asking for new ones.
CACHE_BATCHES = 12

# Inputs of the prompts, as the worker log names them when nothing matches
CACHE_INPUT_LABELS = {
    "people": "people/photos",
    "style": "style image",
    "background": "background",
    "instructions": "instructions",
    "count": "count",
    "expression": "expression",
    "titles": "starred titles",
    "ideas_model": "ideas model",
    "avoid_repeat": "avoid repeat",
    "transcription": "transcription",
    "skeleton_rules": "prompt rules",
    "concept_rules": "prompt rules",
}
# Inputs of the skeleton (identity, style and setting); the concepts depend on all
SKELETON_INPUTS = ("people", "style", "background", "instructions", "skeleton_rules")

# Setting photos in each relationship, to fingerprint the rules written for them
_SETTING_RULE_SAMPLES = [
    [{"reference_role": "environment", "image_index": 1, "environment_mode": "same"}],
    [{"reference_role": "environment", "image_index": index, "environment_mode": "alternatives"} for index in (1, 2)],
    [{"reference_role": "environment", "image_index": index, "environment_mode": "split", "environment_side": side}
     for index, side in ((1, "left"), (2, "right"))],
]


def _digest(*parts) -> str:
    h = hashlib.md5()
    for part in parts:
        h.update(orjson.dumps(part, option=orjson.OPT_SORT_KEYS))
    return h.hexdigest()


def _cache_inputs(
    reference_mapping,
    reference_images_base64,
    has_style_reference,
    cluster_description,
    num_images,
    prompt_config,
    selected_titles,
    transcription_summary,
) -> dict:
    """A fingerprint of each input of the prompts: the cache keys are built
    from them, and a miss is explained by the ones that differ.

    Image numbers are left out of the reference entries (they follow from the
    order: style, people, setting), so new photos of one person do not look
    like a new background. The rules fingerprints change when the prompts or
    answer contracts do, so concepts written under older rules (that lost
    visible text, that invented a place when the setting was given) are not
    reused. The skeleton does not depend on the transcription.
    """
    from gransabio_prompt_generator import (
        CONCEPT_SCHEMA, SKELETON_SCHEMA, _build_concept_setting_section, _build_skeleton_setting_section,
    )

    images = [hashlib.md5(base64.b64decode(image)).hexdigest() for image in reference_images_base64 or []]
    # Images without a mapping (CLI batch processing) show the people
    mapping = reference_mapping or [{} for _ in images]
    if len(mapping) != len(images):
        raise ValueError(f"{len(mapping)} reference mapping entries for {len(images)} reference images")
    groups = {"people": [], "style": [], "background": []}
    for entry, image in zip(mapping, images):
        group = ("style" if entry.get("is_style_reference")
                 else "background" if entry.get("reference_role") == "environment" else "people")
        groups[group].append([{key: value for key, value in entry.items() if key != "image_index"}, image])

    config = prompt_config
    has_background = bool(groups["background"])
    return {
        "people": _digest(groups["people"], cluster_description),
        "style": _digest(groups["style"], has_style_reference),
        "background": _digest(groups["background"]),
        "instructions": _digest(config.custom_instructions if config else None),
        "count": _digest(num_images),
        "expression": _digest(config.facial_expression if config else None),
        "titles": _digest(selected_titles or []),
        "ideas_model": _digest(
            [config.provider, config.model, config.thinking_level if config.thinking_enabled else None]
            if config else None
        ),
        "avoid_repeat": _digest(config.history_prompts or [] if config and config.include_history else None),
        "transcription": _digest(transcription_summary),
        "skeleton_rules": _digest(
            SKELETON_SCHEMA,
            [_build_skeleton_setting_section(sample) for sample in _SETTING_RULE_SAMPLES] if has_background else None,
        ),
        "concept_rules": _digest(
            CONCEPT_SCHEMA,
            [_build_concept_setting_section(sample) for sample in _SETTING_RULE_SAMPLES] if has_background else None,
        ),
    }


def _cache_keys(inputs: dict) -> tuple[str, str]:
    """(skeleton_hash, concepts_hash) of the inputs of a batch."""
    skeleton_hash = _digest([[name, inputs[name]] for name in SKELETON_INPUTS])
    return skeleton_hash, _digest(sorted(inputs.items()))


def _changed_inputs(inputs: dict, batch: dict) -> list[str]:
    """Labels of the inputs that differ from a cached batch's."""
    return list(dict.fromkeys(
        CACHE_INPUT_LABELS[name] for name, value in inputs.items() if batch["inputs"].get(name) != value
    ))


def _explain_miss(inputs: dict, batches: list) -> str:
    """Why no cached batch fits: what differs from the closest one."""
    if not batches:
        return "no cached ideas for this video yet"
    closest = min(batches, key=lambda batch: len(_changed_inputs(inputs, batch)))  # newest wins ties
    return f"the closest of {len(batches)} cached batches differs in: {', '.join(_changed_inputs(inputs, closest))}"


def _concept_to_thumbnail_image(
    concept: dict,
    skeleton: dict,
    reference_mapping,
    fallback_video_title: str,
) -> "ThumbnailImage":
    """Combine the (per-image) concept with the (per-batch) skeleton into the
    flat ThumbnailImage shape downstream code expects."""
    base_cam = skeleton.get("camera_and_lighting_base") or {}
    cam = concept.get("camera_and_lighting_override") or base_cam
    palette = (
        concept.get("colors")
        or skeleton.get("colors_palette_base")
        or ["#FF0000", "#FFFFFF"]
    )
    scene = concept.get("scene") or ""
    pose = concept.get("subject_pose") or ""
    expr = concept.get("subject_expression") or ""
    image_prompt = scene.strip() or f"{pose}. {expr}".strip(". ")
    return ThumbnailImage(
        image_index=concept.get("image_index", 0),
        concept_name=concept.get("concept_name") or f"Image {concept.get('image_index', 0)}",
        thumbnail_concept=concept.get("thumbnail_concept", ""),
        suggested_title=concept.get("suggested_title") or fallback_video_title,
        image_prompt=image_prompt,
        text_overlay=concept["text_in_image"] or "",
        mood=concept.get("mood", "professional"),
        colors=palette,
        key_topics=concept.get("key_topics", []),
        # Scene fields come from the concept
        scene=concept.get("scene"),
        subject_pose=concept.get("subject_pose"),
        subject_expression=concept.get("subject_expression"),
        subject_action=concept.get("subject_action"),
        visual_elements=concept.get("visual_elements"),
        text_in_image=concept["text_in_image"],
        background=concept.get("background"),
        environment_effects=concept.get("environment_effects"),
        # Camera & lighting: per-concept override > skeleton base
        lighting=cam.get("lighting"),
        equipment=cam.get("equipment"),
        framing=cam.get("framing"),
        composition=cam.get("composition"),
        color_palette=palette,
        # Identity comes from the skeleton (constant across the batch); the
        # concept may only dress someone differently
        face_groups=skeleton.get("face_groups"),
        characters=skeleton.get("characters"),
        outfit_changes=concept["outfit_changes"],
        style_source=skeleton.get("style_source"),
        style_analysis=skeleton.get("style_analysis"),
        # Reference mapping travels with each image so the V3 prompt builder
        # can render the "Images N-M: face reference for person_XX" header.
        reference_mapping=reference_mapping,
    )


def _load_batches(images_file: Path) -> list:
    """Cached batches of a video, newest first ([] for none or an unreadable cache)."""
    if not images_file.exists():
        return []
    try:
        cached = orjson.loads(images_file.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        logger.warning(f"Could not load cached {images_file.name}, starting a new cache: {e}")
        return []
    if not isinstance(cached, dict) or "batches" not in cached:
        logger.info(f"{images_file.name} holds a single batch from an older version, starting a new cache")
        return []
    return cached["batches"]


def _save_batches(images_file: Path, batches: list) -> None:
    """Persist the newest CACHE_BATCHES batches to output/{video}/prompts/images.json."""
    images_file.parent.mkdir(parents=True, exist_ok=True)
    images_file.write_bytes(orjson.dumps({"batches": batches[:CACHE_BATCHES]}, option=orjson.OPT_INDENT_2))


def _cached_image(img_data: dict, video_title: str, reference_mapping) -> ThumbnailImage:
    return ThumbnailImage(
        image_index=img_data.get("image_index", 0),
        concept_name=img_data.get("concept_name", ""),
        thumbnail_concept=img_data.get("thumbnail_concept", ""),
        suggested_title=img_data.get("suggested_title", video_title),
        image_prompt=img_data.get("image_prompt", ""),
        text_overlay=img_data["text_in_image"] or "",
        mood=img_data.get("mood", "professional"),
        colors=img_data.get("colors") or ["#FF0000", "#FFFFFF"],
        key_topics=img_data.get("key_topics", []),
        scene=img_data.get("scene"),
        subject_pose=img_data.get("subject_pose"),
        subject_expression=img_data.get("subject_expression"),
        subject_action=img_data.get("subject_action"),
        visual_elements=img_data.get("visual_elements"),
        text_in_image=img_data["text_in_image"],
        background=img_data.get("background"),
        lighting=img_data.get("lighting"),
        equipment=img_data.get("equipment"),
        framing=img_data.get("framing"),
        composition=img_data.get("composition"),
        color_palette=img_data.get("color_palette"),
        face_groups=img_data.get("face_groups"),
        style_source=img_data.get("style_source"),
        style_analysis=img_data.get("style_analysis"),
        characters=img_data.get("characters"),
        outfit_changes=img_data["outfit_changes"],
        environment_effects=img_data.get("environment_effects"),
        reference_mapping=img_data.get("reference_mapping") or reference_mapping,
    )


def generate_thumbnail_images(
    transcription: str,
    video_title: str,
    output: VideoOutput,
    num_images: int = 5,
    cluster_description: Optional[str] = None,
    prompt_config: Optional["PromptGenerationConfig"] = None,
    selected_titles: Optional[list[str]] = None,
    reference_images_base64: Optional[list[str]] = None,
    has_style_reference: bool = False,
    reference_mapping: Optional[list[dict]] = None,
) -> list:
    """Two-stage thumbnail prompt generation with fine-grained caching.

    Stage 1 — SKELETON (1 LLM call): identity, style, setting and base
        camera/lighting. Keyed by `skeleton_hash` (people and their photos,
        style image, setting photos, instructions). Reused as long as the
        inputs that define identity don't change.

    Stage 2 — CONCEPTS (N LLM calls in strict series): one thumbnail concept
        per call, each call aware of the prior concepts in the batch so the
        LLM doesn't repeat itself ("tell me ANOTHER joke" pattern).
        Keyed by `concepts_hash` (every input: count, expression, starred
        titles, ideas model, avoid repeat with its history, transcription...).

    prompts/images.json keeps the newest CACHE_BATCHES batches:
      - A batch with both hashes → its images, no LLM call.
      - A batch with the skeleton hash → its skeleton, new concepts.
      - None → new skeleton and concepts; the worker log says which inputs
        differ from the closest batch.
    Avoid repeat sends the previous thumbnails, which every job adds to, so
    its concepts are always new.

    Returns a list of num_images ThumbnailImage.

    Raises:
        ValueError: the LLM has to be called and no prompt_config was given
        GranSabioError: the skeleton or a concept could not be generated
    """
    logger.info(f"Generating {num_images} thumbnail concept(s) via skeleton+N pipeline...")

    transcription_summary = get_transcription_summary(transcription, 3000)
    inputs = _cache_inputs(
        reference_mapping=reference_mapping,
        reference_images_base64=reference_images_base64,
        has_style_reference=has_style_reference,
        cluster_description=cluster_description,
        num_images=num_images,
        prompt_config=prompt_config,
        selected_titles=selected_titles,
        transcription_summary=transcription_summary,
    )
    skeleton_hash, concepts_hash = _cache_keys(inputs)

    images_file = output.output_dir / "prompts" / "images.json"
    batches = _load_batches(images_file)
    others = [batch for batch in batches
              if (batch["skeleton_hash"], batch["concepts_hash"]) != (skeleton_hash, concepts_hash)]

    # --- Full cache hit -----------------------------------------------------
    hit = next((batch for batch in batches if batch["concepts_hash"] == concepts_hash), None)
    if hit is not None:
        logger.info(f"Cache HIT (full): skeleton + {num_images} concepts reused, no LLM calls")
        try:
            _save_batches(images_file, [hit, *others])  # Newest first: kept the longest
        except OSError as e:
            logger.warning(f"Could not persist {images_file.name} cache: {e}")
        return [_cached_image(img_data, video_title, reference_mapping) for img_data in hit["images"]]

    # --- LLM calls require a config -----------------------------------------
    if prompt_config is None:
        raise ValueError("prompt_config is required: Gran Sabio LLM is the only supported backend")

    from gransabio_prompt_generator import (
        generate_skeleton_gransabio,
        generate_concept_gransabio,
    )
    style_guidance = f"Style guidance:\n{THUMBNAIL_STYLE}"

    # --- Skeleton: cached or new -------------------------------------------
    cached_skeleton = next((batch for batch in batches if batch["skeleton_hash"] == skeleton_hash), None)
    if cached_skeleton is not None:
        skeleton = cached_skeleton["skeleton"]
        changed = ", ".join(_changed_inputs(inputs, cached_skeleton))
        logger.info(f"Cache HIT (skeleton): identity/style reused, new concepts because these changed: {changed}")
    else:
        logger.info(f"Cache MISS: {_explain_miss(inputs, batches)}")
        logger.info("Stage 1/2: requesting skeleton from Gran Sabio LLM...")
        skeleton = generate_skeleton_gransabio(
            transcription=transcription_summary,
            video_title=video_title,
            config=prompt_config,
            cluster_description=cluster_description,
            thumbnail_style=style_guidance,
            reference_images_base64=reference_images_base64,
            has_style_reference=has_style_reference,
            reference_mapping=reference_mapping,
        )

    # --- Concepts: strict series with batch-local anti-repetition ----------
    logger.info(f"Stage 2/2: generating {num_images} concept(s) in strict series...")
    concepts: list[dict] = []
    images: list[ThumbnailImage] = []
    for k in range(1, num_images + 1):
        concept = generate_concept_gransabio(
            transcription=transcription_summary,
            video_title=video_title,
            image_index=k,
            total_images=num_images,
            skeleton=skeleton,
            prior_concepts=concepts,
            config=prompt_config,
            thumbnail_style=style_guidance,
            selected_titles=selected_titles,
            reference_mapping=reference_mapping,
        )
        concept["image_index"] = k
        concepts.append(concept)
        images.append(
            _concept_to_thumbnail_image(concept, skeleton, reference_mapping, video_title)
        )

    # --- Persist cache ------------------------------------------------------
    batch = {
        "skeleton_hash": skeleton_hash,
        "concepts_hash": concepts_hash,
        "inputs": inputs,
        "skeleton": skeleton,
        "images": [img.to_dict() for img in images],
    }
    try:
        _save_batches(images_file, [batch, *others])
    except OSError as e:
        logger.warning(f"Could not persist {images_file.name} cache: {e}")

    logger.success(f"Generated {num_images} thumbnail concept(s) via skeleton+N")
    return images

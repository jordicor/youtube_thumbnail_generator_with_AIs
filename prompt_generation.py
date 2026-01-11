"""
YouTube Thumbnail Generator - Prompt Generation Module
=======================================================
Uses LLM (Claude/GPT) to analyze transcription and generate
thumbnail prompts and titles.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

from config import (
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    THUMBNAIL_STYLE,
    CHARACTER_DNA,
    IDENTITY_PRESERVATION,
    CLOTHING_PRESERVATION,
)
from utils import setup_logger, VideoOutput
from transcription import get_transcription_summary

logger = setup_logger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ThumbnailImage:
    """
    A single thumbnail image to generate.

    This is the new flat structure that replaces the old hierarchical
    ThumbnailConcept → ThumbnailVariation model. Each image is now
    completely independent.
    """
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
    clothing_override: Optional[str] = None        # Different clothing from reference

    # Face identity grouping for same person with different outfits
    face_groups: Optional[dict] = None             # {"group_A": {"physical_description": "...", "characters_with_this_face": [...]}}
    style_source: Optional[str] = None             # Which character defines the outfit for final image

    # V3 fields - characters dict replaces subjects array
    characters: Optional[dict] = None              # {"person_01": {"belongs_to_face": "group_A", "outfit": "...", "identify_in_references": "..."}}
    environment_effects: Optional[str] = None      # Visual effects: glows, holographic elements, particles, etc.

    # Style reference analysis (extracted from user's style reference image)
    style_analysis: Optional[dict] = None          # {"art_type": "kawaii/chibi", "color_palette": [...], "visual_effects": [...], ...}

    def to_dict(self) -> dict:
        return asdict(self)

    def has_v2_fields(self) -> bool:
        """Check if this image has V2 structured fields populated."""
        return self.scene is not None and self.subject_pose is not None


# =============================================================================
# LEGACY DATA CLASSES (kept for backward compatibility with old concepts.json)
# =============================================================================

@dataclass
class ThumbnailVariation:
    """DEPRECATED: Use ThumbnailImage instead. Kept for backward compatibility."""
    variation_index: int
    variation_focus: str
    image_prompt: str
    text_overlay: str
    scene: Optional[str] = None
    subject_pose: Optional[str] = None
    subject_expression: Optional[str] = None
    subject_action: Optional[str] = None
    visual_elements: Optional[list[str]] = None
    text_in_image: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def has_v2_fields(self) -> bool:
        return self.scene is not None and self.subject_pose is not None


@dataclass
class ThumbnailConcept:
    """DEPRECATED: Use ThumbnailImage instead. Kept for backward compatibility."""
    concept_index: int
    concept_name: str
    thumbnail_concept: str
    suggested_title: str
    mood: str
    colors: list[str]
    key_topics: list[str]
    variations: list[ThumbnailVariation]
    subjects: Optional[list[dict]] = None
    background: Optional[str] = None
    lighting: Optional[str] = None
    equipment: Optional[str] = None
    framing: Optional[str] = None
    composition: Optional[str] = None
    color_palette: Optional[list[str]] = None
    style: Optional[str] = None
    quality: Optional[str] = None
    materials: Optional[dict] = None
    clothing_override: Optional[str] = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result['variations'] = [v.to_dict() for v in self.variations]
        return result

    def has_v2_fields(self) -> bool:
        return self.background is not None and self.lighting is not None


# Legacy class for backwards compatibility
@dataclass
class ThumbnailPrompt:
    """Generated thumbnail prompt and metadata (legacy - use ThumbnailConcept instead)"""
    video_title: str
    suggested_title: str
    thumbnail_concept: str
    image_prompt: str
    text_overlay: str
    mood: str
    colors: list[str]
    key_topics: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

ANALYSIS_PROMPT = """You are an expert YouTube thumbnail designer and video content analyst.

Analyze the following video transcription and generate {num_concepts} thumbnail concepts, each with {num_variations} variations.

VIDEO TITLE: {video_title}
{context_section}
TRANSCRIPTION (excerpt):
{transcription}

═══════════════════════════════════════════════════════════════════════════════
TASK: Generate {num_concepts} concepts × {num_variations} variations = {total_images} unique image prompts
═══════════════════════════════════════════════════════════════════════════════

CONCEPTS are different creative ideas for the thumbnail. Examples of concept approaches:
- Emotion/Reaction, Problem/Solution, Intrigue/Curiosity, Achievement, Tutorial, Humor, etc.
- Choose what fits best with the video content - these are just examples, use your creativity.

VARIATIONS are different interpretations of the same concept, like A/B testing for engagement.
Each variation should have a complete, independent image_prompt. Think of variations as:
- Different ways to visually represent the same idea
- Subtle or more noticeable differences depending on what makes sense
- Examples: different expressions, poses, camera angles, compositions, lighting, visual elements
- Use your judgment - sometimes variations can be quite similar with small tweaks,
  other times they might explore the concept from different visual angles.

The goal is to give options to test which thumbnail performs better, not to force
artificial differences. Let the variations flow naturally from the concept.

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

{style_guidance}
"""


# =============================================================================
# LLM PROVIDERS
# =============================================================================

def generate_with_claude(prompt: str) -> Optional[str]:
    """Generate response using Claude API"""

    if not ANTHROPIC_API_KEY:
        logger.warning("Anthropic API key not set")
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.content[0].text

    except ImportError:
        logger.error("anthropic package not installed")
        return None
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return None


def generate_with_openai(prompt: str) -> Optional[str]:
    """Generate response using OpenAI API"""

    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not set")
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content

    except ImportError:
        logger.error("openai package not installed")
        return None
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return None


def generate_with_gemini(prompt: str) -> Optional[str]:
    """Generate response using Gemini API"""

    if not GEMINI_API_KEY:
        logger.warning("Gemini API key not set")
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)

        return response.text

    except ImportError:
        logger.error("google-generativeai package not installed")
        return None
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None


# =============================================================================
# JSON PROMPT BUILDER FOR NANO BANANA PRO
# =============================================================================

def build_json_prompt_for_nano_banana(
    prompt: "ThumbnailPrompt",
    clothing_description: str = None
) -> dict:
    """
    Build a structured JSON prompt optimized for Nano Banana Pro.
    This format provides better face consistency and identity preservation.
    """

    json_prompt = {
        "label": f"youtube-thumbnail-{CHARACTER_DNA['character_id']}",
        "character": {
            "character_id": CHARACTER_DNA["character_id"],
            "traits": CHARACTER_DNA["traits"],
            "face_reference": "uploaded_photo",
            "face_identity_lock": f"{IDENTITY_PRESERVATION['lock_level']}: fixed identity, zero deviation from reference facial features",
            "face_match_accuracy": f"{IDENTITY_PRESERVATION['face_match_accuracy']}% exact match enforced",
            "identity_preservation_details": f"Ensure consistent {', '.join(IDENTITY_PRESERVATION['preserve_features'][:4])} in the generated image"
        },
        "negative_constraints": IDENTITY_PRESERVATION["negative_constraints"],
        "scene": {
            "description": prompt.image_prompt,
            "mood": prompt.mood,
            "expression": _get_expression_from_mood(prompt.mood)
        },
        "clothing": {
            "preserve_from_reference": CLOTHING_PRESERVATION["enabled"],
            "description": clothing_description or "Same clothing as in reference image",
            "preserve_details": CLOTHING_PRESERVATION["preserve_details"]
        },
        "technical": {
            "aspect_ratio": "16:9",
            "resolution": "1280x720",
            "style": "photorealistic, YouTube thumbnail style",
            "quality": "high-resolution, professional"
        },
        "camera": {
            "type": "DSLR",
            "lens": "35mm",
            "aperture": "f/2.8",
            "depth_of_field": "shallow background blur"
        },
        "lighting": _get_lighting_from_mood(prompt.mood),
        "composition": {
            "framing": "rule of thirds",
            "focus_subject": "center-right",
            "style": "bold, eye-catching YouTube thumbnail"
        },
        "color_palette": prompt.colors,
        "preserve_instructions": [
            "Keep the original person's face unchanged and realistic",
            "Preserve skin texture and pores",
            "Maintain 100% identical appearance as in the reference image",
            "The identity must be unmistakable",
            "Strict continuity: same wardrobe/appearance as reference",
            "Keep exact clothing colors, fabric, logos from reference image"
        ]
    }

    return json_prompt


def _get_expression_from_mood(mood: str) -> str:
    """Get facial expression based on mood"""
    expressions = {
        "excited": "excited, surprised, eyes wide open, enthusiastic smile",
        "professional": "confident, slight smile, approachable",
        "fun": "playful smile, happy, energetic",
        "mysterious": "intriguing look, slight smirk, thoughtful",
        "educational": "knowledgeable, trustworthy, friendly",
        "casual": "relaxed, natural smile, approachable",
        "shocked": "surprised, mouth slightly open, wide eyes",
        "serious": "focused, determined, intense gaze",
    }
    return expressions.get(mood.lower(), "engaging, natural expression")


def _get_lighting_from_mood(mood: str) -> dict:
    """Get lighting setup based on mood"""
    lighting_setups = {
        "excited": {
            "type": "bright, vibrant",
            "direction": "front-facing with rim light",
            "quality": "high contrast, energetic"
        },
        "professional": {
            "type": "soft, balanced",
            "direction": "three-point lighting",
            "quality": "clean, polished"
        },
        "fun": {
            "type": "colorful, bright",
            "direction": "front with colored accents",
            "quality": "playful, vibrant"
        },
        "mysterious": {
            "type": "dramatic, moody",
            "direction": "side lighting with shadows",
            "quality": "high contrast, cinematic"
        },
        "educational": {
            "type": "natural, clear",
            "direction": "soft front lighting",
            "quality": "trustworthy, professional"
        },
    }
    return lighting_setups.get(mood.lower(), {
        "type": "balanced",
        "direction": "front-facing",
        "quality": "professional"
    })


def json_prompt_to_text(json_prompt: dict) -> str:
    """
    Convert JSON prompt to text format for image generation.
    """

    parts = []

    # Scene description
    parts.append(json_prompt["scene"]["description"])
    parts.append(f"Expression: {json_prompt['scene']['expression']}")

    # Identity preservation (CRITICAL)
    parts.append("\n--- IDENTITY PRESERVATION (CRITICAL) ---")
    parts.append(f"Character ID: {json_prompt['character']['character_id']}")
    parts.append(json_prompt["character"]["face_identity_lock"])
    parts.append(json_prompt["character"]["face_match_accuracy"])
    parts.append(json_prompt["character"]["identity_preservation_details"])

    # Negative constraints
    parts.append("\n--- DO NOT ---")
    for constraint in json_prompt["negative_constraints"]:
        parts.append(f"- {constraint}")

    # Clothing
    parts.append("\n--- CLOTHING ---")
    parts.append(json_prompt["clothing"]["description"])
    if json_prompt["clothing"]["preserve_from_reference"]:
        parts.append("Preserve exact clothing from reference: " +
                    ", ".join(json_prompt["clothing"]["preserve_details"]))

    # Technical
    parts.append("\n--- TECHNICAL ---")
    parts.append(f"Style: {json_prompt['technical']['style']}")
    parts.append(f"Resolution: {json_prompt['technical']['resolution']}")

    # Lighting
    lighting = json_prompt["lighting"]
    parts.append(f"\nLighting: {lighting['type']}, {lighting['direction']}")

    # Final preservation reminders
    parts.append("\n--- PRESERVE ---")
    for instruction in json_prompt["preserve_instructions"]:
        parts.append(f"* {instruction}")

    return "\n".join(parts)


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def parse_llm_response(response: str) -> Optional[list | dict]:
    """Parse JSON response from LLM (handles both arrays and objects)"""

    if not response:
        return None

    try:
        # Try to extract JSON from response
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

        # Find JSON array (expected for multiple prompts)
        if response.startswith('['):
            # Find matching closing bracket
            bracket_count = 0
            for i, char in enumerate(response):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        response = response[:i + 1]
                        break
        # Find JSON object
        elif response.startswith('{'):
            # Find matching closing brace
            brace_count = 0
            for i, char in enumerate(response):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        response = response[:i + 1]
                        break

        return json.loads(response)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Response was: {response[:1000]}")
        return None


def generate_thumbnail_prompts(
    transcription: str,
    video_title: str,
    output: VideoOutput,
    num_variations: int = 3,
    cluster_description: Optional[str] = None
) -> list[ThumbnailPrompt]:
    """
    Generate multiple thumbnail prompt variations from video transcription.

    Args:
        transcription: Video transcription text
        video_title: Original video title/filename
        output: VideoOutput instance
        num_variations: Number of different prompt concepts to generate
        cluster_description: Optional user-provided context about the reference images

    Returns:
        List of ThumbnailPrompt variations
    """

    logger.info(f"Generating {num_variations} thumbnail prompt variations...")

    # Check if already generated with EXACT same count
    prompts_dir = output.output_dir / "prompts"
    cached_prompts_count = len(list(prompts_dir.glob("prompt_*.json"))) if prompts_dir.exists() else 0

    # Only use cache if we have EXACTLY the number requested (not more, not less)
    if cached_prompts_count == num_variations:
        logger.info(f"Loading {num_variations} cached thumbnail prompts...")
        try:
            prompts = []
            for i in range(num_variations):
                prompt_file = prompts_dir / f"prompt_{i+1}.json"
                if prompt_file.exists():
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        prompts.append(ThumbnailPrompt(video_title=video_title, **data))
            if len(prompts) == num_variations:
                return prompts
            else:
                logger.warning(f"Cache mismatch: expected {num_variations}, got {len(prompts)}. Regenerating...")
        except Exception as e:
            logger.warning(f"Could not load cached prompts: {e}")
    elif cached_prompts_count > 0:
        logger.info(f"Cache has {cached_prompts_count} prompts but {num_variations} requested. Regenerating...")

    # Prepare prompt
    summary = get_transcription_summary(transcription, 3000)

    # Build context section if cluster description is provided
    if cluster_description and cluster_description.strip():
        context_section = """
CONTEXT FOR REFERENCE IMAGES:
The user has provided this note about the person/scene in the reference images that will be sent to the image generation AI:

### START USER CONTEXT ###
{description}
### END USER CONTEXT ###

Use this information as you see fit when crafting your thumbnail concepts and image prompts.
""".format(description=cluster_description.strip())
    else:
        context_section = ""

    prompt = ANALYSIS_PROMPT.format(
        num_variations=num_variations,
        video_title=video_title,
        context_section=context_section,
        transcription=summary,
        style_guidance=f"Style guidance:\n{THUMBNAIL_STYLE}"
    )

    # Try different providers in order
    response = None

    # Try Claude first (usually best for creative tasks)
    if ANTHROPIC_API_KEY:
        logger.info(f"Using Claude to generate {num_variations} prompt variations...")
        response = generate_with_claude(prompt)

    # Fallback to OpenAI
    if not response and OPENAI_API_KEY:
        logger.info(f"Using OpenAI to generate {num_variations} prompt variations...")
        response = generate_with_openai(prompt)

    # Fallback to Gemini
    if not response and GEMINI_API_KEY:
        logger.info(f"Using Gemini to generate {num_variations} prompt variations...")
        response = generate_with_gemini(prompt)

    if not response:
        logger.error("No LLM provider available or all failed")
        return [create_fallback_prompt(video_title, summary, i) for i in range(num_variations)]

    # Parse response (expecting array of prompts)
    data = parse_llm_response(response)

    if not data:
        logger.warning("Could not parse LLM response, using fallback")
        return [create_fallback_prompt(video_title, summary, i) for i in range(num_variations)]

    # Handle both single object and array responses
    if isinstance(data, dict):
        # Single prompt returned - LLM didn't follow instructions properly
        logger.warning(f"LLM returned single object instead of array of {num_variations}. Creating variations...")
        base_prompt = ThumbnailPrompt(
            video_title=video_title,
            suggested_title=data.get('suggested_title', video_title),
            thumbnail_concept=data.get('thumbnail_concept', ''),
            image_prompt=data.get('image_prompt', ''),
            text_overlay=data.get('text_overlay', ''),
            mood=data.get('mood', 'professional'),
            colors=data.get('colors', ['#FF0000', '#FFFFFF']),
            key_topics=data.get('key_topics', [])
        )
        prompts = [base_prompt]
        # Fill remaining with fallbacks
        for i in range(1, num_variations):
            prompts.append(create_fallback_prompt(video_title, summary, i))
        logger.warning(f"Added {num_variations - 1} fallback prompts to reach {num_variations} total")

    elif isinstance(data, list):
        # Multiple prompts returned
        prompts = []
        for item in data[:num_variations]:
            prompts.append(ThumbnailPrompt(
                video_title=video_title,
                suggested_title=item.get('suggested_title', video_title),
                thumbnail_concept=item.get('thumbnail_concept', ''),
                image_prompt=item.get('image_prompt', ''),
                text_overlay=item.get('text_overlay', ''),
                mood=item.get('mood', 'professional'),
                colors=item.get('colors', ['#FF0000', '#FFFFFF']),
                key_topics=item.get('key_topics', [])
            ))

        llm_returned = len(prompts)

        # Fill with fallbacks if LLM returned fewer than requested
        if llm_returned < num_variations:
            logger.warning(f"LLM returned only {llm_returned} concepts instead of {num_variations}. Adding fallbacks...")
            while len(prompts) < num_variations:
                prompts.append(create_fallback_prompt(video_title, summary, len(prompts)))
            logger.info(f"Added {num_variations - llm_returned} fallback prompts to reach {num_variations} total")
        else:
            logger.info(f"LLM successfully returned {llm_returned} concepts as requested")

    else:
        logger.warning("Unexpected response format, using fallback")
        return [create_fallback_prompt(video_title, summary, i) for i in range(num_variations)]

    # Save all prompts
    save_prompts(prompts, output)

    logger.success(f"Generated {len(prompts)} thumbnail prompt variations")

    return prompts


def create_fallback_prompt(video_title: str, transcription: str, variation_num: int = 0) -> ThumbnailPrompt:
    """Create a basic fallback prompt when LLM is not available"""

    logger.warning(f"Creating fallback thumbnail prompt (variation {variation_num + 1})")

    # Extract some keywords from title
    words = video_title.replace('.', ' ').replace('-', ' ').replace('_', ' ').split()
    keywords = [w for w in words if len(w) > 3 and not w.isdigit()][:5]

    moods = ["professional", "exciting", "mysterious"]
    colors_sets = [
        ["#FF0000", "#FFFFFF", "#000000"],
        ["#00FF00", "#000000", "#FFFFFF"],
        ["#0000FF", "#FFFF00", "#FFFFFF"]
    ]

    mood = moods[variation_num % len(moods)]
    colors = colors_sets[variation_num % len(colors_sets)]

    return ThumbnailPrompt(
        video_title=video_title,
        suggested_title=video_title[:60],
        thumbnail_concept=f"A {mood} YouTube thumbnail featuring the video subject",
        image_prompt=f"{mood.capitalize()} YouTube thumbnail, {video_title}, modern design, bold colors, 16:9 aspect ratio, high quality, eye-catching",
        text_overlay=keywords[0] if keywords else "VIDEO",
        mood=mood,
        colors=colors,
        key_topics=keywords
    )


def save_prompts(prompts: list[ThumbnailPrompt], output: VideoOutput):
    """Save generated prompts to files"""

    prompts_dir = output.output_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    for i, prompt in enumerate(prompts):
        # Save basic prompt data
        data = {
            "suggested_title": prompt.suggested_title,
            "thumbnail_concept": prompt.thumbnail_concept,
            "image_prompt": prompt.image_prompt,
            "text_overlay": prompt.text_overlay,
            "mood": prompt.mood,
            "colors": prompt.colors,
            "key_topics": prompt.key_topics
        }

        prompt_file = prompts_dir / f"prompt_{i+1}.json"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Also save the full JSON structured prompt for Nano Banana
        json_prompt = build_json_prompt_for_nano_banana(prompt)
        json_prompt_file = prompts_dir / f"prompt_{i+1}_nano_banana.json"
        with open(json_prompt_file, 'w', encoding='utf-8') as f:
            json.dump(json_prompt, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(prompts)} prompts to: {prompts_dir}")


# =============================================================================
# NEW CONCEPT-BASED GENERATION (with variations)
# =============================================================================

def create_fallback_variation(
    concept_name: str,
    video_title: str,
    variation_index: int
) -> ThumbnailVariation:
    """Create a basic fallback variation when LLM fails"""

    focus_options = ["direct expression", "dynamic pose", "dramatic angle"]
    focus = focus_options[(variation_index - 1) % len(focus_options)]

    return ThumbnailVariation(
        variation_index=variation_index,
        variation_focus=focus,
        image_prompt=f"YouTube thumbnail, {video_title}, {concept_name}, {focus}, modern design, bold colors, 16:9 aspect ratio, high quality, eye-catching",
        text_overlay=video_title.split()[0][:10] if video_title else "VIDEO"
    )


def create_fallback_concept(
    video_title: str,
    transcription: str,
    concept_index: int,
    num_variations: int = 1
) -> ThumbnailConcept:
    """Create a basic fallback concept when LLM is not available"""

    logger.warning(f"Creating fallback thumbnail concept {concept_index}")

    # Extract some keywords from title
    words = video_title.replace('.', ' ').replace('-', ' ').replace('_', ' ').split()
    keywords = [w for w in words if len(w) > 3 and not w.isdigit()][:5]

    concept_types = ["Reaction", "Tutorial", "Mystery", "Achievement", "Behind the scenes"]
    moods = ["exciting", "professional", "mysterious", "fun", "dramatic"]
    colors_sets = [
        ["#FF0000", "#FFFFFF", "#000000"],
        ["#00FF00", "#000000", "#FFFFFF"],
        ["#0000FF", "#FFFF00", "#FFFFFF"],
        ["#FF6B00", "#FFFFFF", "#333333"],
        ["#9B59B6", "#FFFFFF", "#2C3E50"]
    ]

    concept_type = concept_types[(concept_index - 1) % len(concept_types)]
    mood = moods[(concept_index - 1) % len(moods)]
    colors = colors_sets[(concept_index - 1) % len(colors_sets)]

    # Create variations
    variations = [
        create_fallback_variation(concept_type, video_title, i + 1)
        for i in range(num_variations)
    ]

    return ThumbnailConcept(
        concept_index=concept_index,
        concept_name=f"{concept_type} Style",
        thumbnail_concept=f"A {mood} {concept_type.lower()} style thumbnail for the video",
        suggested_title=video_title[:60],
        mood=mood,
        colors=colors,
        key_topics=keywords,
        variations=variations
    )


def save_concepts(concepts: list[ThumbnailConcept], output: VideoOutput):
    """Save generated concepts to files"""

    prompts_dir = output.output_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    # Save full concepts structure
    concepts_file = prompts_dir / "concepts.json"
    with open(concepts_file, 'w', encoding='utf-8') as f:
        json.dump([c.to_dict() for c in concepts], f, indent=2, ensure_ascii=False)

    # Also save individual prompts for compatibility (flat structure)
    prompt_index = 0
    for concept in concepts:
        for variation in concept.variations:
            prompt_index += 1
            data = {
                "concept_index": concept.concept_index,
                "concept_name": concept.concept_name,
                "suggested_title": concept.suggested_title,
                "thumbnail_concept": concept.thumbnail_concept,
                "image_prompt": variation.image_prompt,
                "text_overlay": variation.text_overlay,
                "variation_index": variation.variation_index,
                "variation_focus": variation.variation_focus,
                "mood": concept.mood,
                "colors": concept.colors,
                "key_topics": concept.key_topics
            }

            prompt_file = prompts_dir / f"prompt_{prompt_index}.json"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(concepts)} concepts ({prompt_index} total prompts) to: {prompts_dir}")


# =============================================================================
# NEW IMAGE-BASED GENERATION (flat structure, no variations)
# =============================================================================

def create_fallback_image(
    video_title: str,
    transcription: str,
    image_index: int
) -> ThumbnailImage:
    """Create a basic fallback image when LLM fails"""

    logger.warning(f"Creating fallback thumbnail image {image_index}")

    # Extract some keywords from title
    words = video_title.replace('.', ' ').replace('-', ' ').replace('_', ' ').split()
    keywords = [w for w in words if len(w) > 3 and not w.isdigit()][:5]

    concept_types = ["Reaction", "Tutorial", "Mystery", "Achievement", "Behind the scenes"]
    moods = ["exciting", "professional", "mysterious", "fun", "dramatic"]
    colors_sets = [
        ["#FF0000", "#FFFFFF", "#000000"],
        ["#00FF00", "#000000", "#FFFFFF"],
        ["#0000FF", "#FFFF00", "#FFFFFF"],
        ["#FF6B00", "#FFFFFF", "#333333"],
        ["#9B59B6", "#FFFFFF", "#2C3E50"]
    ]

    concept_type = concept_types[(image_index - 1) % len(concept_types)]
    mood = moods[(image_index - 1) % len(moods)]
    colors = colors_sets[(image_index - 1) % len(colors_sets)]

    return ThumbnailImage(
        image_index=image_index,
        concept_name=f"{concept_type} Style",
        thumbnail_concept=f"A {mood} {concept_type.lower()} style thumbnail for the video",
        suggested_title=video_title[:60],
        image_prompt=f"YouTube thumbnail, {video_title}, {concept_type}, {mood}, modern design, bold colors, 16:9 aspect ratio, high quality, eye-catching",
        text_overlay=video_title.split()[0][:10] if video_title else "VIDEO",
        mood=mood,
        colors=colors,
        key_topics=keywords
    )


def compute_style_reference_hash(
    reference_images_base64: Optional[list[str]],
    has_style_reference: bool
) -> Optional[str]:
    """
    Compute a hash of the style reference image for cache validation.

    Args:
        reference_images_base64: List of reference images (first one is style ref if has_style_reference=True)
        has_style_reference: Whether the first image is a style reference

    Returns:
        MD5 hash of the style reference image, or None if no style reference
    """
    if not has_style_reference or not reference_images_base64:
        return None

    # The style reference is always the first image in the list
    style_image_b64 = reference_images_base64[0]
    return hashlib.md5(style_image_b64.encode()).hexdigest()


def save_images(
    images: list[ThumbnailImage],
    output: VideoOutput,
    style_reference_hash: Optional[str] = None
):
    """Save generated images to files with cache metadata."""

    prompts_dir = output.output_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    # Save full images structure with cache metadata
    images_file = prompts_dir / "images.json"
    cache_data = {
        "style_reference_hash": style_reference_hash,
        "images": [img.to_dict() for img in images]
    }
    with open(images_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    # Also save individual prompts for compatibility
    for img in images:
        data = {
            "image_index": img.image_index,
            "concept_name": img.concept_name,
            "suggested_title": img.suggested_title,
            "thumbnail_concept": img.thumbnail_concept,
            "image_prompt": img.image_prompt,
            "text_overlay": img.text_overlay,
            "mood": img.mood,
            "colors": img.colors,
            "key_topics": img.key_topics,
            # Scene fields
            "scene": img.scene,
            "subject_pose": img.subject_pose,
            "subject_expression": img.subject_expression,
            "subject_action": img.subject_action,
            "visual_elements": img.visual_elements,
            "text_in_image": img.text_in_image,
            "background": img.background,
            "lighting": img.lighting,
            "equipment": img.equipment,
            "framing": img.framing,
            "composition": img.composition,
            "style": img.style,
            "quality": img.quality,
            # Identity fields
            "face_groups": img.face_groups,
            "style_source": img.style_source,
            # V3 fields
            "characters": img.characters,
            "environment_effects": img.environment_effects,
            # Legacy field (deprecated, kept for compatibility)
            "subjects": img.subjects,
        }

        prompt_file = prompts_dir / f"prompt_{img.image_index}.json"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(images)} images to: {prompts_dir}")


def generate_thumbnail_images(
    transcription: str,
    video_title: str,
    output: VideoOutput,
    num_images: int = 5,
    cluster_description: Optional[str] = None,
    prompt_config: Optional["PromptGenerationConfig"] = None,
    selected_titles: Optional[list[str]] = None,
    reference_image_base64: Optional[str] = None,
    reference_images_base64: Optional[list[str]] = None,
    has_style_reference: bool = False
) -> list[ThumbnailImage]:
    """
    Generate thumbnail images (flat structure, no variations).

    Args:
        transcription: Video transcription text
        video_title: Original video title/filename
        output: VideoOutput instance
        num_images: Number of different images to generate
        cluster_description: Optional user-provided context about the reference images
        prompt_config: Optional Gran Sabio LLM configuration for prompt generation
        selected_titles: Optional list of user-selected titles to guide image generation
        reference_image_base64: Optional single external reference image for visual inspiration
        reference_images_base64: Optional list of reference images (max 20)
        has_style_reference: If True, the first reference image is a style guide

    Returns:
        List of ThumbnailImage objects
    """

    logger.info(f"Generating {num_images} thumbnail images...")

    # Compute current style reference hash for cache validation
    current_style_hash = compute_style_reference_hash(reference_images_base64, has_style_reference)

    # Check cache - look for images.json with matching count and style hash
    prompts_dir = output.output_dir / "prompts"
    images_file = prompts_dir / "images.json"

    if images_file.exists():
        try:
            with open(images_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            # Handle both new format (dict with metadata) and legacy format (list)
            if isinstance(cached_data, dict):
                cached_hash = cached_data.get('style_reference_hash')
                images_list = cached_data.get('images', [])
            else:
                # Legacy format: plain list, no hash
                cached_hash = None
                images_list = cached_data

            # Validate cache: count must match AND style reference hash must match
            cache_valid = (
                len(images_list) == num_images and
                cached_hash == current_style_hash
            )

            if cache_valid:
                logger.info(f"Loading cached images ({num_images}, style_hash={cached_hash[:8] if cached_hash else 'none'})...")

                images = []
                for img_data in images_list:
                    images.append(ThumbnailImage(
                        image_index=img_data.get('image_index', 0),
                        concept_name=img_data.get('concept_name', ''),
                        thumbnail_concept=img_data.get('thumbnail_concept', ''),
                        suggested_title=img_data.get('suggested_title', video_title),
                        image_prompt=img_data.get('image_prompt', ''),
                        text_overlay=img_data.get('text_overlay', ''),
                        mood=img_data.get('mood', 'professional'),
                        colors=img_data.get('colors', ['#FF0000', '#FFFFFF']),
                        key_topics=img_data.get('key_topics', []),
                        subjects=img_data.get('subjects'),
                        scene=img_data.get('scene'),
                        subject_pose=img_data.get('subject_pose'),
                        subject_expression=img_data.get('subject_expression'),
                        subject_action=img_data.get('subject_action'),
                        visual_elements=img_data.get('visual_elements'),
                        text_in_image=img_data.get('text_in_image'),
                        background=img_data.get('background'),
                        lighting=img_data.get('lighting'),
                        equipment=img_data.get('equipment'),
                        framing=img_data.get('framing'),
                        composition=img_data.get('composition'),
                        color_palette=img_data.get('color_palette'),
                        style=img_data.get('style'),
                        quality=img_data.get('quality'),
                        materials=img_data.get('materials'),
                        clothing_override=img_data.get('clothing_override'),
                        face_groups=img_data.get('face_groups'),
                        style_source=img_data.get('style_source'),
                        style_analysis=img_data.get('style_analysis')
                    ))
                return images
            else:
                # Log reason for cache miss
                if len(images_list) != num_images:
                    logger.info(f"Cache has {len(images_list)} images but {num_images} requested. Regenerating...")
                elif cached_hash != current_style_hash:
                    logger.info(f"Style reference changed (cached={cached_hash[:8] if cached_hash else 'none'}, current={current_style_hash[:8] if current_style_hash else 'none'}). Regenerating...")
        except Exception as e:
            logger.warning(f"Could not load cached images: {e}")

    # Use Gran Sabio LLM exclusively
    if prompt_config is None:
        logger.error("prompt_config is required - Gran Sabio LLM is the only supported method")
        return None

    from gransabio_prompt_generator import generate_thumbnail_images_gransabio

    logger.info(f"Using Gran Sabio LLM ({prompt_config.provider}) to generate images...")
    data = generate_thumbnail_images_gransabio(
        transcription=get_transcription_summary(transcription, 3000),
        video_title=video_title,
        num_images=num_images,
        cluster_description=cluster_description,
        config=prompt_config,
        thumbnail_style=f"Style guidance:\n{THUMBNAIL_STYLE}",
        selected_titles=selected_titles,
        reference_image_base64=reference_image_base64,
        reference_images_base64=reference_images_base64,
        has_style_reference=has_style_reference
    )

    if data:
        logger.info(f"Gran Sabio LLM successfully generated {len(data)} images")

    if not data:
        logger.error("Gran Sabio LLM failed to generate images - no fallback available")
        return None

    # Parse images from response
    images = []

    if isinstance(data, list):
        for idx, img_data in enumerate(data[:num_images]):
            # V3/V2 fields
            scene = img_data.get('scene')
            subject_pose = img_data.get('subject_pose')
            subject_expression = img_data.get('subject_expression')

            # V3: characters dict (new format)
            characters = img_data.get('characters')

            # V3: environment_effects (new field)
            environment_effects = img_data.get('environment_effects')

            # Build fallback image_prompt from scene if not provided
            image_prompt = img_data.get('image_prompt', '')
            if not image_prompt and scene:
                image_prompt = f"{scene}. {subject_pose or ''}. {subject_expression or ''}"

            images.append(ThumbnailImage(
                image_index=idx + 1,
                concept_name=img_data.get('concept_name', f'Image {idx + 1}'),
                thumbnail_concept=img_data.get('thumbnail_concept', ''),
                suggested_title=img_data.get('suggested_title', video_title),
                image_prompt=image_prompt,
                text_overlay=img_data.get('text_overlay', ''),
                mood=img_data.get('mood', 'professional'),
                colors=img_data.get('colors', ['#FF0000', '#FFFFFF']),
                key_topics=img_data.get('key_topics', []),
                subjects=img_data.get('subjects'),
                scene=scene,
                subject_pose=subject_pose,
                subject_expression=subject_expression,
                subject_action=img_data.get('subject_action'),
                visual_elements=img_data.get('visual_elements'),
                text_in_image=img_data.get('text_in_image'),
                background=img_data.get('background'),
                lighting=img_data.get('lighting'),
                equipment=img_data.get('equipment'),
                framing=img_data.get('framing'),
                composition=img_data.get('composition'),
                color_palette=img_data.get('color_palette') or img_data.get('colors'),
                style=img_data.get('style'),
                quality=img_data.get('quality'),
                materials=img_data.get('materials'),
                clothing_override=img_data.get('clothing_override'),
                face_groups=img_data.get('face_groups'),
                style_source=img_data.get('style_source'),
                characters=characters,
                environment_effects=environment_effects,
                style_analysis=img_data.get('style_analysis')
            ))

        if len(images) < num_images:
            logger.warning(f"Gran Sabio LLM returned {len(images)} images instead of {num_images}")

    else:
        logger.error("Unexpected response format (not a list) - no fallback available")
        return None

    # Save images with style reference hash for cache validation
    save_images(images, output, style_reference_hash=current_style_hash)

    logger.success(f"Generated {len(images)} thumbnail images")

    return images


# =============================================================================
# LEGACY CONCEPT-BASED GENERATION (kept for backward compatibility)
# =============================================================================

def generate_thumbnail_concepts(
    transcription: str,
    video_title: str,
    output: VideoOutput,
    num_concepts: int = 3,
    num_variations_per_concept: int = 1,
    cluster_description: Optional[str] = None,
    prompt_config: Optional["PromptGenerationConfig"] = None,
    selected_titles: Optional[list[str]] = None,
    reference_image_base64: Optional[str] = None,
    reference_images_base64: Optional[list[str]] = None
) -> list[ThumbnailConcept]:
    """
    Generate thumbnail concepts with variations for A/B testing.

    Args:
        transcription: Video transcription text
        video_title: Original video title/filename
        output: VideoOutput instance
        num_concepts: Number of different creative concepts to generate
        num_variations_per_concept: Number of variations per concept
        cluster_description: Optional user-provided context about the reference images
        prompt_config: Optional Gran Sabio LLM configuration for prompt generation
        selected_titles: Optional list of user-selected titles to guide image generation
        reference_image_base64: Optional single external reference image for visual inspiration
        reference_images_base64: Optional list of reference images (max 20)

    Returns:
        List of ThumbnailConcept objects, each with variations
    """

    total_images = num_concepts * num_variations_per_concept
    logger.info(f"Generating {num_concepts} concepts × {num_variations_per_concept} variations = {total_images} prompts...")

    # Check cache - look for concepts.json with matching structure
    prompts_dir = output.output_dir / "prompts"
    concepts_file = prompts_dir / "concepts.json"

    if concepts_file.exists():
        try:
            with open(concepts_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            # Check if cache matches requested structure
            if len(cached_data) == num_concepts:
                first_concept_vars = len(cached_data[0].get('variations', []))
                if first_concept_vars == num_variations_per_concept:
                    logger.info(f"Loading cached concepts ({num_concepts} × {num_variations_per_concept})...")

                    concepts = []
                    for c_data in cached_data:
                        variations = [
                            ThumbnailVariation(
                                variation_index=v.get('variation_index', i + 1),
                                variation_focus=v.get('variation_focus', ''),
                                image_prompt=v.get('image_prompt', ''),
                                text_overlay=v.get('text_overlay', '')
                            )
                            for i, v in enumerate(c_data.get('variations', []))
                        ]
                        concepts.append(ThumbnailConcept(
                            concept_index=c_data.get('concept_index', 0),
                            concept_name=c_data.get('concept_name', ''),
                            thumbnail_concept=c_data.get('thumbnail_concept', ''),
                            suggested_title=c_data.get('suggested_title', video_title),
                            mood=c_data.get('mood', 'professional'),
                            colors=c_data.get('colors', ['#FF0000', '#FFFFFF']),
                            key_topics=c_data.get('key_topics', []),
                            variations=variations
                        ))
                    return concepts
                else:
                    logger.info(f"Cache has {first_concept_vars} variations but {num_variations_per_concept} requested. Regenerating...")
            else:
                logger.info(f"Cache has {len(cached_data)} concepts but {num_concepts} requested. Regenerating...")
        except Exception as e:
            logger.warning(f"Could not load cached concepts: {e}")

    # Prepare prompt
    summary = get_transcription_summary(transcription, 3000)

    # Build context section if cluster description is provided
    if cluster_description and cluster_description.strip():
        context_section = """
CONTEXT FOR REFERENCE IMAGES:
The user has provided this note about the person/scene in the reference images that will be sent to the image generation AI:

### START USER CONTEXT ###
{description}
### END USER CONTEXT ###

Use this information as you see fit when crafting your thumbnail concepts and image prompts.
""".format(description=cluster_description.strip())
    else:
        context_section = ""

    # Use Gran Sabio LLM exclusively (no fallback to direct APIs)
    if prompt_config is None:
        logger.error("prompt_config is required - Gran Sabio LLM is the only supported method")
        return None

    from gransabio_prompt_generator import generate_thumbnail_concepts_gransabio

    logger.info(f"Using Gran Sabio LLM ({prompt_config.provider}) to generate concepts...")
    data = generate_thumbnail_concepts_gransabio(
        transcription=summary,
        video_title=video_title,
        num_concepts=num_concepts,
        num_variations=num_variations_per_concept,
        cluster_description=cluster_description,
        config=prompt_config,
        thumbnail_style=f"Style guidance:\n{THUMBNAIL_STYLE}",
        selected_titles=selected_titles,
        reference_image_base64=reference_image_base64,
        reference_images_base64=reference_images_base64
    )

    if data:
        logger.info(f"Gran Sabio LLM successfully generated {len(data)} concepts")

    if not data:
        logger.error("Gran Sabio LLM failed to generate concepts - no fallback available")
        return None

    # Parse concepts from response
    concepts = []

    if isinstance(data, list):
        for idx, c_data in enumerate(data[:num_concepts]):
            # Parse variations
            variations_data = c_data.get('variations', [])
            variations = []

            for v_idx, v_data in enumerate(variations_data[:num_variations_per_concept]):
                # V2 fields: scene, subject_pose, subject_expression, etc.
                # If V2 fields present but no image_prompt, build a fallback
                scene = v_data.get('scene')
                subject_pose = v_data.get('subject_pose')
                subject_expression = v_data.get('subject_expression')

                # Build fallback image_prompt from V2 fields if not provided
                image_prompt = v_data.get('image_prompt', '')
                if not image_prompt and scene:
                    # Create a basic text prompt from V2 fields for backwards compatibility
                    image_prompt = f"{scene}. {subject_pose or ''}. {subject_expression or ''}"

                variations.append(ThumbnailVariation(
                    variation_index=v_data.get('variation_index', v_idx + 1),
                    variation_focus=v_data.get('variation_focus', f'Variation {v_idx + 1}'),
                    image_prompt=image_prompt,
                    text_overlay=v_data.get('text_overlay', ''),
                    # V2 fields
                    scene=scene,
                    subject_pose=subject_pose,
                    subject_expression=subject_expression,
                    subject_action=v_data.get('subject_action'),
                    visual_elements=v_data.get('visual_elements'),
                    text_in_image=v_data.get('text_in_image')
                ))

            # Log warning if fewer variations than requested (no fallback)
            if len(variations) < num_variations_per_concept:
                logger.warning(f"Concept {idx + 1} has {len(variations)} variations instead of {num_variations_per_concept}")

            concepts.append(ThumbnailConcept(
                concept_index=idx + 1,
                concept_name=c_data.get('concept_name', f'Concept {idx + 1}'),
                thumbnail_concept=c_data.get('thumbnail_concept', ''),
                suggested_title=c_data.get('suggested_title', video_title),
                mood=c_data.get('mood', 'professional'),
                colors=c_data.get('colors', ['#FF0000', '#FFFFFF']),
                key_topics=c_data.get('key_topics', []),
                variations=variations,
                # Subject identification from reference analysis
                subjects=c_data.get('subjects'),
                # V2 fields
                background=c_data.get('background'),
                lighting=c_data.get('lighting'),
                equipment=c_data.get('equipment'),
                framing=c_data.get('framing'),
                composition=c_data.get('composition'),
                color_palette=c_data.get('color_palette') or c_data.get('colors'),
                style=c_data.get('style'),
                quality=c_data.get('quality'),
                materials=c_data.get('materials'),
                clothing_override=c_data.get('clothing_override')
            ))

        # Log warning if fewer concepts than requested (no fallback)
        if len(concepts) < num_concepts:
            logger.warning(f"Gran Sabio LLM returned {len(concepts)} concepts instead of {num_concepts}")

        logger.info(f"Gran Sabio LLM returned {len(data)} concepts (requested {num_concepts})")

    else:
        logger.error("Unexpected response format (not a list) - no fallback available")
        return None

    # Save concepts
    save_concepts(concepts, output)

    logger.success(f"Generated {len(concepts)} concepts with {num_variations_per_concept} variations each")

    return concepts


def enhance_prompt_with_face_description(
    prompt: ThumbnailPrompt,
    best_frame_path: Optional[Path] = None,
    clothing_description: str = None
) -> str:
    """
    Enhance the image prompt with structured JSON format for better
    face/identity preservation in Nano Banana Pro.

    Args:
        prompt: The base ThumbnailPrompt
        best_frame_path: Optional path to best frame for reference
        clothing_description: Optional description of clothing to preserve

    Returns:
        Enhanced prompt string for image generation with identity preservation
    """

    # Build the structured JSON prompt
    json_prompt = build_json_prompt_for_nano_banana(prompt, clothing_description)

    # Convert to text format with all preservation instructions
    enhanced_text = json_prompt_to_text(json_prompt)

    logger.debug(f"Generated enhanced prompt with identity preservation")

    return enhanced_text


def get_json_prompt_for_generation(
    prompt: ThumbnailPrompt,
    clothing_description: str = None
) -> dict:
    """
    Get the full JSON structured prompt for direct use with Nano Banana Pro API.
    """
    return build_json_prompt_for_nano_banana(prompt, clothing_description)


def enhance_prompt_with_face_description_from_text(
    image_prompt: str,
    best_frame_path: Optional[Path] = None,
    mood: str = "professional",
    colors: list = None
) -> str:
    """
    Enhance an image prompt string with identity preservation instructions.

    This is a simplified version that works directly with text prompts
    instead of requiring a ThumbnailPrompt object.

    Args:
        image_prompt: The base image prompt text
        best_frame_path: Optional path to best frame for reference
        mood: The mood/tone of the image (for expression/lighting hints)
        colors: Optional color palette

    Returns:
        Enhanced prompt string for image generation with identity preservation
    """

    # Build identity preservation section
    identity_instructions = f"""
═══════════════════════════════════════════════════════════════════════════════
IDENTITY PRESERVATION - CRITICAL
═══════════════════════════════════════════════════════════════════════════════

You have reference images of a real person. You MUST:

1. FACE: Clone the EXACT face from reference images
   - Same facial structure, features, proportions
   - Same skin tone and texture
   - Same eyes, nose, lips, jawline

2. BODY: Maintain the same body type and build

3. HAIR: Keep the EXACT same hairstyle, color, and length

4. The generated person must be UNMISTAKABLY the same person as in the references

Expression for this image: {_get_expression_from_mood(mood)}

═══════════════════════════════════════════════════════════════════════════════
"""

    # Combine with the original prompt
    enhanced_prompt = f"""{identity_instructions}

SCENE TO GENERATE:
{image_prompt}

TECHNICAL REQUIREMENTS:
- Aspect ratio: 16:9 (1280x720)
- Style: Photorealistic, professional YouTube thumbnail
- Quality: High resolution, sharp, eye-catching
"""

    logger.debug(f"Generated enhanced prompt from text with identity preservation")

    return enhanced_prompt


if __name__ == "__main__":
    # Test with sample transcription
    import sys

    test_transcription = """
    Hoy vamos a hablar sobre cómo empezar un canal de gaming en YouTube.
    Es importante elegir bien tu nicho, ya sea juegos competitivos, walkthroughs,
    o contenido de entretenimiento. La consistencia es clave para crecer.
    También necesitas buen equipo de grabación y edición.
    """

    test_title = "Canales de Gaming, consejos para empezar"

    from config import OUTPUT_DIR

    # Create a dummy output
    output = VideoOutput(Path("test_video.mp4"), OUTPUT_DIR)
    output.setup()

    result = generate_thumbnail_prompt(test_transcription, test_title, output)

    if result:
        print("\nGenerated Thumbnail Prompt:")
        print("-" * 40)
        print(f"Title: {result.suggested_title}")
        print(f"Concept: {result.thumbnail_concept}")
        print(f"Text Overlay: {result.text_overlay}")
        print(f"Mood: {result.mood}")
        print(f"Colors: {result.colors}")
        print(f"Topics: {result.key_topics}")
        print("-" * 40)
        print(f"Image Prompt:\n{result.image_prompt}")

"""
Content Generation Service
===========================
Service for generating YouTube video titles and descriptions using Gran Sabio LLM.

Supports all providers available in Gran Sabio LLM:
- Anthropic (Claude models)
- OpenAI (GPT models)
- Google (Gemini models)
- xAI (Grok models)
"""

import json
import logging
import re
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from config import GRANSABIO_LLM_URL

logger = logging.getLogger(__name__)

# Gran Sabio LLM client singleton
_gransabio_client = None
_gransabio_available = None

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


# =============================================================================
# LANGUAGE CONFIGURATION
# =============================================================================

SUPPORTED_LANGUAGES = {
    "es": "Spanish",
    "en": "English",
    "fr": "French",
    "it": "Italian",
    "de": "German",
    "pt": "Portuguese",
}


# =============================================================================
# TITLE STYLES CONFIGURATION
# =============================================================================

class TitleStyle(str, Enum):
    """Available title generation styles."""
    NEUTRAL = "neutral"
    SEO = "seo"
    CLICKBAIT = "clickbait"
    CUSTOM = "custom"


TITLE_STYLE_PROMPTS = {
    TitleStyle.NEUTRAL: {
        "name": "Neutral",
        "description": "Clear, descriptive titles without sensationalism",
        "prompt": """Generate neutral, descriptive YouTube video titles.
The titles should be:
- Clear and informative
- Professional tone
- Accurately describe the content
- No exaggeration or sensationalism
- Between 40-60 characters for optimal display""",
    },
    TitleStyle.SEO: {
        "name": "SEO Optimized",
        "description": "Keyword-focused titles for search ranking",
        "prompt": """Generate SEO-optimized YouTube video titles.
The titles should be:
- Include relevant keywords naturally
- Front-load important keywords
- Use searchable phrases people actually search for
- Include numbers when relevant (e.g., "5 Tips", "2024 Guide")
- Between 50-70 characters to maximize visibility
- Avoid clickbait but still be compelling""",
    },
    TitleStyle.CLICKBAIT: {
        "name": "Clickbait",
        "description": "Attention-grabbing titles that create curiosity",
        "prompt": """Generate attention-grabbing clickbait-style YouTube titles.
The titles should be:
- Create curiosity and intrigue
- Use emotional triggers
- Include power words (SHOCKING, INCREDIBLE, SECRET, etc.)
- May use brackets for emphasis [MUST SEE]
- Create urgency or FOMO
- Be compelling but not misleading
- Use caps strategically for emphasis
- Between 45-65 characters""",
    },
    TitleStyle.CUSTOM: {
        "name": "Custom",
        "description": "User-defined prompt for title generation",
        "prompt": "",
    },
}


# =============================================================================
# JSON SCHEMAS FOR STRUCTURED OUTPUT
# =============================================================================

TITLE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "titles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "Sequential number of the title (1, 2, 3...)"
                    },
                    "title": {
                        "type": "string",
                        "description": "The generated video title text"
                    }
                },
                "required": ["number", "title"],
                "additionalProperties": False
            }
        }
    },
    "required": ["titles"],
    "additionalProperties": False
}


DESCRIPTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "descriptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "Sequential number of the description (1, 2, 3...)"
                    },
                    "description": {
                        "type": "string",
                        "description": "The complete video description text"
                    }
                },
                "required": ["number", "description"],
                "additionalProperties": False
            }
        }
    },
    "required": ["descriptions"],
    "additionalProperties": False
}


# =============================================================================
# DESCRIPTION STYLES CONFIGURATION
# =============================================================================

class DescriptionStyle(str, Enum):
    """Available description generation styles."""
    INFORMATIVE = "informative"
    SEO = "seo"
    MINIMAL = "minimal"
    CUSTOM = "custom"


DESCRIPTION_STYLE_PROMPTS = {
    DescriptionStyle.INFORMATIVE: {
        "name": "Informative",
        "description": "Complete and professional description",
        "prompt": """Generate an informative YouTube video description.
The description should:
- Start with a compelling hook (first 2 lines are visible before "Show more")
- Summarize the main content and key points
- Be well-structured with clear paragraphs
- Provide context and value to the viewer
- Be professional but engaging""",
    },
    DescriptionStyle.SEO: {
        "name": "SEO Optimized",
        "description": "Optimized for YouTube search",
        "prompt": """Generate an SEO-optimized YouTube video description.
The description should:
- Start with the most important keywords in the first 2 lines
- Include relevant keywords naturally throughout
- Use searchable phrases that viewers actually search for
- Include variations of main keywords
- Be readable and not keyword-stuffed
- Front-load important information""",
    },
    DescriptionStyle.MINIMAL: {
        "name": "Minimal",
        "description": "Short and direct",
        "prompt": """Generate a minimal, concise YouTube video description.
The description should:
- Be brief and to the point (2-4 sentences max)
- Capture the essence of the video
- No fluff or filler content
- Direct and impactful""",
    },
    DescriptionStyle.CUSTOM: {
        "name": "Custom",
        "description": "User-defined prompt for description generation",
        "prompt": "",
    },
}


# Description length options (approximate character counts)
DESCRIPTION_LENGTHS = {
    "short": {"name": "Short", "chars": 200, "description": "~200 characters"},
    "medium": {"name": "Medium", "chars": 500, "description": "~500 characters"},
    "long": {"name": "Long", "chars": 1000, "description": "~1000 characters"},
    "very_long": {"name": "Very Long", "chars": 2000, "description": "~2000 characters"},
}


# =============================================================================
# DATA CLASSES - TITLES
# =============================================================================

@dataclass
class TitleGenerationRequest:
    """Request for title generation."""
    video_title: str
    transcription_summary: Optional[str] = None
    style: TitleStyle = TitleStyle.NEUTRAL
    custom_prompt: Optional[str] = None
    custom_instructions: Optional[str] = None
    language: str = "es"
    num_titles: int = 5
    provider: str = "anthropic"
    model: Optional[str] = None  # None = use provider default via Gran Sabio
    # Thinking mode settings
    thinking_enabled: bool = False
    thinking_level: str = "medium"  # low, medium, high


@dataclass
class TitleGenerationResult:
    """Result of title generation."""
    titles: list[str]
    provider: str
    model: str
    style: str
    language: str
    success: bool
    error: Optional[str] = None


# =============================================================================
# DATA CLASSES - DESCRIPTIONS
# =============================================================================

@dataclass
class DescriptionGenerationRequest:
    """Request for description generation."""
    video_title: str
    transcription_summary: Optional[str] = None
    style: DescriptionStyle = DescriptionStyle.INFORMATIVE
    custom_prompt: Optional[str] = None
    custom_instructions: Optional[str] = None
    language: str = "es"
    length: str = "medium"  # short, medium, long, very_long
    num_descriptions: int = 1
    include_timestamps: bool = False
    include_hashtags: bool = False
    include_emojis: bool = False
    include_social_links: bool = False
    provider: str = "anthropic"
    model: Optional[str] = None  # None = use provider default via Gran Sabio
    # Segmented transcript with real timestamps (from ElevenLabs diarization)
    timestamped_transcript: Optional[str] = None
    # Thinking mode settings
    thinking_enabled: bool = False
    thinking_level: str = "medium"  # low, medium, high


@dataclass
class DescriptionGenerationResult:
    """Result of description generation."""
    descriptions: list[str]
    provider: str
    model: str
    style: str
    language: str
    length: str
    success: bool
    error: Optional[str] = None


# =============================================================================
# SHARED FUNCTIONS
# =============================================================================

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

        if client.is_available():
            _gransabio_client = client
            _gransabio_available = True
            logger.info(f"Gran Sabio LLM client connected for content generation")
            return client
        else:
            _gransabio_available = False
            logger.error(f"Gran Sabio LLM server not available at {GRANSABIO_LLM_URL}")
            return None

    except ImportError as e:
        _gransabio_available = False
        logger.error(f"Gran Sabio LLM client not available: {e}")
        return None
    except Exception as e:
        _gransabio_available = False
        logger.error(f"Could not connect to Gran Sabio LLM: {e}")
        return None


def call_gransabio(
    prompt: str,
    provider: str = "anthropic",
    model: str = None,
    json_schema: dict = None,
    thinking_enabled: bool = False,
    thinking_level: str = "medium"
) -> Optional[str | dict]:
    """
    Call Gran Sabio LLM API with the given prompt.

    Args:
        prompt: The prompt to send
        provider: AI provider (anthropic, openai, google, xai)
        model: Specific model to use (None = use default for provider)
        json_schema: Optional JSON schema for structured output (100% format guarantee)
        thinking_enabled: Enable extended thinking/reasoning mode
        thinking_level: Thinking level (low, medium, high)

    Returns:
        Response content (str if no schema, dict if schema provided)
    """
    client = get_gransabio_client()
    if not client:
        logger.error("Gran Sabio LLM not available")
        return None

    # Default models per provider
    default_models = {
        "anthropic": "claude-sonnet-4-5",
        "openai": "gpt-5.2",
        "google": "gemini-3-flash-preview",
        "xai": "grok-4-1-fast-non-reasoning",
    }

    # Use specified model or default for provider
    generator_model = model or default_models.get(provider, "claude-sonnet-4-5")

    try:
        thinking_info = ""
        if thinking_enabled:
            thinking_info = f" with thinking ({thinking_level})"

        logger.info(f"Calling Gran Sabio LLM ({provider}/{generator_model})"
                    f"{' with JSON schema' if json_schema else ''}{thinking_info}")

        # Build generation kwargs
        gen_kwargs = {
            "prompt": prompt,
            "generator_model": generator_model,
            "qa_layers": [],
            "max_tokens": 8000,
            "temperature": 0.7,
            "max_iterations": 1,
            "verbose": False,
        }

        # Add thinking mode if enabled
        if thinking_enabled:
            model_lower = generator_model.lower()

            if "claude" in model_lower:
                # Claude uses thinking_budget_tokens
                budget = THINKING_BUDGET_TOKENS.get(thinking_level, 8000)
                gen_kwargs["thinking_budget_tokens"] = budget
                logger.info(f"Using Claude thinking mode with {budget} tokens")

            elif any(x in model_lower for x in ["o1", "o3", "gpt-5"]):
                # OpenAI reasoning models use reasoning_effort
                effort = REASONING_EFFORT.get(thinking_level, "medium")
                gen_kwargs["reasoning_effort"] = effort
                logger.info(f"Using OpenAI reasoning mode with effort: {effort}")

        # Add JSON schema if provided (enables structured output)
        if json_schema:
            gen_kwargs["json_output"] = True
            gen_kwargs["json_schema"] = json_schema
            gen_kwargs["json_retry_without_iteration"] = True
        else:
            gen_kwargs["json_output"] = False

        result = client.generate(**gen_kwargs)

        if result and result.get("content"):
            return result["content"]
        else:
            logger.error("Gran Sabio LLM returned empty content")
            return None

    except Exception as e:
        logger.error(f"Gran Sabio LLM error: {e}")
        return None


def parse_lines_response(response: str) -> list[str]:
    """Parse multi-line response into list of items."""
    lines = response.strip().split('\n')
    items = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove common prefixes (numbered lists, bullets, etc.)
        line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
        line = re.sub(r'^[-*•]\s*', '', line)
        line = line.strip('"\'')

        if line:
            items.append(line)

    return items


def parse_description_response(response: str, num_descriptions: int) -> list[str]:
    """Parse description response, handling separators for multiple descriptions."""
    if num_descriptions == 1:
        # Single description - return as-is (cleaned)
        return [response.strip()]

    # Multiple descriptions - look for separators
    # Try to split by common separators
    separators = [
        r'\n---+\n',  # ---
        r'\n===+\n',  # ===
        r'\n\*\*\*+\n',  # ***
        r'\n#{2,}\s*Description\s*\d+',  # ## Description 1
        r'\n#{2,}\s*Option\s*\d+',  # ## Option 1
    ]

    descriptions = [response]
    for sep in separators:
        if len(descriptions) >= num_descriptions:
            break
        parts = re.split(sep, response, flags=re.IGNORECASE)
        if len(parts) > 1:
            descriptions = [p.strip() for p in parts if p.strip()]

    return descriptions[:num_descriptions]


# =============================================================================
# TITLE GENERATION FUNCTIONS
# =============================================================================

def build_title_prompt(
    video_title: str,
    transcription_summary: Optional[str],
    style: TitleStyle,
    custom_prompt: Optional[str],
    custom_instructions: Optional[str],
    language: str,
    num_titles: int,
) -> str:
    """Build the prompt for title generation."""
    language_name = SUPPORTED_LANGUAGES.get(language, "Spanish")

    if style == TitleStyle.CUSTOM and custom_prompt:
        style_prompt = custom_prompt
    else:
        style_prompt = TITLE_STYLE_PROMPTS.get(style, TITLE_STYLE_PROMPTS[TitleStyle.NEUTRAL])["prompt"]

    prompt = f"""You are an expert YouTube title creator. Generate {num_titles} unique video titles.

## VIDEO INFORMATION
Original Title: {video_title}
"""

    if transcription_summary:
        prompt += f"""
Video Content Summary:
{transcription_summary}
"""

    prompt += f"""
## STYLE INSTRUCTIONS
{style_prompt}
"""

    if custom_instructions:
        prompt += f"""
## ADDITIONAL INSTRUCTIONS FROM USER
{custom_instructions}
"""

    prompt += f"""
## OUTPUT REQUIREMENTS
- Generate exactly {num_titles} titles
- All titles must be in {language_name}
- Respond with a JSON object containing a "titles" array
- Each item in the array must have "number" (1, 2, 3...) and "title" (the text)

Example format:
{{
  "titles": [
    {{"number": 1, "title": "Your first title here"}},
    {{"number": 2, "title": "Your second title here"}}
  ]
}}

Generate the titles now:"""

    return prompt


def generate_titles(request: TitleGenerationRequest) -> TitleGenerationResult:
    """Generate video titles using the specified provider and settings."""
    prompt = build_title_prompt(
        video_title=request.video_title,
        transcription_summary=request.transcription_summary,
        style=request.style,
        custom_prompt=request.custom_prompt,
        custom_instructions=request.custom_instructions,
        language=request.language,
        num_titles=request.num_titles,
    )

    titles = None
    error = None

    # Always use Gran Sabio LLM with JSON schema (no fallback)
    response = call_gransabio(
        prompt,
        provider=request.provider,
        model=request.model,
        json_schema=TITLE_JSON_SCHEMA,
        thinking_enabled=request.thinking_enabled,
        thinking_level=request.thinking_level
    )

    if response:
        # Parse JSON response (may be dict or string depending on Gran Sabio response)
        try:
            if isinstance(response, dict):
                data = response
            elif isinstance(response, str):
                data = json.loads(response)
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")

            # Extract titles from structured response
            titles_data = data.get("titles", [])
            titles = [item["title"] for item in titles_data if "title" in item]

            if titles:
                logger.info(f"Generated {len(titles)} titles successfully via Gran Sabio LLM (JSON schema)")
            else:
                error = "Gran Sabio LLM returned empty titles array"
                logger.error(error)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            error = f"Failed to parse JSON response: {e}"
            logger.error(error)
    else:
        error = "Gran Sabio LLM failed to generate titles"
        logger.error(error)

    if titles is None or len(titles) == 0:
        if error is None:
            error = "Failed to generate titles"
        return TitleGenerationResult(
            titles=[],
            provider=request.provider,
            model=request.model,
            style=request.style.value,
            language=request.language,
            success=False,
            error=error,
        )

    return TitleGenerationResult(
        titles=titles,
        provider=request.provider,
        model=request.model,
        style=request.style.value,
        language=request.language,
        success=True,
    )


# =============================================================================
# DESCRIPTION GENERATION FUNCTIONS
# =============================================================================

def build_description_prompt(
    video_title: str,
    transcription_summary: Optional[str],
    style: DescriptionStyle,
    custom_prompt: Optional[str],
    custom_instructions: Optional[str],
    language: str,
    length: str,
    num_descriptions: int,
    include_timestamps: bool,
    include_hashtags: bool,
    include_emojis: bool,
    include_social_links: bool,
    timestamped_transcript: Optional[str] = None,
) -> str:
    """Build the prompt for description generation."""
    language_name = SUPPORTED_LANGUAGES.get(language, "Spanish")
    length_config = DESCRIPTION_LENGTHS.get(length, DESCRIPTION_LENGTHS["medium"])
    target_chars = length_config["chars"]

    if style == DescriptionStyle.CUSTOM and custom_prompt:
        style_prompt = custom_prompt
    else:
        style_prompt = DESCRIPTION_STYLE_PROMPTS.get(
            style, DESCRIPTION_STYLE_PROMPTS[DescriptionStyle.INFORMATIVE]
        )["prompt"]

    prompt = f"""You are an expert YouTube description writer. Generate {num_descriptions} unique video description(s).

## VIDEO INFORMATION
Video Title: {video_title}
"""

    if transcription_summary:
        prompt += f"""
Video Content/Transcript:
{transcription_summary}
"""

    prompt += f"""
## STYLE INSTRUCTIONS
{style_prompt}

## LENGTH REQUIREMENT
Target length: approximately {target_chars} characters per description.
"""

    # Additional options
    extras = []
    if include_timestamps:
        if timestamped_transcript:
            # We have real timestamps from ElevenLabs diarization
            extras.append(f"""
TIMESTAMPS: Include a timestamps section with key moments from the video.
You have access to the REAL timestamps from the video transcription below.
Analyze the content and select the most important moments/topics to include as chapters.

REAL TIMESTAMPED TRANSCRIPT:
{timestamped_transcript}

INSTRUCTIONS FOR TIMESTAMPS:
- Select 5-10 key moments that represent major topic changes or important points
- Use the EXACT timestamps from the transcript above (do NOT invent timestamps)
- Write a brief, descriptive label for each timestamp
- Format as:
00:00 - Introduction
[MM:SS] - [Topic description]
etc.""")
        else:
            # No real timestamps available, ask AI to estimate
            extras.append("""
TIMESTAMPS: Include a timestamps section with key moments from the video.
Format timestamps as:
00:00 - Introduction
01:23 - [Topic 1]
05:45 - [Topic 2]
etc.
Note: Since real timestamps are not available, estimate reasonable timestamps based on the content flow.""")

    if include_hashtags:
        extras.append("""
HASHTAGS: Include 3-5 relevant hashtags at the end of the description.
Format: #hashtag1 #hashtag2 #hashtag3""")

    if include_emojis:
        extras.append("""
EMOJIS: Use relevant emojis throughout the description to make it more engaging and visually appealing.
Don't overuse them - 3-6 emojis total is ideal.""")

    if include_social_links:
        extras.append("""
SOCIAL LINKS SECTION: Include a section for social media links at the end.
Use placeholders like:
---
Follow me:
Instagram: [INSTAGRAM_URL]
Twitter/X: [TWITTER_URL]
TikTok: [TIKTOK_URL]
Website: [WEBSITE_URL]
---""")

    if extras:
        prompt += "\n## ADDITIONAL ELEMENTS\n" + "\n".join(extras)

    if custom_instructions:
        prompt += f"""
## ADDITIONAL INSTRUCTIONS FROM USER
{custom_instructions}
"""

    prompt += f"""
## OUTPUT REQUIREMENTS
- Generate exactly {num_descriptions} description(s)
- All content must be in {language_name}
- Write naturally and engagingly
- Respond with a JSON object containing a "descriptions" array
- Each item in the array must have "number" (1, 2, 3...) and "description" (the full text)

Example format:
{{
  "descriptions": [
    {{"number": 1, "description": "Your first description here..."}},
    {{"number": 2, "description": "Your second description here..."}}
  ]
}}

Generate the description(s) now:"""

    return prompt


def generate_descriptions(request: DescriptionGenerationRequest) -> DescriptionGenerationResult:
    """Generate video descriptions using the specified provider and settings."""
    prompt = build_description_prompt(
        video_title=request.video_title,
        transcription_summary=request.transcription_summary,
        style=request.style,
        custom_prompt=request.custom_prompt,
        custom_instructions=request.custom_instructions,
        language=request.language,
        length=request.length,
        num_descriptions=request.num_descriptions,
        include_timestamps=request.include_timestamps,
        include_hashtags=request.include_hashtags,
        include_emojis=request.include_emojis,
        include_social_links=request.include_social_links,
        timestamped_transcript=request.timestamped_transcript,
    )

    descriptions = None
    error = None

    # Always use Gran Sabio LLM with JSON schema (no fallback)
    response = call_gransabio(
        prompt,
        provider=request.provider,
        model=request.model,
        json_schema=DESCRIPTION_JSON_SCHEMA,
        thinking_enabled=request.thinking_enabled,
        thinking_level=request.thinking_level
    )

    if response:
        # Parse JSON response (may be dict or string depending on Gran Sabio response)
        try:
            if isinstance(response, dict):
                data = response
            elif isinstance(response, str):
                data = json.loads(response)
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")

            # Extract descriptions from structured response
            descriptions_data = data.get("descriptions", [])
            descriptions = [item["description"] for item in descriptions_data if "description" in item]

            if descriptions:
                logger.info(f"Generated {len(descriptions)} description(s) successfully via Gran Sabio LLM (JSON schema)")
            else:
                error = "Gran Sabio LLM returned empty descriptions array"
                logger.error(error)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            error = f"Failed to parse JSON response: {e}"
            logger.error(error)
    else:
        error = "Gran Sabio LLM failed to generate descriptions"
        logger.error(error)

    if descriptions is None or len(descriptions) == 0:
        if error is None:
            error = "Failed to generate descriptions"
        return DescriptionGenerationResult(
            descriptions=[],
            provider=request.provider,
            model=request.model,
            style=request.style.value,
            language=request.language,
            length=request.length,
            success=False,
            error=error,
        )

    return DescriptionGenerationResult(
        descriptions=descriptions,
        provider=request.provider,
        model=request.model,
        style=request.style.value,
        language=request.language,
        length=request.length,
        success=True,
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_available_providers() -> dict:
    """
    Get list of available providers and their models from Gran Sabio LLM.

    Fetches models dynamically from the Gran Sabio LLM API.
    Returns dict mapping provider names to model lists.
    """
    import requests
    from config import GRANSABIO_LLM_URL

    try:
        response = requests.get(f"{GRANSABIO_LLM_URL}/models", timeout=5)
        response.raise_for_status()
        data = response.json()

        # Provider display names
        provider_names = {
            "anthropic": "Anthropic",
            "openai": "OpenAI",
            "google": "Google (Gemini)",
            "xai": "xAI (Grok)",
        }

        # Filter to only include main providers (exclude openrouter, ollama)
        main_providers = ["anthropic", "openai", "google", "xai"]
        result = {}

        for provider in main_providers:
            if provider in data:
                result[provider] = {
                    "name": provider_names.get(provider, provider.capitalize()),
                    "models": [
                        {
                            "id": model.get("key", model.get("model_id", "")),
                            "name": model.get("name", ""),
                            "description": model.get("description", "")[:80] if model.get("description") else ""
                        }
                        for model in data[provider]
                    ]
                }

        logger.info(f"Loaded {sum(len(p['models']) for p in result.values())} models from Gran Sabio LLM for content generation")
        return result

    except Exception as e:
        logger.error(f"Could not fetch models from Gran Sabio LLM: {e}")
        return {}


def get_available_title_styles() -> list[dict]:
    """Get list of available title styles."""
    return [
        {
            "id": style.value,
            "name": config["name"],
            "description": config["description"],
        }
        for style, config in TITLE_STYLE_PROMPTS.items()
    ]


def get_available_description_styles() -> list[dict]:
    """Get list of available description styles."""
    return [
        {
            "id": style.value,
            "name": config["name"],
            "description": config["description"],
        }
        for style, config in DESCRIPTION_STYLE_PROMPTS.items()
    ]


def get_available_description_lengths() -> list[dict]:
    """Get list of available description lengths."""
    return [
        {
            "id": length_id,
            "name": config["name"],
            "description": config["description"],
        }
        for length_id, config in DESCRIPTION_LENGTHS.items()
    ]


def get_available_languages() -> list[dict]:
    """Get list of supported languages."""
    return [
        {"code": code, "name": name}
        for code, name in SUPPORTED_LANGUAGES.items()
    ]

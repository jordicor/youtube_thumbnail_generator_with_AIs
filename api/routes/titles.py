"""
Content Generation API Routes

Endpoints for AI-powered title and description generation.
"""

import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from typing import Optional, Literal, List
from pydantic import BaseModel, Field

from database.db import (
    get_db,
    get_titles_for_video,
    create_titles_batch,
    delete_title as db_delete_title,
    get_descriptions_for_video,
    create_descriptions_batch,
    delete_description as db_delete_description,
)
from config import OUTPUT_DIR
from utils import VideoOutput
from transcription import transcribe_video
from transcript_processing import get_formatted_transcript_with_timestamps, has_multiple_speakers
from services.content_generation_service import (
    # Title imports
    TitleGenerationRequest,
    TitleStyle,
    generate_titles,
    get_available_title_styles,
    # Description imports
    DescriptionGenerationRequest,
    DescriptionStyle,
    generate_descriptions,
    get_available_description_styles,
    get_available_description_lengths,
    # Shared imports
    get_available_providers,
    get_available_languages,
)
from i18n.i18n import translate as t


logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# REQUEST/RESPONSE MODELS - TITLES
# ============================================================================

class GenerateTitlesRequest(BaseModel):
    """Request body for title generation."""
    video_id: int
    style: Literal["neutral", "seo", "clickbait", "custom"] = "neutral"
    custom_prompt: Optional[str] = Field(None)
    custom_instructions: Optional[str] = Field(None)
    language: Literal["es", "en", "fr", "it", "de", "pt"] = "es"
    num_titles: int = Field(5, ge=1, le=10)
    provider: Literal["anthropic", "openai", "google", "xai"] = "anthropic"
    model: Optional[str] = Field(None)
    # Thinking mode settings
    thinking_enabled: bool = False
    thinking_level: Literal["low", "medium", "high"] = "medium"


class GenerateTitlesResponse(BaseModel):
    """Response from title generation."""
    success: bool
    titles: List[str]
    provider: str
    model: str
    style: str
    language: str
    transcription_generated: bool = False
    error: Optional[str] = None


# ============================================================================
# REQUEST/RESPONSE MODELS - DESCRIPTIONS
# ============================================================================

class GenerateDescriptionsRequest(BaseModel):
    """Request body for description generation."""
    video_id: int
    style: Literal["informative", "seo", "minimal", "custom"] = "informative"
    custom_prompt: Optional[str] = Field(None)
    custom_instructions: Optional[str] = Field(None)
    language: Literal["es", "en", "fr", "it", "de", "pt"] = "es"
    length: Literal["short", "medium", "long", "very_long"] = "medium"
    num_descriptions: int = Field(1, ge=1, le=3)
    include_timestamps: bool = False
    include_hashtags: bool = False
    include_emojis: bool = False
    include_social_links: bool = False
    provider: Literal["anthropic", "openai", "google", "xai"] = "anthropic"
    model: Optional[str] = Field(None)
    # Thinking mode settings
    thinking_enabled: bool = False
    thinking_level: Literal["low", "medium", "high"] = "medium"


class GenerateDescriptionsResponse(BaseModel):
    """Response from description generation."""
    success: bool
    descriptions: List[str]
    provider: str
    model: str
    style: str
    language: str
    length: str
    transcription_generated: bool = False
    error: Optional[str] = None


# ============================================================================
# SHARED MODELS
# ============================================================================

class StyleInfo(BaseModel):
    id: str
    name: str
    description: str


class LengthInfo(BaseModel):
    id: str
    name: str
    description: str


class LanguageInfo(BaseModel):
    code: str
    name: str


# ============================================================================
# REQUEST/RESPONSE MODELS - SAVED TITLES
# ============================================================================

class SaveTitlesRequest(BaseModel):
    """Request body for saving generated titles."""
    video_id: int
    titles: List[str]
    style: Optional[str] = None
    language: str = "es"
    provider: Optional[str] = None
    model: Optional[str] = None


class SaveTitlesResponse(BaseModel):
    """Response from saving titles."""
    success: bool
    saved_count: int
    title_ids: List[int]


class SavedTitle(BaseModel):
    """A saved title with metadata."""
    id: int
    title_text: str
    style: Optional[str] = None
    language: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    created_at: str


class GetTitlesResponse(BaseModel):
    """Response with saved titles for a video."""
    success: bool
    titles: List[SavedTitle]
    count: int


# ============================================================================
# REQUEST/RESPONSE MODELS - SAVED DESCRIPTIONS
# ============================================================================

class SaveDescriptionsRequest(BaseModel):
    """Request body for saving generated descriptions."""
    video_id: int
    descriptions: List[str]
    style: Optional[str] = None
    language: str = "es"
    length: str = "medium"
    provider: Optional[str] = None
    model: Optional[str] = None
    include_timestamps: bool = False
    include_hashtags: bool = False
    include_emojis: bool = False
    include_social_links: bool = False


class SaveDescriptionsResponse(BaseModel):
    """Response from saving descriptions."""
    success: bool
    saved_count: int
    description_ids: List[int]


class SavedDescription(BaseModel):
    """A saved description with metadata."""
    id: int
    description_text: str
    style: Optional[str] = None
    language: Optional[str] = None
    length: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    include_timestamps: Optional[bool] = None
    include_hashtags: Optional[bool] = None
    include_emojis: Optional[bool] = None
    include_social_links: Optional[bool] = None
    created_at: str


class GetDescriptionsResponse(BaseModel):
    """Response with saved descriptions for a video."""
    success: bool
    descriptions: List[SavedDescription]
    count: int


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_or_create_transcription(video_path: Path, video_filename: str) -> tuple[Optional[str], bool]:
    """
    Get existing transcription or create one if it doesn't exist.

    Returns:
        tuple: (transcription_text, was_generated)
    """
    output = VideoOutput(video_path, Path(OUTPUT_DIR))

    if output.transcription_file.exists():
        logger.info(f"Loading existing transcription for {video_filename}")
        try:
            with open(output.transcription_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                transcription_lines = [l for l in lines if not l.startswith('#')]
                return '\n'.join(transcription_lines).strip(), False
        except Exception as e:
            logger.error(f"Error reading transcription file: {e}")
            return None, False

    logger.info(f"Generating transcription for {video_filename}...")
    output.setup()
    transcription = transcribe_video(video_path, output)

    if transcription:
        logger.info(f"Transcription generated successfully for {video_filename}")
        return transcription, True
    else:
        logger.warning(f"Failed to generate transcription for {video_filename}")
        return None, False  # was_generated=False because it failed


async def get_video_info(video_id: int) -> tuple[str, Path]:
    """Get video title and path from database."""
    async with get_db() as db:
        query = "SELECT filename, filepath FROM videos WHERE id = ?"
        async with db.execute(query, (video_id,)) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

    video_title = row[0]
    video_path = Path(row[1])

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=t('api.errors.video_file_not_found'))

    return video_title, video_path


# ============================================================================
# TITLE ENDPOINTS
# ============================================================================

@router.post("/generate", response_model=GenerateTitlesResponse)
async def generate_video_titles(request: GenerateTitlesRequest):
    """Generate AI-powered titles for a video."""

    if request.style == "custom" and not request.custom_prompt:
        raise HTTPException(status_code=400, detail=t('api.errors.custom_prompt_required'))

    # Model validation is handled by Gran Sabio LLM internally

    video_title, video_path = await get_video_info(request.video_id)

    # Run blocking transcription in thread pool to avoid blocking event loop
    transcription, was_generated = await run_in_threadpool(
        get_or_create_transcription, video_path, video_title
    )

    gen_request = TitleGenerationRequest(
        video_title=video_title,
        transcription_summary=transcription[:30000] if transcription else None,
        style=TitleStyle(request.style),
        custom_prompt=request.custom_prompt,
        custom_instructions=request.custom_instructions,
        language=request.language,
        num_titles=request.num_titles,
        provider=request.provider,
        model=request.model,  # None = Gran Sabio LLM uses default for provider
        thinking_enabled=request.thinking_enabled,
        thinking_level=request.thinking_level,
    )

    # Run blocking API call in thread pool
    result = await run_in_threadpool(generate_titles, gen_request)

    return GenerateTitlesResponse(
        success=result.success,
        titles=result.titles,
        provider=result.provider,
        model=result.model,
        style=result.style,
        language=result.language,
        transcription_generated=was_generated,
        error=result.error,
    )


# ============================================================================
# SAVED TITLES ENDPOINTS
# ============================================================================

@router.get("/saved/{video_id}", response_model=GetTitlesResponse)
async def get_saved_titles(video_id: int):
    """Get all saved titles for a video."""
    # Verify video exists
    async with get_db() as db:
        query = "SELECT id FROM videos WHERE id = ?"
        async with db.execute(query, (video_id,)) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

    titles = await get_titles_for_video(video_id)

    return GetTitlesResponse(
        success=True,
        titles=[
            SavedTitle(
                id=t['id'],
                title_text=t['title_text'],
                style=t['style'],
                language=t['language'],
                provider=t['provider'],
                model=t['model'],
                created_at=str(t['created_at']) if t['created_at'] else ""
            )
            for t in titles
        ],
        count=len(titles)
    )


@router.post("/save", response_model=SaveTitlesResponse)
async def save_titles(request: SaveTitlesRequest):
    """Save generated titles to the database."""
    # Verify video exists
    async with get_db() as db:
        query = "SELECT id FROM videos WHERE id = ?"
        async with db.execute(query, (request.video_id,)) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

    if not request.titles:
        return SaveTitlesResponse(success=True, saved_count=0, title_ids=[])

    title_ids = await create_titles_batch(
        video_id=request.video_id,
        titles=request.titles,
        style=request.style,
        language=request.language,
        provider=request.provider,
        model=request.model
    )

    logger.info(f"Saved {len(title_ids)} titles for video {request.video_id}")

    return SaveTitlesResponse(
        success=True,
        saved_count=len(title_ids),
        title_ids=title_ids
    )


@router.delete("/delete/{title_id}")
async def delete_title(title_id: int):
    """Delete a saved title by ID."""
    deleted = await db_delete_title(title_id)

    if deleted == 0:
        raise HTTPException(status_code=404, detail=t('api.errors.title_not_found'))

    return {"success": True, "deleted_id": title_id}


@router.delete("/delete-batch")
async def delete_titles_batch(request: dict):
    """Delete multiple titles by IDs."""
    ids = request.get("ids", [])
    if not ids:
        return {"success": True, "deleted_count": 0}

    deleted_count = 0
    for title_id in ids:
        try:
            deleted = await db_delete_title(title_id)
            deleted_count += deleted
        except Exception as e:
            print(f"Error deleting title {title_id}: {e}")

    return {"success": True, "deleted_count": deleted_count}


# ============================================================================
# DESCRIPTION ENDPOINTS
# ============================================================================

@router.post("/generate-description", response_model=GenerateDescriptionsResponse)
async def generate_video_descriptions(request: GenerateDescriptionsRequest):
    """Generate AI-powered descriptions for a video."""

    if request.style == "custom" and not request.custom_prompt:
        raise HTTPException(status_code=400, detail=t('api.errors.custom_prompt_required'))

    # Model validation is handled by Gran Sabio LLM internally

    video_title, video_path = await get_video_info(request.video_id)

    # Run blocking transcription in thread pool to avoid blocking event loop
    transcription, was_generated = await run_in_threadpool(
        get_or_create_transcription, video_path, video_title
    )

    # Load timestamped transcript if timestamps are requested
    timestamped_transcript = None
    if request.include_timestamps:
        output = VideoOutput(video_path, Path(OUTPUT_DIR))
        json_path = output.transcription_file.with_suffix('.json')

        if json_path.exists():
            # Check if video has multiple speakers (for interview/podcast format)
            include_speaker = await run_in_threadpool(has_multiple_speakers, json_path)

            # Get formatted transcript with real timestamps
            timestamped_transcript = await run_in_threadpool(
                get_formatted_transcript_with_timestamps,
                json_path,
                1.5,  # pause_threshold
                True,  # for_ai (optimized format)
                include_speaker
            )

            if timestamped_transcript:
                # Limit size to avoid token issues (keep first ~15000 chars)
                if len(timestamped_transcript) > 15000:
                    timestamped_transcript = timestamped_transcript[:15000] + "\n[... transcript continues ...]"
                logger.info(f"Loaded timestamped transcript ({len(timestamped_transcript)} chars)")
            else:
                logger.warning("Failed to load timestamped transcript, will use estimated timestamps")
        else:
            logger.info("No transcription JSON found, will use estimated timestamps")

    gen_request = DescriptionGenerationRequest(
        video_title=video_title,
        transcription_summary=transcription[:30000] if transcription else None,
        style=DescriptionStyle(request.style),
        custom_prompt=request.custom_prompt,
        custom_instructions=request.custom_instructions,
        language=request.language,
        length=request.length,
        num_descriptions=request.num_descriptions,
        include_timestamps=request.include_timestamps,
        include_hashtags=request.include_hashtags,
        include_emojis=request.include_emojis,
        include_social_links=request.include_social_links,
        provider=request.provider,
        model=request.model,  # None = Gran Sabio LLM uses default for provider
        timestamped_transcript=timestamped_transcript,
        thinking_enabled=request.thinking_enabled,
        thinking_level=request.thinking_level,
    )

    # Run blocking API call in thread pool
    result = await run_in_threadpool(generate_descriptions, gen_request)

    return GenerateDescriptionsResponse(
        success=result.success,
        descriptions=result.descriptions,
        provider=result.provider,
        model=result.model,
        style=result.style,
        language=result.language,
        length=result.length,
        transcription_generated=was_generated,
        error=result.error,
    )


# ============================================================================
# SAVED DESCRIPTIONS ENDPOINTS
# ============================================================================

@router.get("/saved-descriptions/{video_id}", response_model=GetDescriptionsResponse)
async def get_saved_descriptions(video_id: int):
    """Get all saved descriptions for a video."""
    # Verify video exists
    async with get_db() as db:
        query = "SELECT id FROM videos WHERE id = ?"
        async with db.execute(query, (video_id,)) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

    descriptions = await get_descriptions_for_video(video_id)

    return GetDescriptionsResponse(
        success=True,
        descriptions=[
            SavedDescription(
                id=d['id'],
                description_text=d['description_text'],
                style=d['style'],
                language=d['language'],
                length=d['length'],
                provider=d['provider'],
                model=d['model'],
                include_timestamps=bool(d['include_timestamps']) if d['include_timestamps'] is not None else None,
                include_hashtags=bool(d['include_hashtags']) if d['include_hashtags'] is not None else None,
                include_emojis=bool(d['include_emojis']) if d['include_emojis'] is not None else None,
                include_social_links=bool(d['include_social_links']) if d['include_social_links'] is not None else None,
                created_at=str(d['created_at']) if d['created_at'] else ""
            )
            for d in descriptions
        ],
        count=len(descriptions)
    )


@router.post("/save-descriptions", response_model=SaveDescriptionsResponse)
async def save_descriptions(request: SaveDescriptionsRequest):
    """Save generated descriptions to the database."""
    # Verify video exists
    async with get_db() as db:
        query = "SELECT id FROM videos WHERE id = ?"
        async with db.execute(query, (request.video_id,)) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

    if not request.descriptions:
        return SaveDescriptionsResponse(success=True, saved_count=0, description_ids=[])

    description_ids = await create_descriptions_batch(
        video_id=request.video_id,
        descriptions=request.descriptions,
        style=request.style,
        language=request.language,
        length=request.length,
        provider=request.provider,
        model=request.model,
        include_timestamps=request.include_timestamps,
        include_hashtags=request.include_hashtags,
        include_emojis=request.include_emojis,
        include_social_links=request.include_social_links
    )

    logger.info(f"Saved {len(description_ids)} descriptions for video {request.video_id}")

    return SaveDescriptionsResponse(
        success=True,
        saved_count=len(description_ids),
        description_ids=description_ids
    )


@router.delete("/delete-description/{description_id}")
async def delete_description(description_id: int):
    """Delete a saved description by ID."""
    deleted = await db_delete_description(description_id)

    if deleted == 0:
        raise HTTPException(status_code=404, detail=t('api.errors.description_not_found'))

    return {"success": True, "deleted_id": description_id}


@router.delete("/delete-descriptions-batch")
async def delete_descriptions_batch(request: dict):
    """Delete multiple descriptions by IDs."""
    ids = request.get("ids", [])
    if not ids:
        return {"success": True, "deleted_count": 0}

    deleted_count = 0
    for description_id in ids:
        try:
            deleted = await db_delete_description(description_id)
            deleted_count += deleted
        except Exception as e:
            logger.error(f"Error deleting description {description_id}: {e}")

    return {"success": True, "deleted_count": deleted_count}


# ============================================================================
# CHECK ENDPOINTS
# ============================================================================

@router.get("/check-transcription/{video_id}")
async def check_transcription_exists(video_id: int):
    """Quick check if transcription exists for a video."""
    async with get_db() as db:
        query = "SELECT filepath FROM videos WHERE id = ?"
        async with db.execute(query, (video_id,)) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

    video_path = Path(row[0])

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=t('api.errors.video_file_not_found'))

    output = VideoOutput(video_path, Path(OUTPUT_DIR))
    exists = output.transcription_file.exists()

    return {"exists": exists}


@router.get("/transcription/{video_id}")
async def get_transcription(video_id: int):
    """Get transcription content for a video, preferring structured segments format."""
    async with get_db() as db:
        query = "SELECT filepath FROM videos WHERE id = ?"
        async with db.execute(query, (video_id,)) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

    video_path = Path(row[0])

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=t('api.errors.video_file_not_found'))

    output = VideoOutput(video_path, Path(OUTPUT_DIR))
    segments_file = output.output_dir / "transcription_segments.json"

    # Try structured segments file first
    if segments_file.exists():
        try:
            with open(segments_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                segments = data.get('segments', [])
                if segments:
                    # Format as readable lines: [00:00] speaker_0: text
                    lines = []
                    for seg in segments:
                        timestamp = seg.get('timestamp', '00:00')
                        speaker = seg.get('speaker', 'speaker_0')
                        text = seg.get('text', '')
                        lines.append(f"[{timestamp}] {speaker}: {text}")
                    return {"exists": True, "text": '\n'.join(lines)}
        except Exception as e:
            logger.error(f"Error reading segments file: {e}")
            # Fall through to try plain text file

    # Fallback to plain text transcription
    if not output.transcription_file.exists():
        return {"exists": False, "text": None}

    try:
        with open(output.transcription_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Remove header lines (starting with # or =)
            lines = content.split('\n')
            transcription_lines = []
            skip_header = True
            for line in lines:
                if skip_header:
                    if line.startswith('#') or line.startswith('=') or line.strip() == '':
                        continue
                    skip_header = False
                transcription_lines.append(line)
            text = '\n'.join(transcription_lines).strip()
            return {"exists": True, "text": text}
    except Exception as e:
        logger.error(f"Error reading transcription file: {e}")
        raise HTTPException(status_code=500, detail=t('api.errors.error_reading_transcription'))


# ============================================================================
# INFO ENDPOINTS
# ============================================================================

@router.get("/providers")
async def get_providers():
    """Get available AI providers and their models."""
    return get_available_providers()


@router.get("/styles/titles", response_model=List[StyleInfo])
async def get_title_styles():
    """Get available title generation styles."""
    return get_available_title_styles()


@router.get("/styles/descriptions", response_model=List[StyleInfo])
async def get_description_styles():
    """Get available description generation styles."""
    return get_available_description_styles()


@router.get("/lengths", response_model=List[LengthInfo])
async def get_lengths():
    """Get available description lengths."""
    return get_available_description_lengths()


@router.get("/languages", response_model=List[LanguageInfo])
async def get_languages():
    """Get supported languages."""
    return get_available_languages()


# ============================================================================
# GRANSABIO ENDPOINTS
# ============================================================================

@router.get("/gransabio/status")
async def get_gransabio_status():
    """
    Get Gran Sabio LLM server status.

    Returns connection status and available providers.
    """
    from gransabio_prompt_generator import check_gransabio_status
    return check_gransabio_status()


@router.get("/gransabio/models")
async def get_gransabio_models():
    """
    Get available models from Gran Sabio LLM.

    Returns dict mapping provider names to model lists.
    """
    from gransabio_prompt_generator import get_available_models
    return get_available_models()

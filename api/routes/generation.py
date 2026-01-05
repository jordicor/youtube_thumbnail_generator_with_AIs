"""
Generation API Routes

Endpoints for thumbnail generation.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, List, Literal
from pydantic import BaseModel

from database.db import get_db
from services.generation_service import GenerationService
from job_queue.queue import enqueue_generation


router = APIRouter()


# ============================================================================
# MODELS
# ============================================================================

class GenerationRequest(BaseModel):
    cluster_index: int
    num_prompts: int = 5
    num_variations: int = 1
    preferred_expression: Optional[str] = None  # smiling, mouth_closed, neutral
    force_transcription: bool = False
    force_prompts: bool = False

    # Selected titles to guide image generation (optional)
    selected_titles: Optional[List[str]] = None

    # External reference image (optional)
    reference_image_base64: Optional[str] = None  # Base64 encoded image
    reference_image_use_for_prompts: bool = False  # Send to text AI for analysis
    reference_image_include_in_refs: bool = False  # Include in image generation refs

    # Image generation settings
    image_provider: Literal["gemini", "openai", "poe", "replicate"] = "gemini"
    gemini_model: Optional[Literal[
        "gemini-2.5-flash-image",
        "gemini-3-pro-image-preview"
    ]] = None
    openai_model: Optional[Literal[
        "gpt-image-1.5",
        "gpt-image-1",
        "gpt-image-1-mini",
        "dall-e-3"
    ]] = None
    poe_model: Optional[Literal[
        "flux2pro",
        "flux2flex",
        "fluxkontextpro",
        "seedream40",
        "nanobananapro",
        "Ideogram-v3"
    ]] = None

    # Reference images control (None = use model's max)
    num_reference_images: Optional[int] = None

    # Prompt generation AI settings (for generating image prompts)
    prompt_provider: Optional[Literal["anthropic", "openai", "google", "xai"]] = None
    prompt_model: Optional[str] = None  # None = use default for provider
    prompt_thinking_enabled: bool = False
    prompt_thinking_level: Literal["low", "medium", "high"] = "medium"
    prompt_custom_instructions: Optional[str] = None
    prompt_include_history: bool = False


class GenerationStatusResponse(BaseModel):
    job_id: int
    video_id: int
    status: str
    progress: int
    current_step: Optional[str] = None
    thumbnails_generated: int
    total_thumbnails: int
    error_message: Optional[str] = None


class ThumbnailResponse(BaseModel):
    id: int
    filepath: str
    prompt_index: int
    variation_index: int
    suggested_title: Optional[str] = None
    text_overlay: Optional[str] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/{video_id}/start")
async def start_generation(
    video_id: int,
    request: GenerationRequest
):
    """
    Start thumbnail generation for a video.

    Requires the video to be analyzed and have clusters.
    Generation runs in background via Redis job queue.
    """
    async with get_db() as db:
        service = GenerationService(db)

        # Verify video is analyzed
        video = await service.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        if video['status'] not in ('analyzed', 'completed'):
            raise HTTPException(
                status_code=400,
                detail=f"Video must be analyzed first. Current status: {video['status']}"
            )

        # Verify cluster exists
        cluster = await service.get_cluster(video_id, request.cluster_index)
        if not cluster:
            raise HTTPException(
                status_code=404,
                detail=f"Cluster {request.cluster_index} not found"
            )

        # Create job in database
        job = await service.create_generation_job(
            video_id=video_id,
            cluster_id=cluster['id'],
            num_prompts=request.num_prompts,
            num_variations=request.num_variations,
            preferred_expression=request.preferred_expression
        )

    # Enqueue generation job to Redis
    arq_job_id = await enqueue_generation(
        job_id=job['id'],
        force_transcription=request.force_transcription,
        force_prompts=request.force_prompts,
        image_provider=request.image_provider,
        gemini_model=request.gemini_model,
        openai_model=request.openai_model,
        poe_model=request.poe_model,
        num_reference_images=request.num_reference_images,
        # Prompt generation AI settings
        prompt_provider=request.prompt_provider,
        prompt_model=request.prompt_model,
        prompt_thinking_enabled=request.prompt_thinking_enabled,
        prompt_thinking_level=request.prompt_thinking_level,
        prompt_custom_instructions=request.prompt_custom_instructions,
        prompt_include_history=request.prompt_include_history,
        # Selected titles to guide image generation
        selected_titles=request.selected_titles,
        # External reference image
        reference_image_base64=request.reference_image_base64,
        reference_image_use_for_prompts=request.reference_image_use_for_prompts,
        reference_image_include_in_refs=request.reference_image_include_in_refs
    )

    if not arq_job_id:
        # Failed to enqueue - update job status
        async with get_db() as db:
            service = GenerationService(db)
            await service.update_job_status(job['id'], 'error', 0, 'Failed to enqueue generation job')
        raise HTTPException(status_code=500, detail="Failed to enqueue generation job")

    return {
        "job_id": job['id'],
        "video_id": video_id,
        "status": "pending",
        "message": "Generation started",
        "arq_job_id": arq_job_id
    }


@router.get("/jobs/{job_id}/status", response_model=GenerationStatusResponse)
async def get_generation_status(job_id: int):
    """
    Get status of a generation job.
    """
    async with get_db() as db:
        service = GenerationService(db)
        status = await service.get_job_status(job_id)

    if not status:
        raise HTTPException(status_code=404, detail="Job not found")

    return GenerationStatusResponse(**status)


@router.get("/jobs/{job_id}/thumbnails", response_model=List[ThumbnailResponse])
async def get_job_thumbnails(job_id: int):
    """
    Get thumbnails generated by a job.
    """
    async with get_db() as db:
        service = GenerationService(db)
        thumbnails = await service.get_job_thumbnails(job_id)

    return [ThumbnailResponse(**t) for t in thumbnails]


@router.post("/jobs/{job_id}/cancel")
async def cancel_generation(job_id: int):
    """
    Cancel a generation job in progress.
    """
    async with get_db() as db:
        service = GenerationService(db)
        success = await service.cancel_job(job_id)

    if not success:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")

    return {"job_id": job_id, "status": "cancelled"}


@router.get("/{video_id}/history")
async def get_generation_history(video_id: int):
    """
    Get generation history for a video.
    """
    async with get_db() as db:
        service = GenerationService(db)
        jobs = await service.get_video_jobs(video_id)

    return {"video_id": video_id, "jobs": jobs}


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

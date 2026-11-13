"""
Generation API Routes

Endpoints for thumbnail generation.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, model_validator

from database.db import get_db
from gransabio_prompt_generator import TextProvider, resolve_text_model
from services.generation_service import GenerationService, GenerationNotReadyError
from job_queue.queue import enqueue_generation
from i18n.i18n import translate as t
from services.reference_service import ReferenceService, reference_limit
from utils import ReferenceImageError
from starlette.concurrency import run_in_threadpool
from services.comfyui_service import Connection, ComfyUIError, prepare as prepare_comfyui


router = APIRouter()


# ============================================================================
# MODELS
# ============================================================================

class GenerationRequest(BaseModel):
    # One or more cluster database IDs of the video (any view), ordered by
    # intended person_XX role: cluster_ids[0] becomes person_01, and so on.
    # IDs, unlike cluster indices, survive reindexing and identify the view.
    cluster_ids: List[int] = Field(..., min_length=1)
    num_images: int = Field(default=5, ge=1, le=50)
    # Facial expression of the people in the thumbnails (None: each idea chooses)
    preferred_expression: Optional[Literal["smiling", "mouth_closed", "neutral"]] = None
    force_transcription: bool = False
    force_prompts: bool = False

    # Selected titles to guide image generation (optional)
    selected_titles: Optional[List[str]] = None

    # External reference image (optional)
    reference_image_base64: Optional[str] = None  # Base64 encoded image
    reference_image_use_for_prompts: bool = False  # Send to text AI for analysis
    reference_image_include_in_refs: bool = False  # Include in image generation refs

    # Image generation settings
    image_provider: Literal["gemini", "openai", "poe", "replicate", "comfyui"] = "gemini"
    gemini_model: Optional[Literal[
        "gemini-3-pro-image",
        "gemini-nano-banana-2.1"
    ]] = None
    openai_model: Optional[Literal[
        "gpt-image-2.6-sunburst",
        "gpt-image-2.6-flare",
        "gpt-image-2.5-sunburst",
        "gpt-image-2.5-flare",
        "gpt-image-2"
    ]] = None
    poe_model: Optional[Literal[
        "flux2pro",
        "flux2flex",
        "fluxkontextpro",
        "seedream40",
        "nano-banana-pro",
        "nano-banana-2.1",
        "Ideogram-v3"
    ]] = None

    comfyui_workflow: Optional[str] = None
    comfyui: Optional[Connection] = None

    # Reference images control (None = use model's max)
    num_reference_images: Optional[int] = Field(default=None, ge=1)

    # Face swap is an explicit opt-in post-processing step.
    apply_face_swap: bool = False
    restore_faces: bool = False

    # When the chosen image provider fails, try the others (off: the image fails)
    allow_provider_fallback: bool = False

    # Include the first selected person's full environment frame by default.
    include_scene_reference: bool = True

    # Prompt generation AI settings (for generating image prompts)
    prompt_provider: Optional[TextProvider] = None
    prompt_model: Optional[str] = None  # None = use default for provider
    prompt_thinking_enabled: bool = False
    prompt_thinking_level: Literal["low", "medium", "high"] = "medium"
    prompt_custom_instructions: Optional[str] = None
    prompt_include_history: bool = False

    @model_validator(mode="after")
    def _restoration_requires_swap(self):
        resolve_text_model(self.prompt_provider or "anthropic", self.prompt_model)
        if self.image_provider == "comfyui" and self.allow_provider_fallback:
            raise ValueError("ComfyUI does not allow provider fallback")
        if self.restore_faces and not self.apply_face_swap:
            raise ValueError("Face restoration requires face swap to be enabled")
        return self


class GenerationStatusResponse(BaseModel):
    job_id: int
    video_id: int
    status: str
    # A cancelled job whose worker has not stopped yet
    stopping: bool = False
    progress: int
    current_step: Optional[str] = None
    thumbnails_generated: int
    total_thumbnails: int
    error_message: Optional[str] = None


class ThumbnailResponse(BaseModel):
    id: int
    filepath: str
    image_index: int
    suggested_title: Optional[str] = None
    text_overlay: Optional[str] = None
    face_match: Optional[dict] = None  # services/face_match.py face_match_fields; None until scored


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
    # M45: Validate base64 reference image size (10MB base64 ~ 7.5MB image)
    if request.reference_image_base64 and len(request.reference_image_base64) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Reference image too large")

    comfyui_profile = None
    comfyui_config = request.comfyui.model_dump() if request.comfyui else None
    if request.image_provider == "comfyui":
        try:
            comfyui_profile, comfyui_config = await run_in_threadpool(
                prepare_comfyui, request.comfyui_workflow, comfyui_config
            )
        except (ComfyUIError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    job = None
    try:
        async with get_db() as db:
            service = GenerationService(db)

            # Verify video exists
            video = await service.get_video(video_id)
            if not video:
                raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

            # Every cluster must belong to this video. The order the user gave
            # maps to person_01, person_02, ...
            cluster_ids = request.cluster_ids
            for cluster_id in cluster_ids:
                if not await service.get_video_cluster(video_id, cluster_id):
                    raise HTTPException(
                        status_code=404,
                        detail=t('api.errors.cluster_not_found')
                    )

            # Reject duplicates: the same cluster cannot map to two person_XX roles.
            if len(set(cluster_ids)) != len(cluster_ids):
                raise HTTPException(
                    status_code=400,
                    detail="cluster_ids contains duplicates"
                )

            try:
                model = {"gemini": request.gemini_model, "openai": request.openai_model,
                         "poe": request.poe_model, "replicate": None, "comfyui": comfyui_profile["id"] if comfyui_profile else None}[request.image_provider]
                if request.reference_image_base64 and not request.reference_image_include_in_refs:
                    raise ReferenceImageError("Include the style image in generation or remove it.")
                await ReferenceService(db).prepare(
                    video_id, cluster_ids,
                    limit=reference_limit(request.image_provider, model, request.num_reference_images, comfyui_profile),
                    include_environment=request.include_scene_reference,
                    style_base64=request.reference_image_base64,
                )
            except (ReferenceImageError, ValueError) as error:
                raise HTTPException(status_code=400, detail=str(error)) from error

            # Claim and job creation share one transaction, after validation.
            try:
                job = await service.create_generation_job(
                    video_id=video_id,
                    cluster_ids=cluster_ids,
                    num_images=request.num_images,
                    preferred_expression=request.preferred_expression
                )
            except GenerationNotReadyError:
                # Read again: another request may have just claimed it
                status = (await service.get_video(video_id))['status']
                if status == 'generating':
                    raise HTTPException(
                        status_code=409, detail=t('api.errors.generation_in_progress')
                    ) from None
                raise HTTPException(
                    status_code=400,
                    detail=t('api.errors.video_must_be_analyzed', status=status)
                ) from None

        # Enqueue generation job to Redis
        arq_job_id = await enqueue_generation(
            job_id=job['id'],
            force_transcription=request.force_transcription,
            force_prompts=request.force_prompts,
            image_provider=request.image_provider,
            gemini_model=request.gemini_model,
            openai_model=request.openai_model,
            poe_model=request.poe_model,
            comfyui_workflow=comfyui_profile["id"] if comfyui_profile else None,
            comfyui_config=comfyui_config,
            comfyui_profile=comfyui_profile,
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
            reference_image_include_in_refs=request.reference_image_include_in_refs,
            # Face swap post-processing
            apply_face_swap=request.apply_face_swap,
            restore_faces=request.restore_faces,
            allow_provider_fallback=request.allow_provider_fallback,
            include_scene_reference=request.include_scene_reference
        )
        if not arq_job_id:
            raise HTTPException(status_code=500, detail=t('api.errors.failed_enqueue_generation'))
    except BaseException:
        # Also release the claim when enqueue raises or the request is cancelled.
        # A late queue delivery cannot revive the now-terminal database job.
        if job is not None:
            async with get_db() as db:
                service = GenerationService(db)
                await service.fail_enqueue(job['id'], t('api.errors.failed_enqueue_generation'))
        raise

    return {
        "job_id": job['id'],
        "video_id": video_id,
        "status": "pending",
        "message": t('api.messages.generation_started'),
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
        raise HTTPException(status_code=404, detail=t('api.errors.job_not_found'))

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
        raise HTTPException(status_code=400, detail=t('api.errors.job_cannot_cancel'))

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
    return await check_gransabio_status()


@router.get("/gransabio/models")
async def get_gransabio_models():
    """
    Get available models from Gran Sabio LLM.

    Returns dict mapping provider names to model lists.
    """
    from gransabio_prompt_generator import GranSabioError, get_available_models
    try:
        return await get_available_models()
    except GranSabioError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

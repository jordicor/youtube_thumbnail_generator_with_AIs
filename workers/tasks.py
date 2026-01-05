"""
arq task definitions for background job processing.

These tasks are executed by arq workers and communicate progress via Redis pub/sub.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Any

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from database.db import get_db
from services.analysis_service import AnalysisService
from services.generation_service import GenerationService
from job_queue.pubsub import publish_progress, publish_event, CHANNEL_ANALYSIS, CHANNEL_GENERATION

logger = logging.getLogger(__name__)


async def analyze_video(
    ctx: dict,
    video_id: int,
    force_scenes: bool = False,
    force_faces: bool = False,
    force_clustering: bool = False,
    force_transcription: bool = False,
    clustering_eps: float = 0.5,
    clustering_min_samples: int = 3,
    **kwargs  # Ignore arq internal params like _job_timeout
) -> dict:
    """
    Background task for full video analysis.

    This task runs the complete analysis pipeline:
    1. Scene detection
    2. Face extraction
    3. Face clustering
    4. Audio transcription

    Progress is reported via Redis pub/sub to the CHANNEL_ANALYSIS channel.

    Args:
        ctx: arq context (contains job_id, redis connection, etc.)
        video_id: ID of the video to analyze
        force_*: Flags to force re-processing of specific steps
        clustering_eps: DBSCAN epsilon parameter
        clustering_min_samples: DBSCAN min_samples parameter

    Returns:
        dict with status and any error message
    """
    job_id = ctx.get("job_id", "unknown")
    logger.info(f"[Job {job_id}] Starting analysis for video {video_id}")

    try:
        # Notify start
        await publish_progress(
            CHANNEL_ANALYSIS,
            status="analyzing",
            progress=5,
            message="Starting analysis...",
            video_id=video_id
        )

        async with get_db() as db:
            service = AnalysisService(db)

            # Wrap the service to report progress
            # We'll modify the service methods to accept a progress callback
            await service.run_full_analysis(
                video_id=video_id,
                force_scenes=force_scenes,
                force_faces=force_faces,
                force_clustering=force_clustering,
                force_transcription=force_transcription,
                clustering_eps=clustering_eps,
                clustering_min_samples=clustering_min_samples
            )

        # Get final status
        async with get_db() as db:
            service = AnalysisService(db)
            video = await service.get_video(video_id)
            final_status = video.get("status", "error") if video else "error"
            error_message = video.get("error_message") if video else None

        if final_status == "analyzed":
            await publish_event(
                CHANNEL_ANALYSIS,
                "complete",
                {"status": "analyzed", "progress": 100},
                video_id=video_id
            )
            logger.info(f"[Job {job_id}] Analysis completed for video {video_id}")
            return {"status": "success", "video_id": video_id}
        else:
            await publish_event(
                CHANNEL_ANALYSIS,
                "error",
                {"status": final_status, "error": error_message},
                video_id=video_id
            )
            logger.error(f"[Job {job_id}] Analysis failed for video {video_id}: {error_message}")
            return {"status": "error", "video_id": video_id, "error": error_message}

    except Exception as e:
        error_msg = str(e)
        logger.exception(f"[Job {job_id}] Analysis error for video {video_id}: {error_msg}")

        # Update video status to error
        try:
            async with get_db() as db:
                service = AnalysisService(db)
                await service.update_video_status(video_id, "error", error_msg)
        except Exception:
            pass

        # Notify error
        await publish_event(
            CHANNEL_ANALYSIS,
            "error",
            {"status": "error", "error": error_msg},
            video_id=video_id
        )

        return {"status": "error", "video_id": video_id, "error": error_msg}


async def run_generation(
    ctx: dict,
    job_id: int,
    force_transcription: bool = False,
    force_prompts: bool = False,
    image_provider: str = "gemini",
    gemini_model: Optional[str] = None,
    openai_model: Optional[str] = None,
    poe_model: Optional[str] = None,
    num_reference_images: Optional[int] = None,
    # Prompt generation AI settings
    prompt_provider: Optional[str] = None,
    prompt_model: Optional[str] = None,
    prompt_thinking_enabled: bool = False,
    prompt_thinking_level: str = "medium",
    prompt_custom_instructions: Optional[str] = None,
    prompt_include_history: bool = False,
    # Selected titles to guide image generation
    selected_titles: Optional[list] = None,
    # External reference image
    reference_image_base64: Optional[str] = None,
    reference_image_use_for_prompts: bool = False,
    reference_image_include_in_refs: bool = False,
    **kwargs  # Ignore arq internal params like _job_timeout
) -> dict:
    """
    Background task for thumbnail generation.

    This task runs the generation pipeline:
    1. Transcribe video (if needed)
    2. Generate prompts using LLM
    3. Generate thumbnail images

    Progress is reported via Redis pub/sub to the CHANNEL_GENERATION channel.

    Args:
        ctx: arq context
        job_id: ID of the generation job
        force_transcription: Force re-transcription
        force_prompts: Force prompt regeneration
        image_provider: Image generation provider (gemini, openai, poe, replicate)
        *_model: Specific model for each provider
        num_reference_images: Number of reference images to use
        prompt_provider: AI provider for prompt generation (anthropic, openai, google, xai)
        prompt_model: Specific model for prompt generation
        prompt_thinking_enabled: Enable thinking/reasoning mode
        prompt_thinking_level: Thinking level (low, medium, high)
        prompt_custom_instructions: Additional user instructions for prompt generation
        prompt_include_history: Include previous prompts to avoid repetition
        reference_image_base64: External reference image as base64
        reference_image_use_for_prompts: Use reference image for prompt analysis
        reference_image_include_in_refs: Include reference image in generation refs

    Returns:
        dict with status and any error message
    """
    arq_job_id = ctx.get("job_id", "unknown")
    logger.info(f"[arq:{arq_job_id}] Starting generation for job {job_id}")

    try:
        # Notify start
        await publish_progress(
            CHANNEL_GENERATION,
            status="starting",
            progress=5,
            message="Starting generation...",
            job_id=job_id
        )

        async with get_db() as db:
            service = GenerationService(db)

            await service.run_generation_pipeline(
                job_id=job_id,
                force_transcription=force_transcription,
                force_prompts=force_prompts,
                image_provider=image_provider,
                gemini_model=gemini_model,
                openai_model=openai_model,
                poe_model=poe_model,
                num_reference_images=num_reference_images,
                prompt_provider=prompt_provider,
                prompt_model=prompt_model,
                prompt_thinking_enabled=prompt_thinking_enabled,
                prompt_thinking_level=prompt_thinking_level,
                prompt_custom_instructions=prompt_custom_instructions,
                prompt_include_history=prompt_include_history,
                selected_titles=selected_titles,
                reference_image_base64=reference_image_base64,
                reference_image_use_for_prompts=reference_image_use_for_prompts,
                reference_image_include_in_refs=reference_image_include_in_refs
            )

        # Get final status
        async with get_db() as db:
            service = GenerationService(db)
            job = await service.get_job(job_id)
            final_status = job.get("status", "error") if job else "error"
            error_message = job.get("error_message") if job else None

        if final_status == "completed":
            await publish_event(
                CHANNEL_GENERATION,
                "complete",
                {"status": "completed", "progress": 100},
                job_id=job_id
            )
            logger.info(f"[arq:{arq_job_id}] Generation completed for job {job_id}")
            return {"status": "success", "job_id": job_id}
        else:
            await publish_event(
                CHANNEL_GENERATION,
                "error",
                {"status": final_status, "error": error_message},
                job_id=job_id
            )
            logger.error(f"[arq:{arq_job_id}] Generation failed for job {job_id}: {error_message}")
            return {"status": "error", "job_id": job_id, "error": error_message}

    except Exception as e:
        error_msg = str(e)
        logger.exception(f"[arq:{arq_job_id}] Generation error for job {job_id}: {error_msg}")

        # Update job status to error
        try:
            async with get_db() as db:
                service = GenerationService(db)
                await service.update_job_status(job_id, "error", 0, error_msg)
        except Exception:
            pass

        # Notify error
        await publish_event(
            CHANNEL_GENERATION,
            "error",
            {"status": "error", "error": error_msg},
            job_id=job_id
        )

        return {"status": "error", "job_id": job_id, "error": error_msg}

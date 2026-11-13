"""
arq task definitions for background job processing.

These tasks are executed by arq workers. Their progress is what they write to
the database: the web's streams read it from there.
"""

import logging
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from database.db import get_db
from services.analysis_service import ANALYSIS_RUNNING_STATES, AnalysisService
from services.generation_service import (
    GenerationService,
    ImageGenerationError,
    create_sync_cancellation_check,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CANCELLATION CHECK HELPERS
# =============================================================================

def create_sync_analysis_cancellation_check(video_id: int, analysis_run: str):
    """
    Create a synchronous cancellation check function for video analysis.

    This is needed because analysis steps (scene detection, face extraction, etc.)
    are synchronous, but we need to check the database for cancellation status.

    Args:
        video_id: The video ID to check
        analysis_run: The claim on the video this analysis holds

    Returns:
        A callable that returns True once the analysis must stop: it was
        cancelled, or the video is no longer its own
    """
    import sqlite3
    from config import DATABASE_PATH

    def check() -> bool:
        conn = sqlite3.connect(str(DATABASE_PATH), timeout=5)
        try:
            row = conn.execute(
                "SELECT status, analysis_run FROM videos WHERE id = ?",
                [video_id]
            ).fetchone()
            return row is None or row[1] != analysis_run or row[0] not in ANALYSIS_RUNNING_STATES
        except Exception as e:
            logger.warning(f"Error in sync cancellation check for video {video_id}: {e}")
            return False
        finally:
            conn.close()

    return check


async def check_generation_cancelled(job_id: int) -> bool:
    """
    Check if a generation job has been cancelled.

    Args:
        job_id: The generation job ID to check

    Returns:
        True if job status is 'cancelled', False otherwise
    """
    try:
        async with get_db() as db:
            service = GenerationService(db)
            job = await service.get_job(job_id)
            return job and job.get('status') == 'cancelled'
    except Exception as e:
        logger.warning(f"Error checking generation cancellation for job {job_id}: {e}")
        return False


def cancelled_result(task_type: str, task_id: int, arq_job_id: str) -> dict:
    """
    What a cancelled task that has stopped answers arq.

    Nothing the task had finished is deleted: a cancelled analysis keeps its
    files and a cancelled generation the thumbnails it had made. Each task
    frees its video when its job ends (the finally of analyze_video and
    run_generation).

    Args:
        task_type: 'analysis' or 'generation'
        task_id: The task ID (video_id for analysis, job_id for generation)
        arq_job_id: The arq job ID for logging
    """
    logger.info(f"[{arq_job_id}] Task {task_type} {task_id} was cancelled, stopping gracefully")
    id_key = "video_id" if task_type == "analysis" else "job_id"
    return {"status": "cancelled", id_key: task_id}


async def analyze_video(
    ctx: dict,
    video_id: int,
    analysis_run: str,
    reset: bool = False,
    force_scenes: bool = False,
    force_faces: bool = False,
    force_clustering: bool = False,
    force_transcription: bool = False,
    clustering_eps: float = 0.5,
    clustering_min_samples: int = 3,
    refresh_references: bool = False,
    expected_reference_revision: Optional[int] = None,
    **kwargs  # Ignore arq internal params like _job_timeout
) -> dict:
    """
    Background task for full video analysis.

    This task runs the complete analysis pipeline:
    1. Scene detection
    2. Face extraction
    3. Face clustering
    4. Audio transcription

    Args:
        ctx: arq context (contains job_id, redis connection, etc.)
        video_id: ID of the video to analyze
        analysis_run: The claim on the video this job was queued with
            (AnalysisService.claim_analysis). A job whose claim is gone, because
            it was cancelled while queued, does nothing.
        reset: Reanalysis: first delete the previous analysis with its people,
            generation jobs and thumbnails, then run every step again. Refused,
            deleting nothing, when the video file is not there to analyze.
        force_*: Flags to force re-processing of specific steps
        clustering_eps: DBSCAN epsilon parameter
        clustering_min_samples: DBSCAN min_samples parameter

    Returns:
        dict with status and any error message
    """
    job_id = ctx.get("job_id", "unknown")
    logger.info(f"[Job {job_id}] Starting analysis for video {video_id}")

    async with get_db() as db:
        service = AnalysisService(db)
        video = await service.get_video(video_id)
        if not video:
            logger.error(f"[Job {job_id}] Video {video_id} not found")
            return {"status": "error", "video_id": video_id, "error": "Video not found"}

        # A reanalysis deletes all the video has: only when it can be made again
        if reset and not await asyncio.to_thread(os.path.isfile, video["filepath"]):
            error_msg = f"Video file not found: {video['filepath']}. Nothing was deleted."
            await service.unclaim_analysis(video_id, analysis_run, error=error_msg)
            logger.error(f"[Job {job_id}] Reanalysis of video {video_id} refused: {error_msg}")
            return {"status": "error", "video_id": video_id, "error": error_msg}

        first_status = "analyzing_faces" if refresh_references else "analyzing_scenes"
        if not await service.start_analysis_run(video_id, analysis_run, first_status):
            logger.info(f"[Job {job_id}] Analysis of video {video_id} was cancelled before it started: nothing done")
            return {"status": "skipped", "video_id": video_id}

    try:
        async with get_db() as db:
            service = AnalysisService(db)

            if reset:
                await service.reset_analysis(video_id)

            # Create sync cancellation check for use during analysis
            cancellation_check = create_sync_analysis_cancellation_check(video_id, analysis_run)

            await service.run_full_analysis(
                video_id=video_id,
                force_scenes=reset or force_scenes,
                force_faces=reset or force_faces,
                force_clustering=reset or force_clustering,
                force_transcription=reset or force_transcription,
                clustering_eps=clustering_eps,
                clustering_min_samples=clustering_min_samples,
                refresh_references=refresh_references,
                expected_reference_revision=expected_reference_revision,
                cancellation_check=cancellation_check
            )

        # Get final status
        async with get_db() as db:
            service = AnalysisService(db)
            video = await service.get_video(video_id)
            final_status = video.get("status", "error") if video else "error"
            error_message = video.get("error_message") if video else None

        # A reference refresh leaves a video with thumbnails completed
        if final_status in ("analyzed", "completed"):
            logger.info(f"[Job {job_id}] Analysis completed for video {video_id}")
            return {"status": "success", "video_id": video_id}
        elif final_status == "cancelled":
            return cancelled_result("analysis", video_id, job_id)
        else:
            logger.error(f"[Job {job_id}] Analysis failed for video {video_id}: {error_message}")
            return {"status": "error", "video_id": video_id, "error": error_message}

    except asyncio.CancelledError:
        # arq timeouts and shutdowns cancel the coroutine: the analysis is left half done
        async with get_db() as db:
            await AnalysisService(db).interrupt_analysis(video_id, analysis_run)
        raise

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

        return {"status": "error", "video_id": video_id, "error": error_msg}

    finally:
        # However it ended, this analysis no longer owns the video
        async with get_db() as db:
            await AnalysisService(db).release_analysis(video_id, analysis_run)


async def run_generation(
    ctx: dict,
    job_id: int,
    force_transcription: bool = False,
    force_prompts: bool = False,
    image_provider: str = "gemini",
    gemini_model: Optional[str] = None,
    openai_model: Optional[str] = None,
    poe_model: Optional[str] = None,
    comfyui_workflow: Optional[str] = None,
    comfyui_config: Optional[dict] = None,
    comfyui_profile: Optional[dict] = None,
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
    # Face swap post-processing
    apply_face_swap: bool = False,
    restore_faces: bool = False,
    # Try other image providers when the chosen one fails
    allow_provider_fallback: bool = False,
    # With several people, also send a frame of them together
    include_scene_reference: bool = True
) -> dict:
    """
    Background task for thumbnail generation.

    This task runs the generation pipeline:
    1. Transcribe video (if needed)
    2. Generate prompts using LLM
    3. Generate thumbnail images

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
        # Check if already cancelled before starting.
        if await check_generation_cancelled(job_id):
            return cancelled_result("generation", job_id, arq_job_id)

        # Create sync cancellation check for use during generation
        cancellation_check = create_sync_cancellation_check(job_id)

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
                comfyui_workflow=comfyui_workflow,
                comfyui_config=comfyui_config,
                comfyui_profile=comfyui_profile,
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
                reference_image_include_in_refs=reference_image_include_in_refs,
                apply_face_swap=apply_face_swap,
                restore_faces=restore_faces,
                allow_provider_fallback=allow_provider_fallback,
                include_scene_reference=include_scene_reference,
                cancellation_check=cancellation_check
            )

        # Check if cancelled during generation
        if await check_generation_cancelled(job_id):
            return cancelled_result("generation", job_id, arq_job_id)

        # Get final status
        async with get_db() as db:
            service = GenerationService(db)
            job = await service.get_job(job_id)
            final_status = job.get("status", "error") if job else "error"
            error_message = job.get("error_message") if job else None

        if final_status == "completed":
            logger.info(f"[arq:{arq_job_id}] Generation completed for job {job_id}")
            return {"status": "success", "job_id": job_id}
        elif final_status == "cancelled":
            # Handle cancellation detected from final status
            return cancelled_result("generation", job_id, arq_job_id)
        else:
            logger.error(f"[arq:{arq_job_id}] Generation failed for job {job_id}: {error_message}")
            return {"status": "error", "job_id": job_id, "error": error_message}

    except asyncio.CancelledError:
        # arq timeouts/shutdowns cancel the coroutine rather than raising an
        # Exception. Release this job without overwriting a newer generation.
        async with get_db() as db:
            await GenerationService(db).update_job_status(
                job_id, "error", 0, "Generation worker was interrupted"
            )
        raise

    except ImageGenerationError as e:
        # Raised AFTER the job status has been set to 'error' in the database:
        # re-raise so arq registers this as a failed job (j_failed), not completed
        logger.error(f"[arq:{arq_job_id}] Generation failed for job {job_id}: {e}")
        raise

    except Exception as e:
        error_msg = str(e)
        logger.exception(f"[arq:{arq_job_id}] Generation error for job {job_id}: {error_msg}")

        # Persist the failure and release its video before reporting it.
        async with get_db() as db:
            service = GenerationService(db)
            await service.update_job_status(job_id, "error", 0, error_msg)

        # Re-raise so arq registers this as a failed job
        raise

    finally:
        # The pipeline drains any synchronous executor before it exits: only
        # then may the video of a cancelled job generate again. The thumbnails
        # the job had finished stay.
        async with get_db() as db:
            service = GenerationService(db)
            job = await service.get_job(job_id)
            if job and job['status'] == 'cancelled':
                await service.update_job_status(job_id, 'cancelled')

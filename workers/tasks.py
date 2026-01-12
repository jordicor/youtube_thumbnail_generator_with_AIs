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
from job_queue.pubsub import (
    publish_progress,
    publish_event,
    publish_task_event,
    CHANNEL_ANALYSIS,
    CHANNEL_GENERATION,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CANCELLATION CHECK HELPERS
# =============================================================================

def create_sync_cancellation_check(job_id: int):
    """
    Create a synchronous cancellation check function for use in sync code.

    This is needed because generate_thumbnails_from_images is synchronous,
    but we need to check the database for cancellation status.

    Args:
        job_id: The generation job ID to check

    Returns:
        A callable that returns True if the job has been cancelled
    """
    import sqlite3
    from config import DATABASE_PATH

    def check() -> bool:
        try:
            conn = sqlite3.connect(str(DATABASE_PATH), timeout=5)
            cursor = conn.execute(
                "SELECT status FROM generation_jobs WHERE id = ?",
                [job_id]
            )
            row = cursor.fetchone()
            conn.close()
            return row and row[0] == 'cancelled'
        except Exception as e:
            logger.warning(f"Error in sync cancellation check for job {job_id}: {e}")
            return False

    return check


def create_sync_analysis_cancellation_check(video_id: int):
    """
    Create a synchronous cancellation check function for video analysis.

    This is needed because analysis steps (scene detection, face extraction, etc.)
    are synchronous, but we need to check the database for cancellation status.

    Args:
        video_id: The video ID to check

    Returns:
        A callable that returns True if the video analysis has been cancelled
    """
    import sqlite3
    from config import DATABASE_PATH

    def check() -> bool:
        try:
            conn = sqlite3.connect(str(DATABASE_PATH), timeout=5)
            cursor = conn.execute(
                "SELECT status FROM videos WHERE id = ?",
                [video_id]
            )
            row = cursor.fetchone()
            conn.close()
            return row and row[0] == 'cancelled'
        except Exception as e:
            logger.warning(f"Error in sync cancellation check for video {video_id}: {e}")
            return False

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


async def check_analysis_cancelled(video_id: int) -> bool:
    """
    Check if a video analysis has been cancelled.

    Args:
        video_id: The video ID to check

    Returns:
        True if video status is 'cancelled', False otherwise
    """
    try:
        async with get_db() as db:
            service = AnalysisService(db)
            video = await service.get_video(video_id)
            return video and video.get('status') == 'cancelled'
    except Exception as e:
        logger.warning(f"Error checking analysis cancellation for video {video_id}: {e}")
        return False


async def handle_cancellation(
    task_type: str,
    task_id: int,
    video_id: int,
    video_name: str,
    arq_job_id: str
) -> dict:
    """
    Handle task cancellation: cleanup partial resources, emit SSE event, and return response.

    This function:
    1. Cleans up any partial files/DB records created during the interrupted task
    2. Emits cancellation events to both specific and global SSE channels
    3. Returns appropriate response for arq

    Args:
        task_type: 'analysis' or 'generation'
        task_id: The task ID (video_id for analysis, job_id for generation)
        video_id: The video ID
        video_name: The video filename
        arq_job_id: The arq job ID for logging

    Returns:
        dict with cancellation status for arq
    """
    logger.info(f"[{arq_job_id}] Task {task_type} {task_id} was cancelled, stopping gracefully")

    # Clean up partial resources
    cleanup_result = {"cleaned_files": [], "cleaned_db_records": 0}
    try:
        async with get_db() as db:
            if task_type == "analysis":
                service = AnalysisService(db)
                cleanup_result = await service.cleanup_partial_analysis(video_id)
            else:  # generation
                service = GenerationService(db)
                cleanup_result = await service.cleanup_partial_generation(task_id)

        if cleanup_result.get("cleaned_files"):
            logger.info(
                f"[{arq_job_id}] Cleaned up {len(cleanup_result['cleaned_files'])} partial files"
            )
        if cleanup_result.get("cleaned_db_records"):
            logger.info(
                f"[{arq_job_id}] Cleaned up {cleanup_result['cleaned_db_records']} DB records"
            )
    except Exception as e:
        # Cleanup errors should not prevent cancellation from completing
        logger.warning(f"[{arq_job_id}] Error during cleanup (non-fatal): {e}")

    # Emit cancellation event to specific channel
    channel = CHANNEL_ANALYSIS if task_type == "analysis" else CHANNEL_GENERATION
    id_key = "video_id" if task_type == "analysis" else "job_id"

    await publish_event(
        channel,
        "cancelled",
        {"status": "cancelled", "message": "Task cancelled by user"},
        **{id_key: task_id}
    )

    # Emit cancellation event to global task channel
    await publish_task_event(
        task_type=task_type,
        event_type="task_cancelled",
        task_id=task_id,
        video_id=video_id,
        video_name=video_name,
        status="cancelled",
        progress=0
    )

    return {"status": "cancelled", id_key: task_id}


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

    Progress is reported via Redis pub/sub to the CHANNEL_ANALYSIS channel
    and the global CHANNEL_TASKS for the task queue UI.

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

    # Get video name for global task events
    video_name = ""
    try:
        async with get_db() as db:
            service = AnalysisService(db)
            video_info = await service.get_video(video_id)
            video_name = video_info.get("filename", "") if video_info else ""
    except Exception:
        pass

    # Check if already cancelled before starting
    if await check_analysis_cancelled(video_id):
        return await handle_cancellation(
            task_type="analysis",
            task_id=video_id,
            video_id=video_id,
            video_name=video_name,
            arq_job_id=job_id
        )

    try:
        # Notify start (specific channel)
        await publish_progress(
            CHANNEL_ANALYSIS,
            status="analyzing",
            progress=5,
            message="Starting analysis...",
            video_id=video_id
        )

        # Notify start (global task channel)
        await publish_task_event(
            task_type="analysis",
            event_type="task_started",
            task_id=video_id,
            video_id=video_id,
            video_name=video_name,
            status="analyzing",
            progress=5,
            current_step="Starting analysis..."
        )

        async with get_db() as db:
            service = AnalysisService(db)

            # Create sync cancellation check for use during analysis
            cancellation_check = create_sync_analysis_cancellation_check(video_id)

            # Wrap the service to report progress
            # We'll modify the service methods to accept a progress callback
            await service.run_full_analysis(
                video_id=video_id,
                force_scenes=force_scenes,
                force_faces=force_faces,
                force_clustering=force_clustering,
                force_transcription=force_transcription,
                clustering_eps=clustering_eps,
                clustering_min_samples=clustering_min_samples,
                cancellation_check=cancellation_check
            )

        # Check if cancelled during analysis
        if await check_analysis_cancelled(video_id):
            return await handle_cancellation(
                task_type="analysis",
                task_id=video_id,
                video_id=video_id,
                video_name=video_name,
                arq_job_id=job_id
            )

        # Get final status
        async with get_db() as db:
            service = AnalysisService(db)
            video = await service.get_video(video_id)
            final_status = video.get("status", "error") if video else "error"
            error_message = video.get("error_message") if video else None

        if final_status == "analyzed":
            # Notify completion (specific channel)
            await publish_event(
                CHANNEL_ANALYSIS,
                "complete",
                {"status": "analyzed", "progress": 100},
                video_id=video_id
            )

            # Notify completion (global task channel)
            await publish_task_event(
                task_type="analysis",
                event_type="task_completed",
                task_id=video_id,
                video_id=video_id,
                video_name=video_name,
                status="analyzed",
                progress=100
            )

            logger.info(f"[Job {job_id}] Analysis completed for video {video_id}")
            return {"status": "success", "video_id": video_id}
        elif final_status == "cancelled":
            # Handle cancellation detected from final status
            return await handle_cancellation(
                task_type="analysis",
                task_id=video_id,
                video_id=video_id,
                video_name=video_name,
                arq_job_id=job_id
            )
        else:
            # Notify error (specific channel)
            await publish_event(
                CHANNEL_ANALYSIS,
                "error",
                {"status": final_status, "error": error_message},
                video_id=video_id
            )

            # Notify error (global task channel)
            await publish_task_event(
                task_type="analysis",
                event_type="task_error",
                task_id=video_id,
                video_id=video_id,
                video_name=video_name,
                status="error",
                progress=0,
                error_message=error_message
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

        # Notify error (specific channel)
        await publish_event(
            CHANNEL_ANALYSIS,
            "error",
            {"status": "error", "error": error_msg},
            video_id=video_id
        )

        # Notify error (global task channel)
        await publish_task_event(
            task_type="analysis",
            event_type="task_error",
            task_id=video_id,
            video_id=video_id,
            video_name=video_name,
            status="error",
            progress=0,
            error_message=error_msg
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

    Progress is reported via Redis pub/sub to the CHANNEL_GENERATION channel
    and the global CHANNEL_TASKS for the task queue UI.

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

    # Get video info for global task events
    video_id = 0
    video_name = ""
    num_images = 0
    try:
        async with get_db() as db:
            service = GenerationService(db)
            job_info = await service.get_job(job_id)
            if job_info:
                video_id = job_info.get("video_id", 0)
                num_images = job_info.get("num_images", 0)
                video = await service.get_video(video_id)
                video_name = video.get("filename", "") if video else ""
    except Exception:
        pass

    # Check if already cancelled before starting
    if await check_generation_cancelled(job_id):
        return await handle_cancellation(
            task_type="generation",
            task_id=job_id,
            video_id=video_id,
            video_name=video_name,
            arq_job_id=arq_job_id
        )

    try:
        # Notify start (specific channel)
        await publish_progress(
            CHANNEL_GENERATION,
            status="starting",
            progress=5,
            message="Starting generation...",
            job_id=job_id
        )

        # Notify start (global task channel)
        await publish_task_event(
            task_type="generation",
            event_type="task_started",
            task_id=job_id,
            video_id=video_id,
            video_name=video_name,
            status="starting",
            progress=5,
            current_step="Starting generation...",
            thumbnails_generated=0,
            total_thumbnails=num_images
        )

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
                cancellation_check=cancellation_check
            )

        # Check if cancelled during generation
        if await check_generation_cancelled(job_id):
            return await handle_cancellation(
                task_type="generation",
                task_id=job_id,
                video_id=video_id,
                video_name=video_name,
                arq_job_id=arq_job_id
            )

        # Get final status
        async with get_db() as db:
            service = GenerationService(db)
            job = await service.get_job(job_id)
            final_status = job.get("status", "error") if job else "error"
            error_message = job.get("error_message") if job else None

        if final_status == "completed":
            # Notify completion (specific channel)
            await publish_event(
                CHANNEL_GENERATION,
                "complete",
                {"status": "completed", "progress": 100},
                job_id=job_id
            )

            # Notify completion (global task channel)
            await publish_task_event(
                task_type="generation",
                event_type="task_completed",
                task_id=job_id,
                video_id=video_id,
                video_name=video_name,
                status="completed",
                progress=100,
                thumbnails_generated=num_images,
                total_thumbnails=num_images
            )

            logger.info(f"[arq:{arq_job_id}] Generation completed for job {job_id}")
            return {"status": "success", "job_id": job_id}
        elif final_status == "cancelled":
            # Handle cancellation detected from final status
            return await handle_cancellation(
                task_type="generation",
                task_id=job_id,
                video_id=video_id,
                video_name=video_name,
                arq_job_id=arq_job_id
            )
        else:
            # Notify error (specific channel)
            await publish_event(
                CHANNEL_GENERATION,
                "error",
                {"status": final_status, "error": error_message},
                job_id=job_id
            )

            # Notify error (global task channel)
            await publish_task_event(
                task_type="generation",
                event_type="task_error",
                task_id=job_id,
                video_id=video_id,
                video_name=video_name,
                status="error",
                progress=0,
                error_message=error_message
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

        # Notify error (specific channel)
        await publish_event(
            CHANNEL_GENERATION,
            "error",
            {"status": "error", "error": error_msg},
            job_id=job_id
        )

        # Notify error (global task channel)
        await publish_task_event(
            task_type="generation",
            event_type="task_error",
            task_id=job_id,
            video_id=video_id,
            video_name=video_name,
            status="error",
            progress=0,
            error_message=error_msg
        )

        return {"status": "error", "job_id": job_id, "error": error_msg}

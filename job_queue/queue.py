"""
Job queue helpers for enqueueing tasks from the API.

This module provides functions to enqueue jobs to the arq queue
from the FastAPI application.
"""

import logging
from typing import Optional, Any
from datetime import timedelta

from arq import create_pool
from arq.connections import RedisSettings, ArqRedis

from job_queue.settings import get_arq_redis_settings, WorkerSettings

logger = logging.getLogger(__name__)

# Global connection pool for enqueueing
_arq_pool: Optional[ArqRedis] = None


async def get_arq_pool() -> ArqRedis:
    """
    Get or create the arq Redis connection pool.

    This pool is used to enqueue jobs from the API.
    """
    global _arq_pool

    if _arq_pool is None:
        _arq_pool = await create_pool(get_arq_redis_settings())
        logger.info("arq connection pool created")

    return _arq_pool


async def close_arq_pool() -> None:
    """Close the arq connection pool."""
    global _arq_pool

    if _arq_pool is not None:
        await _arq_pool.close()
        _arq_pool = None
        logger.info("arq connection pool closed")


async def enqueue_analysis(
    video_id: int,
    force_scenes: bool = False,
    force_faces: bool = False,
    force_clustering: bool = False,
    force_transcription: bool = False,
    clustering_eps: float = 0.5,
    clustering_min_samples: int = 3
) -> Optional[str]:
    """
    Enqueue a video analysis job.

    Args:
        video_id: ID of the video to analyze
        force_*: Flags to force re-processing of specific steps
        clustering_eps: DBSCAN epsilon parameter
        clustering_min_samples: DBSCAN min_samples parameter

    Returns:
        Job ID if enqueued successfully, None otherwise
    """
    try:
        pool = await get_arq_pool()

        job = await pool.enqueue_job(
            "analyze_video",
            video_id,
            force_scenes=force_scenes,
            force_faces=force_faces,
            force_clustering=force_clustering,
            force_transcription=force_transcription,
            clustering_eps=clustering_eps,
            clustering_min_samples=clustering_min_samples,
            _job_timeout=WorkerSettings.JOB_TIMEOUT_ANALYSIS,
        )

        if job:
            logger.info(f"Enqueued analysis job {job.job_id} for video {video_id}")
            return job.job_id
        else:
            logger.warning(f"Failed to enqueue analysis job for video {video_id}")
            return None

    except Exception as e:
        logger.exception(f"Error enqueueing analysis job for video {video_id}: {e}")
        return None


async def enqueue_generation(
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
    reference_image_include_in_refs: bool = False
) -> Optional[str]:
    """
    Enqueue a thumbnail generation job.

    Args:
        job_id: ID of the generation job (from database)
        force_transcription: Force re-transcription
        force_prompts: Force prompt regeneration
        image_provider: Image generation provider
        *_model: Specific model for each provider
        num_reference_images: Number of reference images to use
        prompt_provider: AI provider for prompt generation (anthropic, openai, google, xai)
        prompt_model: Specific model for prompt generation
        prompt_thinking_enabled: Enable thinking/reasoning mode
        prompt_thinking_level: Thinking level (low, medium, high)
        prompt_custom_instructions: Additional user instructions for prompt generation
        prompt_include_history: Include previous prompts to avoid repetition
        selected_titles: List of user-selected titles to guide image generation
        reference_image_base64: External reference image as base64
        reference_image_use_for_prompts: Use reference image for prompt analysis
        reference_image_include_in_refs: Include reference image in generation refs

    Returns:
        arq Job ID if enqueued successfully, None otherwise
    """
    try:
        pool = await get_arq_pool()

        arq_job = await pool.enqueue_job(
            "run_generation",
            job_id,
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
            _job_timeout=WorkerSettings.JOB_TIMEOUT_GENERATION,
        )

        if arq_job:
            logger.info(f"Enqueued generation arq job {arq_job.job_id} for db job {job_id}")
            return arq_job.job_id
        else:
            logger.warning(f"Failed to enqueue generation job for db job {job_id}")
            return None

    except Exception as e:
        logger.exception(f"Error enqueueing generation job for db job {job_id}: {e}")
        return None


async def get_job_status(job_id: str) -> Optional[dict]:
    """
    Get the status of a queued job.

    Args:
        job_id: arq job ID

    Returns:
        dict with job status info, or None if not found
    """
    try:
        pool = await get_arq_pool()
        job = await pool.job(job_id)

        if job is None:
            return None

        status = await job.status()
        result = await job.result_info()

        return {
            "job_id": job_id,
            "status": status.value if status else "unknown",
            "result": result,
        }

    except Exception as e:
        logger.exception(f"Error getting job status for {job_id}: {e}")
        return None

"""
Tasks API Routes

Unified endpoints for task queue management.
Provides a single interface for viewing and cancelling all running tasks.
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Literal

from database.db import get_db
from services.task_service import TaskService
from i18n.i18n import translate as t


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/active")
async def get_active_tasks():
    """
    Get all currently active (running) tasks.

    Returns both analysis and generation tasks in progress.
    """
    try:
        async with get_db() as db:
            service = TaskService(db)
            tasks = await service.get_active_tasks()
        return {"tasks": tasks, "count": len(tasks)}
    except Exception as e:
        logger.error(f"Error fetching active tasks: {e}")
        raise HTTPException(status_code=503, detail=t('errors.server'))


@router.get("/pending")
async def get_pending_tasks():
    """
    Get all pending tasks waiting to start.
    """
    try:
        async with get_db() as db:
            service = TaskService(db)
            tasks = await service.get_pending_tasks()
        return {"tasks": tasks, "count": len(tasks)}
    except Exception as e:
        logger.error(f"Error fetching pending tasks: {e}")
        raise HTTPException(status_code=503, detail=t('errors.server'))


@router.get("/video/{video_id}")
async def get_video_task(video_id: int):
    """
    Get the active task for a specific video (if any).
    """
    try:
        async with get_db() as db:
            service = TaskService(db)
            task = await service.get_task_for_video(video_id)
        return {"task": task}
    except Exception as e:
        logger.error(f"Error fetching task for video {video_id}: {e}")
        raise HTTPException(status_code=503, detail=t('errors.server'))


@router.post("/{task_type}/{task_id}/cancel")
async def cancel_task(
    task_type: Literal["analysis", "generation"],
    task_id: int
):
    """
    Cancel a task.

    Args:
        task_type: 'analysis' or 'generation'
        task_id: video_id for analysis, job_id for generation
    """
    try:
        async with get_db() as db:
            service = TaskService(db)
            success = await service.cancel_task(task_type, task_id)
    except Exception as e:
        logger.error(f"Error cancelling task {task_type}:{task_id}: {e}")
        raise HTTPException(status_code=503, detail=t('errors.server'))

    if not success:
        raise HTTPException(
            status_code=400,
            detail=t('api.errors.task_cannot_cancel')
        )

    return {
        "task_type": task_type,
        "task_id": task_id,
        "status": "cancelled"
    }

"""
Server-Sent Events (SSE) Routes

Real-time progress updates for analysis and generation operations.
"""

import asyncio
import json
from typing import AsyncGenerator, Optional
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from database.db import get_db
from services.analysis_service import AnalysisService
from services.generation_service import GenerationService
from i18n.i18n import translate as t


router = APIRouter()


# =============================================================================
# SSE HELPERS
# =============================================================================

async def create_sse_message(data: dict, event: str = "message") -> str:
    """Format data as SSE message."""
    json_data = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {json_data}\n\n"


async def create_keepalive() -> str:
    """Create a keepalive comment to prevent timeout."""
    return ": keepalive\n\n"


# =============================================================================
# ANALYSIS EVENTS
# =============================================================================

@router.get("/analysis/{video_id}")
async def stream_analysis_status(video_id: int, request: Request):
    """
    Stream analysis progress updates via SSE.

    Events:
    - progress: Status update with progress percentage
    - complete: Analysis finished successfully
    - error: Analysis failed

    Example client usage:
        const eventSource = new EventSource('/api/events/analysis/1');
        eventSource.addEventListener('progress', (e) => {
            const data = JSON.parse(e.data);
            console.log(`Progress: ${data.progress}%`);
        });
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        last_status = None
        error_count = 0
        max_errors = 5
        keepalive_interval = 15  # seconds

        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    async with get_db() as db:
                        service = AnalysisService(db)
                        status = await service.get_analysis_status(video_id)

                    if not status:
                        yield await create_sse_message(
                            {"error": t('api.errors.video_not_found')},
                            event="error"
                        )
                        break

                    # Only send if status changed
                    current_state = (status['status'], status.get('progress', 0))

                    if current_state != last_status:
                        last_status = current_state

                        # Determine event type
                        if status['status'] == 'error':
                            yield await create_sse_message(
                                {
                                    "video_id": video_id,
                                    "status": status['status'],
                                    "progress": 0,
                                    "error_message": status.get('error_message', t('errors.generic'))
                                },
                                event="error"
                            )
                            break

                        elif status['status'] in ('analyzed', 'completed'):
                            yield await create_sse_message(
                                {
                                    "video_id": video_id,
                                    "status": status['status'],
                                    "progress": 100,
                                    "clusters": status.get('clusters', 0),
                                    "message": t('api.messages.analysis_completed')
                                },
                                event="complete"
                            )
                            break

                        else:
                            # Progress update
                            progress_map = {
                                'pending': 0,
                                'analyzing': 10,
                                'analyzing_scenes': 15,
                                'analyzing_faces': 40,
                                'clustering': 65,
                                'transcribing': 85,
                            }
                            progress = progress_map.get(status['status'], status.get('progress', 0))

                            step_map = {
                                'pending': t('api.steps.waiting'),
                                'analyzing': t('api.steps.analyzing_video'),
                                'analyzing_scenes': t('api.steps.detecting_scenes'),
                                'analyzing_faces': t('api.steps.analyzing_faces'),
                                'clustering': t('api.steps.clustering_faces'),
                                'transcribing': t('api.steps.transcribing_audio'),
                            }
                            current_step = step_map.get(status['status'], status['status'])

                            yield await create_sse_message(
                                {
                                    "video_id": video_id,
                                    "status": status['status'],
                                    "progress": progress,
                                    "current_step": current_step
                                },
                                event="progress"
                            )

                    error_count = 0  # Reset on success

                except Exception as e:
                    error_count += 1
                    if error_count >= max_errors:
                        yield await create_sse_message(
                            {"error": t('api.errors.too_many_errors', error=str(e))},
                            event="error"
                        )
                        break

                # Wait before next check
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# =============================================================================
# GENERATION EVENTS
# =============================================================================

@router.get("/generation/{job_id}")
async def stream_generation_status(job_id: int, request: Request):
    """
    Stream generation progress updates via SSE.

    Events:
    - progress: Status update with progress percentage
    - thumbnail: New thumbnail generated
    - complete: Generation finished successfully
    - error: Generation failed

    Example client usage:
        const eventSource = new EventSource('/api/events/generation/1');
        eventSource.addEventListener('thumbnail', (e) => {
            const data = JSON.parse(e.data);
            appendThumbnail(data.thumbnail_path);
        });
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        last_status = None
        last_thumbnail_count = 0
        error_count = 0
        max_errors = 5

        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    async with get_db() as db:
                        service = GenerationService(db)
                        status = await service.get_job_status(job_id)

                    if not status:
                        yield await create_sse_message(
                            {"error": t('api.errors.job_not_found')},
                            event="error"
                        )
                        break

                    # Check for new thumbnails
                    current_thumbnail_count = status.get('thumbnails_generated', 0)
                    if current_thumbnail_count > last_thumbnail_count:
                        # Get latest thumbnails
                        async with get_db() as db:
                            service = GenerationService(db)
                            thumbnails = await service.get_job_thumbnails(job_id)

                        # Send new thumbnails
                        for thumb in thumbnails[last_thumbnail_count:]:
                            yield await create_sse_message(
                                {
                                    "job_id": job_id,
                                    "thumbnail_id": thumb['id'],
                                    "filepath": thumb['filepath'],
                                    "image_index": thumb.get('image_index'),
                                    "suggested_title": thumb.get('suggested_title'),
                                    "text_overlay": thumb.get('text_overlay')
                                },
                                event="thumbnail"
                            )

                        last_thumbnail_count = current_thumbnail_count

                    # Send status update if changed
                    current_state = (
                        status['status'],
                        status.get('progress', 0),
                        current_thumbnail_count
                    )

                    if current_state != last_status:
                        last_status = current_state

                        if status['status'] == 'error':
                            yield await create_sse_message(
                                {
                                    "job_id": job_id,
                                    "status": status['status'],
                                    "progress": 0,
                                    "error_message": status.get('error_message', t('errors.generic'))
                                },
                                event="error"
                            )
                            break

                        elif status['status'] == 'completed':
                            yield await create_sse_message(
                                {
                                    "job_id": job_id,
                                    "status": status['status'],
                                    "progress": 100,
                                    "thumbnails_generated": status.get('thumbnails_generated', 0),
                                    "total_thumbnails": status.get('total_thumbnails', 0),
                                    "message": t('api.messages.generation_completed')
                                },
                                event="complete"
                            )
                            break

                        elif status['status'] == 'cancelled':
                            yield await create_sse_message(
                                {
                                    "job_id": job_id,
                                    "status": "cancelled",
                                    "message": t('api.messages.generation_cancelled')
                                },
                                event="cancelled"
                            )
                            break

                        else:
                            # Progress update
                            step_map = {
                                'pending': t('api.steps.starting'),
                                'transcribing': t('api.steps.transcribing_audio'),
                                'prompting': t('api.steps.generating_prompts'),
                                'generating': t('api.steps.creating_thumbnails'),
                            }
                            current_step = step_map.get(status['status'], status['status'])

                            yield await create_sse_message(
                                {
                                    "job_id": job_id,
                                    "status": status['status'],
                                    "progress": status.get('progress', 0),
                                    "current_step": current_step,
                                    "thumbnails_generated": current_thumbnail_count,
                                    "total_thumbnails": status.get('total_thumbnails', 0)
                                },
                                event="progress"
                            )

                    error_count = 0

                except Exception as e:
                    error_count += 1
                    if error_count >= max_errors:
                        yield await create_sse_message(
                            {"error": t('api.errors.too_many_errors', error=str(e))},
                            event="error"
                        )
                        break

                # Wait before next check
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# =============================================================================
# MULTI-VIDEO EVENTS (for batch processing)
# =============================================================================

@router.get("/videos/status")
async def stream_all_videos_status(request: Request):
    """
    Stream status updates for all videos.

    Useful for the main video list to show real-time status.

    Events:
    - update: Video status changed
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        last_statuses = {}
        error_count = 0

        try:
            while True:
                if await request.is_disconnected():
                    break

                try:
                    async with get_db() as db:
                        # Get all videos
                        cursor = await db.execute(
                            "SELECT id, filename, status FROM videos ORDER BY updated_at DESC LIMIT 50"
                        )
                        rows = await cursor.fetchall()

                    for row in rows:
                        video_id = row[0]
                        current_status = row[2]

                        if last_statuses.get(video_id) != current_status:
                            last_statuses[video_id] = current_status
                            yield await create_sse_message(
                                {
                                    "video_id": video_id,
                                    "filename": row[1],
                                    "status": current_status
                                },
                                event="update"
                            )

                    error_count = 0

                except Exception:
                    error_count += 1
                    if error_count >= 5:
                        break

                await asyncio.sleep(2)

        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

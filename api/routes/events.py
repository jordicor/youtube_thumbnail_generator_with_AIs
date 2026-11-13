"""
Server-Sent Events (SSE) Routes

Real-time updates of a generation job and of every task.
"""

import asyncio
import orjson
from typing import AsyncGenerator
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from database.db import get_db
from services.generation_service import GenerationService
from services.task_service import TaskService
from services import queue_health
from i18n.i18n import translate as t


router = APIRouter()

# How often the generation and task streams look at the database
POLL_SECONDS = 1


# =============================================================================
# SSE HELPERS
# =============================================================================

async def create_sse_message(data: dict, event: str = "message") -> str:
    """Format data as SSE message."""
    json_data = orjson.dumps(data).decode('utf-8')
    return f"event: {event}\ndata: {json_data}\n\n"


# =============================================================================
# GENERATION EVENTS
# =============================================================================

@router.get("/generation/{job_id}")
async def stream_generation_status(job_id: int, request: Request):
    """
    Stream generation progress updates via SSE.

    Events:
    - progress: Status update with progress percentage. Status 'stopping': the
      job was cancelled and its worker is finishing the thumbnail it was making
    - thumbnail: New thumbnail generated
    - complete: Generation finished successfully
    - cancelled: Generation cancelled, once its worker has stopped
    - failed: Generation failed. Not 'error': EventSource fires that name on
      every connection drop too, and a drop is not a failed generation

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
                            event="failed"
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
                        status['stopping'],
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
                                    "error_message": status.get('error_message') or t('errors.generic')
                                },
                                event="failed"
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

                        elif status['status'] == 'cancelled' and not status['stopping']:
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
                            # Progress update. A cancelled job still here is
                            # stopping: its worker may save one more thumbnail
                            step = TaskService.STOPPING if status['stopping'] else status['status']
                            step_map = {
                                'pending': t('api.steps.starting'),
                                'transcribing': t('api.steps.transcribing_audio'),
                                'prompting': t('api.steps.generating_prompts'),
                                'generating': t('api.steps.creating_thumbnails'),
                                TaskService.STOPPING: t('api.steps.stopping_generation'),
                            }

                            yield await create_sse_message(
                                {
                                    "job_id": job_id,
                                    "status": step,
                                    "progress": status.get('progress', 0),
                                    "current_step": step_map.get(step, step),
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
                            event="failed"
                        )
                        break

                # Wait before next check
                await asyncio.sleep(POLL_SECONDS)

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
# GLOBAL TASK QUEUE EVENTS
# =============================================================================

@router.get("/tasks")
async def stream_all_tasks_status(request: Request):
    """
    Stream status updates for all tasks (analysis + generation).

    This endpoint provides a unified view of all running tasks,
    enabling the task queue UI to show real-time progress across
    all operations regardless of which page the user is viewing.

    Events:
    - tasks_snapshot: Initial state of all active tasks (sent on connect)
    - task_started: New task started
    - task_progress: Task progress update. Status 'stopping': the task was
      cancelled and its worker has not stopped yet
    - task_completed: Task finished successfully
    - task_cancelled: Task was cancelled and has stopped, or never got queued
    - task_error: Task failed, with its reason in error_message
    - queue_status: Whether queued tasks can run ('ok', 'redis_down' or
      'worker_down'), on connect and whenever it changes

    A task that leaves the lists has ended: how it ended is read from the
    database (TaskService.get_task_outcome), never assumed.

    Example client usage:
        const eventSource = new EventSource('/api/events/tasks');
        eventSource.addEventListener('tasks_snapshot', (e) => {
            const data = JSON.parse(e.data);
            initializeTaskList(data.tasks);
        });
        eventSource.addEventListener('task_progress', (e) => {
            const data = JSON.parse(e.data);
            updateTask(data.task_id, data);
        });
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        error_count = 0
        max_errors = 5
        last_task_states = {}
        last_queue_status = None

        try:
            # Send initial snapshot of all active tasks
            try:
                async with get_db() as db:
                    service = TaskService(db)
                    active_tasks = await service.get_active_tasks()
                    pending_tasks = await service.get_pending_tasks()

                all_tasks = active_tasks + pending_tasks

                # Build initial state map
                for task in all_tasks:
                    key = f"{task['type']}:{task['id']}"
                    last_task_states[key] = task

                yield await create_sse_message(
                    {
                        "tasks": all_tasks,
                        "active_count": len(active_tasks),
                        "pending_count": len(pending_tasks)
                    },
                    event="tasks_snapshot"
                )
            except Exception as e:
                yield await create_sse_message(
                    {"error": f"Failed to load initial tasks: {str(e)}"},
                    event="error"
                )

            # Poll the database: it is where tasks write what they do
            while True:
                if await request.is_disconnected():
                    break

                # Whether the queue can run tasks: never waits for Redis
                current_queue_status = queue_health.status()
                if current_queue_status is not None and current_queue_status != last_queue_status:
                    last_queue_status = current_queue_status
                    yield await create_sse_message({"status": current_queue_status}, event="queue_status")

                try:
                    async with get_db() as db:
                        service = TaskService(db)
                        active_tasks = await service.get_active_tasks()
                        pending_tasks = await service.get_pending_tasks()

                    current_tasks = active_tasks + pending_tasks
                    current_task_keys = set()

                    for task in current_tasks:
                        key = f"{task['type']}:{task['id']}"
                        current_task_keys.add(key)

                        prev_task = last_task_states.get(key)

                        if prev_task is None:
                            # New task started
                            yield await create_sse_message(task, event="task_started")
                            last_task_states[key] = task

                        elif (
                            prev_task.get('status') != task.get('status') or
                            prev_task.get('progress') != task.get('progress') or
                            prev_task.get('thumbnails_generated') != task.get('thumbnails_generated')
                        ):
                            yield await create_sse_message(task, event="task_progress")
                            last_task_states[key] = task

                    # A task that left the lists has ended: say how
                    for key in set(last_task_states) - current_task_keys:
                        ended_task = last_task_states[key]
                        async with get_db() as db:
                            outcome = await TaskService(db).get_task_outcome(ended_task['type'], ended_task['id'])
                        del last_task_states[key]
                        yield await create_sse_message(
                            {**ended_task, **outcome},
                            event=f"task_{outcome['status']}"
                        )

                    error_count = 0

                except Exception:
                    error_count += 1
                    if error_count >= max_errors:
                        break

                await asyncio.sleep(POLL_SECONDS)

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

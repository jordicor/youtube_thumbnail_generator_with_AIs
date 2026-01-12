"""
Redis Pub/Sub for real-time progress updates.

Used to communicate between workers and the API server for SSE events.
"""

import json
import logging
from typing import Optional, AsyncGenerator, Any

from job_queue.client import RedisManager, get_redis

logger = logging.getLogger(__name__)

# Channel patterns
CHANNEL_ANALYSIS = "analysis:{video_id}"
CHANNEL_GENERATION = "generation:{job_id}"
CHANNEL_VIDEOS = "videos:status"
CHANNEL_TASKS = "tasks:global"  # Global channel for all task events


def _format_channel(pattern: str, **kwargs) -> str:
    """Format a channel pattern with the given parameters."""
    return pattern.format(**kwargs)


async def publish_event(
    channel: str,
    event_type: str,
    data: dict,
    **channel_params
) -> int:
    """
    Publish an event to a Redis channel.

    Args:
        channel: Channel pattern (e.g., CHANNEL_ANALYSIS)
        event_type: Type of event (e.g., "progress", "complete", "error")
        data: Event data dictionary
        **channel_params: Parameters to format the channel (e.g., video_id=123)

    Returns:
        Number of subscribers that received the message

    Example:
        await publish_event(
            CHANNEL_ANALYSIS,
            "progress",
            {"status": "analyzing_faces", "progress": 45},
            video_id=123
        )
    """
    formatted_channel = _format_channel(channel, **channel_params)
    message = json.dumps({
        "type": event_type,
        "data": data,
    })

    try:
        async with get_redis() as redis:
            subscribers = await redis.publish(formatted_channel, message)
            logger.debug(f"Published to {formatted_channel}: {event_type} ({subscribers} subscribers)")
            return subscribers
    except Exception as e:
        logger.error(f"Failed to publish to {formatted_channel}: {e}")
        return 0


async def publish_progress(
    channel: str,
    status: str,
    progress: int,
    message: Optional[str] = None,
    **channel_params
) -> int:
    """
    Convenience function to publish a progress update.

    Args:
        channel: Channel pattern
        status: Current status string
        progress: Progress percentage (0-100)
        message: Optional human-readable message
        **channel_params: Parameters to format the channel

    Example:
        await publish_progress(
            CHANNEL_ANALYSIS,
            status="analyzing_faces",
            progress=45,
            message="Processing frame 150/300",
            video_id=123
        )
    """
    data = {
        "status": status,
        "progress": progress,
    }
    if message:
        data["message"] = message

    return await publish_event(channel, "progress", data, **channel_params)


async def subscribe_to_channel(
    channel: str,
    **channel_params
) -> AsyncGenerator[dict, None]:
    """
    Subscribe to a Redis channel and yield messages.

    This is an async generator that yields parsed messages from the channel.
    Use this in SSE endpoints.

    Args:
        channel: Channel pattern
        **channel_params: Parameters to format the channel

    Yields:
        Parsed message dictionaries with "type" and "data" keys

    Example:
        async for message in subscribe_to_channel(CHANNEL_ANALYSIS, video_id=123):
            yield f"event: {message['type']}\\ndata: {json.dumps(message['data'])}\\n\\n"
    """
    formatted_channel = _format_channel(channel, **channel_params)

    client = await RedisManager.get_client()
    pubsub = client.pubsub()

    try:
        await pubsub.subscribe(formatted_channel)
        logger.debug(f"Subscribed to {formatted_channel}")

        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    parsed = json.loads(message["data"])
                    yield parsed
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in message: {message['data']}")
                    continue
    finally:
        await pubsub.unsubscribe(formatted_channel)
        await pubsub.close()
        logger.debug(f"Unsubscribed from {formatted_channel}")


# Sync versions for workers (they run in async context but may need sync helpers)

def publish_event_sync(
    channel: str,
    event_type: str,
    data: dict,
    **channel_params
) -> int:
    """
    Sync version of publish_event for use in workers.

    Note: Workers typically run async code, so prefer the async version.
    This is provided for compatibility with sync callbacks.
    """
    from job_queue.client import get_redis_sync

    formatted_channel = _format_channel(channel, **channel_params)
    message = json.dumps({
        "type": event_type,
        "data": data,
    })

    try:
        redis = get_redis_sync()
        subscribers = redis.publish(formatted_channel, message)
        logger.debug(f"Published (sync) to {formatted_channel}: {event_type}")
        return subscribers
    except Exception as e:
        logger.error(f"Failed to publish (sync) to {formatted_channel}: {e}")
        return 0


def publish_progress_sync(
    channel: str,
    status: str,
    progress: int,
    message: Optional[str] = None,
    **channel_params
) -> int:
    """Sync version of publish_progress."""
    data = {
        "status": status,
        "progress": progress,
    }
    if message:
        data["message"] = message

    return publish_event_sync(channel, "progress", data, **channel_params)


# =============================================================================
# GLOBAL TASK CHANNEL HELPERS
# =============================================================================

async def publish_task_event(
    task_type: str,
    event_type: str,
    task_id: int,
    video_id: int,
    video_name: str,
    status: str,
    progress: int = 0,
    current_step: Optional[str] = None,
    error_message: Optional[str] = None,
    thumbnails_generated: Optional[int] = None,
    total_thumbnails: Optional[int] = None
) -> int:
    """
    Publish an event to the global tasks channel.

    This allows the task queue UI to receive updates for all running tasks
    regardless of which page the user is viewing.

    Args:
        task_type: 'analysis' or 'generation'
        event_type: 'task_started', 'task_progress', 'task_completed',
                    'task_cancelled', or 'task_error'
        task_id: video_id for analysis, job_id for generation
        video_id: ID of the video being processed
        video_name: Filename of the video
        status: Current status string
        progress: Progress percentage (0-100)
        current_step: Human-readable current step description
        error_message: Error message if event_type is 'task_error'
        thumbnails_generated: For generation tasks, number completed
        total_thumbnails: For generation tasks, total to generate

    Returns:
        Number of subscribers that received the message
    """
    data = {
        "task_type": task_type,
        "task_id": task_id,
        "video_id": video_id,
        "video_name": video_name,
        "status": status,
        "progress": progress,
    }

    if current_step:
        data["current_step"] = current_step
    if error_message:
        data["error_message"] = error_message
    if thumbnails_generated is not None:
        data["thumbnails_generated"] = thumbnails_generated
    if total_thumbnails is not None:
        data["total_thumbnails"] = total_thumbnails

    return await publish_event(CHANNEL_TASKS, event_type, data)


def publish_task_event_sync(
    task_type: str,
    event_type: str,
    task_id: int,
    video_id: int,
    video_name: str,
    status: str,
    progress: int = 0,
    current_step: Optional[str] = None,
    error_message: Optional[str] = None,
    thumbnails_generated: Optional[int] = None,
    total_thumbnails: Optional[int] = None
) -> int:
    """
    Sync version of publish_task_event for use in workers.

    See publish_task_event for argument documentation.
    """
    data = {
        "task_type": task_type,
        "task_id": task_id,
        "video_id": video_id,
        "video_name": video_name,
        "status": status,
        "progress": progress,
    }

    if current_step:
        data["current_step"] = current_step
    if error_message:
        data["error_message"] = error_message
    if thumbnails_generated is not None:
        data["thumbnails_generated"] = thumbnails_generated
    if total_thumbnails is not None:
        data["total_thumbnails"] = total_thumbnails

    return publish_event_sync(CHANNEL_TASKS, event_type, data)

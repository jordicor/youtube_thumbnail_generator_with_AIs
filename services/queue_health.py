"""
Queue health

Whether queued tasks can run: Redis answers and a worker is alive. The task
stream sends it to every page, which warns when they cannot.
"""

import asyncio
import logging
from time import monotonic

from arq.constants import health_check_key_suffix
from redis.exceptions import RedisError

from job_queue.client import RedisManager
from job_queue.settings import WorkerSettings

logger = logging.getLogger(__name__)


OK = 'ok'
REDIS_DOWN = 'redis_down'
WORKER_DOWN = 'worker_down'

# The arq worker rewrites this key every HEALTH_CHECK_INTERVAL seconds and
# Redis drops it one second later
WORKER_HEALTH_KEY = WorkerSettings.QUEUE_DEFAULT + health_check_key_suffix

# A worker busy for a moment writes its key late: it counts as stopped only
# after this long without one. With the key's own life and the time between
# checks, a stopped worker shows in about a minute.
WORKER_GRACE_SECONDS = WorkerSettings.HEALTH_CHECK_INTERVAL

CHECK_SECONDS = 10
REDIS_TIMEOUT_SECONDS = 3

_status: str | None = None
_checked_at: float | None = None
_worker_seen_at: float | None = None
_refresh_task: asyncio.Task | None = None


async def check() -> str:
    """Ask Redis now: OK, REDIS_DOWN or WORKER_DOWN."""
    global _worker_seen_at

    try:
        client = await RedisManager.get_client()
        worker_alive = await asyncio.wait_for(client.exists(WORKER_HEALTH_KEY), REDIS_TIMEOUT_SECONDS)
    except (OSError, RedisError):
        # Refused, unreachable or too slow (a timeout is an OSError) to queue anything
        return REDIS_DOWN

    now = monotonic()
    if worker_alive:
        _worker_seen_at = now
        return OK
    if _worker_seen_at is not None and now - _worker_seen_at < WORKER_GRACE_SECONDS:
        return OK
    return WORKER_DOWN


async def _refresh() -> None:
    global _status, _checked_at

    status = await check()
    if status != _status:
        log = logger.info if status == OK else logger.warning
        log(f"Task queue: {status}")
    _status = status
    _checked_at = monotonic()


def status() -> str | None:
    """
    The last known state, without waiting: OK, REDIS_DOWN or WORKER_DOWN, or
    None before the first check ends.

    A check older than CHECK_SECONDS starts a new one in the background, so
    callers that poll (the task stream) keep it fresh and none of them waits
    for a Redis that does not answer.
    """
    global _refresh_task

    stale = _checked_at is None or monotonic() - _checked_at >= CHECK_SECONDS
    if stale and (_refresh_task is None or _refresh_task.done()):
        _refresh_task = asyncio.create_task(_refresh())
    return _status

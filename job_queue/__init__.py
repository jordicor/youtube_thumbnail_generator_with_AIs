"""
Redis module for the job queue.

This module provides:
- Redis client connection management
- Job queue configuration for arq workers
- Job enqueueing helpers
"""

from job_queue.client import get_redis, get_redis_sync, RedisManager
from job_queue.settings import get_arq_redis_settings, WorkerSettings
from job_queue.queue import (
    get_arq_pool,
    close_arq_pool,
    enqueue_analysis,
    enqueue_generation,
    get_job_status,
)

__all__ = [
    # Client
    "get_redis",
    "get_redis_sync",
    "RedisManager",
    # Settings
    "get_arq_redis_settings",
    "WorkerSettings",
    # Queue
    "get_arq_pool",
    "close_arq_pool",
    "enqueue_analysis",
    "enqueue_generation",
    "get_job_status",
]

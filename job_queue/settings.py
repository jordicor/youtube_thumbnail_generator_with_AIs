"""
arq worker settings and configuration.

This module defines the settings for arq workers including:
- Redis connection settings
- Job timeouts and retries
- Queue configuration
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional
from datetime import timedelta

from arq.connections import RedisSettings

logger = logging.getLogger(__name__)

# Parse Redis URL components
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def parse_redis_url(url: str) -> dict:
    """Parse a Redis URL into components."""
    # redis://[[username:]password@]host[:port][/database]
    from urllib.parse import urlparse

    parsed = urlparse(url)

    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 6379,
        "database": int(parsed.path.lstrip("/") or 0),
        "password": parsed.password,
        "username": parsed.username,
    }


def get_arq_redis_settings() -> RedisSettings:
    """
    Get Redis settings for arq workers.

    Returns:
        RedisSettings configured from REDIS_URL environment variable
    """
    config = parse_redis_url(REDIS_URL)

    return RedisSettings(
        host=config["host"],
        port=config["port"],
        database=config["database"],
        password=config["password"],
        # arq-specific settings
        conn_timeout=10,
        conn_retries=5,
        conn_retry_delay=1,
    )


@dataclass
class WorkerSettings:
    """
    Configuration for the arq worker.

    These settings control job behavior, timeouts, and retries.
    """

    # Queue names
    QUEUE_DEFAULT: str = "arq:queue"
    QUEUE_ANALYSIS: str = "arq:queue:analysis"
    QUEUE_GENERATION: str = "arq:queue:generation"

    # Job timeouts (in seconds)
    # Analysis can take up to 30 minutes for long videos
    JOB_TIMEOUT_ANALYSIS: int = int(os.getenv("JOB_TIMEOUT_ANALYSIS", "3600"))  # 1 hour

    # Generation can take very long with many images and slow models
    # Default: 6 hours (21600 seconds) to handle large batches
    JOB_TIMEOUT_GENERATION: int = int(os.getenv("JOB_TIMEOUT_GENERATION", "21600"))  # 6 hours

    # Default timeout for other jobs
    JOB_TIMEOUT_DEFAULT: int = int(os.getenv("JOB_TIMEOUT_DEFAULT", "3600"))  # 1 hour

    # Retry configuration
    MAX_RETRIES: int = int(os.getenv("JOB_MAX_RETRIES", "3"))
    RETRY_DELAY_BASE: int = 60  # Base delay in seconds (exponential backoff)

    # Worker concurrency
    # Limited to 2 by default because:
    # - GPU memory constraints (InsightFace, Whisper)
    # - API rate limits
    MAX_CONCURRENT_JOBS: int = int(os.getenv("WORKER_MAX_JOBS", "2"))

    # Health check interval
    HEALTH_CHECK_INTERVAL: int = 30  # seconds

    # Job result TTL (how long to keep results)
    RESULT_TTL: int = 3600  # 1 hour

    @classmethod
    def get_job_timeout(cls, job_type: str) -> int:
        """Get timeout for a specific job type."""
        timeouts = {
            "analysis": cls.JOB_TIMEOUT_ANALYSIS,
            "generation": cls.JOB_TIMEOUT_GENERATION,
        }
        return timeouts.get(job_type, cls.JOB_TIMEOUT_DEFAULT)

    @classmethod
    def get_retry_delay(cls, attempt: int) -> timedelta:
        """
        Calculate retry delay with exponential backoff.

        Delay = base * 2^attempt (capped at 30 minutes)
        Attempt 1: 60s, Attempt 2: 120s, Attempt 3: 240s
        """
        delay = cls.RETRY_DELAY_BASE * (2 ** attempt)
        max_delay = 1800  # 30 minutes max
        return timedelta(seconds=min(delay, max_delay))

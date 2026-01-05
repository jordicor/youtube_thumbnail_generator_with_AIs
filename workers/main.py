"""
arq worker entry point.

Run with:
    arq workers.main.WorkerSettings

Or use the start_worker.bat script.
"""

import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from arq import cron
from arq.connections import RedisSettings

from job_queue.settings import get_arq_redis_settings, WorkerSettings as WS
from workers.tasks import analyze_video, run_generation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)


async def startup(ctx: dict) -> None:
    """
    Called when the worker starts.

    Initialize any resources needed by tasks.
    """
    logger.info("Worker starting up...")

    # Initialize database connection pool (optional, tasks create their own)
    # This just ensures the database exists and migrations are run
    from database.db import init_db
    await init_db()

    logger.info("Worker startup complete")


async def shutdown(ctx: dict) -> None:
    """
    Called when the worker shuts down.

    Clean up any resources.

    NOTE: We intentionally do NOT close RedisManager here because arq may still
    be finalizing jobs (saving results to Redis) when this hook is called.
    The connections will be cleaned up automatically when the process exits.
    """
    logger.info("Worker shutting down...")

    # Close database connections
    from database.db import close_db
    await close_db()

    # NOTE: Don't close RedisManager - arq handles its own Redis connection,
    # and closing our pool here can cause race conditions with arq's finish_job.
    # The process will clean up connections on exit anyway.

    logger.info("Worker shutdown complete")


class WorkerSettings:
    """
    arq worker configuration.

    This class is used by arq to configure the worker.
    Run with: arq workers.main.WorkerSettings
    """

    # Task functions to register
    functions = [
        analyze_video,
        run_generation,
    ]

    # Redis connection settings
    redis_settings = get_arq_redis_settings()

    # Lifecycle hooks
    on_startup = startup
    on_shutdown = shutdown

    # Worker behavior
    max_jobs = WS.MAX_CONCURRENT_JOBS  # Max concurrent jobs (default: 2)
    job_timeout = WS.JOB_TIMEOUT_DEFAULT  # Default timeout (10 min)
    max_tries = WS.MAX_RETRIES + 1  # Total attempts (retries + 1)
    health_check_interval = WS.HEALTH_CHECK_INTERVAL

    # Keep results for 1 hour
    keep_result = WS.RESULT_TTL

    # Retry delay (exponential backoff handled by arq)
    retry_jobs = True

    # Queue name
    queue_name = "arq:queue"

    # Log job results
    after_job_end = None  # Can add a hook here for custom logging

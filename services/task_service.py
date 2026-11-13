"""
Task Service

Unified service for managing analysis and generation tasks.
Provides a single interface for the task queue UI to list and cancel tasks.
"""

import logging
import socket
from typing import List, Optional
import aiosqlite
import psutil


logger = logging.getLogger(__name__)


# error_message of a generation job that was running when the worker stopped
GENERATION_INTERRUPTED_MESSAGE = 'Generation interrupted: the worker stopped'


async def release_interrupted_tasks(db: aiosqlite.Connection) -> dict:
    """Free what a stopped worker left half done. Only the worker calls it, when it starts.

    Analyses run in a single worker. Generations also run directly in the CLI:
    their executor's PID and creation time distinguish live work from abandoned
    work, including a cancelled job whose thread is still stopping. Jobs owned
    by another machine are left for that machine to recover.

    Queued tasks (videos in 'analyzing', jobs in 'pending') are still in the
    queue and keep their claim. The thumbnails an interrupted generation had
    finished stay.

    Returns how many analyses and generation jobs were interrupted.
    """
    from services.analysis_service import ANALYSIS_INTERRUPTED_MESSAGE, ANALYSIS_RUNNING_STATES
    from services.generation_service import STOPPING_JOB_SQL

    running = ','.join('?' * len(ANALYSIS_RUNNING_STATES))
    active = ','.join('?' * len(TaskService.GENERATION_ACTIVE_STATES))
    try:
        await db.execute("BEGIN IMMEDIATE")
        analyses = await db.execute(
            f"""
            UPDATE videos
            SET status = 'pending', error_message = ?, analysis_run = NULL, analysis_previous_status = NULL,
                analysis_outcome = 'error'
            WHERE status IN ({running})
            """,
            [ANALYSIS_INTERRUPTED_MESSAGE, *ANALYSIS_RUNNING_STATES]
        )
        # A terminal analysis may have committed its result just before the worker died.
        await db.execute(
            "UPDATE videos SET analysis_run = NULL, analysis_previous_status = NULL "
            "WHERE analysis_run IS NOT NULL AND status IN ('analyzed', 'completed', 'error')"
        )
        # A cancellation the stopped worker never got to acknowledge
        await db.execute(
            "UPDATE videos SET analysis_run = NULL, analysis_previous_status = NULL, analysis_outcome = 'cancelled' "
            "WHERE status = 'cancelled' AND analysis_run IS NOT NULL"
        )
        async with db.execute(
            f"""SELECT gj.id, gj.video_id, gj.executor_host, gj.executor_pid, gj.executor_started_at
                FROM generation_jobs gj WHERE gj.status IN ({active}) OR {STOPPING_JOB_SQL}""",
            TaskService.GENERATION_ACTIVE_STATES
        ) as cursor:
            generations = await cursor.fetchall()
        live_videos = set()
        interrupted_jobs = 0
        local_host = socket.gethostname()
        for job_id, video_id, host, pid, started_at in generations:
            if pid is not None:
                if host != local_host:
                    live_videos.add(video_id)
                    continue
                try:
                    if psutil.Process(pid).create_time() == started_at:
                        live_videos.add(video_id)
                        continue
                except psutil.NoSuchProcess:
                    pass
            cursor = await db.execute(
                f"UPDATE generation_jobs SET status = 'error', error_message = ? "
                f"WHERE id = ? AND status IN ({active})",
                [GENERATION_INTERRUPTED_MESSAGE, job_id, *TaskService.GENERATION_ACTIVE_STATES]
            )
            interrupted_jobs += cursor.rowcount
        # Videos no job is going to generate for: interrupted or cancelled without acknowledgement
        keep_live = f"AND id NOT IN ({','.join('?' * len(live_videos))})" if live_videos else ''
        await db.execute(f"""
            UPDATE videos
            SET status = CASE WHEN EXISTS (
                SELECT 1 FROM generation_jobs done
                WHERE done.video_id = videos.id AND done.status = 'completed'
            ) THEN 'completed' ELSE 'analyzed' END
            WHERE status = 'generating'
              AND NOT EXISTS (SELECT 1 FROM generation_jobs queued
                              WHERE queued.video_id = videos.id AND queued.status = 'pending')
              {keep_live}
        """, list(live_videos))
        await db.commit()
    except BaseException:
        await db.rollback()
        raise
    return {'analyses': analyses.rowcount, 'generations': interrupted_jobs}


class TaskService:
    """Service for unified task queue management."""

    # Analysis states that indicate work in progress
    ANALYSIS_ACTIVE_STATES = (
        'analyzing',
        'analyzing_scenes',
        'analyzing_faces',
        'clustering',
        'transcribing'
    )

    # Generation states that indicate work in progress
    GENERATION_ACTIVE_STATES = (
        'transcribing',
        'prompting',
        'generating'
    )

    GENERATION_PENDING_STATES = ('pending',)

    # Status of a task the user cancelled whose worker has not stopped yet. It
    # still holds its video, so it stays in the list until the worker lets go.
    STOPPING = 'stopping'

    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    async def get_active_tasks(self) -> List[dict]:
        """
        Get all currently running tasks (analysis + generation), the cancelled
        ones that are still stopping included.

        Returns list of unified task objects with:
        - id: task ID (video_id for analysis, job_id for generation)
        - type: 'analysis' | 'generation'
        - video_id, video_name, status, progress, current_step, started_at
        - For generation: thumbnails_generated, total_thumbnails
        """
        from services.generation_service import STOPPING_JOB_SQL

        tasks = []

        # 1. Get active analysis tasks (videos being analyzed). A cancelled
        # analysis keeps its claim on the video until its worker job ends
        placeholders = ','.join('?' * len(self.ANALYSIS_ACTIVE_STATES))
        analysis_query = f"""
            SELECT
                v.id,
                v.filename,
                v.status,
                v.updated_at as started_at
            FROM videos v
            WHERE v.status IN ({placeholders})
               OR (v.status = 'cancelled' AND v.analysis_run IS NOT NULL)
            ORDER BY v.updated_at DESC
        """
        async with self.db.execute(analysis_query, self.ANALYSIS_ACTIVE_STATES) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                status = self.STOPPING if row[2] == 'cancelled' else row[2]
                tasks.append({
                    'id': row[0],  # video_id is the task ID for analysis
                    'type': 'analysis',
                    'video_id': row[0],
                    'video_name': row[1],
                    'status': status,
                    'progress': self._estimate_analysis_progress(status),
                    'current_step': status,
                    'started_at': row[3]
                })

        # 2. Get active generation tasks
        placeholders = ','.join('?' * len(self.GENERATION_ACTIVE_STATES))
        generation_query = f"""
            SELECT
                gj.id,
                gj.video_id,
                v.filename,
                gj.status,
                gj.progress,
                gj.num_images,
                gj.created_at,
                (SELECT COUNT(*) FROM thumbnails t WHERE t.job_id = gj.id) as thumbnails_done
            FROM generation_jobs gj
            JOIN videos v ON gj.video_id = v.id
            WHERE gj.status IN ({placeholders}) OR {STOPPING_JOB_SQL}
            ORDER BY gj.created_at DESC
        """
        async with self.db.execute(generation_query, self.GENERATION_ACTIVE_STATES) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                status = self.STOPPING if row[3] == 'cancelled' else row[3]
                tasks.append({
                    'id': row[0],  # job_id is the task ID for generation
                    'type': 'generation',
                    'video_id': row[1],
                    'video_name': row[2],
                    'status': status,
                    'progress': row[4],
                    'current_step': status,
                    'started_at': row[6],
                    'thumbnails_generated': row[7],
                    'total_thumbnails': row[5]
                })

        return tasks

    async def get_pending_tasks(self) -> List[dict]:
        """
        Get all pending tasks waiting to start.

        Only generation jobs can be 'pending' - videos go directly
        to 'analyzing' when analysis is triggered.
        """
        tasks = []

        query = """
            SELECT
                gj.id,
                gj.video_id,
                v.filename,
                gj.status,
                gj.num_images,
                gj.created_at
            FROM generation_jobs gj
            JOIN videos v ON gj.video_id = v.id
            WHERE gj.status = 'pending'
            ORDER BY gj.created_at ASC
        """
        async with self.db.execute(query) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                tasks.append({
                    'id': row[0],
                    'type': 'generation',
                    'video_id': row[1],
                    'video_name': row[2],
                    'status': row[3],
                    'total_thumbnails': row[4],
                    'created_at': row[5]
                })

        return tasks

    async def get_task_for_video(self, video_id: int) -> Optional[dict]:
        """
        Get the active task for a specific video (if any).

        Returns the first one found: its analysis, else its generation job
        (running, stopping or queued).
        """
        tasks = await self.get_active_tasks() + await self.get_pending_tasks()
        return next((task for task in tasks if task['video_id'] == video_id), None)

    async def get_task_outcome(self, task_type: str, task_id: int) -> dict:
        """
        How a task that left the lists ended.

        Args:
            task_type: 'analysis' or 'generation'
            task_id: video_id for analysis, job_id for generation

        Returns:
            {'status': 'completed' | 'cancelled' | 'error', 'error_message': the
            reason of an error}. A task with nothing to say (its video or job no
            longer exists, or its analysis never got queued) counts as cancelled.
        """
        if task_type == 'analysis':
            query = "SELECT analysis_outcome, error_message FROM videos WHERE id = ?"
        elif task_type == 'generation':
            query = "SELECT status, error_message FROM generation_jobs WHERE id = ?"
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        async with self.db.execute(query, [task_id]) as cursor:
            row = await cursor.fetchone()

        status = row[0] if row and row[0] in ('completed', 'error') else 'cancelled'
        return {'status': status, 'error_message': row[1] if status == 'error' else None}

    async def cancel_task(self, task_type: str, task_id: int) -> bool:
        """
        Cancel a task.

        Args:
            task_type: 'analysis' or 'generation'
            task_id: video_id for analysis, job_id for generation

        Returns:
            True if cancelled successfully, False otherwise.
        """
        if task_type == 'analysis':
            from services.analysis_service import AnalysisService, ANALYSIS_RUNNING_STATES

            # A queued analysis has done nothing yet: the video goes back as it was
            if await AnalysisService(self.db).unclaim_analysis(task_id):
                logger.info(f"Task analysis:{task_id} cancelled before it started")
                return True

            # A running one stops at its next check and keeps the video until
            # then, so another analysis cannot start over the same files
            placeholders = ','.join('?' * len(ANALYSIS_RUNNING_STATES))
            query = f"""
                UPDATE videos
                SET status = 'cancelled', error_message = 'Cancelled by user'
                WHERE id = ? AND status IN ({placeholders})
            """
            cursor = await self.db.execute(query, (task_id,) + ANALYSIS_RUNNING_STATES)
            await self.db.commit()
            if cursor.rowcount > 0:
                logger.info(f"Task analysis:{task_id} cancelled successfully")
                return True
            else:
                logger.warning(f"Task analysis:{task_id} could not be cancelled (not found or not in active state)")
                return False

        elif task_type == 'generation':
            from services.generation_service import GenerationService

            # Share the atomic terminal transition and video ownership check.
            if await GenerationService(self.db).cancel_job(task_id):
                logger.info(f"Task generation:{task_id} cancelled successfully")
                return True
            else:
                logger.warning(f"Task generation:{task_id} could not be cancelled (not found or not in cancellable state)")
                return False

        logger.warning(f"Unknown task type: {task_type}")
        return False

    def _estimate_analysis_progress(self, status: str) -> int:
        """Estimate analysis progress based on current step."""
        progress_map = {
            'analyzing': 5,
            'analyzing_scenes': 20,
            'analyzing_faces': 50,
            'clustering': 75,
            'transcribing': 90
        }
        return progress_map.get(status, 0)

"""
Task Service

Unified service for managing analysis and generation tasks.
Provides a single interface for the task queue UI to list and cancel tasks.
"""

import logging
from typing import List, Optional
import aiosqlite


logger = logging.getLogger(__name__)


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

    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    async def get_active_tasks(self) -> List[dict]:
        """
        Get all currently running tasks (analysis + generation).

        Returns list of unified task objects with:
        - id: task ID (video_id for analysis, job_id for generation)
        - type: 'analysis' | 'generation'
        - video_id, video_name, status, progress, current_step, started_at
        - For generation: thumbnails_generated, total_thumbnails
        """
        tasks = []

        # 1. Get active analysis tasks (videos being analyzed)
        placeholders = ','.join('?' * len(self.ANALYSIS_ACTIVE_STATES))
        analysis_query = f"""
            SELECT
                v.id,
                v.filename,
                v.status,
                v.updated_at as started_at
            FROM videos v
            WHERE v.status IN ({placeholders})
            ORDER BY v.updated_at DESC
        """
        async with self.db.execute(analysis_query, self.ANALYSIS_ACTIVE_STATES) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                tasks.append({
                    'id': row[0],  # video_id is the task ID for analysis
                    'type': 'analysis',
                    'video_id': row[0],
                    'video_name': row[1],
                    'status': row[2],
                    'progress': self._estimate_analysis_progress(row[2]),
                    'current_step': row[2],
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
            WHERE gj.status IN ({placeholders})
            ORDER BY gj.created_at DESC
        """
        async with self.db.execute(generation_query, self.GENERATION_ACTIVE_STATES) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                tasks.append({
                    'id': row[0],  # job_id is the task ID for generation
                    'type': 'generation',
                    'video_id': row[1],
                    'video_name': row[2],
                    'status': row[3],
                    'progress': row[4],
                    'current_step': row[3],
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

        Returns the first active task found (analysis or generation).
        """
        # Check for active analysis
        placeholders = ','.join('?' * len(self.ANALYSIS_ACTIVE_STATES))
        analysis_query = f"""
            SELECT id, filename, status, updated_at
            FROM videos
            WHERE id = ? AND status IN ({placeholders})
        """
        async with self.db.execute(
            analysis_query,
            (video_id,) + self.ANALYSIS_ACTIVE_STATES
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'type': 'analysis',
                    'video_id': row[0],
                    'video_name': row[1],
                    'status': row[2],
                    'progress': self._estimate_analysis_progress(row[2]),
                    'current_step': row[2],
                    'started_at': row[3]
                }

        # Check for active or pending generation
        all_gen_states = self.GENERATION_PENDING_STATES + self.GENERATION_ACTIVE_STATES
        placeholders = ','.join('?' * len(all_gen_states))
        generation_query = f"""
            SELECT
                gj.id,
                v.filename,
                gj.status,
                gj.progress,
                gj.num_images,
                gj.created_at,
                (SELECT COUNT(*) FROM thumbnails t WHERE t.job_id = gj.id) as thumbnails_done
            FROM generation_jobs gj
            JOIN videos v ON gj.video_id = v.id
            WHERE gj.video_id = ? AND gj.status IN ({placeholders})
            ORDER BY gj.created_at DESC
            LIMIT 1
        """
        async with self.db.execute(
            generation_query,
            (video_id,) + all_gen_states
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'type': 'generation',
                    'video_id': video_id,
                    'video_name': row[1],
                    'status': row[2],
                    'progress': row[3],
                    'current_step': row[2],
                    'started_at': row[5],
                    'thumbnails_generated': row[6],
                    'total_thumbnails': row[4]
                }

        return None

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
            # For analysis, reset video to pending
            placeholders = ','.join('?' * len(self.ANALYSIS_ACTIVE_STATES))
            query = f"""
                UPDATE videos
                SET status = 'pending', error_message = 'Cancelled by user'
                WHERE id = ? AND status IN ({placeholders})
            """
            cursor = await self.db.execute(query, (task_id,) + self.ANALYSIS_ACTIVE_STATES)
            await self.db.commit()
            if cursor.rowcount > 0:
                logger.info(f"Task analysis:{task_id} cancelled successfully")
                return True
            else:
                logger.warning(f"Task analysis:{task_id} could not be cancelled (not found or not in active state)")
                return False

        elif task_type == 'generation':
            # For generation, mark job as cancelled
            all_gen_states = self.GENERATION_PENDING_STATES + self.GENERATION_ACTIVE_STATES
            placeholders = ','.join('?' * len(all_gen_states))
            query = f"""
                UPDATE generation_jobs
                SET status = 'cancelled'
                WHERE id = ? AND status IN ({placeholders})
            """
            cursor = await self.db.execute(
                query,
                (task_id,) + all_gen_states
            )
            await self.db.commit()
            if cursor.rowcount > 0:
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

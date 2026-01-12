"""
E2E Tests for Generation Flow

Tests that verify the complete generation task lifecycle through the Task Queue system.
"""

import pytest
from services.task_service import TaskService


# Import helpers from parent conftest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import insert_video, insert_job, insert_cluster


# =============================================================================
# GENERATION VISIBILITY IN TASK QUEUE
# =============================================================================

class TestGenerationVisibilityInTaskQueue:
    """Tests for generation tasks appearing correctly in task queue."""

    async def test_pending_job_appears_in_pending_tasks(self, test_db):
        """Generation job in pending state should appear in pending tasks."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        job_id = await insert_job(test_db, video_id, cluster_id, "pending")

        pending_tasks = await service.get_pending_tasks()

        assert len(pending_tasks) == 1
        assert pending_tasks[0]['id'] == job_id
        assert pending_tasks[0]['type'] == 'generation'
        assert pending_tasks[0]['status'] == 'pending'

    async def test_pending_job_not_in_active_tasks(self, test_db):
        """Generation job in pending state should NOT appear in active tasks."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        await insert_job(test_db, video_id, cluster_id, "pending")

        active_tasks = await service.get_active_tasks()

        assert len(active_tasks) == 0

    async def test_generating_job_appears_in_active_tasks(self, test_db):
        """Generation job in generating state should appear in active tasks."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        job_id = await insert_job(test_db, video_id, cluster_id, "generating", progress=50)

        active_tasks = await service.get_active_tasks()

        assert len(active_tasks) == 1
        assert active_tasks[0]['id'] == job_id
        assert active_tasks[0]['type'] == 'generation'
        assert active_tasks[0]['status'] == 'generating'
        assert active_tasks[0]['progress'] == 50

    async def test_all_generation_active_states(self, test_db):
        """All active generation states should appear in active tasks."""
        service = TaskService(test_db)

        active_states = ['transcribing', 'prompting', 'generating']
        job_ids = []

        for i, status in enumerate(active_states):
            video_id = await insert_video(test_db, f"video_{i}.mp4", "analyzed")
            cluster_id = await insert_cluster(test_db, video_id, cluster_index=i)
            job_id = await insert_job(test_db, video_id, cluster_id, status)
            job_ids.append(job_id)

        active_tasks = await service.get_active_tasks()

        assert len(active_tasks) == 3
        active_job_ids = {t['id'] for t in active_tasks}
        assert active_job_ids == set(job_ids)

    async def test_completed_job_not_in_any_list(self, test_db):
        """Completed generation job should NOT appear in any task list."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "completed")
        cluster_id = await insert_cluster(test_db, video_id)
        await insert_job(test_db, video_id, cluster_id, "completed")

        active_tasks = await service.get_active_tasks()
        pending_tasks = await service.get_pending_tasks()

        assert len(active_tasks) == 0
        assert len(pending_tasks) == 0


# =============================================================================
# GENERATION CANCELLATION FLOW
# =============================================================================

class TestGenerationCancellationFlow:
    """Tests for the complete generation cancellation flow."""

    async def test_cancel_pending_job(self, test_db):
        """Should be able to cancel a pending generation job."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        job_id = await insert_job(test_db, video_id, cluster_id, "pending")

        # Verify it appears in pending
        pending_before = await service.get_pending_tasks()
        assert len(pending_before) == 1

        # Cancel
        result = await service.cancel_task('generation', job_id)
        assert result is True

        # Verify it no longer appears
        pending_after = await service.get_pending_tasks()
        assert len(pending_after) == 0

        # Verify status changed to cancelled
        async with test_db.execute(
            "SELECT status FROM generation_jobs WHERE id = ?",
            (job_id,)
        ) as cursor:
            row = await cursor.fetchone()
            assert row[0] == 'cancelled'

    async def test_cancel_generating_job(self, test_db):
        """Should be able to cancel an in-progress generation job."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        job_id = await insert_job(test_db, video_id, cluster_id, "generating", progress=60)

        # Verify it appears in active
        active_before = await service.get_active_tasks()
        assert len(active_before) == 1

        # Cancel
        result = await service.cancel_task('generation', job_id)
        assert result is True

        # Verify it no longer appears
        active_after = await service.get_active_tasks()
        assert len(active_after) == 0

    async def test_cancel_all_generation_states(self, test_db):
        """Should be able to cancel from any generation state."""
        service = TaskService(test_db)

        cancellable_states = ['pending', 'transcribing', 'prompting', 'generating']

        for i, status in enumerate(cancellable_states):
            video_id = await insert_video(test_db, f"{status}.mp4", "analyzed")
            cluster_id = await insert_cluster(test_db, video_id, cluster_index=i)
            job_id = await insert_job(test_db, video_id, cluster_id, status)

            result = await service.cancel_task('generation', job_id)

            assert result is True, f"Failed to cancel from state: {status}"

            # Verify cancelled
            async with test_db.execute(
                "SELECT status FROM generation_jobs WHERE id = ?",
                (job_id,)
            ) as cursor:
                row = await cursor.fetchone()
                assert row[0] == 'cancelled', f"Job not cancelled from state: {status}"

    async def test_cannot_cancel_completed_job(self, test_db):
        """Cannot cancel a job that is already completed."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "completed")
        cluster_id = await insert_cluster(test_db, video_id)
        job_id = await insert_job(test_db, video_id, cluster_id, "completed")

        result = await service.cancel_task('generation', job_id)

        assert result is False

    async def test_cannot_cancel_already_cancelled_job(self, test_db):
        """Cannot cancel a job that is already cancelled."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        job_id = await insert_job(test_db, video_id, cluster_id, "cancelled")

        result = await service.cancel_task('generation', job_id)

        assert result is False


# =============================================================================
# GET TASK FOR VIDEO (GENERATION)
# =============================================================================

class TestGetTaskForVideoGeneration:
    """Tests for retrieving generation task for a specific video."""

    async def test_returns_pending_generation_task(self, test_db):
        """Should return pending generation task for video."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        job_id = await insert_job(test_db, video_id, cluster_id, "pending")

        task = await service.get_task_for_video(video_id)

        assert task is not None
        assert task['type'] == 'generation'
        assert task['id'] == job_id
        assert task['status'] == 'pending'

    async def test_returns_active_generation_task(self, test_db):
        """Should return active generation task for video."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        job_id = await insert_job(test_db, video_id, cluster_id, "generating", progress=75)

        task = await service.get_task_for_video(video_id)

        assert task is not None
        assert task['type'] == 'generation'
        assert task['id'] == job_id
        assert task['status'] == 'generating'
        assert task['progress'] == 75

    async def test_returns_none_for_completed_job(self, test_db):
        """Should return None when only completed jobs exist."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "completed")
        cluster_id = await insert_cluster(test_db, video_id)
        await insert_job(test_db, video_id, cluster_id, "completed")

        task = await service.get_task_for_video(video_id)

        assert task is None

    async def test_returns_most_recent_job(self, test_db):
        """Should return the most recent job when multiple exist."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)

        # Create old completed job
        await insert_job(test_db, video_id, cluster_id, "completed")

        # Create new pending job
        new_job_id = await insert_job(test_db, video_id, cluster_id, "pending")

        task = await service.get_task_for_video(video_id)

        assert task is not None
        assert task['id'] == new_job_id


# =============================================================================
# MIXED ANALYSIS AND GENERATION
# =============================================================================

class TestMixedAnalysisAndGeneration:
    """Tests for scenarios with both analysis and generation tasks."""

    async def test_analysis_prioritized_over_generation(self, test_db):
        """When both exist, analysis task should be returned for get_task_for_video."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzing_faces")
        cluster_id = await insert_cluster(test_db, video_id)
        await insert_job(test_db, video_id, cluster_id, "pending")

        task = await service.get_task_for_video(video_id)

        assert task is not None
        assert task['type'] == 'analysis'
        assert task['status'] == 'analyzing_faces'

    async def test_both_appear_in_active_tasks(self, test_db):
        """Both analysis and generation tasks should appear in active tasks."""
        service = TaskService(test_db)

        # Video being analyzed
        v1 = await insert_video(test_db, "analyzing.mp4", "analyzing_scenes")

        # Video with active generation
        v2 = await insert_video(test_db, "generating.mp4", "analyzed")
        c2 = await insert_cluster(test_db, v2)
        await insert_job(test_db, v2, c2, "generating")

        active_tasks = await service.get_active_tasks()

        assert len(active_tasks) == 2
        types = {t['type'] for t in active_tasks}
        assert types == {'analysis', 'generation'}

    async def test_pending_generation_separate_from_active_analysis(self, test_db):
        """Pending generation should be in pending_tasks, not active_tasks with analysis."""
        service = TaskService(test_db)

        # Active analysis
        await insert_video(test_db, "analyzing.mp4", "analyzing_faces")

        # Pending generation (different video)
        v2 = await insert_video(test_db, "waiting.mp4", "analyzed")
        c2 = await insert_cluster(test_db, v2)
        await insert_job(test_db, v2, c2, "pending")

        active_tasks = await service.get_active_tasks()
        pending_tasks = await service.get_pending_tasks()

        assert len(active_tasks) == 1
        assert active_tasks[0]['type'] == 'analysis'

        assert len(pending_tasks) == 1
        assert pending_tasks[0]['type'] == 'generation'


# =============================================================================
# MULTIPLE GENERATION JOBS
# =============================================================================

class TestMultipleGenerationJobs:
    """Tests for handling multiple generation jobs."""

    async def test_multiple_pending_jobs_appear_in_order(self, test_db):
        """Multiple pending jobs should appear ordered by creation time."""
        service = TaskService(test_db)

        job_ids = []
        for i in range(3):
            video_id = await insert_video(test_db, f"video_{i}.mp4", "analyzed")
            cluster_id = await insert_cluster(test_db, video_id, cluster_index=i)
            job_id = await insert_job(test_db, video_id, cluster_id, "pending")
            job_ids.append(job_id)

        pending_tasks = await service.get_pending_tasks()

        assert len(pending_tasks) == 3
        # Should be in creation order (ASC)
        for i, task in enumerate(pending_tasks):
            assert task['id'] == job_ids[i]

    async def test_cancel_one_job_keeps_others(self, test_db):
        """Cancelling one job should not affect other jobs."""
        service = TaskService(test_db)

        # Create two generating jobs
        v1 = await insert_video(test_db, "video1.mp4", "analyzed")
        c1 = await insert_cluster(test_db, v1, cluster_index=0)
        j1 = await insert_job(test_db, v1, c1, "generating")

        v2 = await insert_video(test_db, "video2.mp4", "analyzed")
        c2 = await insert_cluster(test_db, v2, cluster_index=1)
        j2 = await insert_job(test_db, v2, c2, "generating")

        # Cancel first job
        await service.cancel_task('generation', j1)

        # Second job should still be active
        active_tasks = await service.get_active_tasks()
        assert len(active_tasks) == 1
        assert active_tasks[0]['id'] == j2

"""
Unit Tests for TaskService

Tests for the unified task queue management service.
"""

import pytest
from services.task_service import TaskService
from tests.conftest import insert_video, insert_job, insert_cluster


# =============================================================================
# GET ACTIVE TASKS TESTS
# =============================================================================

class TestGetActiveTasks:
    """Tests for TaskService.get_active_tasks()"""

    async def test_returns_empty_list_when_no_tasks(self, test_db):
        """Should return empty list when no active tasks exist."""
        service = TaskService(test_db)

        tasks = await service.get_active_tasks()

        assert tasks == []

    async def test_returns_analyzing_video_as_task(self, test_db):
        """Should return video in analyzing state as an active task."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzing_faces")

        tasks = await service.get_active_tasks()

        assert len(tasks) == 1
        assert tasks[0]['id'] == video_id
        assert tasks[0]['type'] == 'analysis'
        assert tasks[0]['video_name'] == "test.mp4"
        assert tasks[0]['status'] == 'analyzing_faces'
        assert tasks[0]['progress'] == 50  # analyzing_faces = 50%

    async def test_returns_all_analysis_states(self, test_db):
        """Should return videos in all analysis states."""
        service = TaskService(test_db)

        # Insert videos in different analysis states
        states = ['analyzing', 'analyzing_scenes', 'analyzing_faces', 'clustering', 'transcribing']
        for i, status in enumerate(states):
            await insert_video(test_db, f"video_{i}.mp4", status)

        tasks = await service.get_active_tasks()

        assert len(tasks) == 5
        assert all(t['type'] == 'analysis' for t in tasks)

    async def test_does_not_return_pending_videos(self, test_db):
        """Should not return videos in pending or completed states."""
        service = TaskService(test_db)
        await insert_video(test_db, "pending.mp4", "pending")
        await insert_video(test_db, "analyzed.mp4", "analyzed")
        await insert_video(test_db, "completed.mp4", "completed")

        tasks = await service.get_active_tasks()

        assert tasks == []

    async def test_returns_generating_job_as_task(self, test_db):
        """Should return generation job in generating state as an active task."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        job_id = await insert_job(test_db, video_id, cluster_id, "generating", progress=60)

        tasks = await service.get_active_tasks()

        assert len(tasks) == 1
        assert tasks[0]['id'] == job_id
        assert tasks[0]['type'] == 'generation'
        assert tasks[0]['video_id'] == video_id
        assert tasks[0]['video_name'] == "test.mp4"
        assert tasks[0]['status'] == 'generating'
        assert tasks[0]['progress'] == 60

    async def test_returns_both_analysis_and_generation_tasks(self, test_db):
        """Should return both analysis and generation tasks."""
        service = TaskService(test_db)

        # Analysis task
        await insert_video(test_db, "analyzing.mp4", "analyzing_scenes")

        # Generation task
        video_id = await insert_video(test_db, "generating.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        await insert_job(test_db, video_id, cluster_id, "generating")

        tasks = await service.get_active_tasks()

        assert len(tasks) == 2
        types = {t['type'] for t in tasks}
        assert types == {'analysis', 'generation'}

    async def test_does_not_return_pending_generation_jobs(self, test_db):
        """Pending generation jobs should not appear in active tasks."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        await insert_job(test_db, video_id, cluster_id, "pending")

        tasks = await service.get_active_tasks()

        assert tasks == []


# =============================================================================
# GET PENDING TASKS TESTS
# =============================================================================

class TestGetPendingTasks:
    """Tests for TaskService.get_pending_tasks()"""

    async def test_returns_empty_list_when_no_pending(self, test_db):
        """Should return empty list when no pending tasks exist."""
        service = TaskService(test_db)

        tasks = await service.get_pending_tasks()

        assert tasks == []

    async def test_returns_pending_generation_job(self, test_db):
        """Should return generation jobs in pending state."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        job_id = await insert_job(test_db, video_id, cluster_id, "pending")

        tasks = await service.get_pending_tasks()

        assert len(tasks) == 1
        assert tasks[0]['id'] == job_id
        assert tasks[0]['type'] == 'generation'
        assert tasks[0]['status'] == 'pending'

    async def test_does_not_return_active_generation_jobs(self, test_db):
        """Should not return jobs that are already in progress."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        await insert_job(test_db, video_id, cluster_id, "generating")

        tasks = await service.get_pending_tasks()

        assert tasks == []

    async def test_returns_multiple_pending_jobs_in_order(self, test_db):
        """Should return pending jobs ordered by creation time (oldest first)."""
        service = TaskService(test_db)

        # Create multiple videos and pending jobs
        for i in range(3):
            video_id = await insert_video(test_db, f"video_{i}.mp4", "analyzed")
            cluster_id = await insert_cluster(test_db, video_id, cluster_index=i)
            await insert_job(test_db, video_id, cluster_id, "pending")

        tasks = await service.get_pending_tasks()

        assert len(tasks) == 3
        # Jobs should be ordered by created_at ASC (oldest first)
        for i, task in enumerate(tasks):
            assert task['video_name'] == f"video_{i}.mp4"


# =============================================================================
# GET TASK FOR VIDEO TESTS
# =============================================================================

class TestGetTaskForVideo:
    """Tests for TaskService.get_task_for_video()"""

    async def test_returns_none_when_no_task(self, test_db):
        """Should return None when video has no active task."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "pending")

        task = await service.get_task_for_video(video_id)

        assert task is None

    async def test_returns_none_for_nonexistent_video(self, test_db):
        """Should return None for non-existent video ID."""
        service = TaskService(test_db)

        task = await service.get_task_for_video(99999)

        assert task is None

    async def test_returns_analysis_task(self, test_db):
        """Should return analysis task when video is being analyzed."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzing_faces")

        task = await service.get_task_for_video(video_id)

        assert task is not None
        assert task['type'] == 'analysis'
        assert task['video_id'] == video_id
        assert task['status'] == 'analyzing_faces'

    async def test_returns_generation_task(self, test_db):
        """Should return generation task when video has active generation."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        job_id = await insert_job(test_db, video_id, cluster_id, "generating", progress=50)

        task = await service.get_task_for_video(video_id)

        assert task is not None
        assert task['type'] == 'generation'
        assert task['id'] == job_id
        assert task['status'] == 'generating'
        assert task['progress'] == 50

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

    async def test_prioritizes_analysis_over_generation(self, test_db):
        """Should return analysis task even if generation job exists."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzing_scenes")
        cluster_id = await insert_cluster(test_db, video_id)
        await insert_job(test_db, video_id, cluster_id, "pending")

        task = await service.get_task_for_video(video_id)

        assert task is not None
        assert task['type'] == 'analysis'
        assert task['status'] == 'analyzing_scenes'


# =============================================================================
# CANCEL TASK TESTS
# =============================================================================

class TestCancelTask:
    """Tests for TaskService.cancel_task()"""

    async def test_cancel_analysis_resets_video_to_pending(self, test_db):
        """Cancelling analysis should reset video status to pending."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzing_faces")

        result = await service.cancel_task('analysis', video_id)

        assert result is True

        # Verify video status was reset
        async with test_db.execute("SELECT status FROM videos WHERE id = ?", (video_id,)) as cursor:
            row = await cursor.fetchone()
            assert row[0] == 'pending'

    async def test_cancel_generation_marks_job_cancelled(self, test_db):
        """Cancelling generation should mark job as cancelled."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        job_id = await insert_job(test_db, video_id, cluster_id, "generating")

        result = await service.cancel_task('generation', job_id)

        assert result is True

        # Verify job status was changed
        async with test_db.execute("SELECT status FROM generation_jobs WHERE id = ?", (job_id,)) as cursor:
            row = await cursor.fetchone()
            assert row[0] == 'cancelled'

    async def test_cancel_pending_generation_job(self, test_db):
        """Should be able to cancel pending generation jobs."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")
        cluster_id = await insert_cluster(test_db, video_id)
        job_id = await insert_job(test_db, video_id, cluster_id, "pending")

        result = await service.cancel_task('generation', job_id)

        assert result is True

        # Verify job status
        async with test_db.execute("SELECT status FROM generation_jobs WHERE id = ?", (job_id,)) as cursor:
            row = await cursor.fetchone()
            assert row[0] == 'cancelled'

    async def test_cancel_nonexistent_analysis_returns_false(self, test_db):
        """Cancelling non-existent analysis task should return False."""
        service = TaskService(test_db)

        result = await service.cancel_task('analysis', 99999)

        assert result is False

    async def test_cancel_nonexistent_generation_returns_false(self, test_db):
        """Cancelling non-existent generation task should return False."""
        service = TaskService(test_db)

        result = await service.cancel_task('generation', 99999)

        assert result is False

    async def test_cancel_completed_job_returns_false(self, test_db):
        """Cannot cancel already completed job."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "completed")
        cluster_id = await insert_cluster(test_db, video_id)
        job_id = await insert_job(test_db, video_id, cluster_id, "completed")

        result = await service.cancel_task('generation', job_id)

        assert result is False

    async def test_cancel_invalid_task_type_returns_false(self, test_db):
        """Invalid task type should return False."""
        service = TaskService(test_db)

        result = await service.cancel_task('invalid_type', 1)

        assert result is False


# =============================================================================
# PROGRESS ESTIMATION TESTS
# =============================================================================

class TestEstimateAnalysisProgress:
    """Tests for TaskService._estimate_analysis_progress()"""

    async def test_progress_values(self, test_db):
        """Should return correct progress values for each state."""
        service = TaskService(test_db)

        expected = {
            'analyzing': 5,
            'analyzing_scenes': 20,
            'analyzing_faces': 50,
            'clustering': 75,
            'transcribing': 90,
            'unknown_state': 0
        }

        for status, expected_progress in expected.items():
            progress = service._estimate_analysis_progress(status)
            assert progress == expected_progress, f"Expected {expected_progress} for '{status}', got {progress}"

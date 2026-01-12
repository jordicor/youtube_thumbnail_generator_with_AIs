"""
E2E Tests for Analysis Flow

Tests that verify the complete analysis task lifecycle through the Task Queue system.
"""

import pytest
from services.task_service import TaskService


# Import helpers from parent conftest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import insert_video, insert_cluster


# =============================================================================
# ANALYSIS VISIBILITY IN TASK QUEUE
# =============================================================================

class TestAnalysisVisibilityInTaskQueue:
    """Tests for analysis tasks appearing correctly in task queue."""

    async def test_pending_video_not_in_active_tasks(self, test_db):
        """Video in pending state should NOT appear in active tasks."""
        service = TaskService(test_db)
        await insert_video(test_db, "pending.mp4", "pending")

        active_tasks = await service.get_active_tasks()

        assert len(active_tasks) == 0

    async def test_analyzing_video_appears_in_active_tasks(self, test_db):
        """Video being analyzed should appear in active tasks."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "analyzing.mp4", "analyzing_scenes")

        active_tasks = await service.get_active_tasks()

        assert len(active_tasks) == 1
        assert active_tasks[0]['id'] == video_id
        assert active_tasks[0]['type'] == 'analysis'
        assert active_tasks[0]['status'] == 'analyzing_scenes'

    async def test_analyzed_video_not_in_active_tasks(self, test_db):
        """Completed analysis should NOT appear in active tasks."""
        service = TaskService(test_db)
        await insert_video(test_db, "done.mp4", "analyzed")

        active_tasks = await service.get_active_tasks()

        assert len(active_tasks) == 0

    async def test_all_analysis_states_appear_as_active(self, test_db):
        """All intermediate analysis states should appear as active."""
        service = TaskService(test_db)

        states = ['analyzing', 'analyzing_scenes', 'analyzing_faces', 'clustering', 'transcribing']
        video_ids = []
        for i, status in enumerate(states):
            vid = await insert_video(test_db, f"video_{i}.mp4", status)
            video_ids.append(vid)

        active_tasks = await service.get_active_tasks()

        assert len(active_tasks) == 5
        active_ids = {t['id'] for t in active_tasks}
        assert active_ids == set(video_ids)


# =============================================================================
# ANALYSIS CANCELLATION FLOW
# =============================================================================

class TestAnalysisCancellationFlow:
    """Tests for the complete analysis cancellation flow."""

    async def test_cancel_resets_video_to_pending(self, test_db):
        """Cancelling analysis should reset video status to pending."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzing_faces")

        # Verify it appears in active tasks
        active_before = await service.get_active_tasks()
        assert len(active_before) == 1

        # Cancel the task
        result = await service.cancel_task('analysis', video_id)
        assert result is True

        # Verify it no longer appears in active tasks
        active_after = await service.get_active_tasks()
        assert len(active_after) == 0

        # Verify video status is now pending
        async with test_db.execute(
            "SELECT status, error_message FROM videos WHERE id = ?",
            (video_id,)
        ) as cursor:
            row = await cursor.fetchone()
            assert row[0] == 'pending'
            assert row[1] == 'Cancelled by user'

    async def test_cancel_all_analysis_states(self, test_db):
        """Should be able to cancel from any analysis state."""
        service = TaskService(test_db)

        states = ['analyzing', 'analyzing_scenes', 'analyzing_faces', 'clustering', 'transcribing']

        for status in states:
            video_id = await insert_video(test_db, f"{status}.mp4", status)

            result = await service.cancel_task('analysis', video_id)

            assert result is True, f"Failed to cancel from state: {status}"

            # Verify reset to pending
            async with test_db.execute(
                "SELECT status FROM videos WHERE id = ?",
                (video_id,)
            ) as cursor:
                row = await cursor.fetchone()
                assert row[0] == 'pending', f"Video not reset from state: {status}"

    async def test_cannot_cancel_pending_video(self, test_db):
        """Cannot cancel a video that is not being analyzed."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "pending.mp4", "pending")

        result = await service.cancel_task('analysis', video_id)

        assert result is False

    async def test_cannot_cancel_analyzed_video(self, test_db):
        """Cannot cancel a video that is already analyzed."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "done.mp4", "analyzed")

        result = await service.cancel_task('analysis', video_id)

        assert result is False


# =============================================================================
# GET TASK FOR VIDEO (ANALYSIS)
# =============================================================================

class TestGetTaskForVideoAnalysis:
    """Tests for retrieving analysis task for a specific video."""

    async def test_returns_analysis_task_when_analyzing(self, test_db):
        """Should return analysis task when video is being analyzed."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzing_faces")

        task = await service.get_task_for_video(video_id)

        assert task is not None
        assert task['type'] == 'analysis'
        assert task['video_id'] == video_id
        assert task['status'] == 'analyzing_faces'
        assert task['progress'] == 50  # analyzing_faces = 50%

    async def test_returns_none_when_pending(self, test_db):
        """Should return None when video is pending."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "pending")

        task = await service.get_task_for_video(video_id)

        assert task is None

    async def test_returns_none_when_analyzed(self, test_db):
        """Should return None when video is already analyzed."""
        service = TaskService(test_db)
        video_id = await insert_video(test_db, "test.mp4", "analyzed")

        task = await service.get_task_for_video(video_id)

        assert task is None


# =============================================================================
# ANALYSIS PROGRESS ESTIMATION
# =============================================================================

class TestAnalysisProgressEstimation:
    """Tests for analysis progress values based on status."""

    async def test_progress_increases_through_states(self, test_db):
        """Progress should increase as analysis moves through states."""
        service = TaskService(test_db)

        # States in order with expected progress
        states_progress = [
            ('analyzing', 5),
            ('analyzing_scenes', 20),
            ('analyzing_faces', 50),
            ('clustering', 75),
            ('transcribing', 90),
        ]

        for status, expected_progress in states_progress:
            video_id = await insert_video(test_db, f"{status}.mp4", status)
            task = await service.get_task_for_video(video_id)

            assert task['progress'] == expected_progress, \
                f"Expected {expected_progress}% for {status}, got {task['progress']}%"


# =============================================================================
# MULTIPLE VIDEOS ANALYSIS
# =============================================================================

class TestMultipleVideosAnalysis:
    """Tests for handling multiple videos being analyzed simultaneously."""

    async def test_multiple_videos_all_appear_in_active(self, test_db):
        """All videos being analyzed should appear in active tasks."""
        service = TaskService(test_db)

        # Create 3 videos in different analysis states
        v1 = await insert_video(test_db, "video1.mp4", "analyzing_scenes")
        v2 = await insert_video(test_db, "video2.mp4", "analyzing_faces")
        v3 = await insert_video(test_db, "video3.mp4", "transcribing")

        active_tasks = await service.get_active_tasks()

        assert len(active_tasks) == 3
        active_ids = {t['id'] for t in active_tasks}
        assert active_ids == {v1, v2, v3}

    async def test_cancel_one_video_keeps_others_active(self, test_db):
        """Cancelling one video should not affect other active analyses."""
        service = TaskService(test_db)

        v1 = await insert_video(test_db, "video1.mp4", "analyzing_scenes")
        v2 = await insert_video(test_db, "video2.mp4", "analyzing_faces")

        # Cancel first video
        await service.cancel_task('analysis', v1)

        # Second video should still be active
        active_tasks = await service.get_active_tasks()
        assert len(active_tasks) == 1
        assert active_tasks[0]['id'] == v2

    async def test_mixed_states_only_active_appear(self, test_db):
        """Only videos in active states should appear, not pending/completed."""
        service = TaskService(test_db)

        await insert_video(test_db, "pending.mp4", "pending")
        v_active = await insert_video(test_db, "active.mp4", "analyzing_faces")
        await insert_video(test_db, "done.mp4", "analyzed")
        await insert_video(test_db, "completed.mp4", "completed")

        active_tasks = await service.get_active_tasks()

        assert len(active_tasks) == 1
        assert active_tasks[0]['id'] == v_active

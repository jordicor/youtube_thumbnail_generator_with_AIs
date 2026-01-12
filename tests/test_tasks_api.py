"""
Unit Tests for Tasks API Endpoints

Tests for the REST API endpoints in api/routes/tasks.py
Uses mocking to isolate the HTTP layer from the service layer.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from contextlib import asynccontextmanager


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_task_service():
    """Create a mock TaskService."""
    service = MagicMock()
    service.get_active_tasks = AsyncMock(return_value=[])
    service.get_pending_tasks = AsyncMock(return_value=[])
    service.get_task_for_video = AsyncMock(return_value=None)
    service.cancel_task = AsyncMock(return_value=True)
    return service


@pytest.fixture
async def test_client(mock_task_service):
    """Create test client with mocked dependencies."""
    # Mock database context manager
    @asynccontextmanager
    async def mock_get_db():
        yield MagicMock()

    with patch('api.routes.tasks.get_db', mock_get_db):
        with patch('api.routes.tasks.TaskService', return_value=mock_task_service):
            # Import app after patching
            from api.main import app

            # Mock Redis and DB init for app startup
            with patch('database.db.init_db', new_callable=AsyncMock):
                with patch('job_queue.client.RedisManager.health_check', new_callable=AsyncMock) as mock_redis:
                    mock_redis.return_value = False

                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                        yield client, mock_task_service


# =============================================================================
# GET /api/tasks/active TESTS
# =============================================================================

class TestGetActiveTasksEndpoint:
    """Tests for GET /api/tasks/active"""

    async def test_returns_empty_tasks_list(self, test_client):
        """Should return empty list when no active tasks."""
        client, mock_service = test_client
        mock_service.get_active_tasks.return_value = []

        response = await client.get("/api/tasks/active")

        assert response.status_code == 200
        data = response.json()
        assert data["tasks"] == []
        assert data["count"] == 0

    async def test_returns_active_tasks(self, test_client):
        """Should return list of active tasks."""
        client, mock_service = test_client
        mock_service.get_active_tasks.return_value = [
            {
                "id": 1,
                "type": "analysis",
                "video_id": 1,
                "video_name": "test.mp4",
                "status": "analyzing_faces",
                "progress": 50
            },
            {
                "id": 2,
                "type": "generation",
                "video_id": 2,
                "video_name": "video2.mp4",
                "status": "generating",
                "progress": 30
            }
        ]

        response = await client.get("/api/tasks/active")

        assert response.status_code == 200
        data = response.json()
        assert len(data["tasks"]) == 2
        assert data["count"] == 2
        assert data["tasks"][0]["type"] == "analysis"
        assert data["tasks"][1]["type"] == "generation"

    async def test_handles_service_error(self, test_client):
        """Should return 503 when service fails."""
        client, mock_service = test_client
        mock_service.get_active_tasks.side_effect = Exception("DB error")

        response = await client.get("/api/tasks/active")

        assert response.status_code == 503


# =============================================================================
# GET /api/tasks/pending TESTS
# =============================================================================

class TestGetPendingTasksEndpoint:
    """Tests for GET /api/tasks/pending"""

    async def test_returns_empty_pending_list(self, test_client):
        """Should return empty list when no pending tasks."""
        client, mock_service = test_client
        mock_service.get_pending_tasks.return_value = []

        response = await client.get("/api/tasks/pending")

        assert response.status_code == 200
        data = response.json()
        assert data["tasks"] == []
        assert data["count"] == 0

    async def test_returns_pending_tasks(self, test_client):
        """Should return list of pending tasks."""
        client, mock_service = test_client
        mock_service.get_pending_tasks.return_value = [
            {
                "id": 5,
                "type": "generation",
                "video_id": 3,
                "video_name": "pending.mp4",
                "status": "pending",
                "total_thumbnails": 5
            }
        ]

        response = await client.get("/api/tasks/pending")

        assert response.status_code == 200
        data = response.json()
        assert len(data["tasks"]) == 1
        assert data["count"] == 1
        assert data["tasks"][0]["status"] == "pending"

    async def test_handles_service_error(self, test_client):
        """Should return 503 when service fails."""
        client, mock_service = test_client
        mock_service.get_pending_tasks.side_effect = Exception("DB error")

        response = await client.get("/api/tasks/pending")

        assert response.status_code == 503


# =============================================================================
# GET /api/tasks/video/{video_id} TESTS
# =============================================================================

class TestGetVideoTaskEndpoint:
    """Tests for GET /api/tasks/video/{video_id}"""

    async def test_returns_null_when_no_task(self, test_client):
        """Should return null task when video has no active task."""
        client, mock_service = test_client
        mock_service.get_task_for_video.return_value = None

        response = await client.get("/api/tasks/video/123")

        assert response.status_code == 200
        data = response.json()
        assert data["task"] is None

    async def test_returns_task_for_video(self, test_client):
        """Should return the active task for a video."""
        client, mock_service = test_client
        mock_service.get_task_for_video.return_value = {
            "id": 1,
            "type": "analysis",
            "video_id": 123,
            "video_name": "test.mp4",
            "status": "analyzing",
            "progress": 20
        }

        response = await client.get("/api/tasks/video/123")

        assert response.status_code == 200
        data = response.json()
        assert data["task"] is not None
        assert data["task"]["video_id"] == 123
        assert data["task"]["type"] == "analysis"

    async def test_handles_service_error(self, test_client):
        """Should return 503 when service fails."""
        client, mock_service = test_client
        mock_service.get_task_for_video.side_effect = Exception("DB error")

        response = await client.get("/api/tasks/video/123")

        assert response.status_code == 503


# =============================================================================
# POST /api/tasks/{task_type}/{task_id}/cancel TESTS
# =============================================================================

class TestCancelTaskEndpoint:
    """Tests for POST /api/tasks/{task_type}/{task_id}/cancel"""

    async def test_cancel_analysis_success(self, test_client):
        """Should cancel analysis task successfully."""
        client, mock_service = test_client
        mock_service.cancel_task.return_value = True

        response = await client.post("/api/tasks/analysis/1/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["task_type"] == "analysis"
        assert data["task_id"] == 1
        assert data["status"] == "cancelled"

    async def test_cancel_generation_success(self, test_client):
        """Should cancel generation task successfully."""
        client, mock_service = test_client
        mock_service.cancel_task.return_value = True

        response = await client.post("/api/tasks/generation/5/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["task_type"] == "generation"
        assert data["task_id"] == 5
        assert data["status"] == "cancelled"

    async def test_cancel_nonexistent_task_returns_400(self, test_client):
        """Should return 400 when task cannot be cancelled."""
        client, mock_service = test_client
        mock_service.cancel_task.return_value = False

        response = await client.post("/api/tasks/analysis/999/cancel")

        assert response.status_code == 400

    async def test_cancel_invalid_task_type_returns_422(self, test_client):
        """Should return 422 for invalid task type."""
        client, _ = test_client

        response = await client.post("/api/tasks/invalid_type/1/cancel")

        assert response.status_code == 422

    async def test_handles_service_error(self, test_client):
        """Should return 503 when service fails."""
        client, mock_service = test_client
        mock_service.cancel_task.side_effect = Exception("DB error")

        response = await client.post("/api/tasks/analysis/1/cancel")

        assert response.status_code == 503


# =============================================================================
# RESPONSE FORMAT TESTS
# =============================================================================

class TestResponseFormats:
    """Tests to verify API response formats match expected structure."""

    async def test_active_tasks_response_format(self, test_client):
        """Verify active tasks response has correct structure."""
        client, mock_service = test_client
        mock_service.get_active_tasks.return_value = [
            {
                "id": 1,
                "type": "generation",
                "video_id": 1,
                "video_name": "test.mp4",
                "status": "generating",
                "progress": 60,
                "current_step": "generating",
                "started_at": "2024-01-01T10:00:00",
                "thumbnails_generated": 3,
                "total_thumbnails": 5
            }
        ]

        response = await client.get("/api/tasks/active")

        assert response.status_code == 200
        data = response.json()

        # Verify structure
        assert "tasks" in data
        assert "count" in data
        assert isinstance(data["tasks"], list)
        assert isinstance(data["count"], int)

        # Verify task fields
        task = data["tasks"][0]
        assert "id" in task
        assert "type" in task
        assert "video_id" in task
        assert "video_name" in task
        assert "status" in task
        assert "progress" in task

    async def test_cancel_response_format(self, test_client):
        """Verify cancel response has correct structure."""
        client, mock_service = test_client
        mock_service.cancel_task.return_value = True

        response = await client.post("/api/tasks/generation/1/cancel")

        assert response.status_code == 200
        data = response.json()

        assert "task_type" in data
        assert "task_id" in data
        assert "status" in data
        assert data["status"] == "cancelled"

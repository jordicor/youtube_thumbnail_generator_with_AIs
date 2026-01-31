"""
Tests for SSE Task Queue Events

Tests for the Server-Sent Events endpoint /api/events/tasks
"""

import pytest
import orjson
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from contextlib import asynccontextmanager


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_task_service():
    """Create a mock TaskService with configurable responses."""
    service = MagicMock()
    service.get_active_tasks = AsyncMock(return_value=[])
    service.get_pending_tasks = AsyncMock(return_value=[])
    return service


@pytest.fixture
async def sse_client(mock_task_service):
    """Create a test client configured for SSE testing."""
    @asynccontextmanager
    async def mock_get_db():
        yield MagicMock()

    with patch('api.routes.events.get_db', mock_get_db):
        with patch('api.routes.events.TaskService', return_value=mock_task_service):
            with patch('database.db.init_db', new_callable=AsyncMock):
                with patch('job_queue.client.RedisManager.health_check', new_callable=AsyncMock) as mock_redis:
                    mock_redis.return_value = False

                    from api.main import app
                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                        yield client, mock_task_service


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_sse_events(content: str) -> list:
    """Parse SSE content into list of (event_type, data) tuples."""
    events = []
    current_event = "message"
    current_data = []

    for line in content.split('\n'):
        if line.startswith('event:'):
            current_event = line[6:].strip()
        elif line.startswith('data:'):
            current_data.append(line[5:].strip())
        elif line == '' and current_data:
            # End of event
            data_str = '\n'.join(current_data)
            try:
                data = orjson.loads(data_str)
            except orjson.JSONDecodeError:
                data = data_str
            events.append((current_event, data))
            current_data = []
            current_event = "message"
        elif line.startswith(':'):
            # Comment/keepalive, ignore
            pass

    return events


async def read_sse_with_timeout(response, timeout: float = 2.0) -> str:
    """Read SSE response with timeout."""
    content = b""
    try:
        async with asyncio.timeout(timeout):
            async for chunk in response.aiter_bytes():
                content += chunk
    except asyncio.TimeoutError:
        pass
    return content.decode('utf-8')


# =============================================================================
# CONNECTION TESTS
# =============================================================================

class TestSSEConnection:
    """Tests for SSE endpoint connection behavior."""

    async def test_returns_event_stream_content_type(self, sse_client):
        """SSE endpoint should return text/event-stream content type."""
        client, _ = sse_client

        async with client.stream("GET", "/api/events/tasks") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

    async def test_returns_correct_headers(self, sse_client):
        """SSE endpoint should return proper caching headers."""
        client, _ = sse_client

        async with client.stream("GET", "/api/events/tasks") as response:
            assert response.headers.get("cache-control") == "no-cache"


# =============================================================================
# SNAPSHOT TESTS
# =============================================================================

class TestTasksSnapshot:
    """Tests for initial tasks_snapshot event."""

    async def test_sends_snapshot_on_connect(self, sse_client):
        """Should send tasks_snapshot event immediately on connect."""
        client, mock_service = sse_client
        mock_service.get_active_tasks.return_value = []
        mock_service.get_pending_tasks.return_value = []

        async with client.stream("GET", "/api/events/tasks") as response:
            content = await read_sse_with_timeout(response, timeout=1.0)

        events = parse_sse_events(content)

        # First event should be tasks_snapshot
        assert len(events) >= 1
        event_type, data = events[0]
        assert event_type == "tasks_snapshot"

    async def test_snapshot_contains_empty_tasks_list(self, sse_client):
        """Snapshot should contain empty tasks list when no tasks exist."""
        client, mock_service = sse_client
        mock_service.get_active_tasks.return_value = []
        mock_service.get_pending_tasks.return_value = []

        async with client.stream("GET", "/api/events/tasks") as response:
            content = await read_sse_with_timeout(response, timeout=1.0)

        events = parse_sse_events(content)
        _, data = events[0]

        assert data["tasks"] == []
        assert data["active_count"] == 0
        assert data["pending_count"] == 0

    async def test_snapshot_contains_active_tasks(self, sse_client):
        """Snapshot should include active tasks."""
        client, mock_service = sse_client
        mock_service.get_active_tasks.return_value = [
            {
                "id": 1,
                "type": "analysis",
                "video_id": 1,
                "video_name": "test.mp4",
                "status": "analyzing_faces",
                "progress": 50
            }
        ]
        mock_service.get_pending_tasks.return_value = []

        async with client.stream("GET", "/api/events/tasks") as response:
            content = await read_sse_with_timeout(response, timeout=1.0)

        events = parse_sse_events(content)
        _, data = events[0]

        assert len(data["tasks"]) == 1
        assert data["active_count"] == 1
        assert data["pending_count"] == 0
        assert data["tasks"][0]["type"] == "analysis"

    async def test_snapshot_contains_pending_tasks(self, sse_client):
        """Snapshot should include pending tasks."""
        client, mock_service = sse_client
        mock_service.get_active_tasks.return_value = []
        mock_service.get_pending_tasks.return_value = [
            {
                "id": 5,
                "type": "generation",
                "video_id": 2,
                "video_name": "video2.mp4",
                "status": "pending",
                "total_thumbnails": 5
            }
        ]

        async with client.stream("GET", "/api/events/tasks") as response:
            content = await read_sse_with_timeout(response, timeout=1.0)

        events = parse_sse_events(content)
        _, data = events[0]

        assert len(data["tasks"]) == 1
        assert data["active_count"] == 0
        assert data["pending_count"] == 1
        assert data["tasks"][0]["status"] == "pending"

    async def test_snapshot_contains_both_active_and_pending(self, sse_client):
        """Snapshot should include both active and pending tasks."""
        client, mock_service = sse_client
        mock_service.get_active_tasks.return_value = [
            {
                "id": 1,
                "type": "analysis",
                "video_id": 1,
                "video_name": "analyzing.mp4",
                "status": "analyzing",
                "progress": 10
            }
        ]
        mock_service.get_pending_tasks.return_value = [
            {
                "id": 10,
                "type": "generation",
                "video_id": 3,
                "video_name": "pending.mp4",
                "status": "pending",
                "total_thumbnails": 3
            }
        ]

        async with client.stream("GET", "/api/events/tasks") as response:
            content = await read_sse_with_timeout(response, timeout=1.0)

        events = parse_sse_events(content)
        _, data = events[0]

        assert len(data["tasks"]) == 2
        assert data["active_count"] == 1
        assert data["pending_count"] == 1


# =============================================================================
# SNAPSHOT FORMAT TESTS
# =============================================================================

class TestSnapshotFormat:
    """Tests for tasks_snapshot event data structure."""

    async def test_snapshot_has_required_fields(self, sse_client):
        """Snapshot event should have tasks, active_count, pending_count."""
        client, mock_service = sse_client
        mock_service.get_active_tasks.return_value = []
        mock_service.get_pending_tasks.return_value = []

        async with client.stream("GET", "/api/events/tasks") as response:
            content = await read_sse_with_timeout(response, timeout=1.0)

        events = parse_sse_events(content)
        _, data = events[0]

        assert "tasks" in data
        assert "active_count" in data
        assert "pending_count" in data
        assert isinstance(data["tasks"], list)
        assert isinstance(data["active_count"], int)
        assert isinstance(data["pending_count"], int)

    async def test_task_object_has_required_fields(self, sse_client):
        """Each task in snapshot should have required fields."""
        client, mock_service = sse_client
        mock_service.get_active_tasks.return_value = [
            {
                "id": 1,
                "type": "generation",
                "video_id": 1,
                "video_name": "test.mp4",
                "status": "generating",
                "progress": 60,
                "current_step": "generating",
                "thumbnails_generated": 3,
                "total_thumbnails": 5
            }
        ]
        mock_service.get_pending_tasks.return_value = []

        async with client.stream("GET", "/api/events/tasks") as response:
            content = await read_sse_with_timeout(response, timeout=1.0)

        events = parse_sse_events(content)
        _, data = events[0]

        task = data["tasks"][0]
        assert "id" in task
        assert "type" in task
        assert "video_id" in task
        assert "video_name" in task
        assert "status" in task


# =============================================================================
# EVENT TYPE TESTS
# =============================================================================

class TestEventTypes:
    """Tests for SSE event types."""

    async def test_snapshot_event_type(self, sse_client):
        """First event should be tasks_snapshot type."""
        client, mock_service = sse_client
        mock_service.get_active_tasks.return_value = []
        mock_service.get_pending_tasks.return_value = []

        async with client.stream("GET", "/api/events/tasks") as response:
            content = await read_sse_with_timeout(response, timeout=1.0)

        events = parse_sse_events(content)

        assert len(events) >= 1
        assert events[0][0] == "tasks_snapshot"


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestSSEErrorHandling:
    """Tests for SSE error handling."""

    async def test_handles_service_error_gracefully(self, sse_client):
        """Should handle service errors without crashing."""
        client, mock_service = sse_client
        mock_service.get_active_tasks.side_effect = Exception("DB error")

        async with client.stream("GET", "/api/events/tasks") as response:
            # Should still connect
            assert response.status_code == 200
            content = await read_sse_with_timeout(response, timeout=1.0)

        # Should receive error event
        events = parse_sse_events(content)
        assert len(events) >= 1
        # First event will be error due to exception
        event_type, data = events[0]
        assert event_type == "error" or "error" in str(data)

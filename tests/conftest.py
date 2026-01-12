"""
Test Configuration and Fixtures

Shared fixtures for all tests in the project.
"""

import pytest
import aiosqlite
from pathlib import Path
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport

# Project paths
ROOT_DIR = Path(__file__).parent.parent
SCHEMA_PATH = ROOT_DIR / "database" / "schema.sql"


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture
async def test_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """
    Create an in-memory SQLite database with the full schema.

    Yields:
        aiosqlite.Connection: Database connection ready for use
    """
    # Create in-memory database
    db = await aiosqlite.connect(":memory:")

    # Enable foreign keys
    await db.execute("PRAGMA foreign_keys = ON")

    # Load and execute schema
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        await db.executescript(schema_sql)

    yield db

    await db.close()


@pytest.fixture
async def sample_video(test_db: aiosqlite.Connection) -> dict:
    """
    Create a sample video in the test database.

    Returns:
        dict: Video record with id, filename, filepath, status
    """
    cursor = await test_db.execute(
        """
        INSERT INTO videos (filename, filepath, status)
        VALUES (?, ?, ?)
        """,
        ("test_video.mp4", "/tmp/test_video.mp4", "pending")
    )
    await test_db.commit()
    video_id = cursor.lastrowid

    return {
        "id": video_id,
        "filename": "test_video.mp4",
        "filepath": "/tmp/test_video.mp4",
        "status": "pending"
    }


@pytest.fixture
async def analyzing_video(test_db: aiosqlite.Connection) -> dict:
    """
    Create a video in 'analyzing' state.

    Returns:
        dict: Video record in analyzing state
    """
    cursor = await test_db.execute(
        """
        INSERT INTO videos (filename, filepath, status)
        VALUES (?, ?, ?)
        """,
        ("analyzing_video.mp4", "/tmp/analyzing_video.mp4", "analyzing_faces")
    )
    await test_db.commit()
    video_id = cursor.lastrowid

    return {
        "id": video_id,
        "filename": "analyzing_video.mp4",
        "filepath": "/tmp/analyzing_video.mp4",
        "status": "analyzing_faces"
    }


@pytest.fixture
async def analyzed_video(test_db: aiosqlite.Connection) -> dict:
    """
    Create a video that has been analyzed (ready for generation).

    Returns:
        dict: Video record with analyzed status
    """
    cursor = await test_db.execute(
        """
        INSERT INTO videos (filename, filepath, status)
        VALUES (?, ?, ?)
        """,
        ("analyzed_video.mp4", "/tmp/analyzed_video.mp4", "analyzed")
    )
    await test_db.commit()
    video_id = cursor.lastrowid

    return {
        "id": video_id,
        "filename": "analyzed_video.mp4",
        "filepath": "/tmp/analyzed_video.mp4",
        "status": "analyzed"
    }


@pytest.fixture
async def sample_cluster(test_db: aiosqlite.Connection, analyzed_video: dict) -> dict:
    """
    Create a sample cluster for a video.

    Returns:
        dict: Cluster record
    """
    cursor = await test_db.execute(
        """
        INSERT INTO clusters (video_id, cluster_index, num_frames, representative_frame, cluster_type, view_mode)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (analyzed_video["id"], 0, 10, "/tmp/frame_001.jpg", "person", "person")
    )
    await test_db.commit()
    cluster_id = cursor.lastrowid

    return {
        "id": cluster_id,
        "video_id": analyzed_video["id"],
        "cluster_index": 0,
        "num_frames": 10,
        "representative_frame": "/tmp/frame_001.jpg"
    }


@pytest.fixture
async def sample_generation_job(test_db: aiosqlite.Connection, analyzed_video: dict, sample_cluster: dict) -> dict:
    """
    Create a sample generation job in 'pending' state.

    Returns:
        dict: Generation job record
    """
    cursor = await test_db.execute(
        """
        INSERT INTO generation_jobs (video_id, cluster_id, num_images, status, progress)
        VALUES (?, ?, ?, ?, ?)
        """,
        (analyzed_video["id"], sample_cluster["id"], 5, "pending", 0)
    )
    await test_db.commit()
    job_id = cursor.lastrowid

    return {
        "id": job_id,
        "video_id": analyzed_video["id"],
        "cluster_id": sample_cluster["id"],
        "num_images": 5,
        "status": "pending",
        "progress": 0
    }


@pytest.fixture
async def generating_job(test_db: aiosqlite.Connection, analyzed_video: dict, sample_cluster: dict) -> dict:
    """
    Create a generation job in 'generating' state.

    Returns:
        dict: Generation job record with generating status
    """
    cursor = await test_db.execute(
        """
        INSERT INTO generation_jobs (video_id, cluster_id, num_images, status, progress)
        VALUES (?, ?, ?, ?, ?)
        """,
        (analyzed_video["id"], sample_cluster["id"], 5, "generating", 40)
    )
    await test_db.commit()
    job_id = cursor.lastrowid

    return {
        "id": job_id,
        "video_id": analyzed_video["id"],
        "cluster_id": sample_cluster["id"],
        "num_images": 5,
        "status": "generating",
        "progress": 40
    }


# =============================================================================
# APP AND CLIENT FIXTURES
# =============================================================================

@pytest.fixture
async def app(test_db: aiosqlite.Connection):
    """
    Create FastAPI app with test database.

    This fixture patches the database module to use the test database.
    """
    import sys
    from unittest.mock import patch, MagicMock, AsyncMock
    from contextlib import asynccontextmanager

    # Create a mock for get_db that returns our test_db
    @asynccontextmanager
    async def mock_get_db():
        yield test_db

    # Patch the database module
    with patch('database.db.get_db', mock_get_db):
        with patch('database.db._db_connection', test_db):
            # Import the app after patching
            from api.main import app as fastapi_app

            # Skip Redis check in tests
            with patch('job_queue.client.RedisManager.health_check', new_callable=AsyncMock) as mock_health:
                mock_health.return_value = False  # Redis not available in tests
                yield fastapi_app


@pytest.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """
    Create async HTTP client for testing API endpoints.

    Yields:
        AsyncClient: httpx client configured for the test app
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# =============================================================================
# TASK SERVICE FIXTURES
# =============================================================================

@pytest.fixture
async def task_service(test_db: aiosqlite.Connection):
    """
    Create TaskService instance with test database.

    Returns:
        TaskService: Service instance ready for testing
    """
    from unittest.mock import patch
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_get_db():
        yield test_db

    with patch('database.db.get_db', mock_get_db):
        from services.task_service import TaskService
        service = TaskService(test_db)
        yield service


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def insert_video(db: aiosqlite.Connection, filename: str, status: str) -> int:
    """
    Helper to insert a video with given status.

    Args:
        db: Database connection
        filename: Video filename
        status: Video status

    Returns:
        int: Video ID
    """
    cursor = await db.execute(
        """
        INSERT INTO videos (filename, filepath, status)
        VALUES (?, ?, ?)
        """,
        (filename, f"/tmp/{filename}", status)
    )
    await db.commit()
    return cursor.lastrowid


async def insert_job(
    db: aiosqlite.Connection,
    video_id: int,
    cluster_id: int,
    status: str,
    progress: int = 0
) -> int:
    """
    Helper to insert a generation job.

    Args:
        db: Database connection
        video_id: Video ID
        cluster_id: Cluster ID
        status: Job status
        progress: Job progress (0-100)

    Returns:
        int: Job ID
    """
    cursor = await db.execute(
        """
        INSERT INTO generation_jobs (video_id, cluster_id, num_images, status, progress)
        VALUES (?, ?, ?, ?, ?)
        """,
        (video_id, cluster_id, 5, status, progress)
    )
    await db.commit()
    return cursor.lastrowid


async def insert_cluster(db: aiosqlite.Connection, video_id: int, cluster_index: int = 0) -> int:
    """
    Helper to insert a cluster for a video.

    Args:
        db: Database connection
        video_id: Video ID
        cluster_index: Cluster index

    Returns:
        int: Cluster ID
    """
    cursor = await db.execute(
        """
        INSERT INTO clusters (video_id, cluster_index, num_frames, representative_frame, cluster_type, view_mode)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (video_id, cluster_index, 10, f"/tmp/frame_{cluster_index}.jpg", "person", "person")
    )
    await db.commit()
    return cursor.lastrowid

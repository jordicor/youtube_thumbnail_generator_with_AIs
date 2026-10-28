"""Downloading a job's or a video's thumbnails as a ZIP."""

import io
import zipfile
from pathlib import Path

import pytest

import config
from tests.conftest import insert_video


async def _insert_job(db, video_id: int) -> int:
    cursor = await db.execute(
        "INSERT INTO generation_jobs (video_id, status) VALUES (?, 'completed')", (video_id,)
    )
    return cursor.lastrowid


async def _insert_thumbnail(db, job_id: int, index: int, title, name: str, content) -> None:
    """A thumbnail of the job; content None leaves its file missing."""
    stored = f"zip_video/thumbnails/{name}"
    if content is not None:
        path = Path(config.OUTPUT_DIR) / stored
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    await db.execute(
        "INSERT INTO thumbnails (job_id, image_index, filepath, suggested_title) VALUES (?, ?, ?, ?)",
        (job_id, index, stored, title),
    )


@pytest.fixture
async def thumbnails(test_db):
    video_id = await insert_video(test_db, "zip_video.mp4", "completed")
    job_id = await _insert_job(test_db, video_id)
    await _insert_thumbnail(test_db, job_id, 0, "Mi título/raro", "a.png", b"png-a")
    await _insert_thumbnail(test_db, job_id, 1, None, "b.jpg", b"jpg-b")
    await _insert_thumbnail(test_db, job_id, 2, "Borrada", "c.png", None)
    await test_db.commit()
    return video_id, job_id


def _entries(response) -> dict:
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        # Already compressed images are stored as they are
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}
        return {name: archive.read(name) for name in archive.namelist()}


async def test_job_thumbnails_download_as_zip(client, thumbnails):
    _, job_id = thumbnails

    entries = _entries(await client.get(f"/api/thumbnails/job/{job_id}/download-all"))

    # Thumbnails whose file is gone are left out
    assert entries == {"01_Mi título_raro.png": b"png-a", "02_thumbnail_1.jpg": b"jpg-b"}


async def test_video_thumbnails_download_as_zip_by_job(client, thumbnails):
    video_id, job_id = thumbnails

    entries = _entries(await client.get(f"/api/thumbnails/video/{video_id}/download-all"))

    assert entries == {f"job{job_id}/01_Mi título_raro.png": b"png-a", f"job{job_id}/02_thumbnail_1.jpg": b"jpg-b"}

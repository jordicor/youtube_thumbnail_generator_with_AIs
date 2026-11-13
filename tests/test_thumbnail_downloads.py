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
    # Thumbnails are numbered from 1, and titles carry what Windows forbids in a name
    await _insert_thumbnail(test_db, job_id, 1, "Mi título/raro", "a.png", b"png-a")
    await _insert_thumbnail(test_db, job_id, 2, None, "b.jpg", b"jpg-b")
    await _insert_thumbnail(test_db, job_id, 3, "Borrada", "c.png", None)
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
    assert entries == {"01_Mi título_raro.png": b"png-a", "02_thumbnail_2.jpg": b"jpg-b"}


async def test_video_thumbnails_download_as_zip_by_job(client, thumbnails):
    video_id, job_id = thumbnails

    entries = _entries(await client.get(f"/api/thumbnails/video/{video_id}/download-all"))

    assert entries == {f"job{job_id}/01_Mi título_raro.png": b"png-a", f"job{job_id}/02_thumbnail_2.jpg": b"jpg-b"}


def test_a_title_becomes_a_file_name_windows_accepts():
    from api.routes.thumbnails import _zip_title

    assert _zip_title('¿Qué: "es" <esto>? A|B*', 4) == '¿Qué_ _es_ _esto__ A_B_'
    assert _zip_title('Termina en puntos... ', 4) == 'Termina en puntos'
    assert _zip_title('Con\ttabulador', 4) == 'Con_tabulador'
    assert _zip_title('...', 4) == 'thumbnail_4'
    assert _zip_title(None, 4) == 'thumbnail_4'


async def test_a_thumbnail_is_served_with_the_type_of_its_file(client, thumbnails, test_db):
    async with test_db.execute("SELECT id, filepath FROM thumbnails ORDER BY image_index") as cursor:
        (png_id, _), (jpg_id, _), _ = await cursor.fetchall()

    assert (await client.get(f"/api/thumbnails/{png_id}")).headers["content-type"] == "image/png"
    assert (await client.get(f"/api/thumbnails/{jpg_id}")).headers["content-type"] == "image/jpeg"

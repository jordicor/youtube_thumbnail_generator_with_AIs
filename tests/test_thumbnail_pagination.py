"""Thumbnail pagination against synthetic records, including timestamp ties."""

import pytest

from tests.conftest import insert_cluster, insert_job, insert_video


@pytest.fixture
async def thumbnail_pages(test_db, sample_generation_job):
    ids = []
    for index, timestamp in enumerate([
        "2026-01-01 00:00:00", "2026-01-02 00:00:00", "2026-01-02 00:00:00"
    ]):
        cursor = await test_db.execute(
            "INSERT INTO thumbnails (job_id, image_index, filepath, created_at) VALUES (?, ?, ?, ?)",
            (sample_generation_job["id"], index, f"/tmp/synthetic_{index}.png", timestamp),
        )
        ids.append(cursor.lastrowid)

    other_video = await insert_video(test_db, "other.mp4", "completed")
    other_cluster = await insert_cluster(test_db, other_video)
    other_job = await insert_job(test_db, other_video, other_cluster, "completed")
    await test_db.execute(
        "INSERT INTO thumbnails (job_id, image_index, filepath, created_at) VALUES (?, ?, ?, ?)",
        (other_job, 0, "/tmp/other.png", "2026-01-03 00:00:00"),
    )
    await test_db.commit()
    return sample_generation_job["video_id"], list(reversed(ids))


async def test_pages_are_distinct_and_stably_ordered(client, thumbnail_pages):
    video_id, expected_ids = thumbnail_pages
    pages = []
    for skip in range(4):
        response = await client.get(f"/api/thumbnails/video/{video_id}", params={"limit": 1, "skip": skip})
        assert response.status_code == 200
        pages.append([item["id"] for item in response.json()["thumbnails"]])
    assert pages == [[item] for item in expected_ids] + [[]]


async def test_page_size_and_default_offset(client, thumbnail_pages):
    video_id, expected_ids = thumbnail_pages
    response = await client.get(f"/api/thumbnails/video/{video_id}", params={"limit": 2})
    assert response.status_code == 200
    assert [item["id"] for item in response.json()["thumbnails"]] == expected_ids[:2]


@pytest.mark.parametrize("params", [{"skip": -1}, {"limit": 0}, {"limit": 501}])
async def test_rejects_invalid_pagination(client, params):
    response = await client.get("/api/thumbnails/video/1", params=params)
    assert response.status_code == 422

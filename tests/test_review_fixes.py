"""Small fixes from the 2026-11 review: failures that answered as successes or
as server errors, and work that blocked the server. Synthetic records only."""

import sqlite3
import threading
from pathlib import Path

import pytest

import config
from i18n.i18n import translate as t
from tests.conftest import insert_cluster, insert_job, insert_video, provision_uploaded_face


async def test_the_face_of_a_person_whose_photo_is_gone_is_not_found(client, test_db):
    """The primary photo's image missing on disk used to answer 500."""
    video_id = await insert_video(test_db, "faces.mp4", "analyzed")
    person_id = await insert_cluster(test_db, video_id)
    await provision_uploaded_face(test_db, person_id)
    assert (await client.get(f"/api/people/{video_id}/{person_id}/face")).status_code == 200

    for original in (Path(config.OUTPUT_DIR) / "faces" / "references" / "originals").iterdir():
        original.unlink()

    response = await client.get(f"/api/people/{video_id}/{person_id}/face")
    assert response.status_code == 404 and response.json()["detail"]


async def test_merging_with_a_person_that_no_longer_exists_says_so(client, test_db):
    """It used to answer 200 "merged" with the people unchanged."""
    video_id = await insert_video(test_db, "people.mp4", "analyzed")
    await insert_cluster(test_db, video_id, cluster_index=0)
    await insert_cluster(test_db, video_id, cluster_index=1)

    response = await client.post(
        f"/api/analysis/{video_id}/clusters/merge",
        json={"cluster_indices": [0, 1, 7], "target_index": 0},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == t("api.errors.source_clusters_not_found")
    # Not even the people that do exist were merged
    async with test_db.execute("SELECT COUNT(*) FROM clusters WHERE video_id = ?", [video_id]) as cursor:
        assert (await cursor.fetchone())[0] == 2


async def test_a_scan_walks_and_probes_videos_off_the_event_loop(test_db, tmp_path, monkeypatch):
    """ffprobe takes up to 30 s per new video: on the loop, no other page answered meanwhile."""
    from services.directory_service import DirectoryService

    folder = tmp_path / "Videos"
    folder.mkdir()
    videos = [folder / "a.mp4", folder / "b.mp4"]
    threads = {"find": [], "probe": []}

    def find_videos(path, extensions, excluded):
        threads["find"].append(threading.get_ident())
        return videos

    def get_video_info(path):
        threads["probe"].append(threading.get_ident())
        return {"duration": 12.5}

    monkeypatch.setattr("utils.find_videos", find_videos)
    monkeypatch.setattr("utils.get_video_info", get_video_info)
    cursor = await test_db.execute("INSERT INTO directories (path) VALUES (?)", [str(folder)])
    await test_db.commit()

    result = await DirectoryService(test_db).scan_directory(cursor.lastrowid)

    assert (result["scanned"], result["new"], result["existing"]) == (2, 2, 0)
    loop_thread = threading.get_ident()
    assert len(threads["find"]) == 1 and len(threads["probe"]) == 2
    assert loop_thread not in threads["find"] + threads["probe"]
    async with test_db.execute("SELECT filename, duration_seconds FROM videos ORDER BY filename") as cursor:
        assert await cursor.fetchall() == [("a.mp4", 12.5), ("b.mp4", 12.5)]


def test_a_history_that_cannot_be_read_is_an_error_not_an_empty_history():
    """Answered as empty, "avoid repeats" stopped applying without a word."""
    from gransabio_prompt_generator import get_prompt_history_for_video

    # The isolated database of the tests does not exist
    with pytest.raises(sqlite3.Error):
        get_prompt_history_for_video(1)


async def _thumbnail_with_copies(db, job_id, name, index):
    """A thumbnail of the job with every copy the app keeps of it. Returns (id, its files)."""
    folder = Path(config.OUTPUT_DIR) / "copies" / "thumbnails"
    prefix, rest = name.split("_", 1)
    files = [
        folder / name,
        folder / "raw_swap" / name,
        folder / "raw" / f"{prefix}_raw_{rest}",
        (folder / name).with_suffix(".comfyui.json"),
    ]
    for file in files:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"synthetic")
    cursor = await db.execute(
        "INSERT INTO thumbnails (job_id, image_index, filepath, raw_path) VALUES (?, ?, ?, ?)",
        (job_id, index, f"copies/thumbnails/{name}", f"copies/thumbnails/raw_swap/{name}"),
    )
    await db.commit()
    return cursor.lastrowid, files


async def test_deleting_a_thumbnail_deletes_the_copies_kept_of_it(client, test_db):
    """The image from before the face swap and the provider's one used to stay on disk for ever."""
    video_id = await insert_video(test_db, "copies.mp4", "completed")
    job_id = await insert_job(test_db, video_id, await insert_cluster(test_db, video_id), "completed")
    deleted, its_files = await _thumbnail_with_copies(test_db, job_id, "gpt-image_2026_11_11_1.png", 1)
    kept, other_files = await _thumbnail_with_copies(test_db, job_id, "gpt-image_2026_11_11_2.png", 2)

    response = await client.delete(f"/api/thumbnails/{deleted}")

    assert response.status_code == 200, response.text
    assert not any(file.exists() for file in its_files)
    assert all(file.exists() for file in other_files)
    async with test_db.execute("SELECT id FROM thumbnails") as cursor:
        assert await cursor.fetchall() == [(kept,)]


async def test_a_thumbnail_of_a_running_job_is_not_deleted(client, test_db):
    video_id = await insert_video(test_db, "copies.mp4", "generating")
    job_id = await insert_job(test_db, video_id, await insert_cluster(test_db, video_id), "generating")
    thumbnail, files = await _thumbnail_with_copies(test_db, job_id, "gpt-image_2026_11_11_1.png", 1)

    response = await client.delete(f"/api/thumbnails/{thumbnail}")

    assert response.status_code == 409
    assert response.json()["detail"] == t("api.errors.generation_in_progress")
    assert all(file.exists() for file in files)


@pytest.mark.parametrize("newer_job", [False, True])
async def test_a_cancelled_thumbnail_is_kept_until_its_worker_stops(client, test_db, newer_job):
    from services.generation_service import GenerationService

    video_id = await insert_video(test_db, "copies.mp4", "generating")
    cluster_id = await insert_cluster(test_db, video_id)
    job_id = await insert_job(test_db, video_id, cluster_id, "generating")
    thumbnail, files = await _thumbnail_with_copies(test_db, job_id, "gpt-image_2026_11_12_1.png", 1)
    service = GenerationService(test_db)
    assert await service.cancel_job(job_id)

    response = await client.delete(f"/api/thumbnails/{thumbnail}")

    assert response.status_code == 409
    assert response.json()["detail"] == t("api.errors.generation_in_progress")
    assert all(file.exists() for file in files)
    async with test_db.execute("SELECT id FROM thumbnails WHERE id = ?", [thumbnail]) as cursor:
        assert await cursor.fetchone() == (thumbnail,)

    # The worker acknowledges the cancellation only after its last write.
    await service.update_job_status(job_id, "cancelled")
    if newer_job:
        # A later generation owns its own thumbnails, not this stopped job's.
        next_job = await service.create_generation_job(video_id, [cluster_id])
        await service.update_job_status(next_job["id"], "generating")

    response = await client.delete(f"/api/thumbnails/{thumbnail}")

    assert response.status_code == 200, response.text
    assert not any(file.exists() for file in files)
    async with test_db.execute("SELECT id FROM thumbnails WHERE id = ?", [thumbnail]) as cursor:
        assert await cursor.fetchone() is None


async def test_thumbnails_of_a_merged_person_still_know_who_is_in_them(client, test_db):
    """Merging B into A deleted B's row in the old jobs: "fix face" then found nobody to fix."""
    video_id = await insert_video(test_db, "merged.mp4", "completed")
    ana = await insert_cluster(test_db, video_id, cluster_index=0)
    ana_again = await insert_cluster(test_db, video_id, cluster_index=1)
    only_the_other = await insert_job(test_db, video_id, ana_again, "completed")
    both = await insert_job(test_db, video_id, ana, "completed")
    await test_db.execute(
        "INSERT INTO generation_job_clusters (job_id, cluster_id, person_id, cluster_order) VALUES (?, ?, 'person_02', 1)",
        [both, ana_again],
    )
    await test_db.commit()

    response = await client.post(
        f"/api/analysis/{video_id}/clusters/merge",
        json={"cluster_indices": [0, 1], "target_index": 0},
    )

    assert response.status_code == 200, response.text
    async with test_db.execute(
        "SELECT job_id, cluster_id, person_id FROM generation_job_clusters ORDER BY job_id"
    ) as cursor:
        # The job that had both keeps one row for the person they now are
        assert await cursor.fetchall() == [(only_the_other, ana, "person_01"), (both, ana, "person_01")]


async def test_the_gallery_gets_small_copies_of_the_thumbnails(client, test_db):
    """A dozen cards used to load a dozen whole PNGs: 9.5 MB to show them 320 px wide."""
    import io

    from PIL import Image

    video_id = await insert_video(test_db, "small.mp4", "completed")
    job_id = await insert_job(test_db, video_id, await insert_cluster(test_db, video_id), "completed")
    path = Path(config.OUTPUT_DIR) / "small" / "thumbnails" / "a.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (1280, 720), (124, 108, 246)).save(path, "PNG")
    cursor = await test_db.execute(
        "INSERT INTO thumbnails (job_id, image_index, filepath) VALUES (?, 1, 'small/thumbnails/a.png')", [job_id]
    )
    await test_db.commit()
    url = f"/api/thumbnails/{cursor.lastrowid}"

    small = await client.get(f"{url}?width=640")

    assert small.status_code == 200 and small.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(small.content)) as image:
        assert image.size == (640, 360)
    # The browser keeps the copy while the file is the same one
    again = await client.get(f"{url}?width=640", headers={"If-None-Match": small.headers["etag"]})
    assert again.status_code == 304 and not again.content

    # A face swap rewrites the file: the copy is another one
    Image.new("RGB", (1280, 720), (10, 10, 10)).save(path, "PNG")
    changed = await client.get(f"{url}?width=640", headers={"If-None-Match": small.headers["etag"]})
    assert changed.status_code == 200 and changed.headers["etag"] != small.headers["etag"]

    # Without a width, the file itself: the viewer and the downloads
    assert (await client.get(url)).content == path.read_bytes()
    assert (await client.get(f"{url}?width=5000")).status_code == 422

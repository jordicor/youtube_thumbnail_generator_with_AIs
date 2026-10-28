"""Library listing: status-group filters, counts, search, sort and per-video totals."""

import pytest


async def _insert_directory(db) -> int:
    cursor = await db.execute(
        "INSERT INTO directories (path, name) VALUES (?, ?)", ("/library/videos", "videos")
    )
    return cursor.lastrowid


async def _insert_video(db, directory_id: int, filename: str, status: str, *, hidden: bool = False,
                        duration: float | None = None, created_at: str, updated_at: str) -> int:
    cursor = await db.execute(
        """
        INSERT INTO videos (filename, filepath, status, directory_id, is_hidden,
                            duration_seconds, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (filename, f"/library/videos/{filename}", status, directory_id, hidden,
         duration, created_at, updated_at),
    )
    return cursor.lastrowid


@pytest.fixture
async def library(test_db):
    directory_id = await _insert_directory(test_db)
    ids = {}
    rows = [
        # name, status, hidden, duration, created, updated
        ("Bravo.mp4", "pending", False, 300, "2026-01-01 00:00:00", "2026-01-05 00:00:00"),
        ("alpha.mp4", "cancelled", False, None, "2026-01-02 00:00:00", "2026-01-02 00:00:00"),
        ("TAG DEL NOVIO%21 %23JuntosEsMejor.mp4", "analyzed", False, 90, "2026-01-03 00:00:00", "2026-01-03 00:00:00"),
        ("Charlie 100%_done.mp4", "completed", False, 600, "2026-01-04 00:00:00", "2026-01-04 00:00:00"),
        ("delta.mp4", "analyzing_faces", False, 120, "2026-01-05 00:00:00", "2026-01-01 00:00:00"),
        ("echo.mp4", "error", False, 60, "2026-01-06 00:00:00", "2026-01-06 00:00:00"),
        ("foxtrot.mp4", "completed", True, 30, "2026-01-07 00:00:00", "2026-01-07 00:00:00"),
    ]
    for name, status, hidden, duration, created, updated in rows:
        ids[name] = await _insert_video(test_db, directory_id, name, status, hidden=hidden,
                                        duration=duration, created_at=created, updated_at=updated)

    # Two thumbnails and two person clusters (plus one scene cluster) for "Charlie"
    charlie = ids["Charlie 100%_done.mp4"]
    for index in range(2):
        await test_db.execute(
            """
            INSERT INTO clusters (video_id, cluster_index, view_mode, num_frames, representative_frame)
            VALUES (?, ?, 'person', 1, 'charlie/frames/rep.jpg')
            """,
            (charlie, index),
        )
    await test_db.execute(
        """
        INSERT INTO clusters (video_id, cluster_index, view_mode, num_frames, representative_frame)
        VALUES (?, 0, 'person_scene', 1, 'charlie/frames/rep.jpg')
        """,
        (charlie,),
    )
    cursor = await test_db.execute(
        "INSERT INTO generation_jobs (video_id, num_images, status) VALUES (?, 2, 'completed')", (charlie,)
    )
    for index in range(2):
        await test_db.execute(
            "INSERT INTO thumbnails (job_id, image_index, filepath) VALUES (?, ?, ?)",
            (cursor.lastrowid, index, f"charlie/thumbnails/{index}.png"),
        )
    await test_db.commit()
    return directory_id, ids


async def _get(client, directory_id, **params):
    response = await client.get(f"/api/directories/{directory_id}/videos", params=params)
    assert response.status_code == 200, response.text
    return response.json()


async def test_counts_every_filter_and_groups_add_up(client, library):
    directory_id, _ = library
    data = await _get(client, directory_id)
    assert data["counts"] == {
        "all": 6, "pending": 2, "processing": 1, "analyzed": 1,
        "completed": 1, "error": 1, "hidden": 1,
    }
    assert data["total"] == 6


@pytest.mark.parametrize("status, expected", [
    ("pending", {"Bravo.mp4", "alpha.mp4"}),
    ("processing", {"delta.mp4"}),
    ("analyzed", {"TAG DEL NOVIO%21 %23JuntosEsMejor.mp4"}),
    ("completed", {"Charlie 100%_done.mp4"}),
    ("error", {"echo.mp4"}),
    ("hidden", {"foxtrot.mp4"}),
])
async def test_filters_select_one_status_group(client, library, status, expected):
    directory_id, _ = library
    data = await _get(client, directory_id, status=status)
    assert {v["filename"] for v in data["videos"]} == expected
    assert data["total"] == len(expected)


async def test_search_matches_display_text_and_literal_wildcards(client, library):
    directory_id, _ = library
    # "!" and "#" are stored percent-encoded in downloaded names
    data = await _get(client, directory_id, search="novio! #juntos")
    assert [v["filename"] for v in data["videos"]] == ["TAG DEL NOVIO%21 %23JuntosEsMejor.mp4"]
    # "%" and "_" match literally, not as LIKE wildcards
    data = await _get(client, directory_id, search="100%_")
    assert [v["filename"] for v in data["videos"]] == ["Charlie 100%_done.mp4"]
    data = await _get(client, directory_id, search="0%d")
    assert data["videos"] == []
    # Counts follow the search
    data = await _get(client, directory_id, search="  ALPHA  ")
    assert data["counts"]["all"] == 1 and data["counts"]["pending"] == 1


@pytest.mark.parametrize("sort, expected", [
    ("recent", ["echo.mp4", "delta.mp4", "Charlie 100%_done.mp4", "TAG DEL NOVIO%21 %23JuntosEsMejor.mp4",
                "alpha.mp4", "Bravo.mp4"]),
    ("name", ["alpha.mp4", "Bravo.mp4", "Charlie 100%_done.mp4", "delta.mp4", "echo.mp4",
              "TAG DEL NOVIO%21 %23JuntosEsMejor.mp4"]),
    ("duration", ["Charlie 100%_done.mp4", "Bravo.mp4", "delta.mp4", "TAG DEL NOVIO%21 %23JuntosEsMejor.mp4",
                  "echo.mp4", "alpha.mp4"]),
    ("activity", ["echo.mp4", "Bravo.mp4", "Charlie 100%_done.mp4", "TAG DEL NOVIO%21 %23JuntosEsMejor.mp4",
                  "alpha.mp4", "delta.mp4"]),
])
async def test_sort_orders(client, library, sort, expected):
    directory_id, _ = library
    data = await _get(client, directory_id, sort=sort)
    assert [v["filename"] for v in data["videos"]] == expected


async def test_videos_carry_thumbnail_and_person_counts(client, library):
    directory_id, _ = library
    data = await _get(client, directory_id, status="completed")
    video = data["videos"][0]
    assert (video["thumbnail_count"], video["person_count"]) == (2, 2)
    data = await _get(client, directory_id, status="pending")
    assert all((v["thumbnail_count"], v["person_count"]) == (0, 0) for v in data["videos"])


@pytest.mark.parametrize("params", [{"status": "ready"}, {"sort": "size"}, {"search": "x" * 201}])
async def test_rejects_unknown_filter_sort_and_long_search(client, library, params):
    directory_id, _ = library
    response = await client.get(f"/api/directories/{directory_id}/videos", params=params)
    assert response.status_code == 422


async def test_unknown_video_status_fails_loudly(client, library, test_db):
    directory_id, ids = library
    await test_db.execute("UPDATE videos SET status = 'mystery' WHERE id = ?", (ids["echo.mp4"],))
    await test_db.commit()
    with pytest.raises(ValueError, match="mystery"):
        await client.get(f"/api/directories/{directory_id}/videos")

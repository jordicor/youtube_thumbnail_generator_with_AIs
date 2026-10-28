"""Editing a directory: renaming it, pointing it at the same folder reached another
way, and switching by itself to an alternative path when its path does not answer."""

import json
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# A path that no longer answers, like a share mounted by software that is gone
OLD_ROOT = str(Path("/unreachable/share/Videos").resolve())

VIDEOS = ["a.mp4", str(Path("Project 1") / "b.mp4"), str(Path("Project 2") / "deep" / "c.mp4")]


@pytest.fixture
async def directory(test_db):
    """A directory under OLD_ROOT with three videos, one analyzed with a person."""
    cursor = await test_db.execute(
        "INSERT INTO directories (path, name, video_count) VALUES (?, 'Videos', 3)", (OLD_ROOT,)
    )
    directory_id = cursor.lastrowid
    video_ids = {}
    for relative in VIDEOS:
        cursor = await test_db.execute(
            """
            INSERT INTO videos (filename, filepath, status, directory_id, updated_at)
            VALUES (?, ?, 'analyzed', ?, '2026-01-01 00:00:00')
            """,
            (Path(relative).name, str(Path(OLD_ROOT) / relative), directory_id),
        )
        video_ids[relative] = cursor.lastrowid
    await test_db.execute(
        """
        INSERT INTO clusters (video_id, cluster_index, view_mode, num_frames, representative_frame)
        VALUES (?, 0, 'person', 1, 'a/frames/rep.jpg')
        """,
        (video_ids["a.mp4"],),
    )
    await test_db.commit()
    return directory_id, video_ids


def _make_folder(root: Path, videos=VIDEOS) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative in videos:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(b"video")
    return root


def _put(client, directory_id, path, name="Videos", alternatives=(), **extra):
    return client.put(
        f"/api/directories/{directory_id}",
        json={"name": name, "path": str(path), "alternative_paths": [str(p) for p in alternatives], **extra},
    )


async def _video_paths(db) -> dict:
    async with db.execute("SELECT id, filepath FROM videos") as cursor:
        return dict(await cursor.fetchall())


async def _directory_row(db, directory_id) -> tuple:
    async with db.execute("SELECT path, name FROM directories WHERE id = ?", (directory_id,)) as cursor:
        return await cursor.fetchone()


async def _set_directory(db, directory_id, path, alternatives) -> None:
    """Point the directory and its videos at path, with these alternative paths."""
    await db.execute(
        "UPDATE directories SET path = ?, alternative_paths = ? WHERE id = ?",
        (str(path), json.dumps([str(p) for p in alternatives]), directory_id),
    )
    async with db.execute("SELECT id, filepath FROM videos WHERE directory_id = ?", (directory_id,)) as cursor:
        for video_id, filepath in await cursor.fetchall():
            # Writing the same path would count as activity on the video
            new_filepath = str(Path(path) / Path(filepath).relative_to(OLD_ROOT))
            if new_filepath != filepath:
                await db.execute("UPDATE videos SET filepath = ? WHERE id = ?", (new_filepath, video_id))
    await db.commit()


async def _listed(client, directory_id) -> dict:
    response = await client.get("/api/directories")
    assert response.status_code == 200, response.text
    return next(d for d in response.json()["directories"] if d["id"] == directory_id)


# =============================================================================
# EDITING THE PATH
# =============================================================================

async def test_new_path_moves_every_video_and_keeps_their_analysis(client, test_db, directory, tmp_path):
    directory_id, video_ids = directory
    new_root = _make_folder(tmp_path / "Mapped" / "Videos")

    response = await _put(client, directory_id, new_root, name="Mapped videos")

    assert response.status_code == 200, response.text
    assert response.json()["path"] == str(new_root)
    assert await _directory_row(test_db, directory_id) == (str(new_root), "Mapped videos")
    paths = await _video_paths(test_db)
    for relative, video_id in video_ids.items():
        assert paths[video_id] == str(new_root / relative)
    # Status, people and the library's activity order stay as they were
    async with test_db.execute("SELECT DISTINCT status, updated_at FROM videos") as cursor:
        assert await cursor.fetchall() == [("analyzed", "2026-01-01 00:00:00")]
    async with test_db.execute("SELECT video_id FROM clusters") as cursor:
        assert await cursor.fetchall() == [(video_ids["a.mp4"],)]


async def test_missing_videos_need_confirmation(client, test_db, directory, tmp_path):
    directory_id, video_ids = directory
    new_root = _make_folder(tmp_path / "Videos", videos=VIDEOS[:1])
    await test_db.execute("UPDATE videos SET is_hidden = TRUE WHERE id = ?", (video_ids[VIDEOS[1]],))
    await test_db.commit()
    before = await _video_paths(test_db)

    response = await _put(client, directory_id, new_root)

    assert response.status_code == 409
    # Visible videos first: hidden ones are usually gone on purpose
    assert response.json()["detail"] == {"missing": [VIDEOS[2], VIDEOS[1]], "hidden": 1, "total": 3}
    assert await _video_paths(test_db) == before
    assert (await _directory_row(test_db, directory_id))[0] == OLD_ROOT

    response = await _put(client, directory_id, new_root, allow_missing_videos=True)

    assert response.status_code == 200, response.text
    assert (await _video_paths(test_db))[video_ids[VIDEOS[2]]] == str(new_root / VIDEOS[2])


async def test_rename_only_works_while_the_path_is_unreachable(client, test_db, directory):
    directory_id, _ = directory
    before = await _video_paths(test_db)

    response = await _put(client, directory_id, OLD_ROOT, name="  Old share  ")

    assert response.status_code == 200, response.text
    assert await _directory_row(test_db, directory_id) == (OLD_ROOT, "Old share")
    assert await _video_paths(test_db) == before


async def test_blank_name_becomes_the_folder_name(client, test_db, directory, tmp_path):
    directory_id, _ = directory
    new_root = _make_folder(tmp_path / "Shared videos")

    response = await _put(client, directory_id, new_root, name=" ")

    assert response.status_code == 200, response.text
    assert response.json()["name"] == "Shared videos"


@pytest.mark.parametrize("make_target", [
    lambda tmp_path: tmp_path / "does-not-exist",
    lambda tmp_path: _make_folder(tmp_path, videos=["file.mp4"]) / "file.mp4",
])
async def test_new_path_must_be_an_existing_directory(client, test_db, directory, tmp_path, make_target):
    directory_id, _ = directory
    before = await _video_paths(test_db)

    response = await _put(client, directory_id, make_target(tmp_path))

    assert response.status_code == 400
    assert await _video_paths(test_db) == before
    assert (await _directory_row(test_db, directory_id))[0] == OLD_ROOT


async def test_path_of_another_directory_is_refused(client, test_db, directory, tmp_path):
    directory_id, _ = directory
    other_root = _make_folder(tmp_path / "Other")
    await test_db.execute("INSERT INTO directories (path, name) VALUES (?, 'Other')", (str(other_root),))
    await test_db.commit()

    response = await _put(client, directory_id, other_root)

    assert response.status_code == 400
    assert "already used by another directory" in response.json()["detail"]


async def test_clash_with_another_video_changes_nothing(client, test_db, directory, tmp_path):
    directory_id, _ = directory
    new_root = _make_folder(tmp_path / "Videos")
    # A video registered on its own under the new path, e.g. by a scan of another directory
    await test_db.execute(
        "INSERT INTO videos (filename, filepath, status) VALUES ('c.mp4', ?, 'pending')",
        (str(new_root / VIDEOS[2]),),
    )
    await test_db.commit()
    before = await _video_paths(test_db)

    response = await _put(client, directory_id, new_root, name="Renamed")

    assert response.status_code == 400
    assert "already has the new path" in response.json()["detail"]
    assert await _video_paths(test_db) == before
    assert await _directory_row(test_db, directory_id) == (OLD_ROOT, "Videos")


async def test_video_outside_its_directory_is_refused(client, test_db, directory, tmp_path):
    directory_id, video_ids = directory
    await test_db.execute(
        "UPDATE videos SET filepath = ? WHERE id = ?",
        (str(Path("/elsewhere/a.mp4").resolve()), video_ids["a.mp4"]),
    )
    await test_db.commit()
    new_root = _make_folder(tmp_path / "Videos")

    response = await _put(client, directory_id, new_root)

    assert response.status_code == 400
    assert "not inside its directory" in response.json()["detail"]
    assert (await _directory_row(test_db, directory_id))[0] == OLD_ROOT


async def test_scan_after_moving_finds_the_same_videos(client, test_db, directory, tmp_path, monkeypatch):
    directory_id, _ = directory
    new_root = _make_folder(tmp_path / "Videos")
    monkeypatch.setattr("utils.get_video_info", lambda path: {"duration": 1.0})
    response = await _put(client, directory_id, new_root)
    assert response.status_code == 200, response.text

    response = await client.post(f"/api/directories/{directory_id}/scan")

    assert response.status_code == 200, response.text
    assert response.json() == {"scanned": 3, "new": 0, "existing": 3, "total": 3}


async def test_added_directory_keeps_the_path_as_typed(client, tmp_path, monkeypatch):
    folder = tmp_path / "Videos"
    (folder / "sub").mkdir(parents=True)
    # Path.resolve() would turn a mapped drive into its network share
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: Path(r"\\server\share\Videos"))

    response = await client.post("/api/directories", json={"path": str(folder / "sub" / "..")})

    assert response.status_code == 200, response.text
    assert response.json()["path"] == str(folder)


# =============================================================================
# ALTERNATIVE PATHS
# =============================================================================

async def test_alternatives_are_saved_without_existing_here(client, test_db, directory, tmp_path):
    directory_id, _ = directory
    mac, other = tmp_path / "Mac share" / "Videos", tmp_path / "Other machine"

    response = await _put(client, directory_id, OLD_ROOT, alternatives=[str(mac) + os.sep, other])

    assert response.status_code == 200, response.text
    assert response.json()["alternative_paths"] == [str(mac), str(other)]
    assert (await _listed(client, directory_id))["alternative_paths"] == [str(mac), str(other)]
    # Saving them moves nothing
    assert (await _directory_row(test_db, directory_id))[0] == OLD_ROOT


@pytest.mark.parametrize("make_alternatives, error", [
    (lambda tmp_path: [Path("relative") / "Videos"], "not absolute"),
    (lambda tmp_path: [OLD_ROOT], "listed twice"),
    (lambda tmp_path: [tmp_path / "A", tmp_path / "A"], "listed twice"),
])
async def test_invalid_alternatives_are_refused(client, test_db, directory, tmp_path, make_alternatives, error):
    directory_id, _ = directory

    response = await _put(client, directory_id, OLD_ROOT, alternatives=make_alternatives(tmp_path))

    assert response.status_code == 400
    assert error in response.json()["detail"]


async def test_paths_cannot_be_shared_with_another_directory(client, test_db, directory, tmp_path):
    directory_id, _ = directory
    other_path, other_alternative = tmp_path / "Other", tmp_path / "Other elsewhere"
    _make_folder(other_alternative, videos=[])
    await test_db.execute(
        "INSERT INTO directories (path, name, alternative_paths) VALUES (?, 'Other', ?)",
        (str(other_path), json.dumps([str(other_alternative)])),
    )
    await test_db.commit()

    for alternative in (other_path, other_alternative):
        response = await _put(client, directory_id, OLD_ROOT, alternatives=[alternative])
        assert response.status_code == 400
        assert "already used by another directory" in response.json()["detail"]
    # Nor as the path of a new directory
    response = await client.post("/api/directories", json={"path": str(other_alternative)})
    assert response.status_code == 400


async def test_alternative_where_a_video_clashes_is_refused(client, test_db, directory, tmp_path):
    directory_id, _ = directory
    alternative = tmp_path / "Elsewhere"
    await test_db.execute(
        "INSERT INTO videos (filename, filepath, status) VALUES ('a.mp4', ?, 'pending')", (str(alternative / "a.mp4"),)
    )
    await test_db.commit()

    response = await _put(client, directory_id, OLD_ROOT, alternatives=[alternative])

    assert response.status_code == 400
    assert "already has the new path" in response.json()["detail"]


async def test_listing_switches_to_the_first_alternative_that_answers(client, test_db, directory, tmp_path):
    directory_id, video_ids = directory
    gone, here, also_here = tmp_path / "Gone", _make_folder(tmp_path / "Here"), _make_folder(tmp_path / "Also")
    await _set_directory(test_db, directory_id, OLD_ROOT, [gone, here, also_here])

    listed = await _listed(client, directory_id)

    assert listed["path"] == str(here)
    assert listed["switched_from"] == OLD_ROOT
    # The old path takes the place of the one now in use
    assert listed["alternative_paths"] == [str(gone), OLD_ROOT, str(also_here)]
    paths = await _video_paths(test_db)
    assert [paths[video_ids[relative]] for relative in VIDEOS] == [str(here / relative) for relative in VIDEOS]
    async with test_db.execute("SELECT DISTINCT updated_at FROM videos") as cursor:
        assert await cursor.fetchall() == [("2026-01-01 00:00:00",)]

    # Once switched, it stays
    listed = await _listed(client, directory_id)
    assert (listed["path"], listed["switched_from"]) == (str(here), None)


async def test_switching_back_and_forth_between_machines(client, test_db, directory, tmp_path):
    directory_id, video_ids = directory
    pc, mac = _make_folder(tmp_path / "PC"), _make_folder(tmp_path / "Mac")
    await _set_directory(test_db, directory_id, pc, [mac])

    # On the other machine only its own path answers
    shutil.rmtree(pc)
    listed = await _listed(client, directory_id)
    assert (listed["path"], listed["alternative_paths"]) == (str(mac), [str(pc)])

    _make_folder(pc)
    shutil.rmtree(mac)
    listed = await _listed(client, directory_id)
    assert (listed["path"], listed["alternative_paths"]) == (str(pc), [str(mac)])
    assert (await _video_paths(test_db))[video_ids[VIDEOS[2]]] == str(pc / VIDEOS[2])


async def test_switch_moves_videos_missing_in_the_new_path(client, test_db, directory, tmp_path):
    directory_id, video_ids = directory
    here = _make_folder(tmp_path / "Here", videos=VIDEOS[:1])
    await _set_directory(test_db, directory_id, OLD_ROOT, [here])

    listed = await _listed(client, directory_id)

    assert listed["path"] == str(here)
    assert (await _video_paths(test_db))[video_ids[VIDEOS[2]]] == str(here / VIDEOS[2])


@pytest.mark.parametrize("answering", ["path", "none"])
async def test_listing_keeps_the_path(client, test_db, directory, tmp_path, answering):
    directory_id, _ = directory
    path = _make_folder(tmp_path / "Path") if answering == "path" else tmp_path / "Gone"
    alternative = _make_folder(tmp_path / "Alternative") if answering == "path" else tmp_path / "Gone too"
    await _set_directory(test_db, directory_id, path, [alternative])

    listed = await _listed(client, directory_id)

    assert (listed["path"], listed["alternative_paths"], listed["switched_from"]) == (str(path), [str(alternative)], None)


async def test_scan_switches_to_an_alternative_first(client, test_db, directory, tmp_path, monkeypatch):
    directory_id, _ = directory
    here = _make_folder(tmp_path / "Here")
    await _set_directory(test_db, directory_id, OLD_ROOT, [here])
    monkeypatch.setattr("utils.get_video_info", lambda path: {"duration": 1.0})

    response = await client.post(f"/api/directories/{directory_id}/scan")

    assert response.status_code == 200, response.text
    assert response.json()["existing"] == 3
    assert (await _directory_row(test_db, directory_id))[0] == str(here)


@pytest.mark.parametrize("route, module", [
    ("/api/analysis/{video_id}/start", "api.routes.analysis"),
    ("/api/videos/{video_id}/reanalyze", "api.routes.videos"),
])
async def test_analysis_switches_to_an_alternative_first(client, test_db, directory, tmp_path, monkeypatch,
                                                         route, module):
    directory_id, video_ids = directory
    here = _make_folder(tmp_path / "Here")
    await _set_directory(test_db, directory_id, OLD_ROOT, [here])
    enqueue = AsyncMock(return_value="job-1")
    monkeypatch.setattr(f"{module}.enqueue_analysis", enqueue)

    response = await client.post(route.format(video_id=video_ids["a.mp4"]))

    assert response.status_code == 200, response.text
    enqueue.assert_awaited_once()
    assert (await _video_paths(test_db))[video_ids["a.mp4"]] == str(here / "a.mp4")


# =============================================================================
# MIGRATIONS
# =============================================================================

async def test_migration_makes_the_timestamp_trigger_skip_path_changes(test_db, directory):
    from database.db import _run_migrations

    _, video_ids = directory
    # The trigger as databases created before migration 17 have it
    await test_db.execute("DROP TRIGGER update_videos_timestamp")
    await test_db.execute("""
        CREATE TRIGGER update_videos_timestamp AFTER UPDATE ON videos
        BEGIN UPDATE videos SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id; END
    """)
    await test_db.commit()

    await _run_migrations(test_db)

    moved, analyzed = video_ids["a.mp4"], video_ids[VIDEOS[1]]
    await test_db.execute("UPDATE videos SET filepath = '/new/a.mp4' WHERE id = ?", (moved,))
    await test_db.execute("UPDATE videos SET status = 'completed' WHERE id = ?", (analyzed,))
    await test_db.commit()
    async with test_db.execute("SELECT id, updated_at FROM videos") as cursor:
        updated = dict(await cursor.fetchall())
    assert updated[moved] == "2026-01-01 00:00:00"
    assert updated[analyzed] != "2026-01-01 00:00:00"


async def test_migration_gives_existing_directories_no_alternatives(test_db, directory):
    from database.db import _run_migrations

    directory_id, _ = directory
    await test_db.execute("ALTER TABLE directories DROP COLUMN alternative_paths")
    await test_db.commit()

    await _run_migrations(test_db)

    async with test_db.execute("SELECT alternative_paths FROM directories WHERE id = ?", (directory_id,)) as cursor:
        assert await cursor.fetchone() == ("[]",)

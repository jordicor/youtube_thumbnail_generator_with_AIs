"""Cluster listing: person_scene clusters name the person that shares most of their frames."""

import pytest

from tests.conftest import insert_video


@pytest.mark.parametrize('reference_only', [False, True])
async def test_cli_frames_use_current_visual_references(test_db, monkeypatch, reference_only):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import database.db
    import main
    from services.reference_service import ReferenceService, VisualReference

    video_id = await insert_video(test_db, 'listing_video.mp4', 'analyzed')
    frame_ids = [await _insert_frame(test_db, video_id, f'f{i}') for i in range(1, 4)]
    person = await _insert_cluster(test_db, video_id, 0, 'person', frame_ids)
    await ReferenceService(test_db).insert(video_id, [
        VisualReference(role='face', person_id=person, source_type='frame', source=name,
                        primary=index == 0).model_dump()
        for index, name in enumerate(['f2.jpg', 'f1.jpg', 'f2.jpg'])
    ])

    @asynccontextmanager
    async def get_db():
        yield test_db

    result = {}
    monkeypatch.setattr(database.db, 'get_db', get_db)
    monkeypatch.setattr(main, '_open_db', AsyncMock())
    monkeypatch.setattr(main, '_emit_json', result.update)
    assert await main.cmd_list_frames(SimpleNamespace(cluster=person, reference_only=reference_only, json=True)) == 0
    assert [frame['id'] for frame in result['frames']] == (
        [frame_ids[1], frame_ids[0]] if reference_only else [frame_ids[1], frame_ids[0], frame_ids[2]])
    assert [frame['is_reference'] for frame in result['frames']] == ([1, 1] if reference_only else [1, 1, 0])


async def _insert_frame(db, video_id: int, name: str) -> int:
    cursor = await db.execute(
        "INSERT INTO video_frames (video_id, frame_path, quality_score) VALUES (?, ?, ?)",
        (video_id, f"listing_video/frames/{name}.jpg", 80),
    )
    return cursor.lastrowid


async def _insert_cluster(db, video_id: int, index: int, view_mode: str, frame_ids: list,
                          label: str | None = None, scene_index: int | None = None) -> int:
    cursor = await db.execute(
        """
        INSERT INTO clusters (video_id, cluster_index, label, view_mode, scene_index,
                              representative_frame_id, num_frames, representative_frame)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (video_id, index, label, view_mode, scene_index, frame_ids[0], len(frame_ids),
         "listing_video/frames/representative.jpg"),
    )
    cluster_id = cursor.lastrowid
    for frame_id in frame_ids:
        await db.execute(
            "INSERT INTO cluster_frame_assignments (cluster_id, frame_id) VALUES (?, ?)",
            (cluster_id, frame_id),
        )
    return cluster_id


@pytest.fixture
async def scene_view(test_db):
    video_id = await insert_video(test_db, "listing_video.mp4", "analyzed")
    f1, f2, f3, f4, f5 = [await _insert_frame(test_db, video_id, f"f{i}") for i in range(1, 6)]
    await _insert_cluster(test_db, video_id, 0, "person", [f1, f2], label="Ana")
    await _insert_cluster(test_db, video_id, 1, "person", [f3, f5])
    # Scene clusters reuse the person indices space independently (V2 views)
    await _insert_cluster(test_db, video_id, 0, "person_scene", [f2, f1], scene_index=3)
    await _insert_cluster(test_db, video_id, 1, "person_scene", [f3], scene_index=0)
    # A manual scene-view cluster whose frames no person cluster holds
    await _insert_cluster(test_db, video_id, 2, "person_scene", [f4])
    # Its representative frame f1 is Ana's, but most of its frames are the second person's
    await _insert_cluster(test_db, video_id, 3, "person_scene", [f1, f3, f5], scene_index=7)
    await test_db.commit()
    return video_id


async def test_scene_clusters_carry_their_current_person(client, scene_view):
    response = await client.get(f"/api/analysis/{scene_view}/clusters", params={"view_mode": "person_scene"})
    assert response.status_code == 200
    parents = [
        (c["cluster_index"], c["scene_index"], c["parent_cluster_index"], c["parent_label"])
        for c in response.json()["clusters"]
    ]
    assert parents == [(0, 3, 0, "Ana"), (1, 0, 1, None), (2, None, None, None), (3, 7, 1, None)]


async def test_clusters_count_their_frames_and_references(client, scene_view, test_db):
    from services.reference_service import ReferenceService
    async with test_db.execute("""
        SELECT cluster_id FROM cluster_frame_assignments
        WHERE frame_id=(SELECT id FROM video_frames WHERE frame_path LIKE '%/f1.jpg')
    """) as cursor:
        ids = [row[0] for row in await cursor.fetchall()]
    for person_id in ids:
        await ReferenceService(test_db).replace_person_faces(scene_view, person_id, ["f1.jpg"])
    # A person left without frames
    await test_db.execute(
        "INSERT INTO clusters (video_id, cluster_index, view_mode, num_frames, representative_frame) "
        "VALUES (?, 2, 'person', 0, ?)",
        (scene_view, "listing_video/frames/representative.jpg"),
    )
    await test_db.commit()

    counts = {}
    for view_mode in ("person", "person_scene"):
        response = await client.get(f"/api/analysis/{scene_view}/clusters", params={"view_mode": view_mode})
        assert response.status_code == 200
        counts[view_mode] = [
            (c["cluster_index"], c["num_frames"], c["reference_count"]) for c in response.json()["clusters"]
        ]

    assert counts["person"] == [(0, 2, 1), (1, 2, 0), (2, 0, 0)]
    assert counts["person_scene"] == [(0, 2, 1), (1, 1, 0), (2, 1, 0), (3, 3, 1)]


async def test_person_clusters_have_no_parent(client, scene_view):
    response = await client.get(f"/api/analysis/{scene_view}/clusters", params={"view_mode": "person"})
    assert response.status_code == 200
    clusters = response.json()["clusters"]
    assert [c["label"] for c in clusters] == ["Ana", None]
    assert all(c["parent_cluster_index"] is None and c["parent_label"] is None for c in clusters)

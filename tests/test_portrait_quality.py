"""Characters, pictures and non-destructive reference refresh regressions."""
from types import SimpleNamespace

import numpy as np
import pytest

from services.analysis_service import AnalysisService
from services.face_selection import character_groups, is_picture, look_vector, same_look_frames
from services.reference_service import ReferenceService
from tests.conftest import portrait
from tests.test_body_references import pose, segmented_pose
from tests.test_person_editor import people_video as people_video


def _look(hair_lightness, skin_lightness=190.0):
    return {"hair": [hair_lightness, 140.0, 150.0], "skin": [skin_lightness, 140.0, 150.0],
            "clothes": None, "hair_share": 0.2}


def test_a_wig_is_another_character_but_lighting_is_not():
    rng = np.random.default_rng(3)
    # Light hair under changing light (ratio 0.3-0.7) and a black wig (ratio < 0.1)
    light = {f"mel{i}": look_vector({"look": _look(v * 190)}) for i, v in enumerate(rng.uniform(0.3, 0.7, 60))}
    dark = {f"juan{i}": look_vector({"look": _look(v * 190)}) for i, v in enumerate(rng.uniform(0.0, 0.08, 40))}
    groups = character_groups({**light, **dark, "unmeasured": None})
    assert [len(group) for group in groups] == [61, 40]
    assert set(groups[1]) == set(dark) and "unmeasured" in groups[0]
    assert character_groups(light) == [list(light)]
    # A few dark frames (a shadow, a hat) are not a character
    assert len(character_groups({**light, **dict(list(dark.items())[:8])})) == 1


def test_the_same_look_follows_hair_against_the_skin_not_the_light():
    unknown = portrait()
    unknown["look"] = None
    index = SimpleNamespace(portraits=(portrait(look=_look(120, 190)), portrait(look=_look(60, 100)),
                                       portrait(look=_look(15, 190)), unknown))
    matches = {name: SimpleNamespace(row=row) for row, name in enumerate(("bright", "dim", "wig", "unknown"))}
    assert same_look_frames(matches, index, "bright") == {"bright", "dim"}
    assert same_look_frames(matches, index, None) == set()


def test_a_face_that_never_moves_is_a_picture():
    still = [(f"scene_001_frame_{n:06d}.jpg", [500.0, 100.0, 560.0, 170.0 + n % 2 * 0.3]) for n in range(6)]
    moving = [(name, [box[0] + 8 * i, box[1], box[2] + 8 * i, box[3]]) for i, (name, box) in enumerate(still)]
    assert is_picture(still)
    assert not is_picture(moving)
    assert not is_picture(still[:2])  # Two frames cannot tell
    assert not is_picture(still + [(f"scene_002_frame_{n:06d}.jpg", box) for n, (_, box) in enumerate(moving)])


def test_empty_body_mask_is_a_rejected_candidate():
    from tests.test_body_references import person_mask
    record = segmented_pose(pose(), [[350, 100, 450, 200]], obstacles=np.ones(person_mask().shape, dtype=bool))[0]
    assert record["status"] == "empty_mask"


def test_textured_clothing_cannot_rescue_a_blurred_body_reference():
    record = segmented_pose(pose(), [[350, 100, 450, 200]],
                            portrait={"score": 85, "eye_openness": .3, "occlusion": 0, "issues": ["blurred"]})[0]
    assert record["status"] == "blurred"


def test_tiny_predicted_legs_do_not_prove_full_body_coverage():
    from services.body_references import pose_description
    from tests.test_body_references import person_mask
    points = pose()
    for hip, knee, ankle in [(23, 25, 27), (24, 26, 28)]:
        points[knee].y = points[hip].y + .04
        points[ankle].y = points[knee].y + .04
    assert pose_description(points, points, person_mask(), face_height=100)[0] == "torso"


async def test_refresh_keeps_people_notes_and_transcripts(test_db, people_video, monkeypatch):
    import services.analysis_service as analysis
    import services.body_references as bodies
    import services.portrait_quality as portraits

    video = people_video["video_id"]
    folder = people_video["dir"]
    transcript = folder / "transcription.txt"
    transcript.write_text("Existing transcript", encoding="utf-8")
    await test_db.execute("UPDATE clusters SET label='Custom name',description='Keep these notes' WHERE id=?",
                          [people_video["ana"]])
    await test_db.commit()
    async with test_db.execute("SELECT id,label,description,embedding_centroid FROM clusters WHERE video_id=? ORDER BY id",
                               [video]) as cursor:
        before = await cursor.fetchall()
    monkeypatch.setattr(analysis, "process_faces", lambda *args: None)
    monkeypatch.setattr(portraits, "enrich_portraits", lambda *args: None)
    monkeypatch.setattr(bodies, "ensure_body_index", lambda *args: {"frames": {}})
    monkeypatch.setattr(analysis, "transcribe_video", lambda *args: pytest.fail("A reference refresh must not transcribe."))
    monkeypatch.setattr(analysis, "process_video_scenes", lambda *args: pytest.fail("Do not repeat scene extraction."))
    service = AnalysisService(test_db)
    await service.run_full_analysis(video, refresh_references=True)
    assert (await service.get_video(video))["status"] == "analyzed"
    assert (await service.get_video(video))["analysis_outcome"] == "completed"
    async with test_db.execute("SELECT id,label,description,embedding_centroid FROM clusters WHERE video_id=? ORDER BY id",
                               [video]) as cursor:
        assert await cursor.fetchall() == before
    assert transcript.read_text(encoding="utf-8") == "Existing transcript"
    assert (await service.get_video(video))["reference_revision"] == 1
    assert await ReferenceService(test_db).list(video, people_video["ana"])


async def test_cancellation_preserves_preexisting_transcripts_and_people(test_db, people_video, monkeypatch):
    from contextlib import asynccontextmanager
    from services.task_service import TaskService
    from workers import tasks

    video = people_video["video_id"]
    paths = [people_video["dir"] / name for name in ("transcription.txt", "transcription.json", "clustering_result.json")]
    for path in paths:
        path.write_bytes(b"Previously completed data")
    refs = await ReferenceService(test_db).list(video)
    service = AnalysisService(test_db)
    run = await service.claim_analysis(video)

    @asynccontextmanager
    async def get_db():
        yield test_db

    monkeypatch.setattr(tasks, "get_db", get_db)

    async def cancelled_while_running(self, video_id, **kwargs):
        assert await TaskService(self.db).cancel_task("analysis", video_id)

    monkeypatch.setattr(AnalysisService, "run_full_analysis", cancelled_while_running)
    result = await tasks.analyze_video({}, video, run)

    assert result["status"] == "cancelled"
    assert all(path.read_bytes() == b"Previously completed data" for path in paths)
    assert await ReferenceService(test_db).list(video) == refs


async def test_queued_refresh_does_not_overwrite_newer_reference_edits(test_db, people_video, monkeypatch):
    import services.analysis_service as analysis

    video = people_video["video_id"]
    refs = await ReferenceService(test_db).list(video)
    await test_db.execute("UPDATE videos SET reference_revision=2 WHERE id=?", [video])
    await test_db.commit()
    monkeypatch.setattr(analysis, "process_faces", lambda *args: pytest.fail("Do not process a stale refresh."))
    with pytest.raises(ValueError, match="changed after this refresh was queued"):
        await AnalysisService(test_db).refresh_visual_references(video, expected_reference_revision=1)
    assert await ReferenceService(test_db).list(video) == refs


def test_body_blur_is_measured_at_a_common_scale():
    import cv2
    from services.body_references import body_sharpness

    image = np.random.default_rng(12).integers(0, 255, (512, 256, 3), dtype=np.uint8)
    mask = np.ones(image.shape[:2], dtype=bool)
    sharp = body_sharpness(image, mask, [0, 0, 256, 512])
    enlarged = cv2.resize(image, (512, 1024), interpolation=cv2.INTER_NEAREST)
    assert body_sharpness(enlarged, np.ones((1024, 512), dtype=bool), [0, 0, 512, 1024]) == pytest.approx(sharp)
    blurred = cv2.GaussianBlur(image, (21, 21), 5)
    assert body_sharpness(blurred, mask, [0, 0, 256, 512]) < 20


def test_head_and_shoulders_is_not_a_body_proportions_reference():
    from services.body_references import person_body_candidates

    face = [100, 100, 300, 400]
    data = {"frames": {"portrait.jpg": [{"status": "matched", "face_box": face,
                                        "box": [0, 0, 500, 600], "quality": 90}]}}
    matches = {"portrait.jpg": SimpleNamespace(similarity=.9, box=face)}
    result = person_body_candidates(data, matches)
    assert result["candidates"] == []
    assert result["excluded"] == {"insufficient_coverage": 1}


async def _refresh(test_db, video, monkeypatch):
    import services.analysis_service as analysis
    import services.body_references as bodies
    import services.portrait_quality as portraits

    monkeypatch.setattr(analysis, "process_faces", lambda *args: None)
    monkeypatch.setattr(portraits, "enrich_portraits", lambda *args: None)
    monkeypatch.setattr(bodies, "ensure_body_index", lambda *args: {"frames": {}})
    await AnalysisService(test_db).refresh_visual_references(video)


async def test_refresh_moves_a_second_character_to_a_new_person(test_db, people_video, monkeypatch):
    import orjson

    import services.face_selection as selection

    monkeypatch.setattr(selection, "CHARACTER_MIN_FACES", 1)
    monkeypatch.setattr(selection, "CHARACTER_MIN_SHARE", 0)
    video, (f1, f2, f3, f4, f5, *_) = people_video["video_id"], people_video["names"]
    faces_file = people_video["dir"] / "faces.json"
    data = orjson.loads(faces_file.read_bytes())
    for face in data["all_faces"]:
        if face["frame_path"].endswith((f3, f4)):
            face["portrait_quality"]["look"] = _look(10.0)  # A black wig
    faces_file.write_bytes(orjson.dumps(data))
    await test_db.execute("UPDATE clusters SET label='Mel' WHERE id=?", [people_video["ana"]])
    await test_db.commit()

    await _refresh(test_db, video, monkeypatch)

    people = {c["id"]: c for c in await AnalysisService(test_db).get_clusters(video, "person")}
    assert people[people_video["ana"]]["label"] == "Mel" and people[people_video["ana"]]["num_frames"] == 3
    new = [c for c in people.values() if c["id"] not in (people_video["ana"], people_video["bea"])]
    assert len(new) == 1 and new[0]["num_frames"] == 2 and new[0]["label"] is None
    refs = await ReferenceService(test_db).list(video)
    # Mel keeps the photos the user chose; the new character gets automatic ones
    assert [r["source"] for r in refs if r["person_id"] == people_video["ana"]] == [f2, f1]
    theirs = [r for r in refs if r["person_id"] == new[0]["id"]]
    assert {r["source"] for r in theirs} <= {f3, f4} and theirs[0]["primary"] and theirs[0]["automatic"]
    async with test_db.execute("SELECT COUNT(*) FROM clusters WHERE video_id=? AND view_mode='person_scene'",
                               [video]) as cursor:
        assert (await cursor.fetchone())[0] == 1  # Scene clusters are left as they are


async def test_refresh_replaces_only_automatic_choices(test_db, people_video, monkeypatch):
    from services.reference_service import VisualReference

    video = people_video["video_id"]
    service = ReferenceService(test_db)
    automatic = VisualReference(role="environment", source_type="frame", source=people_video["names"][7],
                                automatic=True).model_dump()
    await service.insert(video, [automatic])
    await test_db.commit()

    await _refresh(test_db, video, monkeypatch)

    refs = await service.list(video)
    settings = [r for r in refs if r["role"] == "environment"]
    assert len(settings) == 1 and settings[0]["automatic"] and settings[0]["id"] != automatic["id"]
    bea = [r for r in refs if r["person_id"] == people_video["bea"]]
    assert bea and all(r["automatic"] for r in bea)

    # A setting the user saves (changed in any way) is theirs from then on
    async with test_db.execute("SELECT reference_revision FROM videos WHERE id=?", [video]) as cursor:
        revision = (await cursor.fetchone())[0]
    document = await service.save_environment(video, settings, "same", revision)
    assert [r["automatic"] for r in document["references"] if r["role"] == "environment"] == [True]
    changed = [dict(settings[0], crop=[0, 0, 100, 100])]
    document = await service.save_environment(video, changed, "same", revision + 1)
    kept = [r for r in document["references"] if r["role"] == "environment"]
    assert [r["automatic"] for r in kept] == [False]

    await _refresh(test_db, video, monkeypatch)
    assert [r["id"] for r in await service.list(video) if r["role"] == "environment"] == [kept[0]["id"]]


async def test_refresh_keeps_a_video_with_thumbnails_completed(test_db, people_video, monkeypatch):
    video = people_video["video_id"]
    await test_db.execute("INSERT INTO generation_jobs (video_id, status) VALUES (?, 'completed')", [video])
    await test_db.execute("UPDATE videos SET status='analyzing' WHERE id=?", [video])  # as queued
    await test_db.commit()
    await _refresh(test_db, video, monkeypatch)
    assert (await AnalysisService(test_db).get_video(video))["status"] == "completed"

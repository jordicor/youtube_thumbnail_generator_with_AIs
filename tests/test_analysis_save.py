"""Saving a clustering result: each frame once, both views, references by quality."""

import re
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import orjson
import pytest
from PIL import Image

import config
from services.analysis_service import AnalysisService
from services.face_index import centroid_from_blob
from tests.conftest import insert_video, portrait
from utils import VideoOutput

LEFT_FACE = [40.0, 40.0, 100.0, 110.0]
RIGHT_FACE = [200.0, 50.0, 250.0, 110.0]
ANA = (lambda v: v / np.linalg.norm(v))(np.random.default_rng(1).normal(size=512))
BEA = (lambda v: v / np.linalg.norm(v))(np.random.default_rng(2).normal(size=512))
SKIN = (200, 140, 110)


def _name(scene: int, number: int) -> str:
    return f"scene_{scene:03d}_frame_{number:06d}.jpg"


def _frame(output: VideoOutput, scene: int, number: int, quality: float, expression: str = 'neutral') -> dict:
    return {
        'frame_path': str(output.frames_dir / _name(scene, number)),
        'quality_score': quality,
        'expression': expression,
    }


def _face(name: str, direction: np.ndarray, box: list, quality: float) -> dict:
    noise = np.random.default_rng(int(quality)).normal(scale=0.02, size=512)
    return {'frame_path': f'save_video/frames/{name}', 'bbox': box, 'embedding': (direction + noise).tolist(),
            'quality_score': quality, 'expression': 'neutral',
            'portrait_quality': portrait(quality)}


@pytest.fixture
async def clustered_video(test_db, monkeypatch):
    """A video and the clustering result _run_clustering gets for it.

    Ana is person 0 and Bea person 1. The frame of scene 1 / 30 shows both of
    them (Ana on the left); every frame is in colour.
    """
    video_id = await insert_video(test_db, "save_video.mp4", "analyzing")
    output = VideoOutput(Path("save_video.mp4"), Path(config.OUTPUT_DIR))
    output.frames_dir.mkdir(parents=True)
    for scene, number in [(0, 10), (0, 11), (1, 30), (1, 31), (2, 50)]:
        Image.new('RGB', (320, 180), SKIN).save(output.frames_dir / _name(scene, number))
    output.faces_file.write_bytes(orjson.dumps({'all_faces': [
        _face(_name(0, 10), ANA, LEFT_FACE, 50), _face(_name(0, 11), ANA, LEFT_FACE, 90),
        _face(_name(1, 30), ANA, LEFT_FACE, 70), _face(_name(1, 30), BEA, RIGHT_FACE, 71),
        _face(_name(1, 31), ANA, LEFT_FACE, 60), _face(_name(2, 50), BEA, RIGHT_FACE, 40),
    ]}))

    output.scenes_file.write_bytes(orjson.dumps({'scenes': [
        {'index': scene, 'start_frame': numbers[0] - 1, 'end_frame': numbers[-1] + 1,
         'duration': 5, 'frame_indices': numbers}
        for scene, numbers in [(0, [10, 11]), (1, [30, 31]), (2, [50])]
    ]}))

    shared = _frame(output, 1, 30, 70)
    result = {'clusters': [
        {
            'cluster_index': 0,
            'representative_frame': shared['frame_path'],
            'centroid': (ANA * 3).tolist(),  # DBSCAN's mean is not normalized
            'is_picture': False,
            'frames': [_frame(output, 0, 10, 50), _frame(output, 0, 11, 90, 'smiling'), shared, _frame(output, 1, 31, 60)],
        },
        {
            'cluster_index': 1,
            'representative_frame': '',
            'centroid': BEA.tolist(),
            'is_picture': False,
            'frames': [shared, _frame(output, 2, 50, 40)],
        },
    ]}
    monkeypatch.setitem(sys.modules, 'face_clustering', SimpleNamespace(run_clustering_pipeline=lambda **kwargs: result))
    return video_id, output


async def _save(db, video_id: int, output: VideoOutput) -> None:
    await AnalysisService(db)._run_clustering(video_id, output, force=True, eps=0.5, min_samples=3)


async def _assignments(db, video_id: int) -> list:
    async with db.execute("""
        SELECT c.view_mode, c.cluster_index, c.scene_index, vf.frame_path, CASE WHEN vr.id IS NULL THEN 0 ELSE 1 END, vr.position + 1
        FROM cluster_frame_assignments cfa
        JOIN clusters c ON c.id = cfa.cluster_id
        JOIN video_frames vf ON vf.id = cfa.frame_id
        LEFT JOIN visual_references vr ON vr.person_id=c.id AND vr.role='face'
            AND json_extract(vr.data, '$.source')=substr(vf.frame_path, length('save_video/frames/')+1)
        WHERE c.video_id = ?
        ORDER BY c.view_mode, c.cluster_index, vr.position IS NULL, vr.position, vf.frame_path
    """, [video_id]) as cursor:
        rows = await cursor.fetchall()
    return [(view, index, scene, PurePosixPath(path).name, ref, order) for view, index, scene, path, ref, order in rows]


async def test_every_frame_is_saved_once(test_db, clustered_video):
    video_id, output = clustered_video

    await _save(test_db, video_id, output)

    async with test_db.execute(
        "SELECT frame_path, quality_score, expression, scene_index FROM video_frames WHERE video_id = ? ORDER BY id",
        [video_id],
    ) as cursor:
        frames = await cursor.fetchall()
    async with test_db.execute(
        "SELECT reference_format FROM clusters WHERE video_id = ?",
        [video_id],
    ) as cursor:
        defaults = await cursor.fetchall()
    assert defaults and all(row == ('photos',) for row in defaults)
    assert frames == [
        (f"save_video/frames/{_name(0, 10)}", 50, 'neutral', 0),
        (f"save_video/frames/{_name(0, 11)}", 90, 'smiling', 0),
        (f"save_video/frames/{_name(1, 30)}", 70, 'neutral', 1),
        (f"save_video/frames/{_name(1, 31)}", 60, 'neutral', 1),
        (f"save_video/frames/{_name(2, 50)}", 40, 'neutral', 2),
    ]


async def test_both_views_get_their_frames_best_first(test_db, clustered_video):
    video_id, output = clustered_video

    await _save(test_db, video_id, output)

    assignments = await _assignments(test_db, video_id)
    assert len(assignments) == 12
    for view, index in {row[:2] for row in assignments}:
        group = [row for row in assignments if row[:2] == (view, index)]
        assert sum(row[4] for row in group) == 1
    assert [row[3] for row in assignments if row[:2] == ("person", 0) and row[4]] == [_name(0, 11)]


    async with test_db.execute("""
        SELECT c.cluster_index, vf.frame_path
        FROM clusters c JOIN video_frames vf ON vf.id = c.representative_frame_id
        WHERE c.video_id = ? AND c.view_mode = 'person'
        ORDER BY c.cluster_index
    """, [video_id]) as cursor:
        representatives = [(index, PurePosixPath(path).name) for index, path in await cursor.fetchall()]
    # Listing and editor use the same chosen primary.
    assert representatives == [(0, _name(0, 11)), (1, _name(1, 30))]


async def test_each_person_keeps_the_face_dbscan_grouped(test_db, clustered_video):
    video_id, output = clustered_video

    await _save(test_db, video_id, output)

    async with test_db.execute(
        "SELECT view_mode, cluster_index, embedding_centroid FROM clusters WHERE video_id = ? ORDER BY view_mode, cluster_index",
        [video_id],
    ) as cursor:
        faces = {(view, index): centroid_from_blob(blob) for view, index, blob in await cursor.fetchall()}
    # Normalized, and shared by the person's scene clusters (scene 1 shows both)
    assert np.isclose(np.linalg.norm(faces[('person', 0)]), 1)
    assert float(faces[('person', 0)] @ ANA) > 0.99 and float(faces[('person', 1)] @ BEA) > 0.99
    assert float(faces[('person_scene', 1)] @ ANA) > 0.99 and float(faces[('person_scene', 2)] @ BEA) > 0.99


async def test_black_and_white_frames_are_not_picked_as_references(test_db, clustered_video):
    video_id, output = clustered_video
    # Ana's best frame is a black and white insert: its skin has no colour
    data = orjson.loads(output.faces_file.read_bytes())
    for face in data['all_faces']:
        if face['frame_path'].endswith(_name(0, 11)):
            face['portrait_quality']['look']['skin'] = [150.0, 128.0, 129.0]
    output.faces_file.write_bytes(orjson.dumps(data))

    await _save(test_db, video_id, output)

    references = [row for row in await _assignments(test_db, video_id) if row[0] == 'person' and row[1] == 0]
    assert [(name, order) for _, _, _, name, ref, order in references if ref] == [(_name(1, 30), 1)]



async def test_rerunning_from_the_cache_needs_the_faces(test_db, clustered_video):
    video_id, output = clustered_video
    (output.output_dir / 'clustering_result.json').write_bytes(orjson.dumps({'faces_fingerprint': [output.faces_file.stat().st_mtime_ns, output.faces_file.stat().st_size], 'clusters': [
        {'cluster_index': 0, 'frames': [], 'representative_frame': ''}
    ]}))

    with pytest.raises(ValueError, match='lacks the face embedding'):
        await AnalysisService(test_db)._run_clustering(video_id, output, force=False, eps=0.5, min_samples=3)


async def test_saving_again_replaces_the_previous_result(test_db, clustered_video):
    video_id, output = clustered_video

    await _save(test_db, video_id, output)
    first = await _assignments(test_db, video_id)
    await _save(test_db, video_id, output)

    assert await _assignments(test_db, video_id) == first
    async with test_db.execute("SELECT COUNT(*) FROM video_frames WHERE video_id = ?", [video_id]) as cursor:
        assert (await cursor.fetchone())[0] == 5


async def test_the_frame_a_cluster_is_represented_by_is_indexed(test_db):
    # Deleting a frame sets representative_frame_id to NULL in the clusters
    # pointing at it: without this index every cluster is scanned per frame
    async with test_db.execute("PRAGMA index_info(idx_clusters_representative_frame)") as cursor:
        assert [row[2] for row in await cursor.fetchall()] == ['representative_frame_id']


@pytest.mark.parametrize("entry_point", ["analysis", "editor"])
@pytest.mark.parametrize("variation", ["angles", "expressions", "unclear_expression", "duplicates"])
async def test_automatic_faces_choose_useful_angles_or_expressions(test_db, clustered_video, monkeypatch, entry_point, variation):
    from services.person_service import PersonService
    from services.reference_service import ReferenceService

    video_id, output = clustered_video
    faces, frames = [], []
    for number in range(32):
        name = _name(0, 100 + number)
        Image.new("RGB", (320, 180), SKIN).save(output.frames_dir / name)
        face = _face(name, ANA, LEFT_FACE, 100 - number)
        face["pose"] = [0, 0 if variation != "angles" or number < 30 else 40 if number == 30 else 70, 0]
        if variation in {"expressions", "unclear_expression"} and number >= 30:
            face["expression"] = "smiling" if number == 30 else "mouth_open"
            face["portrait_quality"] = portrait(100 - number, **(
                {"mouthSmileLeft": 0.6, "mouthSmileRight": 0.6} if number == 30 else {"lip_gap": 0.3}))
            if variation == "unclear_expression" and number == 30:
                face["portrait_quality"]["sharpness"] = 8.0  # Too blurred to add an expression
        faces.append(face)
        frames.append(_frame(output, 0, 100 + number, 100 - number))
    output.faces_file.write_bytes(orjson.dumps({"all_faces": faces}))
    result = {"clusters": [{"cluster_index": 0, "centroid": ANA.tolist(), "is_picture": False, "frames": frames}]}
    monkeypatch.setitem(sys.modules, "face_clustering", SimpleNamespace(run_clustering_pipeline=lambda **kwargs: result))
    await _save(test_db, video_id, output)
    async with test_db.execute("SELECT id FROM clusters WHERE video_id=? AND view_mode='person'", [video_id]) as cursor:
        person_id = (await cursor.fetchone())[0]
    if entry_point == "analysis":
        selected = [r for r in await ReferenceService(test_db).list(video_id, person_id) if r["role"] == "face"]
    else:
        selected = await PersonService(test_db).automatic_references(
            video_id, "person", person_id, [Path(frame["frame_path"]).name for frame in frames])
    # Identical photos add nothing; a wide open mouth or a blurred face never fill an extra slot
    expected = ([_name(0, 100)] if variation in {"duplicates", "unclear_expression"} else
                [_name(0, 100), _name(0, 130)] if variation == "expressions" else
                [_name(0, 100), _name(0, 130), _name(0, 131)])
    assert [r["source"] for r in selected] == expected
    assert selected[0]["primary"] and selected[0]["view"] == "front"
    if variation == "expressions":
        assert [r["expression"] for r in selected] == ["neutral", "smiling"]


async def test_analysis_selects_body_and_shared_environment_from_cached_evidence(test_db, clustered_video):
    from services.body_references import CACHE_NAME, _fingerprint
    from services.reference_service import ReferenceService, VisualReference

    video_id, output = clustered_video
    for name in (_name(0, 10), _name(2, 50)):
        Image.new("RGB", (320, 360), SKIN).save(output.frames_dir / name)
    body_data = {"fingerprint": _fingerprint(output.output_dir)[0], "frames": {
        _name(0, 10): [{"status": "matched", "face_box": LEFT_FACE, "box": [0, 0, 150, 360],
                         "coverage": "torso", "view": "front", "posture": "seated", "quality": 80}],
        _name(2, 50): [{"status": "matched", "face_box": RIGHT_FACE, "box": [180, 0, 300, 360],
                         "coverage": "torso", "view": "front", "posture": "standing", "quality": 85}],
    }}
    (output.output_dir / CACHE_NAME).write_bytes(orjson.dumps(body_data))
    await _save(test_db, video_id, output)
    service = ReferenceService(test_db)
    async with test_db.execute("SELECT id FROM clusters WHERE video_id=? AND view_mode='person' ORDER BY cluster_index", [video_id]) as cursor:
        people = [row[0] for row in await cursor.fetchall()]
    prepared = await service.prepare(video_id, people, limit=8)
    assert [r["reference_role"] for r in prepared] == ["face", "body", "face", "body", "environment"]
    assert [(r["person_id"], r["source_frame"]) for r in prepared if r["reference_role"] == "body"] == [
        ("person_01", _name(0, 10)), ("person_02", _name(2, 50))]
    # The setting photo is the frame where people cover least of it (only Bea, narrower)
    assert prepared[-1]["source_frame"] == _name(2, 50)

    # Reanalysis keeps the user's shared setting instead of replacing it automatically.
    await test_db.execute("DELETE FROM visual_references WHERE video_id=? AND role='environment'", [video_id])
    manual = VisualReference(role="environment", source_type="frame", source=_name(1, 31)).model_dump()
    await service.insert(video_id, [manual])
    await test_db.commit()
    await _save(test_db, video_id, output)
    settings = [r for r in await service.list(video_id) if r["role"] == "environment"]
    assert len(settings) == 1 and settings[0]["id"] == manual["id"]


@pytest.mark.parametrize("entry_point", ["analysis", "editor", "new_editor"])
@pytest.mark.parametrize("blocked_role", ["primary", "additional"])
async def test_automatic_faces_exclude_overlapping_people(test_db, clustered_video, entry_point, blocked_role):
    from services.person_service import PersonService
    from services.reference_service import ReferenceService

    video_id, output = clustered_video
    data = orjson.loads(output.faces_file.read_bytes())
    blocked_name = _name(0, 11)
    data["all_faces"].append(_face(blocked_name, BEA, [80, 40, 140, 110], 95))
    for face in data["all_faces"]:
        angled = {_name(1, 30), blocked_name} if blocked_role == "additional" else {_name(1, 30)}
        face["pose"] = [0, 40 if Path(face["frame_path"]).name in angled else 0, 0]
    output.faces_file.write_bytes(orjson.dumps(data))
    await _save(test_db, video_id, output)
    async with test_db.execute("SELECT id FROM clusters WHERE video_id=? AND view_mode='person' AND cluster_index=0", [video_id]) as cursor:
        person_id = (await cursor.fetchone())[0]
    service = ReferenceService(test_db)
    members = [_name(0, 10), blocked_name, _name(1, 30), _name(1, 31)]
    if entry_point == "analysis":
        selected = [ref for ref in await service.list(video_id, person_id) if ref["role"] == "face"]
    else:
        selected = await PersonService(test_db).automatic_references(
            video_id, "person_scene" if entry_point == "new_editor" else "person",
            None if entry_point == "new_editor" else person_id, members)
    assert [ref["source"] for ref in selected] == [_name(1, 31), _name(1, 30)]
    for ref in selected:
        assert await service.crop_bytes(video_id, ref)


async def test_automatic_faces_fail_when_no_isolated_face_is_available(test_db, clustered_video):
    from services.person_service import PersonEditError, PersonService

    video_id, output = clustered_video
    await _save(test_db, video_id, output)
    async with test_db.execute("SELECT id FROM clusters WHERE video_id=? AND view_mode='person' AND cluster_index=0", [video_id]) as cursor:
        person_id = (await cursor.fetchone())[0]
    data = orjson.loads(output.faces_file.read_bytes())
    name = _name(0, 11)
    data["all_faces"].append(_face(name, BEA, [80, 40, 140, 110], 95))
    output.faces_file.write_bytes(orjson.dumps(data))
    from i18n.i18n import translate as t
    with pytest.raises(PersonEditError, match=re.escape(t('api.errors.no_clear_face'))):
        await PersonService(test_db).automatic_references(video_id, "person", person_id, [name])


async def test_analysis_picks_the_best_face_there_is_when_every_one_has_problems(test_db, clustered_video):
    video_id, output = clustered_video
    data = orjson.loads(output.faces_file.read_bytes())
    for face in data["all_faces"]:
        face["portrait_quality"]["occlusion"] = 0.6
    output.faces_file.write_bytes(orjson.dumps(data))
    await _save(test_db, video_id, output)
    async with test_db.execute("""SELECT representative_frame_id,representative_frame FROM clusters
                                  WHERE video_id=? AND view_mode='person'""", [video_id]) as cursor:
        assert all(frame_id is not None and path for frame_id, path in await cursor.fetchall())
    refs = [ref for ref in await __import__("services.reference_service", fromlist=["x"]).ReferenceService(test_db).list(video_id)
            if ref["role"] == "face" and ref["primary"]]
    assert refs and all(ref["automatic"] for ref in refs)

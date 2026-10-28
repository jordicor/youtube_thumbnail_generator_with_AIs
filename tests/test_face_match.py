"""Face match of thumbnails: scoring, references, storage, gallery orders and CLI."""

import argparse
import sys
from types import SimpleNamespace

import numpy as np
import orjson
import pytest
from PIL import Image

import config
from services import face_match
from services.face_match import (
    MATCHED,
    MISSING,
    NO_REFERENCE,
    PersonReference,
    face_match_band,
    load_job_references,
    score_existing_thumbnails,
    score_image,
    score_job_thumbnails,
)
from services.face_index import MissingCentroidError, centroid_to_blob
from tests.conftest import insert_video


def _direction(seed: int) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=512)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


ANA = _direction(1)
BEA = _direction(2)
STRANGER = _direction(3)


def _near(direction: np.ndarray, seed: int) -> np.ndarray:
    vector = direction + np.random.default_rng(seed).normal(scale=0.005, size=512)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


class FakeFaceApp:
    """Stands in for InsightFace: returns the faces given for each image file name."""

    def __init__(self):
        self.faces = {}
        self.images = []

    def get(self, image):
        name = self.images.pop(0)
        return [SimpleNamespace(normed_embedding=e) for e in self.faces.get(name, [])]


@pytest.fixture
def face_app(monkeypatch):
    """Detector that maps the image read (by file name, in reading order) to faces."""
    app = FakeFaceApp()
    monkeypatch.setitem(sys.modules, 'face_extraction', SimpleNamespace(get_face_app=lambda: app))
    real_read = face_match.read_image

    def read_image(path):
        app.images.append(path.name)
        return real_read(path)

    monkeypatch.setattr(face_match, 'read_image', read_image)
    return app


def _image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (64, 36), (90, 90, 90)).save(path)
    return path


# =============================================================================
# SCORING ONE IMAGE
# =============================================================================

def test_the_persons_face_matches_and_a_stranger_is_bad(tmp_path, face_app):
    ana = [PersonReference('person_01', 1, ANA)]
    face_app.faces = {'same.png': [_near(ANA, 10)], 'other.png': [STRANGER]}

    same = score_image(_image(tmp_path / 'same.png'), ana)
    assert same.status == MATCHED and same.score > 0.9
    assert face_match_band(same.status, same.score) == 'good'

    other = score_image(_image(tmp_path / 'other.png'), ana)
    assert other.status == MATCHED
    assert face_match_band(other.status, other.score) == 'bad'
    assert other.people == [{'person_id': 'person_01', 'cluster_id': 1, 'similarity': other.score}]


def test_each_person_gets_their_own_face_and_extras_are_ignored(tmp_path, face_app):
    people = [PersonReference('person_01', 1, ANA), PersonReference('person_02', 2, BEA)]
    # Bea first, then Ana, then a stranger in the background
    face_app.faces = {'two.png': [_near(BEA, 11), _near(ANA, 12), STRANGER]}

    result = score_image(_image(tmp_path / 'two.png'), people)
    assert result.status == MATCHED
    assert [p['person_id'] for p in result.people] == ['person_01', 'person_02']
    assert all(p['similarity'] > 0.9 for p in result.people)
    assert result.score == min(p['similarity'] for p in result.people)


@pytest.mark.parametrize('faces, found', [([], []), ([_near(BEA, 13)], ['person_02'])])
def test_a_person_without_a_face_in_the_image_is_missing(tmp_path, face_app, faces, found):
    people = [PersonReference('person_01', 1, ANA), PersonReference('person_02', 2, BEA)]
    face_app.faces = {'few.png': faces}

    result = score_image(_image(tmp_path / 'few.png'), people)
    assert result.status == MISSING and result.score is None
    assert [p['person_id'] for p in result.people if p['similarity'] is not None] == found
    assert face_match_band(result.status, result.score) == 'missing'


def test_people_without_faces_have_nothing_to_compare(tmp_path, face_app):
    # The image is not even read: a missing file would fail otherwise
    result = score_image(tmp_path / 'absent.png', [PersonReference('person_01', 1, None)])
    assert (result.status, result.score, result.people) == (NO_REFERENCE, None, [])


def test_a_person_without_faces_is_left_out_of_the_match(tmp_path, face_app):
    people = [PersonReference('person_01', 1, None), PersonReference('person_02', 2, BEA)]
    face_app.faces = {'one.png': [_near(BEA, 14)]}

    result = score_image(_image(tmp_path / 'one.png'), people)
    assert result.status == MATCHED
    assert [p['person_id'] for p in result.people] == ['person_02']


def test_an_unreadable_image_fails(tmp_path, face_app):
    path = tmp_path / 'broken.png'
    path.write_bytes(b'not an image')
    with pytest.raises(ValueError):
        score_image(path, [PersonReference('person_01', 1, ANA)])


@pytest.mark.parametrize('score, band', [(0.45, 'good'), (0.449, 'doubtful'), (0.25, 'doubtful'), (0.249, 'bad')])
def test_bands_follow_the_thresholds(score, band):
    assert face_match_band(MATCHED, score) == band
    assert face_match_band(None, None) is None
    assert face_match_band(NO_REFERENCE, None) == NO_REFERENCE


# =============================================================================
# JOBS AND STORAGE
# =============================================================================

@pytest.fixture
async def scored_video(test_db):
    """A video whose faces.json shows Ana (f1, f2) and Bea (f3); f4 shows no face.

    People: Ana, Bea, and a "hand" made only of f4. Job 1 asks for Ana and Bea.
    """
    video_id = await insert_video(test_db, 'match_video.mp4', 'completed')
    video_dir = config.OUTPUT_DIR / 'match_video'
    names = ['scene_000_frame_000001.jpg', 'scene_000_frame_000002.jpg',
             'scene_001_frame_000003.jpg', 'scene_001_frame_000004.jpg']
    faces = [
        {'frame_path': f'match_video/frames/{names[0]}', 'bbox': [0, 0, 10, 10], 'embedding': _near(ANA, 1).tolist()},
        {'frame_path': f'match_video/frames/{names[1]}', 'bbox': [0, 0, 10, 10], 'embedding': _near(ANA, 2).tolist()},
        {'frame_path': f'match_video/frames/{names[2]}', 'bbox': [0, 0, 10, 10], 'embedding': _near(BEA, 3).tolist()},
    ]
    video_dir.mkdir(parents=True)
    (video_dir / 'faces.json').write_bytes(orjson.dumps({'all_faces': faces}))

    frame_ids = {}
    for name in names:
        cursor = await test_db.execute(
            "INSERT INTO video_frames (video_id, frame_path) VALUES (?, ?)",
            (video_id, f'match_video/frames/{name}'),
        )
        frame_ids[name] = cursor.lastrowid

    clusters = {}
    people = {'ana': (names[:2], ANA), 'bea': (names[2:3], BEA), 'hand': (names[3:], None)}
    for index, (person, (members, face)) in enumerate(people.items()):
        cursor = await test_db.execute(
            "INSERT INTO clusters (video_id, cluster_index, num_frames, representative_frame, view_mode, embedding_centroid) "
            "VALUES (?, ?, ?, ?, 'person', ?)",
            (video_id, index, len(members), f'match_video/frames/{members[0]}',
             None if face is None else centroid_to_blob(face)),
        )
        clusters[person] = cursor.lastrowid
        for name in members:
            await test_db.execute(
                "INSERT INTO cluster_frame_assignments (cluster_id, frame_id) VALUES (?, ?)",
                (cursor.lastrowid, frame_ids[name]),
            )
    await test_db.commit()
    return {'video_id': video_id, 'dir': video_dir, 'clusters': clusters}


async def _job(db, video, people, status='completed'):
    cursor = await db.execute(
        "INSERT INTO generation_jobs (video_id, num_images, status) VALUES (?, 1, ?)", (video['video_id'], status)
    )
    job_id = cursor.lastrowid
    for order, person in enumerate(people):
        await db.execute(
            "INSERT INTO generation_job_clusters (job_id, cluster_id, person_id, cluster_order) VALUES (?, ?, ?, ?)",
            (job_id, video['clusters'][person], f'person_{order + 1:02d}', order),
        )
    await db.commit()
    return job_id


async def _thumbnail(db, video, job_id, name, raw=False, created_at='2026-01-01 00:00:00'):
    _image(video['dir'] / 'thumbnails' / name)
    raw_path =_image(video['dir'] / 'thumbnails' / 'raw_swap' / name) if raw else None
    cursor = await db.execute(
        "INSERT INTO thumbnails (job_id, image_index, filepath, raw_path, created_at) VALUES (?, 1, ?, ?, ?)",
        (job_id, f'match_video/thumbnails/{name}',
         f'match_video/thumbnails/raw_swap/{name}' if raw_path else None, created_at),
    )
    await db.commit()
    return cursor.lastrowid


async def _stored(db, thumbnail_id):
    async with db.execute(
        "SELECT face_match_status, face_match_score, face_match_people FROM thumbnails WHERE id = ?", [thumbnail_id]
    ) as cursor:
        status, score, people = await cursor.fetchone()
    return status, score, orjson.loads(people) if people else None


async def test_references_follow_the_people_of_the_job(test_db, scored_video):
    job_id = await _job(test_db, scored_video, ['bea', 'ana', 'hand'])
    references = await load_job_references(test_db, job_id)

    assert [r.person_id for r in references] == ['person_01', 'person_02', 'person_03']
    assert float(references[0].embedding @ BEA) > 0.95
    assert float(references[1].embedding @ ANA) > 0.95
    assert references[2].embedding is None  # Its frames show no face


async def test_a_person_with_faces_needs_a_stored_face(test_db, scored_video):
    job_id = await _job(test_db, scored_video, ['ana'])
    await test_db.execute("UPDATE clusters SET embedding_centroid = NULL WHERE id = ?", [scored_video['clusters']['ana']])

    with pytest.raises(MissingCentroidError, match='rerun face analysis'):
        await load_job_references(test_db, job_id)


async def test_references_need_the_face_analysis(test_db, scored_video):
    job_id = await _job(test_db, scored_video, ['ana'])
    (scored_video['dir'] / 'faces.json').unlink()
    with pytest.raises(FileNotFoundError):
        await load_job_references(test_db, job_id)


async def test_a_swapped_thumbnail_is_scored_on_its_image_as_generated(test_db, scored_video, face_app):
    job_id = await _job(test_db, scored_video, ['ana'])
    thumbnail_id = await _thumbnail(test_db, scored_video, job_id, 'a.png', raw=True)
    # The swap pasted Ana back on a stranger: only the raw image tells
    face_app.faces = {'a.png': [STRANGER]}

    bands = await score_job_thumbnails(test_db, job_id)
    assert bands == {'bad': 1}
    status, score, people = await _stored(test_db, thumbnail_id)
    assert status == MATCHED and score < 0.25
    assert people == [{'person_id': 'person_01', 'cluster_id': scored_video['clusters']['ana'], 'similarity': score}]


async def test_job_scoring_stops_when_cancelled(test_db, scored_video, face_app):
    from services.generation_service import GenerationCancelledError

    job_id = await _job(test_db, scored_video, ['ana'])
    thumbnail_id = await _thumbnail(test_db, scored_video, job_id, 'a.png')
    with pytest.raises(GenerationCancelledError):
        await score_job_thumbnails(test_db, job_id, cancellation_check=lambda: True)
    assert (await _stored(test_db, thumbnail_id))[0] is None


async def test_existing_thumbnails_of_finished_jobs_are_scored_once(test_db, scored_video, face_app):
    done = await _job(test_db, scored_video, ['ana'])
    running = await _job(test_db, scored_video, ['ana'], status='generating')
    first = await _thumbnail(test_db, scored_video, done, 'a.png')
    second = await _thumbnail(test_db, scored_video, done, 'b.png')
    busy = await _thumbnail(test_db, scored_video, running, 'c.png')
    face_app.faces = {'a.png': [_near(ANA, 20)], 'b.png': []}

    progress = []
    bands, missing_files = await score_existing_thumbnails(
        test_db, on_scored=lambda n, total: progress.append((n, total))
    )
    assert (bands, missing_files) == ({'good': 1, 'missing': 1}, [])
    assert progress == [(1, 2), (2, 2)]
    assert (await _stored(test_db, first))[0] == MATCHED
    assert (await _stored(test_db, second))[0] == MISSING
    assert (await _stored(test_db, busy))[0] is None  # Its job may still swap it

    # Scored ones are skipped, unless scoring again
    assert await score_existing_thumbnails(test_db) == ({}, [])
    face_app.faces = {'a.png': [STRANGER], 'b.png': [_near(ANA, 21)]}
    assert await score_existing_thumbnails(test_db, video_id=scored_video['video_id'], rescore=True) == (
        {'bad': 1, 'good': 1}, []
    )
    assert await score_existing_thumbnails(test_db, video_id=scored_video['video_id'] + 1, rescore=True) == ({}, [])


async def test_thumbnails_without_image_are_listed_and_left_unscored(test_db, scored_video, face_app):
    job_id = await _job(test_db, scored_video, ['ana'])
    lost = await _thumbnail(test_db, scored_video, job_id, 'lost.png')
    kept = await _thumbnail(test_db, scored_video, job_id, 'kept.png')
    (scored_video['dir'] / 'thumbnails' / 'lost.png').unlink()
    face_app.faces = {'kept.png': [_near(ANA, 22)]}

    progress = []
    bands, missing_files = await score_existing_thumbnails(
        test_db, on_scored=lambda n, total: progress.append((n, total))
    )
    assert bands == {'good': 1}
    assert missing_files == [{
        'thumbnail_id': lost,
        'video_id': scored_video['video_id'],
        'path': str(scored_video['dir'] / 'thumbnails' / 'lost.png'),
    }]
    assert progress == [(1, 2), (2, 2)]
    assert (await _stored(test_db, lost))[0] is None
    assert (await _stored(test_db, kept))[0] == MATCHED

    # A swapped thumbnail is scored on its pre-swap copy: that one must exist
    raw = await _thumbnail(test_db, scored_video, job_id, 'swapped.png', raw=True)
    (scored_video['dir'] / 'thumbnails' / 'raw_swap' / 'swapped.png').unlink()
    _, missing_files = await score_existing_thumbnails(test_db)
    assert [m['thumbnail_id'] for m in missing_files] == [lost, raw]


# =============================================================================
# GALLERY ORDERS
# =============================================================================

@pytest.fixture
async def gallery(test_db, scored_video):
    """Thumbnails of one job, oldest first: good, bad, missing, unscored, doubtful."""
    job_id = await _job(test_db, scored_video, ['ana'])
    scores = [(MATCHED, 0.8), (MATCHED, 0.05), (MISSING, None), (None, None), (MATCHED, 0.3)]
    ids = []
    for day, (status, score) in enumerate(scores, start=1):
        thumbnail_id = await _thumbnail(
            test_db, scored_video, job_id, f'{day}.png', created_at=f'2026-01-0{day} 00:00:00'
        )
        if status:
            people = [{'person_id': 'person_01', 'cluster_id': 1, 'similarity': score}]
            await test_db.execute(
                "UPDATE thumbnails SET face_match_status = ?, face_match_score = ?, face_match_people = ? WHERE id = ?",
                (status, score, orjson.dumps(people).decode(), thumbnail_id),
            )
        ids.append(thumbnail_id)
    await test_db.commit()
    good, bad, missing, unscored, doubtful = ids
    return {'video_id': scored_video['video_id'], 'good': good, 'bad': bad, 'missing': missing,
            'unscored': unscored, 'doubtful': doubtful}


@pytest.mark.parametrize('sort, expected', [
    ('recent', ['doubtful', 'unscored', 'missing', 'bad', 'good']),
    ('failures_last', ['doubtful', 'unscored', 'good', 'missing', 'bad']),
    ('match', ['good', 'doubtful', 'bad', 'missing', 'unscored']),
])
async def test_gallery_orders(client, gallery, sort, expected):
    response = await client.get(f"/api/thumbnails/video/{gallery['video_id']}", params={'sort': sort})
    assert response.status_code == 200
    assert [item['id'] for item in response.json()['thumbnails']] == [gallery[name] for name in expected]

    # Pages follow the same order
    pages = []
    for skip in range(len(expected)):
        page = await client.get(
            f"/api/thumbnails/video/{gallery['video_id']}", params={'sort': sort, 'limit': 1, 'skip': skip}
        )
        pages.extend(item['id'] for item in page.json()['thumbnails'])
    assert pages == [gallery[name] for name in expected]


async def test_gallery_shows_the_face_match(client, gallery):
    response = await client.get(f"/api/thumbnails/video/{gallery['video_id']}")
    by_id = {item['id']: item for item in response.json()['thumbnails']}
    assert by_id[gallery['unscored']]['face_match'] is None
    assert by_id[gallery['good']]['face_match'] == {
        'status': MATCHED, 'band': 'good', 'score': 0.8,
        'people': [{'person_id': 'person_01', 'cluster_id': 1, 'similarity': 0.8}],
    }
    assert by_id[gallery['missing']]['face_match']['band'] == 'missing'


async def test_job_thumbnails_show_the_face_match(client, gallery, test_db):
    async with test_db.execute("SELECT job_id FROM thumbnails WHERE id = ?", [gallery['bad']]) as cursor:
        job_id = (await cursor.fetchone())[0]
    response = await client.get(f"/api/generation/jobs/{job_id}/thumbnails")
    assert response.status_code == 200
    bands = {item['id']: item['face_match'] and item['face_match']['band'] for item in response.json()}
    assert bands[gallery['bad']] == 'bad'
    assert bands[gallery['unscored']] is None


async def test_gallery_rejects_an_unknown_order(client):
    response = await client.get("/api/thumbnails/video/1", params={'sort': 'random'})
    assert response.status_code == 422


# =============================================================================
# CLI
# =============================================================================

async def test_cli_scores_existing_thumbnails(test_db, scored_video, face_app, monkeypatch, capfd):
    import importlib
    import logging
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock

    import utils
    from database import db as database_module

    if 'main' not in sys.modules:
        monkeypatch.setattr(utils, 'setup_logger', lambda *args, **kwargs: logging.getLogger('main'))
    main = importlib.import_module('main')
    monkeypatch.setattr(main, '_open_db', AsyncMock())

    @asynccontextmanager
    async def get_db():
        yield test_db

    monkeypatch.setattr(database_module, 'get_db', get_db)
    job_id = await _job(test_db, scored_video, ['ana'])
    await _thumbnail(test_db, scored_video, job_id, 'a.png')
    face_app.faces = {'a.png': [_near(ANA, 30)]}

    args = argparse.Namespace(video=None, rescore=False, json=True)
    assert await main.cmd_score_thumbnails(args) == 0
    output = orjson.loads(capfd.readouterr().out)
    assert output == {'video_id': None, 'rescore': False, 'scored': 1, 'bands': {'good': 1}, 'missing_files': []}

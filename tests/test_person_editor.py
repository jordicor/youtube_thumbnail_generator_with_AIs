"""Person editor: face matching, listing frames and editing people."""

import io
from pathlib import Path

import numpy as np
import orjson
import pytest
from PIL import Image

import config
from gransabio_prompt_generator import _build_reference_mapping_section
from image_generation import _build_face_source_header
from services.generation_service import GenerationService
from services.face_index import (
    MissingCentroidError,
    centroid_from_blob,
    centroid_to_blob,
    claimed_faces,
    face_crop_box,
    load_face_index,
    match_faces,
    person_centroid,
    portrait_crop_box,
)
from tests.conftest import insert_video, portrait

FRAME_SIZE = (320, 180)
LEFT_FACE = [40.0, 40.0, 100.0, 110.0]
RIGHT_FACE = [200.0, 50.0, 250.0, 110.0]


def _direction(seed: int) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=512)
    return vector / np.linalg.norm(vector)


ANA = _direction(1)
BEA = _direction(2)


def _face(name: str, video_dir: str, direction: np.ndarray, box: list, quality: float, seed: int) -> dict:
    noise = np.random.default_rng(seed).normal(scale=0.02, size=512)
    return {
        'frame_path': f'{video_dir}/frames/{name}',
        'bbox': box,
        'embedding': (direction + noise).tolist(),
        'quality_score': quality,
        'portrait_quality': portrait(quality),
        'expression': 'neutral',
    }


@pytest.fixture
async def people_video(test_db):
    """A video with two people, Ana (person 0) and Bea (person 1).

    f1-f4 show Ana alone, f5 shows Ana (left) and Bea (right), f6-f7 show Bea,
    f8 has no face and f9 shows Ana but clustering left it out.
    """
    video_id = await insert_video(test_db, 'people_video.mp4', 'analyzed')
    video_dir = config.OUTPUT_DIR / 'people_video'
    frames_dir = video_dir / 'frames'
    frames_dir.mkdir(parents=True)
    names = [f'scene_{(i - 1) // 3:03d}_frame_{i:06d}.jpg' for i in range(1, 10)]
    for name in names:
        Image.new('RGB', FRAME_SIZE, (90, 90, 90)).save(frames_dir / name)
    f1, f2, f3, f4, f5, f6, f7, f8, f9 = names

    faces = [_face(name, 'people_video', ANA, LEFT_FACE, 60 + i, i) for i, name in enumerate([f1, f2, f3, f4])]
    faces += [
        _face(f5, 'people_video', ANA, LEFT_FACE, 50, 10),
        _face(f5, 'people_video', BEA, RIGHT_FACE, 90, 11),
        _face(f6, 'people_video', BEA, RIGHT_FACE, 70, 12),
        _face(f7, 'people_video', BEA, RIGHT_FACE, 71, 13),
        _face(f9, 'people_video', ANA, LEFT_FACE, 66, 14),
    ]
    (video_dir / 'faces.json').write_bytes(orjson.dumps({'all_faces': faces}))
    (video_dir / 'scenes.json').write_bytes(orjson.dumps({'scenes': [
        {'index': scene, 'start_frame': scene * 3, 'end_frame': (scene + 1) * 3 + 1,
         'duration': 4, 'frame_indices': list(range(scene * 3 + 1, scene * 3 + 4))}
        for scene in range(3)
    ]}))


    frame_ids = {}
    for name in [f1, f2, f3, f4, f5, f6, f7]:
        cursor = await test_db.execute(
            "INSERT INTO video_frames (video_id, frame_path, quality_score, expression) VALUES (?, ?, ?, ?)",
            (video_id, f'people_video/frames/{name}', 80, 'neutral'),
        )
        frame_ids[name] = cursor.lastrowid

    async def cluster(index, view_mode, members, references, face):
        cursor = await test_db.execute(
            """INSERT INTO clusters (video_id, cluster_index, view_mode, num_frames,
                                     representative_frame, representative_frame_id, embedding_centroid)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (video_id, index, view_mode, len(members), f'people_video/frames/{members[0]}', frame_ids[members[0]],
             centroid_to_blob(face)),
        )
        person_id = cursor.lastrowid
        for name in members:
            await test_db.execute(
                "INSERT INTO cluster_frame_assignments (cluster_id, frame_id) VALUES (?, ?)",
                (person_id, frame_ids[name]),
            )
        from services.reference_service import ReferenceService
        await ReferenceService(test_db).replace_person_faces(video_id, person_id, references)
        return cursor.lastrowid

    ana = await cluster(0, 'person', [f1, f2, f3, f4, f5], [f2, f1], ANA)
    bea = await cluster(1, 'person', [f5, f6, f7], [], BEA)
    scene = await cluster(0, 'person_scene', [f1], [f1], ANA)
    await test_db.commit()
    return {'video_id': video_id, 'dir': video_dir, 'names': names, 'ana': ana, 'bea': bea, 'scene': scene}


# =============================================================================
# FACE INDEX
# =============================================================================

async def test_person_centroid_finds_the_shared_face(people_video):
    f1, f2, f3, f4, f5, f6, *_ = people_video['names']
    index = load_face_index(people_video['dir'] / 'faces.json')
    centroid = person_centroid(index, [f1, f5, f2])

    assert float(centroid @ ANA) > 0.95
    matches = match_faces(index, centroid, [f5, f6, people_video['names'][7]])
    assert matches[f5].box == LEFT_FACE
    assert matches[f5].similarity > 0.9
    assert matches[f6].similarity < 0.3
    assert people_video['names'][7] not in matches  # No detected face


async def test_without_a_person_the_best_face_is_shown(people_video):
    f5 = people_video['names'][4]
    index = load_face_index(people_video['dir'] / 'faces.json')
    match = match_faces(index, None, [f5])[f5]
    assert match.box == RIGHT_FACE  # Bea's face has the better quality
    assert match.similarity is None


def test_two_people_in_every_frame_keep_their_own_faces(tmp_path):
    names = [f'scene_000_frame_{i:06d}.jpg' for i in range(1, 5)]
    faces = []
    for i, name in enumerate(names):
        faces += [_face(name, 'duo', ANA, LEFT_FACE, 60 + i, 100 + i), _face(name, 'duo', BEA, RIGHT_FACE, 80 + i, 200 + i)]
    (tmp_path / 'faces.json').write_bytes(orjson.dumps({'all_faces': faces}))
    index = load_face_index(tmp_path / 'faces.json')

    # From the frames alone they share everything; from each one's face they stay apart
    assert float(person_centroid(index, names, ANA) @ ANA) > 0.95
    assert float(person_centroid(index, names, BEA) @ BEA) > 0.95
    # A new person of these frames takes the faces Ana does not have
    claimed = claimed_faces(index, [(ANA, names)])
    assert float(person_centroid(index, names, claimed=claimed) @ BEA) > 0.95


def test_portrait_crop_leaves_out_the_face_beside():
    # Three face widths around the face and room for the hair, inside the frame
    assert portrait_crop_box(LEFT_FACE, [], 320, 180) == (0, 0, 160, 180)
    # Each side stops halfway to the face beside it
    assert portrait_crop_box(LEFT_FACE, [RIGHT_FACE], 320, 180) == (0, 0, 150, 180)
    assert portrait_crop_box(RIGHT_FACE, [LEFT_FACE], 320, 180) == (150, 0, 300, 180)


def test_face_crop_box_stays_inside_the_frame():
    assert face_crop_box([0, 0, 50, 60], 320, 180) == (0, 0, 108, 108)
    left, top, right, bottom = face_crop_box([280, 120, 320, 180], 320, 180)
    assert right <= 320 and bottom <= 180 and right - left == bottom - top


# =============================================================================
# LISTING
# =============================================================================

async def test_frames_of_a_person_carry_its_face(client, people_video):
    f1, f2, f3, f4, f5, f6, f7, f8, f9 = people_video['names']
    response = await client.get(f"/api/people/{people_video['video_id']}/frames",
                                params={'person_id': people_video['ana']})
    assert response.status_code == 200
    data = response.json()
    frames = {frame['filename']: frame for frame in data['frames']}

    assert data['view_mode'] == 'person'
    assert list(frames) == people_video['names']
    assert frames[f5]['face_box'] == LEFT_FACE
    assert frames[f5]['people'] == [people_video['ana'], people_video['bea']]
    assert frames[f9]['people'] == [] and frames[f9]['similarity'] > 0.9
    assert frames[f6]['similarity'] < 0.3
    assert frames[f8]['face_box'] is None and frames[f8]['quality_score'] is None
    assert (frames[f1]['scene_index'], frames[f1]['frame_number']) == (0, 1)
    assert [person['id'] for person in data['people']] == [people_video['ana'], people_video['bea']]


async def test_match_follows_the_frames_of_a_new_person(client, people_video):
    f1, f2, f3, f4, f5, f6, f7, f8, f9 = people_video['names']
    response = await client.post(f"/api/people/{people_video['video_id']}/frames/match",
                                 json={'view_mode': 'person', 'filenames': [f6, f7]})
    assert response.status_code == 200
    matches = response.json()['matches']
    assert matches[f5]['face_box'] == RIGHT_FACE
    assert matches[f1]['similarity'] < 0.3

    unknown = await client.post(f"/api/people/{people_video['video_id']}/frames/match",
                                json={'view_mode': 'person', 'filenames': ['missing.jpg']})
    assert unknown.status_code == 400


async def test_person_lists_members_and_ordered_references(client, people_video):
    f1, f2, f3, f4, f5, *_ = people_video['names']
    response = await client.get(f"/api/people/{people_video['video_id']}/{people_video['ana']}")
    assert response.status_code == 200
    person = response.json()
    assert person['members'] == [f1, f2, f3, f4, f5]
    assert [ref['source'] for ref in person['references']] == [f2, f1]

    other_video = await client.get(f"/api/people/{people_video['video_id'] + 1}/{people_video['ana']}")
    assert other_video.status_code == 404


async def test_frame_image_and_face_avatar(client, people_video):
    video_id = people_video['video_id']
    image = await client.get(f"/api/people/{video_id}/frames/{people_video['names'][0]}/image")
    assert image.status_code == 200
    assert (await client.get(f"/api/people/{video_id}/frames/..%5Cfaces.json/image")).status_code == 404
    assert (await client.get(f"/api/people/{video_id}/frames/nope.jpg/image")).status_code == 404

    avatar = await client.get(f"/api/people/{video_id}/{people_video['ana']}/face")
    assert avatar.status_code == 200
    assert avatar.headers['content-type'] == 'image/png'
    width, height = Image.open(io.BytesIO(avatar.content)).size
    assert width > 0 and height > 0



def test_face_similarity_is_not_rounded_across_identity_threshold(tmp_path):
    from services.face_index import MIN_IDENTITY_SIMILARITY
    from services.face_selection import select_face_references

    centroid = np.zeros(512, dtype=np.float32)
    centroid[0] = 1
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 0.2998
    embedding[1] = np.sqrt(1 - embedding[0] ** 2)
    name = "scene_000_frame_000001.jpg"
    (tmp_path / "faces.json").write_bytes(orjson.dumps({"all_faces": [{
        "frame_path": name, "bbox": LEFT_FACE, "embedding": embedding.tolist(),
        "quality_score": 99, "pose": [0, 0, 0], "expression": "neutral",
    }]}))
    index = load_face_index(tmp_path / "faces.json")
    matches = match_faces(index, centroid, [name])
    assert matches[name].similarity < MIN_IDENTITY_SIMILARITY
    assert select_face_references(tmp_path, matches, index) == []

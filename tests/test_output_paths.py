"""Stored output paths are relative to OUTPUT_DIR, so the project can move."""

import shutil
from pathlib import Path

import orjson
import pytest

from utils import resolve_output_path, to_output_relative


def output_dir() -> Path:
    import config

    return Path(config.OUTPUT_DIR)


def move_output_folder(monkeypatch, destination: Path) -> None:
    """Simulate moving the project: same files, new OUTPUT_DIR."""
    import config
    from api.routes import thumbnails

    shutil.move(str(output_dir()), str(destination))
    monkeypatch.setattr(config, 'OUTPUT_DIR', destination)
    monkeypatch.setattr(thumbnails, 'OUTPUT_DIR', destination)


# =============================================================================
# PATH HELPERS
# =============================================================================

def test_output_paths_round_trip():
    frame = output_dir() / 'Video A' / 'frames' / 'scene_001_frame_000010.jpg'
    assert to_output_relative(frame) == 'Video A/frames/scene_001_frame_000010.jpg'
    assert resolve_output_path('Video A/frames/scene_001_frame_000010.jpg') == frame


def test_paths_outside_output_dir_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        to_output_relative(tmp_path / 'elsewhere' / 'frame.jpg')
    with pytest.raises(ValueError):
        to_output_relative('Video A/frames/frame.jpg')


@pytest.mark.parametrize('stored', [
    '', '/abs/frame.jpg', 'C:/abs/frame.jpg', '../escape/frame.jpg',
    'Video A\\frames\\frame.jpg',
])
def test_malformed_stored_paths_are_rejected(stored):
    with pytest.raises(ValueError):
        resolve_output_path(stored)


def test_relative_settings_resolve_against_the_project():
    import config

    assert config._project_path('database/thumbnails.db') == config.PROJECT_DIR / 'database' / 'thumbnails.db'
    absolute = config.PROJECT_DIR.anchor + 'elsewhere'
    assert config._project_path(absolute) == Path(absolute)


# =============================================================================
# MOVING THE PROJECT
# =============================================================================

@pytest.mark.asyncio
async def test_thumbnail_is_served_after_moving_the_output_folder(test_db, monkeypatch, tmp_path):
    from contextlib import asynccontextmanager
    from api.routes import thumbnails

    @asynccontextmanager
    async def get_db():
        yield test_db

    monkeypatch.setattr(thumbnails, 'get_db', get_db)
    await test_db.executescript("""
        INSERT INTO videos (id, filename, filepath, status) VALUES (1, 'v.mp4', '/videos/v.mp4', 'completed');
        INSERT INTO generation_jobs (id, video_id, status) VALUES (1, 1, 'completed');
        INSERT INTO thumbnails (id, job_id, image_index, filepath) VALUES (1, 1, 1, 'v/thumbnails/a.png');
    """)
    image = output_dir() / 'v' / 'thumbnails' / 'a.png'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'png')

    assert Path((await thumbnails.get_thumbnail(1)).path) == image.resolve()

    moved = tmp_path / 'moved_project' / 'output'
    move_output_folder(monkeypatch, moved)
    assert Path((await thumbnails.get_thumbnail(1)).path) == (moved / 'v' / 'thumbnails' / 'a.png').resolve()


def test_faces_cache_stays_valid_after_moving_the_output_folder(monkeypatch, tmp_path):
    import face_extraction
    from utils import VideoOutput

    def frames_of(output):
        frames = [output.frames_dir / f'scene_000_frame_00000{i}.jpg' for i in range(2)]
        return frames

    output = VideoOutput(Path('/videos/v.mp4'), output_dir())
    output.frames_dir.mkdir(parents=True)
    for frame in frames_of(output):
        frame.write_bytes(b'jpg')

    def analyze(frame_path):
        return [face_extraction.FaceInfo(
            frame_path=str(frame_path), bbox=[0, 0, 1, 1], confidence=0.9,
            quality_score=50.0, pose=[0, 0, 0], embedding=[0.1] * 512,
            portrait_quality={"eye_openness": 0.3, "score": 50.0, "issues": []},
        )]

    monkeypatch.setattr(face_extraction, 'analyze_frame', analyze)
    face_extraction.extract_faces(frames_of(output), output)
    cached = orjson.loads(output.faces_file.read_bytes())
    assert cached['all_faces'][0]['frame_path'] == 'v/frames/scene_000_frame_000000.jpg'

    moved = tmp_path / 'moved_project' / 'output'
    move_output_folder(monkeypatch, moved)

    def must_not_reanalyze(frame_path):
        raise AssertionError('A valid cache must not be recomputed after moving')

    monkeypatch.setattr(face_extraction, 'analyze_frame', must_not_reanalyze)
    moved_output = VideoOutput(Path('/videos/v.mp4'), moved)
    result = face_extraction.extract_faces(frames_of(moved_output), moved_output)
    assert [face.frame_path for face in result.all_faces] == [str(p) for p in frames_of(moved_output)]


# =============================================================================
# MIGRATION OF ABSOLUTE PATHS
# =============================================================================

LEGACY_PATHS = {
    'C:\\AcertingAI\\project\\output\\Video A\\frames\\scene_001_frame_000010.jpg':
        'Video A/frames/scene_001_frame_000010.jpg',
    # A parent folder that happens to be called "output"
    'D:\\output\\project\\output\\Video B\\frames\\scene_002_frame_000020.jpg':
        'Video B/frames/scene_002_frame_000020.jpg',
    # A video folder that happens to be called "output"
    'E:\\project\\output\\output\\thumbnails\\raw_swap\\t.png':
        'output/thumbnails/raw_swap/t.png',
    '/home/user/project/output/Video C/clusters/cluster_0/preview/r.jpg':
        'Video C/clusters/cluster_0/preview/r.jpg',
}


async def insert_legacy_rows(db, frame_path, thumbnail_path):
    await db.execute(
        "INSERT INTO videos (id, filename, filepath, status) VALUES (1, 'v.mp4', '/videos/v.mp4', 'completed')"
    )
    await db.execute(
        "INSERT INTO clusters (id, video_id, cluster_index, num_frames, representative_frame, view_mode, cluster_type) "
        "VALUES (1, 1, 0, 1, ?, 'person', 'person')", [frame_path]
    )
    await db.execute(
        "INSERT INTO video_frames (id, video_id, frame_path) VALUES (1, 1, ?)", [frame_path]
    )
    await db.execute(
        "INSERT INTO cluster_frames (cluster_id, frame_path) VALUES (1, ?)", [frame_path]
    )
    await db.execute("INSERT INTO generation_jobs (id, video_id, status) VALUES (1, 1, 'completed')")
    await db.execute(
        "INSERT INTO thumbnails (job_id, image_index, filepath, raw_path) VALUES (1, 1, ?, ?)",
        [thumbnail_path, thumbnail_path],
    )
    await db.commit()


async def stored_paths(db):
    values = []
    for table, column in (
        ('video_frames', 'frame_path'), ('cluster_frames', 'frame_path'),
        ('clusters', 'representative_frame'), ('thumbnails', 'filepath'), ('thumbnails', 'raw_path'),
    ):
        async with db.execute(f"SELECT {column} FROM {table}") as cursor:
            values.extend(row[0] for row in await cursor.fetchall())
    return values


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy, relative', list(LEGACY_PATHS.items()))
async def test_migration_converts_absolute_paths_from_a_former_location(test_db, legacy, relative):
    from database.db import _run_migrations

    await insert_legacy_rows(test_db, legacy, legacy)
    await _run_migrations(test_db)
    assert await stored_paths(test_db) == [relative] * 5

    # Already relative values are left alone on later starts
    await _run_migrations(test_db)
    assert await stored_paths(test_db) == [relative] * 5


@pytest.mark.asyncio
async def test_migration_converts_json_caches(test_db):
    from database.db import _run_migrations

    frame = 'C:\\old\\project\\output\\v\\frames\\scene_000_frame_000001.jpg'
    representative = 'C:\\old\\project\\output\\v\\clusters\\cluster_0\\representative.jpg'
    video_dir = output_dir() / 'v'
    (video_dir / 'clusters').mkdir(parents=True)
    caches = {
        video_dir / 'faces.json': {
            'all_faces': [{'frame_path': frame, 'embedding': [0.5]}], 'processed_frames': [frame],
        },
        video_dir / 'clustering_result.json': {
            'clusters': [{'representative_frame': frame, 'top_frames': [frame]}],
        },
        video_dir / 'clusters' / 'clusters.json': {
            'clusters': [{'representative': representative}],
        },
    }
    for path, data in caches.items():
        path.write_bytes(orjson.dumps(data))
    await insert_legacy_rows(test_db, frame, frame.replace('frames', 'thumbnails'))

    await _run_migrations(test_db)

    faces = orjson.loads((video_dir / 'faces.json').read_bytes())
    assert faces == {
        'all_faces': [{'frame_path': 'v/frames/scene_000_frame_000001.jpg', 'embedding': [0.5]}],
        'processed_frames': ['v/frames/scene_000_frame_000001.jpg'],
    }
    clustering = orjson.loads((video_dir / 'clustering_result.json').read_bytes())
    assert clustering['clusters'][0]['top_frames'] == ['v/frames/scene_000_frame_000001.jpg']
    clusters = orjson.loads((video_dir / 'clusters' / 'clusters.json').read_bytes())
    assert clusters['clusters'][0]['representative'] == 'v/clusters/cluster_0/representative.jpg'


@pytest.mark.asyncio
async def test_unrecognized_absolute_path_fails_without_changing_the_database(test_db):
    from database.db import _run_migrations

    unknown = 'C:\\somewhere\\frame.jpg'
    await insert_legacy_rows(test_db, unknown, unknown)
    with pytest.raises(ValueError, match='Cannot locate the output folder'):
        await _run_migrations(test_db)
    assert await stored_paths(test_db) == [unknown] * 5


async def test_output_mount_serves_the_configured_directory(tmp_path, monkeypatch):
    import runpy

    from httpx import ASGITransport, AsyncClient

    import config
    import api.main

    output = tmp_path / "custom" / "generated"
    monkeypatch.setattr(config, "OUTPUT_DIR", output)
    application = runpy.run_path(api.main.__file__)["app"]
    assert output.is_dir()
    (output / "thumbnail.png").write_bytes(b"synthetic thumbnail")

    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://localhost") as client:
        response = await client.get("/output/thumbnail.png")

    assert response.status_code == 200
    assert response.content == b"synthetic thumbnail"

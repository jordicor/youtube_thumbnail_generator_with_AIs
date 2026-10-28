"""Multi-cluster jobs and face swap under the generation ownership rules."""

import argparse
import importlib
import logging
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException
from pydantic import ValidationError
import pytest

from services.face_swap_service import FaceSwapResult
from services.generation_service import GenerationCancelledError, GenerationService
from utils import to_output_relative


@pytest.fixture
async def generation_db(test_db):
    await test_db.execute(
        "INSERT INTO videos (id, filename, filepath, status) "
        "VALUES (1, 'fixture.mp4', '/fake/fixture.mp4', 'analyzed')"
    )
    for cluster_id, cluster_index in ((1, 0), (2, 1)):
        await test_db.execute(
            "INSERT INTO clusters "
            "(id, video_id, cluster_index, num_frames, representative_frame, view_mode, cluster_type) "
            "VALUES (?, 1, ?, 1, 'fixture/frames/frame.png', 'person', 'person')",
            [cluster_id, cluster_index],
        )
    from tests.conftest import provision_uploaded_face
    for cluster_id in [1, 2]:
        await provision_uploaded_face(test_db, cluster_id)
    await test_db.commit()
    return test_db


def inject_database(monkeypatch, module, db):
    @asynccontextmanager
    async def get_db():
        yield db

    monkeypatch.setattr(module, 'get_db', get_db)


async def video_status(db):
    async with db.execute('SELECT status FROM videos WHERE id = 1') as cursor:
        return (await cursor.fetchone())[0]


async def count_rows(db, table):
    async with db.execute(f'SELECT COUNT(*) FROM {table}') as cursor:
        return (await cursor.fetchone())[0]


# =============================================================================
# MIGRATION
# =============================================================================

@pytest.mark.asyncio
async def test_legacy_job_cluster_migration_keeps_jobs_and_thumbnails(test_db, monkeypatch):
    from database.db import _run_migrations

    await test_db.execute("PRAGMA foreign_keys = OFF")
    # Raw string: legacy rows hold Windows paths from a former project location
    await test_db.executescript(r"""
        DROP TABLE generation_jobs;
        CREATE TABLE generation_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            num_images INTEGER DEFAULT 5,
            preferred_expression TEXT,
            status TEXT DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
            FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
        );
        INSERT INTO videos (id, filename, filepath, status) VALUES (1, 'v.mp4', '/fake/v.mp4', 'completed');
        INSERT INTO clusters (id, video_id, cluster_index, num_frames, representative_frame, view_mode, cluster_type)
            VALUES (7, 1, 0, 1, 'C:\old\project\output\v\frames\frame.png', 'person', 'person');
        INSERT INTO generation_jobs (id, video_id, cluster_id, status) VALUES (41, 1, 7, 'completed');
        INSERT INTO thumbnails (job_id, image_index, filepath)
            VALUES (41, 1, 'C:\old\project\output\v\thumbnails\a.png');
    """)
    await test_db.execute("PRAGMA foreign_keys = ON")
    # Windows consoles redirected to a file or pipe use a legacy code page.
    import io
    monkeypatch.setattr(sys, 'stdout', io.TextIOWrapper(io.BytesIO(), encoding='cp1252'))

    await _run_migrations(test_db)

    async with test_db.execute("PRAGMA table_info(generation_jobs)") as cursor:
        assert 'cluster_id' not in [row[1] for row in await cursor.fetchall()]
    assert await GenerationService(test_db).get_job_clusters(41) == [
        {'cluster_id': 7, 'person_id': 'person_01', 'cluster_order': 0}
    ]
    async with test_db.execute("SELECT filepath FROM thumbnails") as cursor:
        assert [row[0] for row in await cursor.fetchall()] == ['v/thumbnails/a.png']
    async with test_db.execute("PRAGMA foreign_key_check") as cursor:
        assert await cursor.fetchall() == []


# =============================================================================
# JOB CREATION
# =============================================================================

@pytest.mark.asyncio
async def test_job_maps_clusters_to_person_ids_in_request_order(generation_db):
    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, [2, 1])
    assert await service.get_job_clusters(job['id']) == [
        {'cluster_id': 2, 'person_id': 'person_01', 'cluster_order': 0},
        {'cluster_id': 1, 'person_id': 'person_02', 'cluster_order': 1},
    ]
    assert await video_status(generation_db) == 'generating'


@pytest.mark.asyncio
async def test_empty_cluster_list_never_claims_video(generation_db):
    with pytest.raises(ValueError):
        await GenerationService(generation_db).create_generation_job(1, [])
    assert await video_status(generation_db) == 'analyzed'
    assert await count_rows(generation_db, 'generation_jobs') == 0


def test_request_requires_at_least_one_cluster():
    from api.routes.generation import GenerationRequest

    with pytest.raises(ValidationError):
        GenerationRequest(cluster_ids=[])


@pytest.mark.asyncio
@pytest.mark.parametrize('cluster_ids, status_code', [([1, 1], 400), ([1, 999], 404), ([3], 404)])
async def test_invalid_cluster_list_never_claims_video(generation_db, monkeypatch, cluster_ids, status_code):
    from api.routes import generation

    # Cluster 3 exists but belongs to another video
    await generation_db.execute(
        "INSERT INTO videos (id, filename, filepath, status) VALUES (2, 'other.mp4', '/fake/other.mp4', 'analyzed')"
    )
    await generation_db.execute(
        "INSERT INTO clusters (id, video_id, cluster_index, num_frames, representative_frame, view_mode) "
        "VALUES (3, 2, 0, 1, 'other/frames/frame.png', 'person')"
    )
    await generation_db.commit()
    inject_database(monkeypatch, generation, generation_db)
    enqueue = AsyncMock()
    monkeypatch.setattr(generation, 'enqueue_generation', enqueue)
    with pytest.raises(HTTPException) as error:
        await generation.start_generation(1, generation.GenerationRequest(cluster_ids=cluster_ids))
    assert error.value.status_code == status_code
    assert await video_status(generation_db) == 'analyzed'
    assert await count_rows(generation_db, 'generation_jobs') == 0
    enqueue.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", [None, False, True])
async def test_start_enqueues_multi_cluster_job_with_face_swap_choice(generation_db, monkeypatch, choice):
    from api.routes import generation

    inject_database(monkeypatch, generation, generation_db)
    enqueue = AsyncMock(return_value='fake-queue-id')
    monkeypatch.setattr(generation, 'enqueue_generation', enqueue)
    result = await generation.start_generation(
        1, generation.GenerationRequest(cluster_ids=[2, 1], **({} if choice is None else {'apply_face_swap': choice}))
    )
    assert enqueue.await_args.kwargs['apply_face_swap'] is (choice is True)
    assert enqueue.await_args.kwargs['restore_faces'] is False
    assert enqueue.await_args.kwargs['include_scene_reference'] is True
    clusters = await GenerationService(generation_db).get_job_clusters(result['job_id'])
    assert [(c['cluster_id'], c['person_id']) for c in clusters] == [(2, 'person_01'), (1, 'person_02')]
    assert await video_status(generation_db) == 'generating'


@pytest.mark.asyncio
async def test_scene_view_cluster_is_used_as_chosen(generation_db, monkeypatch):
    """A person_scene cluster shares its index with a person cluster; its ID picks it exactly."""
    from api.routes import generation

    await generation_db.execute(
        "INSERT INTO clusters (id, video_id, cluster_index, num_frames, representative_frame, view_mode, scene_index) "
        "VALUES (5, 1, 0, 1, 'fixture/frames/scene.png', 'person_scene', 3)"
    )
    await generation_db.commit()
    inject_database(monkeypatch, generation, generation_db)
    monkeypatch.setattr(generation, 'enqueue_generation', AsyncMock(return_value='fake-queue-id'))
    from tests.conftest import provision_uploaded_face
    await provision_uploaded_face(generation_db, 5)
    result = await generation.start_generation(1, generation.GenerationRequest(cluster_ids=[5]))
    clusters = await GenerationService(generation_db).get_job_clusters(result['job_id'])
    assert [(c['cluster_id'], c['person_id']) for c in clusters] == [(5, 'person_01')]


# =============================================================================
# FACE SWAP STAGE
# =============================================================================

@pytest.fixture
async def stub_pipeline(monkeypatch, tmp_path, generation_db):
    """Replace transcription, prompt and image providers and face match scoring with synthetic stages."""
    import config
    from services import face_match

    import numpy as np
    import orjson
    from PIL import Image
    from services.face_index import centroid_to_blob

    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    from services import analysis_service, body_references
    monkeypatch.setattr(analysis_service, 'OUTPUT_DIR', tmp_path)
    # These tests exercise generation ownership, not pose inference on blank fixtures.
    monkeypatch.setattr(body_references, 'ensure_body_index', lambda *args, **kwargs: {'frames': {}})
    frames_dir = tmp_path / 'fixture' / 'frames'
    frames_dir.mkdir(parents=True)
    faces = []
    for cluster_id in (1, 2):
        embedding = np.zeros(512, dtype=np.float32)
        embedding[cluster_id] = 1
        path = frames_dir / f'person_{cluster_id}.png'
        Image.new('RGB', (640, 360), (40 * cluster_id, 80, 110)).save(path)
        faces.append({
            'frame_path': str(path), 'bbox': [200, 80, 300, 200],
            'embedding': embedding.tolist(), 'quality_score': 90, 'expression': 'neutral',
        })
        await generation_db.execute(
            'UPDATE clusters SET embedding_centroid = ? WHERE id = ?',
            [centroid_to_blob(embedding), cluster_id],
        )
        frame = await generation_db.execute(
            "INSERT INTO video_frames (video_id, frame_path, quality_score, expression) VALUES (1, ?, 90, 'neutral')",
            [to_output_relative(path)],
        )
        await generation_db.execute(
            'INSERT INTO cluster_frame_assignments (cluster_id, frame_id) VALUES (?, ?)',
            [cluster_id, frame.lastrowid],
        )
        from services.reference_service import ReferenceService
        await ReferenceService(generation_db).replace_person_faces(1, cluster_id, [path.name])
    (frames_dir.parent / 'faces.json').write_bytes(orjson.dumps({'all_faces': faces}))
    await generation_db.commit()
    monkeypatch.setattr(face_match, 'score_job_thumbnails', AsyncMock(return_value={}))
    stubs = {
        'transcription': SimpleNamespace(transcribe_video=lambda video_path, output: 'transcript'),
        'prompt_generation': SimpleNamespace(generate_thumbnail_images=lambda **kwargs: [SimpleNamespace(reference_mapping=None)]),
        'image_generation': SimpleNamespace(
            generate_thumbnails_from_images=lambda **kwargs: [tmp_path / 'fixture' / 'thumbnails' / 'a.png']
        ),
        'gransabio_prompt_generator': SimpleNamespace(
            PromptGenerationConfig=lambda **kwargs: kwargs,
            get_prompt_history_for_video=lambda video_id: None,
            require_vision_model=AsyncMock(),
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    return tmp_path


@pytest.mark.asyncio
async def test_face_swap_failure_does_not_fail_completed_job(generation_db, stub_pipeline, monkeypatch):
    from services import face_swap_pipeline

    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, [1])
    monkeypatch.setattr(
        face_swap_pipeline, 'apply_face_swap_to_job', AsyncMock(side_effect=RuntimeError('GPU missing'))
    )
    await service.run_generation_pipeline(job['id'])
    assert (await service.get_job(job['id']))['status'] == 'completed'
    assert await video_status(generation_db) == 'completed'


@pytest.mark.asyncio
async def test_the_job_keeps_what_the_image_provider_said(generation_db, stub_pipeline, monkeypatch):
    # Stands in for image_generation's, which the stubbed module does not have
    class ImageProviderError(RuntimeError):
        pass

    reason = 'Image 1 of 1: OpenAI gpt-image-2: rejected by the safety system. safety_violations=[sexual]. (moderation_blocked)'

    def refuse(**kwargs):
        raise ImageProviderError(reason)

    monkeypatch.setitem(sys.modules, 'image_generation', SimpleNamespace(generate_thumbnails_from_images=refuse))
    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, [1])
    with pytest.raises(ImageProviderError):
        await service.run_generation_pipeline(job['id'], image_provider='openai', openai_model='gpt-image-2')
    stored = await service.get_job(job['id'])
    assert (stored['status'], stored['error_message']) == ('error', reason)
    assert (await service.get_job_status(job['id']))['error_message'] == reason


@pytest.mark.asyncio
async def test_face_match_is_scored_before_the_face_swap(generation_db, stub_pipeline, monkeypatch):
    from services import face_match, face_swap_pipeline

    stages = []

    async def score(db, job_id, cancellation_check=None):
        stages.append('face_match')
        return {}

    async def swap(db, job_id, restore_faces=False, cancellation_check=None):
        stages.append('face_swap')
        return {'processed': 0, 'swapped': 0, 'skipped': 0, 'errors': 0}

    monkeypatch.setattr(face_match, 'score_job_thumbnails', score)
    monkeypatch.setattr(face_swap_pipeline, 'apply_face_swap_to_job', swap)
    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, [1])
    await service.run_generation_pipeline(job['id'], apply_face_swap=True)
    assert stages == ['face_match', 'face_swap']
    assert (await service.get_job(job['id']))['status'] == 'completed'


@pytest.mark.asyncio
async def test_face_match_failure_fails_the_job(generation_db, stub_pipeline, monkeypatch):
    from services import face_match, face_swap_pipeline

    monkeypatch.setattr(
        face_match, 'score_job_thumbnails', AsyncMock(side_effect=FileNotFoundError('no face analysis'))
    )
    swap = AsyncMock()
    monkeypatch.setattr(face_swap_pipeline, 'apply_face_swap_to_job', swap)
    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, [1])
    with pytest.raises(FileNotFoundError):
        await service.run_generation_pipeline(job['id'])
    swap.assert_not_called()
    assert (await service.get_job(job['id']))['status'] == 'error'
    assert await video_status(generation_db) == 'analyzed'


@pytest.mark.asyncio
async def test_disabled_face_swap_is_skipped(generation_db, stub_pipeline, monkeypatch):
    from services import face_swap_pipeline

    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, [1])
    swap = AsyncMock()
    monkeypatch.setattr(face_swap_pipeline, 'apply_face_swap_to_job', swap)
    await service.run_generation_pipeline(job['id'], apply_face_swap=False)
    swap.assert_not_called()
    assert (await service.get_job(job['id']))['status'] == 'completed'


@pytest.mark.asyncio
async def test_cancellation_during_face_swap_is_cleaned_before_release(generation_db, stub_pipeline, monkeypatch):
    from services import face_swap_pipeline
    from workers import tasks

    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, [1, 2])
    thumbnails_dir = stub_pipeline / 'fixture' / 'thumbnails'
    thumbnails_dir.mkdir(parents=True)
    for index, name in enumerate(('a.png', 'b.png'), start=1):
        (thumbnails_dir / name).write_bytes(b'generated')
        await generation_db.execute(
            "INSERT INTO thumbnails (job_id, image_index, filepath) VALUES (?, ?, ?)",
            [job['id'], index, to_output_relative(thumbnails_dir / name)],
        )
    await generation_db.commit()

    cancelled = False
    swapped = []

    async def swap_then_cancel(db, thumbnail_path, job_clusters, thresholds=None, restore_faces=False):
        nonlocal cancelled
        assert [c['person_id'] for c in job_clusters] == ['person_01', 'person_02']
        raw_path = thumbnail_path.parent / 'raw_swap' / thumbnail_path.name
        raw_path.parent.mkdir(exist_ok=True)
        raw_path.write_bytes(b'raw')
        swapped.append(thumbnail_path.name)
        # The user cancels while the first thumbnail is being swapped.
        assert await GenerationService(db).cancel_job(job['id'])
        cancelled = True
        return FaceSwapResult(success=True, output_path=thumbnail_path, raw_path=raw_path)

    monkeypatch.setattr(face_swap_pipeline, 'apply_face_swap_to_thumbnail', swap_then_cancel)
    inject_database(monkeypatch, tasks, generation_db)
    for name in ('publish_progress', 'publish_event', 'publish_task_event'):
        monkeypatch.setattr(tasks, name, AsyncMock())
    monkeypatch.setattr(tasks, 'create_sync_cancellation_check', lambda job_id: lambda: cancelled)

    await tasks.run_generation({}, job['id'], apply_face_swap=True)

    assert swapped == ['a.png']
    assert (await service.get_job(job['id']))['status'] == 'cancelled'
    assert await video_status(generation_db) == 'analyzed'
    assert await count_rows(generation_db, 'thumbnails') == 0
    assert list(thumbnails_dir.rglob('*.png')) == []


@pytest.mark.asyncio
async def test_job_face_swap_stops_before_next_thumbnail_when_cancelled(generation_db, monkeypatch):
    from services import face_swap_pipeline

    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, [1])
    await generation_db.execute(
        "INSERT INTO thumbnails (job_id, image_index, filepath) VALUES (?, 1, 'fixture/thumbnails/a.png')",
        [job['id']]
    )
    await generation_db.commit()
    swap = AsyncMock()
    monkeypatch.setattr(face_swap_pipeline, 'apply_face_swap_to_thumbnail', swap)
    with pytest.raises(GenerationCancelledError):
        await face_swap_pipeline.apply_face_swap_to_job(
            generation_db, job['id'], cancellation_check=lambda: True
        )
    swap.assert_not_called()


@pytest.mark.asyncio
async def test_fix_face_refuses_thumbnail_of_running_job(generation_db, monkeypatch):
    import config
    from api.routes import thumbnails

    inject_database(monkeypatch, thumbnails, generation_db)
    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, [1])
    await service.update_job_status(job['id'], 'generating', 95)
    thumbnail = Path(config.OUTPUT_DIR) / 'fixture' / 'thumbnails' / 'a.png'
    raw = thumbnail.parent / 'raw_swap' / 'a.png'
    raw.parent.mkdir(parents=True)
    thumbnail.write_bytes(b'swapped')
    raw.write_bytes(b'raw')
    cursor = await generation_db.execute(
        "INSERT INTO thumbnails (job_id, image_index, filepath, raw_path) VALUES (?, 1, ?, ?)",
        [job['id'], 'fixture/thumbnails/a.png', 'fixture/thumbnails/raw_swap/a.png'],
    )
    await generation_db.commit()

    with pytest.raises(HTTPException) as error:
        await thumbnails.fix_thumbnail_face(cursor.lastrowid)
    assert error.value.status_code == 409
    assert thumbnail.read_bytes() == b'swapped'


# =============================================================================
# CLI GENERATE SUBCOMMAND
# =============================================================================

@pytest.fixture
def cli(monkeypatch, generation_db):
    """Import main.py without its file logger and bind it to the test database."""
    import utils
    from database import db as database_module

    if 'main' not in sys.modules:
        monkeypatch.setattr(utils, 'setup_logger', lambda *args, **kwargs: logging.getLogger('main'))
    main = importlib.import_module('main')
    monkeypatch.setattr(main, '_open_db', AsyncMock())
    inject_database(monkeypatch, database_module, generation_db)
    return main


def generate_args(clusters):
    return argparse.Namespace(
        video=1, clusters=clusters, view_mode='person', num_images=1,
        image_provider='gemini', gemini_model=None, openai_model=None,
        poe_model=None, comfyui_url=None, comfyui_workflow=None, comfyui_config=None,
        force_prompts=False, provider_fallback=False, no_scene_reference=False, json=True,
    )


@pytest.mark.asyncio
async def test_cli_generate_claims_video_once(cli, generation_db, monkeypatch):
    async def complete(self, job_id, **kwargs):
        await self.update_job_status(job_id, 'completed', 100)

    monkeypatch.setattr(GenerationService, 'run_generation_pipeline', complete)
    assert await cli.cmd_generate(generate_args('1,0')) == 0
    clusters = await GenerationService(generation_db).get_job_clusters(1)
    assert [c['cluster_id'] for c in clusters] == [2, 1]
    assert await video_status(generation_db) == 'completed'


@pytest.mark.asyncio
async def test_cli_generate_releases_video_after_cancelled_pipeline(cli, generation_db, monkeypatch):
    async def cancelled_by_user(self, job_id, **kwargs):
        await self.update_job_status(job_id, 'prompting', 30)
        assert await self.cancel_job(job_id)

    monkeypatch.setattr(GenerationService, 'run_generation_pipeline', cancelled_by_user)
    assert await cli.cmd_generate(generate_args('0')) == 1
    assert (await GenerationService(generation_db).get_job(1))['status'] == 'cancelled'
    assert await video_status(generation_db) == 'analyzed'


@pytest.mark.asyncio
async def test_cli_subcommand_closes_init_connection_and_keeps_json_stdout_clean(cli, monkeypatch, capfd):
    from database import db as database_module

    close_db = AsyncMock()
    monkeypatch.setattr(database_module, 'close_db', close_db)

    async def noisy_handler(args):
        print('Database initialized: noise')
        cli._emit_json({'status': 'ok'})
        return 0

    assert await cli._run_subcommand(noisy_handler, argparse.Namespace(json=True)) == 0
    close_db.assert_awaited_once()
    captured = capfd.readouterr()
    assert captured.out == '{\n  "status": "ok"\n}\n'
    assert 'Database initialized: noise' in captured.err


@pytest.mark.asyncio
@pytest.mark.parametrize('clusters, video_state', [('0,0', 'analyzed'), ('0', 'pending')])
async def test_cli_generate_rejects_without_claiming(cli, generation_db, monkeypatch, clusters, video_state):
    await generation_db.execute('UPDATE videos SET status = ? WHERE id = 1', [video_state])
    await generation_db.commit()
    pipeline = AsyncMock()
    monkeypatch.setattr(GenerationService, 'run_generation_pipeline', pipeline)
    assert await cli.cmd_generate(generate_args(clusters)) == 1
    pipeline.assert_not_called()
    assert await video_status(generation_db) == video_state
    assert await count_rows(generation_db, 'generation_jobs') == 0


@pytest.mark.parametrize('problem', ['missing_file', 'corrupt_file', 'no_frames', 'bad_upload', 'too_few_slots'])
async def test_reference_preflight_stops_before_any_paid_stage(generation_db, stub_pipeline, monkeypatch, problem):
    from unittest.mock import Mock

    transcribe = Mock(side_effect=AssertionError('Transcription must not start'))
    prompts = Mock(side_effect=AssertionError('Prompt generation must not start'))
    images = Mock(side_effect=AssertionError('Image generation must not start'))
    monkeypatch.setattr(sys.modules['transcription'], 'transcribe_video', transcribe)
    monkeypatch.setattr(sys.modules['prompt_generation'], 'generate_thumbnail_images', prompts)
    monkeypatch.setattr(sys.modules['image_generation'], 'generate_thumbnails_from_images', images)
    kwargs = {'apply_face_swap': False}
    frame = stub_pipeline / 'fixture' / 'frames' / 'person_1.png'
    if problem == 'missing_file':
        frame.unlink()
    elif problem == 'corrupt_file':
        frame.write_bytes(b'broken image')
    elif problem == 'no_frames':
        await generation_db.execute('DELETE FROM visual_references WHERE person_id = 1')
        await generation_db.commit()
    elif problem == 'bad_upload':
        kwargs.update(reference_image_base64='invalid base64!', reference_image_use_for_prompts=True)
    else:
        kwargs['num_reference_images'] = 1

    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, [1, 2])
    with pytest.raises((ValueError, OSError)):
        await service.run_generation_pipeline(job['id'], **kwargs)
    for call in (transcribe, prompts, images):
        call.assert_not_called()
    assert (await service.get_job(job['id']))['status'] == 'error'


async def test_an_ideas_model_that_does_not_see_images_stops_before_any_paid_stage(
        generation_db, stub_pipeline, monkeypatch):
    from unittest.mock import Mock

    transcribe = Mock(side_effect=AssertionError('Transcription must not start'))
    monkeypatch.setattr(sys.modules['transcription'], 'transcribe_video', transcribe)
    blind = AsyncMock(side_effect=RuntimeError('The model that thinks up the thumbnails (Gemini 3.7) does not see images'))
    monkeypatch.setattr(sys.modules['gransabio_prompt_generator'], 'require_vision_model', blind)

    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, [1])
    with pytest.raises(RuntimeError, match='does not see images'):
        await service.run_generation_pipeline(job['id'], apply_face_swap=False, prompt_provider='google',
                                              prompt_model='gemini-3.7-flash')
    blind.assert_awaited_once_with('google', 'gemini-3.7-flash')
    transcribe.assert_not_called()
    failed = await service.get_job(job['id'])
    assert failed['status'] == 'error' and 'does not see images' in failed['error_message']


@pytest.mark.parametrize('include_style', [False, True])
async def test_mapping_matches_actual_image_attachments(generation_db, stub_pipeline, monkeypatch, include_style):
    import base64
    import io
    from PIL import Image
    from unittest.mock import Mock

    style = io.BytesIO()
    Image.new('RGB', (75, 45), 'blue').save(style, format='PNG')
    image = SimpleNamespace(reference_mapping=None)
    prompts = Mock(return_value=[image])
    images = Mock(return_value=[stub_pipeline / 'result.png'])
    monkeypatch.setattr(sys.modules['prompt_generation'], 'generate_thumbnail_images', prompts)
    monkeypatch.setattr(sys.modules['image_generation'], 'generate_thumbnails_from_images', images)
    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, [2, 1])
    await service.run_generation_pipeline(
        job['id'], apply_face_swap=False, include_scene_reference=False, num_reference_images=2 + int(include_style),
        reference_image_base64=base64.b64encode(style.getvalue()).decode('ascii') if include_style else None,
        reference_image_use_for_prompts=include_style, reference_image_include_in_refs=include_style,
    )
    prompt_mapping = prompts.call_args.kwargs['reference_mapping']
    assert [entry['cluster_id'] for entry in prompt_mapping] == ([None] if include_style else []) + [2, 1]
    paths = images.call_args.kwargs['best_frames']
    mapping = image.reference_mapping
    assert len(paths) == len(mapping) == 2 + int(include_style)
    assert [entry['image_index'] for entry in mapping] == list(range(1, len(paths) + 1))
    assert [entry['cluster_id'] for entry in mapping] == ([None] if include_style else []) + [2, 1]
    offset = 1 + int(include_style)
    assert [(entry['image_index'], entry['person_id']) for entry in mapping if entry['person_id']] == [
        (offset, 'person_01'), (offset + 1, 'person_02'),
    ]
    attached = [path.read_bytes() for path in paths]
    analyzed = [base64.b64decode(value) for value in prompts.call_args.kwargs['reference_images_base64']]
    assert attached == analyzed
    assert prompt_mapping == mapping


def test_restoration_requires_the_swap():
    from api.routes.generation import GenerationRequest

    assert GenerationRequest(cluster_ids=[1]).restore_faces is False
    with pytest.raises(ValidationError, match='requires face swap'):
        GenerationRequest(cluster_ids=[1], apply_face_swap=False, restore_faces=True)


async def test_restoration_choice_reaches_the_worker_and_swap(generation_db, stub_pipeline, monkeypatch):
    from api.routes import generation
    from job_queue import queue
    from services import face_swap_pipeline
    from workers import tasks

    inject_database(monkeypatch, generation, generation_db)
    enqueue = AsyncMock(return_value=SimpleNamespace(job_id='fake-queue-id'))
    monkeypatch.setattr(queue, 'get_arq_pool', AsyncMock(return_value=SimpleNamespace(enqueue_job=enqueue)))
    await generation.start_generation(1, generation.GenerationRequest(cluster_ids=[1], apply_face_swap=True, restore_faces=True))
    assert enqueue.await_args.kwargs['restore_faces'] is True
    job_id = enqueue.await_args.args[1]
    inject_database(monkeypatch, tasks, generation_db)
    for name in ('publish_progress', 'publish_event', 'publish_task_event'):
        monkeypatch.setattr(tasks, name, AsyncMock())
    monkeypatch.setattr(tasks, 'create_sync_cancellation_check', lambda job_id: lambda: False)
    swap = AsyncMock(return_value={'processed': 1, 'swapped': 1, 'skipped': 0, 'errors': 0})
    monkeypatch.setattr(face_swap_pipeline, 'apply_face_swap_to_job', swap)

    await tasks.run_generation({}, job_id, apply_face_swap=True, restore_faces=True)

    assert swap.await_args.kwargs['restore_faces'] is True
    assert (await GenerationService(generation_db).get_job(job_id))['status'] == 'completed'


@pytest.mark.parametrize('restore_faces, missing_original', [(False, False), (True, False), (False, True)])
async def test_fix_face_starts_from_the_original_and_forwards_restoration(
    generation_db, monkeypatch, restore_faces, missing_original,
):
    import config
    from api.routes import thumbnails
    from services import face_swap_pipeline

    inject_database(monkeypatch, thumbnails, generation_db)
    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, [1])
    await service.update_job_status(job['id'], 'completed', 100)
    thumbnail = Path(config.OUTPUT_DIR) / 'fixture' / 'thumbnails' / 'a.png'
    raw = thumbnail.parent / 'raw_swap' / 'a.png'
    raw.parent.mkdir(parents=True)
    thumbnail.write_bytes(b'previous swap')
    if not missing_original:
        raw.write_bytes(b'original generation')
    cursor = await generation_db.execute(
        "INSERT INTO thumbnails (job_id, image_index, filepath, raw_path) VALUES (?, 1, ?, ?)",
        [job['id'], to_output_relative(thumbnail), to_output_relative(raw)],
    )
    await generation_db.commit()

    async def swap(db, path, clusters, restore_faces=False):
        assert path.read_bytes() == b'original generation'
        return FaceSwapResult(
            success=True, output_path=path, raw_path=raw, restoration_applied=restore_faces,
        )

    swap_call = AsyncMock(side_effect=swap)
    monkeypatch.setattr(face_swap_pipeline, 'apply_face_swap_to_thumbnail', swap_call)
    if missing_original:
        with pytest.raises(HTTPException) as error:
            await thumbnails.fix_thumbnail_face(cursor.lastrowid, restore_faces=restore_faces)
        assert error.value.status_code == 404
        assert thumbnail.read_bytes() == b'previous swap'
        swap_call.assert_not_called()
    else:
        result = await thumbnails.fix_thumbnail_face(cursor.lastrowid, restore_faces=restore_faces)
        assert result['success'] is True
        assert swap_call.await_args.kwargs['restore_faces'] is restore_faces
        assert raw.read_bytes() == b'original generation'

@pytest.mark.parametrize('reference_limit', [2, 3])
async def test_body_and_environment_reach_both_providers_without_losing_identity(
    generation_db, stub_pipeline, monkeypatch, reference_limit
):
    from unittest.mock import Mock
    from services import body_references
    from utils import ReferenceImageError
    candidate = {
        'face_box': [200.0, 80.0, 300.0, 200.0], 'box': [50, 0, 550, 360],
        'coverage': 'full_body', 'posture': 'standing', 'status': 'matched', 'quality': 90,
    }
    monkeypatch.setattr(body_references, 'ensure_body_index',
                        lambda *args, **kwargs: {'frames': {'person_1.png': [candidate]}})
    from services.reference_service import ReferenceService, VisualReference
    await ReferenceService(generation_db).insert(1, [
        VisualReference(role="body", person_id=1, source_type="frame", source="person_1.png",
                        crop=candidate["box"], coverage="full_body", posture="standing", identity_confirmed=True).model_dump(),
        VisualReference(role="environment", source_type="frame", source="person_1.png").model_dump(),
    ])
    await generation_db.commit()
    image = SimpleNamespace(reference_mapping=None)
    prompts = Mock(return_value=[image])
    images = Mock(return_value=[stub_pipeline / 'result.png'])
    monkeypatch.setattr(sys.modules['prompt_generation'], 'generate_thumbnail_images', prompts)
    monkeypatch.setattr(sys.modules['image_generation'], 'generate_thumbnails_from_images', images)
    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, [1])
    if reference_limit == 2:
        from i18n.i18n import translate as t
        with pytest.raises(ReferenceImageError, match=re.escape(t('api.errors.references_over_limit', count=3, limit=2))):
            await service.run_generation_pipeline(job['id'], apply_face_swap=False, num_reference_images=reference_limit)
        prompts.assert_not_called()
        images.assert_not_called()
        return
    await service.run_generation_pipeline(job['id'], apply_face_swap=False, num_reference_images=reference_limit)
    mapping = image.reference_mapping
    assert [entry['reference_role'] for entry in mapping] == ['face', 'body', 'environment']
    assert [entry['person_id'] for entry in mapping] == ['person_01', 'person_01', None]
    assert [entry['image_index'] for entry in mapping] == [1, 2, 3]
    assert prompts.call_args.kwargs['reference_mapping'] == mapping
    from PIL import Image
    paths = images.call_args.kwargs['best_frames']
    assert Image.open(paths[1]).size == (500, 360)
    assert Image.open(paths[2]).size == (640, 360)

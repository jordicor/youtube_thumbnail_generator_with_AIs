"""Generation ownership regressions using synthetic records and mocked queues."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import sqlite3
import threading
from unittest.mock import AsyncMock

import aiosqlite
from fastapi import HTTPException
import pytest

from services.generation_service import (
    GenerationCancelledError,
    GenerationNotReadyError,
    GenerationService,
)
from services.task_service import TaskService


@pytest.fixture
async def generation_db(test_db):
    await test_db.execute(
        "INSERT INTO videos (id, filename, filepath, status) "
        "VALUES (1, 'fixture.mp4', '/fake/fixture.mp4', 'analyzed')"
    )
    await test_db.execute(
        "INSERT INTO clusters "
        "(id, video_id, cluster_index, num_frames, representative_frame, view_mode, cluster_type) "
        "VALUES (1, 1, 0, 1, '/fake/frame.png', 'person', 'person')"
    )
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


@pytest.mark.asyncio
async def test_invalid_cluster_never_claims_video(generation_db, monkeypatch):
    from api.routes import generation

    inject_database(monkeypatch, generation, generation_db)
    enqueue = AsyncMock()
    monkeypatch.setattr(generation, 'enqueue_generation', enqueue)
    with pytest.raises(HTTPException) as error:
        await generation.start_generation(1, generation.GenerationRequest(cluster_index=999))
    assert error.value.status_code == 404
    assert await video_status(generation_db) == 'analyzed'
    enqueue.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [None, RuntimeError('Queue unavailable'), asyncio.CancelledError()])
async def test_enqueue_failure_releases_video(generation_db, monkeypatch, failure):
    from api.routes import generation

    inject_database(monkeypatch, generation, generation_db)
    enqueue = AsyncMock(return_value=None) if failure is None else AsyncMock(side_effect=failure)
    monkeypatch.setattr(generation, 'enqueue_generation', enqueue)
    expected = HTTPException if failure is None else type(failure)
    with pytest.raises(expected):
        await generation.start_generation(1, generation.GenerationRequest(cluster_index=0))
    assert await video_status(generation_db) == 'analyzed'
    job = await GenerationService(generation_db).get_job(1)
    assert job['status'] == 'error'


@pytest.mark.asyncio
async def test_cancellation_during_connection_exit_releases_committed_claim(generation_db, monkeypatch):
    from api.routes import generation

    exits = 0

    @asynccontextmanager
    async def interrupted_get_db():
        nonlocal exits
        try:
            yield generation_db
        finally:
            exits += 1
            if exits == 1:
                raise asyncio.CancelledError()

    monkeypatch.setattr(generation, 'get_db', interrupted_get_db)
    enqueue = AsyncMock()
    monkeypatch.setattr(generation, 'enqueue_generation', enqueue)
    with pytest.raises(asyncio.CancelledError):
        await generation.start_generation(1, generation.GenerationRequest(cluster_index=0))
    enqueue.assert_not_called()
    assert (await GenerationService(generation_db).get_job(1))['status'] == 'error'
    assert await video_status(generation_db) == 'analyzed'


@pytest.mark.asyncio
async def test_failed_job_insert_rolls_back_claim(generation_db):
    service = GenerationService(generation_db)
    with pytest.raises(sqlite3.IntegrityError):
        await service.create_generation_job(1, 999)
    assert await video_status(generation_db) == 'analyzed'
    assert not generation_db.in_transaction


@pytest.mark.asyncio
async def test_cancellation_after_commit_compensates_persisted_job(generation_db, monkeypatch):
    service = GenerationService(generation_db)
    original_commit = generation_db.commit
    interrupted = False

    async def commit_then_cancel():
        nonlocal interrupted
        await original_commit()
        if not interrupted:
            interrupted = True
            raise asyncio.CancelledError()

    monkeypatch.setattr(generation_db, 'commit', commit_then_cancel)
    with pytest.raises(asyncio.CancelledError):
        await service.create_generation_job(1, 1)
    assert (await service.get_job(1))['status'] == 'error'
    assert await video_status(generation_db) == 'analyzed'


@pytest.mark.asyncio
@pytest.mark.parametrize('journal_mode', ['DELETE', 'WAL'])
async def test_concurrent_requests_only_create_one_job(generation_db, tmp_path, monkeypatch, journal_mode):
    from api.routes import generation
    from database import db as database_module

    database_path = tmp_path / 'concurrent.sqlite'
    async with aiosqlite.connect(database_path) as target:
        await generation_db.backup(target)
        await target.execute(f'PRAGMA journal_mode = {journal_mode}')
    monkeypatch.setattr(database_module, 'get_db_path', lambda: database_path)
    monkeypatch.setattr(generation, 'get_db', database_module.get_db)
    enqueue = AsyncMock(return_value='fake-queue-id')
    monkeypatch.setattr(generation, 'enqueue_generation', enqueue)

    results = await asyncio.gather(
        generation.start_generation(1, generation.GenerationRequest(cluster_index=0)),
        generation.start_generation(1, generation.GenerationRequest(cluster_index=0)),
        return_exceptions=True,
    )
    assert sum(isinstance(result, dict) for result in results) == 1, repr(results)
    failures = [result for result in results if isinstance(result, HTTPException)]
    assert len(failures) == 1 and failures[0].status_code == 400, repr(results)
    enqueue.assert_awaited_once()
    async with database_module.get_db() as db:
        async with db.execute('SELECT COUNT(*) FROM generation_jobs') as cursor:
            assert (await cursor.fetchone())[0] == 1
        assert await video_status(db) == 'generating'


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal', ['error', 'cancelled', 'completed'])
async def test_terminal_transition_releases_video(generation_db, terminal):
    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, 1)
    assert await service.update_job_status(job['id'], terminal, 100)
    assert await video_status(generation_db) == ('completed' if terminal == 'completed' else 'analyzed')


@pytest.mark.asyncio
async def test_cancelled_regeneration_preserves_completed_video(generation_db):
    service = GenerationService(generation_db)
    first = await service.create_generation_job(1, 1)
    await service.update_job_status(first['id'], 'completed', 100)
    second = await service.create_generation_job(1, 1)
    assert await TaskService(generation_db).cancel_task('generation', second['id'])
    assert await video_status(generation_db) == 'completed'


@pytest.mark.asyncio
async def test_active_cancellation_keeps_claim_until_worker_acknowledges(generation_db):
    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, 1)
    await service.update_job_status(job['id'], 'prompting', 30)
    assert await TaskService(generation_db).cancel_task('generation', job['id'])
    assert (await service.get_job(job['id']))['status'] == 'cancelled'
    assert await video_status(generation_db) == 'generating'
    with pytest.raises(GenerationNotReadyError):
        await service.create_generation_job(1, 1)
    await service.update_job_status(job['id'], 'cancelled')
    assert await video_status(generation_db) == 'analyzed'


@pytest.mark.asyncio
async def test_ambiguous_enqueue_failure_keeps_running_worker_claim(generation_db, monkeypatch):
    from api.routes import generation

    service = GenerationService(generation_db)
    inject_database(monkeypatch, generation, generation_db)

    async def accepted_then_timeout(**kwargs):
        await service.update_job_status(kwargs['job_id'], 'transcribing', 10)
        raise TimeoutError('Queue response was lost')

    monkeypatch.setattr(generation, 'enqueue_generation', accepted_then_timeout)
    with pytest.raises(TimeoutError):
        await generation.start_generation(1, generation.GenerationRequest(cluster_index=0))
    assert (await service.get_job(1))['status'] == 'cancelled'
    assert await video_status(generation_db) == 'generating'


@pytest.mark.asyncio
async def test_stale_worker_cannot_complete_or_release_newer_job(generation_db):
    service = GenerationService(generation_db)
    old = await service.create_generation_job(1, 1)
    assert await service.cancel_job(old['id'])
    current = await service.create_generation_job(1, 1)
    assert not await service.update_job_status(old['id'], 'completed', 100)
    assert not await service.update_job_status(old['id'], 'error')
    with pytest.raises(GenerationCancelledError):
        await service.update_job_status(old['id'], 'transcribing', 10)
    assert (await service.get_job(old['id']))['status'] == 'cancelled'
    assert (await service.get_job(current['id']))['status'] == 'pending'
    assert await video_status(generation_db) == 'generating'


@pytest.mark.asyncio
async def test_old_active_job_does_not_release_newer_owner(generation_db):
    service = GenerationService(generation_db)
    old = await service.create_generation_job(1, 1)
    await generation_db.execute(
        "INSERT INTO generation_jobs (video_id, cluster_id, status) VALUES (1, 1, 'pending')"
    )
    await generation_db.commit()
    await service.update_job_status(old['id'], 'error')
    assert await video_status(generation_db) == 'generating'


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [RuntimeError('Worker failed'), asyncio.CancelledError()])
async def test_worker_failure_releases_video(generation_db, monkeypatch, failure):
    from workers import tasks

    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, 1)
    inject_database(monkeypatch, tasks, generation_db)
    for name in ('publish_progress', 'publish_event', 'publish_task_event'):
        monkeypatch.setattr(tasks, name, AsyncMock())
    monkeypatch.setattr(tasks, 'create_sync_cancellation_check', lambda job_id: lambda: False)
    monkeypatch.setattr(GenerationService, 'run_generation_pipeline', AsyncMock(side_effect=failure))
    with pytest.raises(type(failure)):
        await tasks.run_generation({}, job['id'])
    assert (await service.get_job(job['id']))['status'] == 'error'
    assert await video_status(generation_db) == 'analyzed'


@pytest.mark.asyncio
@pytest.mark.parametrize('executor_fails', [False, True])
async def test_worker_cancellation_waits_for_executor_before_releasing(generation_db, monkeypatch, executor_fails):
    from workers import tasks

    service = GenerationService(generation_db)
    job = await service.create_generation_job(1, 1)
    inject_database(monkeypatch, tasks, generation_db)
    for name in ('publish_progress', 'publish_event', 'publish_task_event'):
        monkeypatch.setattr(tasks, name, AsyncMock())
    monkeypatch.setattr(tasks, 'create_sync_cancellation_check', lambda job_id: lambda: False)
    started = asyncio.Event()
    allow_finish = threading.Event()
    loop = asyncio.get_running_loop()

    def blocking_step():
        loop.call_soon_threadsafe(started.set)
        assert allow_finish.wait(5), 'The synthetic executor was not released'
        if executor_fails:
            raise RuntimeError('The executor failed while stopping')

    async def pipeline(self, job_id, **kwargs):
        await self.update_job_status(job_id, 'prompting', 30)
        await self._run_generation_step(job_id, blocking_step)

    monkeypatch.setattr(GenerationService, 'run_generation_pipeline', pipeline)
    running = asyncio.create_task(tasks.run_generation({}, job['id']))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        running.cancel()
        for _ in range(100):
            if (await service.get_job(job['id']))['status'] == 'cancelled':
                break
            await asyncio.sleep(0.001)
        assert (await service.get_job(job['id']))['status'] == 'cancelled'
        assert not running.done()
        assert await video_status(generation_db) == 'generating'
        with pytest.raises(GenerationNotReadyError):
            await service.create_generation_job(1, 1)
    finally:
        allow_finish.set()
        with pytest.raises(RuntimeError if executor_fails else asyncio.CancelledError):
            await asyncio.wait_for(running, timeout=2)
    assert await video_status(generation_db) == 'analyzed'


@pytest.mark.parametrize('status', ['generating', 'cancelled'])
def test_callback_closes_connection_and_respects_cancellation(tmp_path, monkeypatch, status):
    import services.generation_service as module

    database_path = tmp_path / 'callback.sqlite'
    with sqlite3.connect(database_path) as db:
        db.execute('CREATE TABLE generation_jobs (id INTEGER, progress INTEGER, status TEXT)')
        db.execute('INSERT INTO generation_jobs VALUES (1, 0, ?)', [status])
        db.execute('CREATE TABLE thumbnails (job_id, image_index, filepath, prompt_text, suggested_title, text_overlay)')
    db.close()
    original_connect = sqlite3.connect
    opened = []

    def tracked_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(module.sqlite3, 'connect', tracked_connect)
    info = {'path': Path('/fake/thumbnail.png'), 'image_index': 1}
    if status == 'cancelled':
        with pytest.raises(GenerationCancelledError):
            GenerationService.save_generation_progress(database_path, 1, 70, info)
    else:
        GenerationService.save_generation_progress(database_path, 1, 70, info)
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute('SELECT 1')
    with original_connect(database_path) as check:
        assert check.execute('SELECT COUNT(*) FROM thumbnails').fetchone()[0] == (1 if status == 'generating' else 0)
    check.close()


@pytest.mark.asyncio
async def test_scoped_database_connections_rollback_and_close(tmp_path, monkeypatch):
    from database import db as database_module

    database_path = tmp_path / 'scoped.sqlite'
    monkeypatch.setattr(database_module, 'get_db_path', lambda: database_path)
    async with database_module.get_db() as first:
        await first.execute('CREATE TABLE fixture (id INTEGER)')
        await first.commit()
        async with database_module.get_db() as second:
            assert first is not second
        with pytest.raises(ValueError):
            await second.execute('SELECT 1')
        await first.execute('INSERT INTO fixture VALUES (1)')
    async with database_module.get_db() as check:
        async with check.execute('SELECT COUNT(*) FROM fixture') as cursor:
            assert (await cursor.fetchone())[0] == 0

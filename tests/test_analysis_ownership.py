"""One analysis owns a video, only the worker frees interrupted work, and a
reanalysis deletes nothing it cannot make again. Synthetic records, no queue."""

import asyncio
import sqlite3
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from i18n.i18n import translate as t
from services.analysis_service import ANALYSIS_INTERRUPTED_MESSAGE, AnalysisService
from services.task_service import GENERATION_INTERRUPTED_MESSAGE, TaskService, release_interrupted_tasks
from tests.conftest import insert_cluster, insert_job, insert_video


async def video_row(db, video_id):
    async with db.execute(
        "SELECT status, analysis_run, analysis_previous_status FROM videos WHERE id = ?", [video_id]
    ) as cursor:
        return await cursor.fetchone()


async def data_counts(db, video_id):
    """(people, generation jobs, thumbnails) of a video."""
    counts = []
    for query in (
        "SELECT COUNT(*) FROM clusters WHERE video_id = ?",
        "SELECT COUNT(*) FROM generation_jobs WHERE video_id = ?",
        "SELECT COUNT(*) FROM thumbnails t JOIN generation_jobs j ON j.id = t.job_id WHERE j.video_id = ?",
    ):
        async with db.execute(query, [video_id]) as cursor:
            counts.append((await cursor.fetchone())[0])
    return tuple(counts)


@pytest.fixture
def worker(test_db, monkeypatch):
    """The worker tasks module on the test database."""
    from workers import tasks

    @asynccontextmanager
    async def get_db():
        yield test_db

    monkeypatch.setattr(tasks, "get_db", get_db)
    # The real check opens the database file; these tests use one in memory
    monkeypatch.setattr(tasks, "create_sync_analysis_cancellation_check", lambda video_id, run: lambda: False)
    return tasks


@pytest.fixture
async def real_video(test_db, tmp_path):
    """A video with thumbnails whose file exists: a person, a finished job and a thumbnail."""
    path = tmp_path / "videos" / "real.mp4"
    path.parent.mkdir()
    path.write_bytes(b"synthetic video")
    cursor = await test_db.execute(
        "INSERT INTO videos (filename, filepath, status) VALUES ('real.mp4', ?, 'completed')", [str(path)]
    )
    video_id = cursor.lastrowid
    cluster_id = await insert_cluster(test_db, video_id)
    job_id = await insert_job(test_db, video_id, cluster_id, "completed", progress=100)
    await test_db.execute(
        "INSERT INTO thumbnails (job_id, image_index, filepath) VALUES (?, 1, 'real/thumbnails/a.png')", [job_id]
    )
    await test_db.commit()
    return {"id": video_id, "path": path}


# =============================================================================
# THE CLAIM
# =============================================================================

async def test_only_one_analysis_owns_a_video(test_db):
    service = AnalysisService(test_db)
    video = await insert_video(test_db, "video.mp4", "completed")

    run = await service.claim_analysis(video)

    assert await video_row(test_db, video) == ("analyzing", run, "completed")
    assert await service.claim_analysis(video) is None


@pytest.mark.parametrize("status", ["analyzing_scenes", "transcribing", "generating"])
async def test_a_video_with_a_running_task_cannot_be_claimed(test_db, status):
    video = await insert_video(test_db, "video.mp4", status)

    assert await AnalysisService(test_db).claim_analysis(video) is None
    assert await video_row(test_db, video) == (status, None, None)


async def test_unclaiming_gives_the_video_back_as_it_was(test_db):
    service = AnalysisService(test_db)
    video = await insert_video(test_db, "video.mp4", "completed")
    run = await service.claim_analysis(video)

    assert not await service.unclaim_analysis(video, "the run of another analysis")
    assert await service.unclaim_analysis(video, run)

    assert await video_row(test_db, video) == ("completed", None, None)


def test_the_cancellation_check_stops_an_analysis_that_lost_its_video(tmp_path, monkeypatch):
    import config
    from workers import tasks

    path = tmp_path / "check.sqlite"
    monkeypatch.setattr(config, "DATABASE_PATH", path)

    def store(statement, values=()):
        db = sqlite3.connect(path)
        with db:
            db.execute(statement, values)
        db.close()

    store("CREATE TABLE videos (id INTEGER, status TEXT, analysis_run TEXT)")
    store("INSERT INTO videos VALUES (1, 'analyzing_faces', 'mine')")
    check = tasks.create_sync_analysis_cancellation_check(1, "mine")
    assert check() is False

    # Cancelled by the user, taken by another analysis, freed by a worker start
    for status, run in [("cancelled", "mine"), ("analyzing_faces", "another"), ("pending", None)]:
        store("UPDATE videos SET status = ?, analysis_run = ?", [status, run])
        assert check() is True, (status, run)
    store("DELETE FROM videos")
    assert check() is True


# =============================================================================
# CANCELLING
# =============================================================================

async def test_an_analysis_cancelled_while_queued_never_runs(test_db, worker, monkeypatch):
    video = await insert_video(test_db, "video.mp4", "analyzed")
    run = await AnalysisService(test_db).claim_analysis(video)

    assert await TaskService(test_db).cancel_task("analysis", video)
    assert await video_row(test_db, video) == ("analyzed", None, None)

    analysis = AsyncMock()
    monkeypatch.setattr(AnalysisService, "run_full_analysis", analysis)
    result = await worker.analyze_video({}, video, run)

    assert result == {"status": "skipped", "video_id": video}
    analysis.assert_not_called()
    assert await video_row(test_db, video) == ("analyzed", None, None)


async def test_a_cancelled_running_analysis_keeps_the_video_until_it_stops(test_db, worker, monkeypatch):
    service = AnalysisService(test_db)
    video = await insert_video(test_db, "video.mp4", "pending")
    run = await service.claim_analysis(video)
    seen = {}

    async def cancelled_during_a_step(self, video_id, **kwargs):
        # The user cancels while a step runs and asks for another analysis at once
        assert await TaskService(self.db).cancel_task("analysis", video_id)
        seen["while_stopping"] = await video_row(self.db, video_id)
        seen["second_claim"] = await AnalysisService(self.db).claim_analysis(video_id)

    monkeypatch.setattr(AnalysisService, "run_full_analysis", cancelled_during_a_step)
    result = await worker.analyze_video({}, video, run)

    assert result["status"] == "cancelled"
    assert seen["while_stopping"] == ("cancelled", run, "pending")
    assert seen["second_claim"] is None
    # The job has ended: the video can be analyzed again
    assert await video_row(test_db, video) == ("cancelled", None, None)
    assert await service.claim_analysis(video)


async def test_an_interrupted_analysis_frees_its_video(test_db, worker, monkeypatch):
    video = await insert_video(test_db, "video.mp4", "pending")
    run = await AnalysisService(test_db).claim_analysis(video)
    monkeypatch.setattr(AnalysisService, "run_full_analysis", AsyncMock(side_effect=asyncio.CancelledError()))

    with pytest.raises(asyncio.CancelledError):
        await worker.analyze_video({}, video, run)

    assert await video_row(test_db, video) == ("pending", None, None)
    async with test_db.execute("SELECT error_message FROM videos WHERE id = ?", [video]) as cursor:
        assert (await cursor.fetchone())[0] == ANALYSIS_INTERRUPTED_MESSAGE


@pytest.mark.parametrize("stage, executor_fails", [("scenes", False), ("scenes", True), ("transcription", False)])
async def test_interruption_keeps_analysis_owned_until_its_thread_stops(test_db, worker, monkeypatch,
                                                                     stage, executor_fails):
    from services import analysis_service as analysis

    video = await insert_video(test_db, "synthetic.mp4", "pending")
    service = AnalysisService(test_db)
    run = await service.claim_analysis(video)
    started = asyncio.Event()
    allow_finish = threading.Event()
    loop = asyncio.get_running_loop()

    def blocking_stage(*args):
        loop.call_soon_threadsafe(started.set)
        assert allow_finish.wait(5), "The synthetic stage was not released"
        if executor_fails:
            raise RuntimeError("The stage failed while stopping")
        return (True, []) if stage == "scenes" else ""

    monkeypatch.setattr(analysis, "process_video_scenes", lambda *args: (True, []))
    monkeypatch.setattr(analysis, "process_faces", lambda *args: SimpleNamespace(all_faces=[]))
    monkeypatch.setattr(analysis, "transcribe_video", lambda *args: "")
    monkeypatch.setattr(analysis, "process_video_scenes" if stage == "scenes" else "transcribe_video", blocking_stage)
    running = asyncio.create_task(worker.analyze_video({}, video, run))
    try:
        await asyncio.wait_for(started.wait(), 2)
        for _ in range(2):
            running.cancel()
            await asyncio.sleep(0.01)
            assert not running.done()
            assert (await video_row(test_db, video))[1] == run
            assert await service.claim_analysis(video) is None
    finally:
        allow_finish.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 2)

    assert await video_row(test_db, video) == ("pending", None, None)
    assert await TaskService(test_db).get_task_outcome("analysis", video) == {
        "status": "error", "error_message": ANALYSIS_INTERRUPTED_MESSAGE,
    }


async def test_cancelling_during_transcription_stays_cancelled(test_db, worker, monkeypatch):
    from services import analysis_service as analysis

    video = await insert_video(test_db, "synthetic.mp4", "pending")
    run = await AnalysisService(test_db).claim_analysis(video)
    started = asyncio.Event()
    allow_finish = threading.Event()
    cancelled = threading.Event()
    loop = asyncio.get_running_loop()

    def transcribe(*args):
        loop.call_soon_threadsafe(started.set)
        assert allow_finish.wait(5)
        return "Already transcribed"

    monkeypatch.setattr(analysis, "process_video_scenes", lambda *args: (True, []))
    monkeypatch.setattr(analysis, "process_faces", lambda *args: SimpleNamespace(all_faces=[]))
    monkeypatch.setattr(analysis, "transcribe_video", transcribe)
    monkeypatch.setattr(worker, "create_sync_analysis_cancellation_check", lambda *args: cancelled.is_set)
    running = asyncio.create_task(worker.analyze_video({}, video, run))
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert await TaskService(test_db).cancel_task("analysis", video)
        cancelled.set()
    finally:
        allow_finish.set()
        result = await asyncio.wait_for(running, 2)

    assert result["status"] == "cancelled"
    assert await TaskService(test_db).get_task_outcome("analysis", video) == {
        "status": "cancelled", "error_message": None,
    }


# =============================================================================
# REANALYSIS
# =============================================================================


async def test_reanalyze_queues_the_job_and_deletes_nothing_yet(client, test_db, real_video, monkeypatch):
    enqueue = AsyncMock(return_value="job-1")
    monkeypatch.setattr("api.routes.videos.enqueue_analysis", enqueue)

    response = await client.post(f"/api/videos/{real_video['id']}/reanalyze")

    assert response.status_code == 200, response.text
    status, run, previous = await video_row(test_db, real_video["id"])
    assert (status, previous) == ("analyzing", "completed")
    enqueue.assert_awaited_once_with(video_id=real_video["id"], analysis_run=run, reset=True)
    assert await data_counts(test_db, real_video["id"]) == (1, 1, 1)


async def test_reanalyze_refuses_a_video_whose_file_is_gone(client, test_db, real_video, monkeypatch):
    """The owner moves published videos away: what they have must stay."""
    enqueue = AsyncMock(return_value="job-1")
    monkeypatch.setattr("api.routes.videos.enqueue_analysis", enqueue)
    real_video["path"].unlink()

    response = await client.post(f"/api/videos/{real_video['id']}/reanalyze")

    assert response.status_code == 409
    assert response.json()["detail"] == t("api.errors.video_file_missing")
    enqueue.assert_not_called()
    assert await video_row(test_db, real_video["id"]) == ("completed", None, None)
    assert await data_counts(test_db, real_video["id"]) == (1, 1, 1)


@pytest.mark.parametrize("status, run, reason", [
    ("analyzing_faces", "running", "api.errors.analysis_in_progress"),
    ("analyzing", "queued", "api.errors.analysis_in_progress"),
    ("generating", None, "api.errors.generation_in_progress"),
    ("cancelled", "stopping", "api.errors.analysis_stopping"),
])
@pytest.mark.parametrize("route", ["/api/videos/{id}/reanalyze", "/api/analysis/{id}/start"])
async def test_a_video_with_a_task_is_not_analyzed_again(client, test_db, real_video, monkeypatch,
                                                        route, status, run, reason):
    enqueue = AsyncMock(return_value="job-1")
    monkeypatch.setattr("api.routes.videos.enqueue_analysis", enqueue)
    monkeypatch.setattr("api.routes.analysis.enqueue_analysis", enqueue)
    await test_db.execute(
        "UPDATE videos SET status = ?, analysis_run = ? WHERE id = ?", [status, run, real_video["id"]]
    )
    await test_db.commit()

    response = await client.post(route.format(id=real_video["id"]))

    assert response.status_code == 409
    assert response.json()["detail"] == t(reason)
    enqueue.assert_not_called()
    assert (await video_row(test_db, real_video["id"]))[:2] == (status, run)
    assert await data_counts(test_db, real_video["id"]) == (1, 1, 1)


@pytest.mark.parametrize("route, module", [
    ("/api/videos/{id}/reanalyze", "api.routes.videos"),
    ("/api/analysis/{id}/start", "api.routes.analysis"),
])
async def test_an_analysis_that_cannot_be_queued_changes_nothing(client, test_db, real_video, monkeypatch,
                                                                route, module):
    monkeypatch.setattr(f"{module}.enqueue_analysis", AsyncMock(return_value=None))

    response = await client.post(route.format(id=real_video["id"]))

    assert response.status_code == 500
    assert await video_row(test_db, real_video["id"]) == ("completed", None, None)
    assert await data_counts(test_db, real_video["id"]) == (1, 1, 1)


async def test_a_reanalysis_deletes_the_old_analysis_when_it_starts(test_db, worker, real_video, monkeypatch):
    import config

    folder = config.OUTPUT_DIR / "real"
    for name in ("frames", "thumbnails", "clusters"):
        (folder / name).mkdir(parents=True)
        (folder / name / "old").write_bytes(b"made by the previous analysis")
    run = await AnalysisService(test_db).claim_analysis(real_video["id"])
    seen = {}

    async def analysis(self, video_id, **kwargs):
        seen.update(kwargs, data=await data_counts(self.db, video_id))
        await self.update_video_status(video_id, "analyzed")

    monkeypatch.setattr(AnalysisService, "run_full_analysis", analysis)
    result = await worker.analyze_video({}, real_video["id"], run, reset=True)

    assert result == {"status": "success", "video_id": real_video["id"]}
    assert seen["data"] == (0, 0, 0)
    assert all(seen[stage] for stage in ("force_scenes", "force_faces", "force_clustering", "force_transcription"))
    assert not any((folder / name).exists() for name in ("frames", "thumbnails", "clusters"))
    assert await video_row(test_db, real_video["id"]) == ("analyzed", None, None)


async def test_a_reanalysis_of_a_video_moved_meanwhile_deletes_nothing(test_db, worker, real_video, monkeypatch):
    run = await AnalysisService(test_db).claim_analysis(real_video["id"])
    real_video["path"].unlink()
    analysis = AsyncMock()
    monkeypatch.setattr(AnalysisService, "run_full_analysis", analysis)

    result = await worker.analyze_video({}, real_video["id"], run, reset=True)

    assert result["status"] == "error" and "Nothing was deleted" in result["error"]
    analysis.assert_not_called()
    assert await video_row(test_db, real_video["id"]) == ("completed", None, None)
    assert await data_counts(test_db, real_video["id"]) == (1, 1, 1)


async def test_a_video_with_a_task_is_not_deleted(client, test_db, real_video):
    await test_db.execute("UPDATE videos SET status = 'generating' WHERE id = ?", [real_video["id"]])
    await test_db.commit()

    response = await client.delete(f"/api/videos/{real_video['id']}")

    assert response.status_code == 409
    assert await data_counts(test_db, real_video["id"]) == (1, 1, 1)


# =============================================================================
# WHAT A STOPPED WORKER LEFT
# =============================================================================

async def test_starting_the_web_or_the_cli_leaves_running_tasks_alone(tmp_path, monkeypatch):
    """Their database start-up used to reset the tasks the worker was running."""
    from database import db as database

    path = tmp_path / "live.sqlite"
    monkeypatch.setattr(database, "get_db_path", lambda: path)
    await database.init_db()
    async with database.get_db() as db:
        await db.execute(
            "INSERT INTO videos (filename, filepath, status, analysis_run) "
            "VALUES ('analyzing.mp4', '/videos/analyzing.mp4', 'analyzing_faces', 'run')"
        )
        await db.execute(
            "INSERT INTO videos (filename, filepath, status) VALUES ('generating.mp4', '/videos/generating.mp4', 'generating')"
        )
        await db.commit()
    await database.close_db()

    await database.init_db()
    await database.close_db()

    async with database.get_db() as db:
        async with db.execute("SELECT status, analysis_run, error_message FROM videos ORDER BY id") as cursor:
            assert await cursor.fetchall() == [("analyzing_faces", "run", None), ("generating", None, None)]


async def test_a_worker_start_frees_what_the_stopped_worker_left(test_db):
    service = AnalysisService(test_db)
    tasks = TaskService(test_db)

    running = await insert_video(test_db, "running.mp4", "pending")
    await service.start_analysis_run(running, await service.claim_analysis(running), "analyzing_faces")
    queued = await insert_video(test_db, "queued.mp4", "analyzed")
    queued_run = await service.claim_analysis(queued)
    stopping = await insert_video(test_db, "stopping.mp4", "pending")
    await service.start_analysis_run(stopping, await service.claim_analysis(stopping), "analyzing_scenes")
    assert await tasks.cancel_task("analysis", stopping)

    generating = await insert_video(test_db, "generating.mp4", "generating")
    interrupted_job = await insert_job(test_db, generating, await insert_cluster(test_db, generating), "generating", 60)
    await test_db.execute(
        "INSERT INTO thumbnails (job_id, image_index, filepath) VALUES (?, 1, 'generating/thumbnails/a.png')",
        [interrupted_job],
    )
    waiting = await insert_video(test_db, "waiting.mp4", "generating")
    waiting_job = await insert_job(test_db, waiting, await insert_cluster(test_db, waiting), "pending")
    again = await insert_video(test_db, "again.mp4", "generating")
    again_cluster = await insert_cluster(test_db, again)
    await insert_job(test_db, again, again_cluster, "completed", 100)
    await insert_job(test_db, again, again_cluster, "prompting", 30)
    await test_db.commit()

    assert await release_interrupted_tasks(test_db) == {"analyses": 1, "generations": 2}

    assert await video_row(test_db, running) == ("pending", None, None)
    async with test_db.execute("SELECT error_message FROM videos WHERE id = ?", [running]) as cursor:
        assert (await cursor.fetchone())[0] == ANALYSIS_INTERRUPTED_MESSAGE
    # Still in the queue: the worker will run it
    assert await video_row(test_db, queued) == ("analyzing", queued_run, "analyzed")
    assert await video_row(test_db, stopping) == ("cancelled", None, None)

    async with test_db.execute(
        "SELECT status, error_message FROM generation_jobs WHERE id = ?", [interrupted_job]
    ) as cursor:
        assert await cursor.fetchone() == ("error", GENERATION_INTERRUPTED_MESSAGE)
    assert (await video_row(test_db, generating))[0] == "analyzed"
    # The thumbnails it had finished stay
    assert (await data_counts(test_db, generating))[2] == 1
    async with test_db.execute("SELECT status FROM generation_jobs WHERE id = ?", [waiting_job]) as cursor:
        assert (await cursor.fetchone())[0] == "pending"
    assert (await video_row(test_db, waiting))[0] == "generating"
    assert (await video_row(test_db, again))[0] == "completed"

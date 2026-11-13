"""A task that ends is told as it ended (completed, cancelled or failed with its
reason), and a cancelled one stays in the list, stopping, until its worker lets
go of the video. Synthetic records, no queue."""

import asyncio
from contextlib import asynccontextmanager

import orjson
import pytest

from api.routes import events
from services.analysis_service import ANALYSIS_INTERRUPTED_MESSAGE, AnalysisService
from services.generation_service import GenerationNotReadyError, GenerationService
from services.task_service import TaskService, release_interrupted_tasks
from tests.conftest import insert_cluster, insert_job, insert_video


async def analysis_outcome(db, video_id):
    """(status, analysis_outcome, error_message) of a video."""
    async with db.execute(
        "SELECT status, analysis_outcome, error_message FROM videos WHERE id = ?", [video_id]
    ) as cursor:
        return await cursor.fetchone()


async def running_analysis(db, filename="video.mp4"):
    """A video whose analysis the worker has started: (video id, run)."""
    service = AnalysisService(db)
    video = await insert_video(db, filename, "pending")
    run = await service.claim_analysis(video)
    assert await service.start_analysis_run(video, run, "analyzing_faces")
    return video, run


async def running_generation(db, job_status="generating"):
    """A video that is generating with one job: (video id, job id)."""
    video = await insert_video(db, "video.mp4", "generating")
    job = await insert_job(db, video, await insert_cluster(db, video), job_status, progress=60)
    return video, job


async def add_thumbnail(db, job_id, index):
    await db.execute(
        "INSERT INTO thumbnails (job_id, image_index, filepath) VALUES (?, ?, ?)",
        [job_id, index, f"video/thumbnails/{index}.png"],
    )
    await db.commit()


@pytest.fixture
def worker(test_db, monkeypatch):
    """The worker tasks module on the test database."""
    from workers import tasks

    @asynccontextmanager
    async def get_db():
        yield test_db

    monkeypatch.setattr(tasks, "get_db", get_db)
    monkeypatch.setattr(tasks, "create_sync_analysis_cancellation_check", lambda video_id, run: lambda: False)
    return tasks


class Stream:
    """One of the SSE streams of api/routes/events.py, read event by event."""

    def __init__(self, response):
        self.events = response.body_iterator

    async def next(self):
        """The next event as (name, data). The stream only speaks when something changed."""
        message = await asyncio.wait_for(anext(self.events), timeout=2)
        name, data = message.strip().split("\n")
        return name.removeprefix("event: "), orjson.loads(data.removeprefix("data: "))

    async def ended(self):
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(self.events), timeout=2)
        return True


class Browser:
    """A page that stays connected."""

    async def is_disconnected(self):
        return False


@pytest.fixture
def streams(test_db, monkeypatch):
    """Open the streams on the test database, polling without waiting."""

    @asynccontextmanager
    async def get_db():
        yield test_db

    monkeypatch.setattr(events, "get_db", get_db)
    monkeypatch.setattr(events, "POLL_SECONDS", 0)
    queue = {"status": None}
    monkeypatch.setattr(events.queue_health, "status", lambda: queue["status"])

    class Streams:
        queue_status = queue

        async def tasks(self):
            stream = Stream(await events.stream_all_tasks_status(Browser()))
            name, data = await stream.next()
            assert name == "tasks_snapshot"
            return stream, data["tasks"]

        async def generation(self, job_id):
            return Stream(await events.stream_generation_status(job_id, Browser()))

    return Streams()


# =============================================================================
# HOW AN ANALYSIS ENDED
# =============================================================================

async def test_an_analysis_in_progress_has_no_outcome(test_db):
    video = await insert_video(test_db, "video.mp4", "completed")
    await test_db.execute("UPDATE videos SET analysis_outcome = 'completed' WHERE id = ?", [video])

    await AnalysisService(test_db).claim_analysis(video)

    assert await analysis_outcome(test_db, video) == ("analyzing", None, None)


async def test_cancelling_a_queued_analysis_is_its_outcome(test_db):
    video = await insert_video(test_db, "video.mp4", "completed")
    await AnalysisService(test_db).claim_analysis(video)

    assert await TaskService(test_db).cancel_task("analysis", video)

    assert await analysis_outcome(test_db, video) == ("completed", "cancelled", None)


async def test_an_analysis_that_could_not_start_keeps_its_reason(test_db):
    service = AnalysisService(test_db)
    video = await insert_video(test_db, "video.mp4", "completed")
    run = await service.claim_analysis(video)

    assert await service.unclaim_analysis(video, run, error="Video file not found")

    # The video is as it was: only the reason says that something went wrong
    assert await analysis_outcome(test_db, video) == ("completed", "error", "Video file not found")


@pytest.mark.parametrize("final_status, outcome", [
    ("analyzed", "completed"),
    ("completed", "completed"),
    ("cancelled", "cancelled"),
    ("error", "error"),
])
async def test_the_worker_records_how_its_analysis_ended(test_db, final_status, outcome):
    video, run = await running_analysis(test_db)
    service = AnalysisService(test_db)
    await service.update_video_status(video, final_status)

    await service.release_analysis(video, run)

    assert (await analysis_outcome(test_db, video))[:2] == (final_status, outcome)


async def test_an_interrupted_analysis_ends_in_error(test_db):
    video, run = await running_analysis(test_db)

    await AnalysisService(test_db).interrupt_analysis(video, run)

    assert await analysis_outcome(test_db, video) == ("pending", "error", ANALYSIS_INTERRUPTED_MESSAGE)


@pytest.mark.parametrize("status, outcome", [("analyzed", "completed"), ("error", "error")])
async def test_worker_restart_frees_a_finished_analysis_without_rewriting_its_result(test_db, status, outcome):
    video, _ = await running_analysis(test_db)
    service = AnalysisService(test_db)
    await service.update_video_status(video, status, "Synthetic failure" if status == "error" else None)

    assert await release_interrupted_tasks(test_db) == {"analyses": 0, "generations": 0}
    row = await service.get_video(video)
    assert (row["status"], row["analysis_outcome"], row["analysis_run"]) == (status, outcome, None)
    assert await service.claim_analysis(video)


async def test_a_worker_start_records_what_the_stopped_worker_left(test_db):
    interrupted, _ = await running_analysis(test_db, "interrupted.mp4")
    stopping, _ = await running_analysis(test_db, "stopping.mp4")
    assert await TaskService(test_db).cancel_task("analysis", stopping)

    await release_interrupted_tasks(test_db)

    assert (await analysis_outcome(test_db, interrupted))[:2] == ("pending", "error")
    assert (await analysis_outcome(test_db, stopping))[:2] == ("cancelled", "cancelled")


async def test_a_refused_reanalysis_says_why(test_db, worker, tmp_path):
    """The video file went away between the request and the worker: nothing is deleted."""
    cursor = await test_db.execute(
        "INSERT INTO videos (filename, filepath, status) VALUES ('gone.mp4', ?, 'completed')",
        [str(tmp_path / "gone.mp4")],
    )
    video = cursor.lastrowid
    run = await AnalysisService(test_db).claim_analysis(video)

    result = await worker.analyze_video({}, video, run, reset=True)

    status, outcome, reason = await analysis_outcome(test_db, video)
    assert (status, outcome) == ("completed", "error")
    assert reason == result["error"] and "Nothing was deleted" in reason
    assert await TaskService(test_db).get_task_outcome("analysis", video) == {
        "status": "error", "error_message": reason,
    }


# =============================================================================
# STOPPING: CANCELLED, BUT THE WORKER HAS NOT LET GO YET
# =============================================================================

async def test_a_cancelled_analysis_is_stopping_until_the_worker_releases_it(test_db):
    tasks = TaskService(test_db)
    video, run = await running_analysis(test_db)

    assert await tasks.cancel_task("analysis", video)
    [task] = await tasks.get_active_tasks()

    assert (task["type"], task["id"], task["status"]) == ("analysis", video, TaskService.STOPPING)
    assert (await tasks.get_task_for_video(video))["status"] == TaskService.STOPPING

    await AnalysisService(test_db).release_analysis(video, run)

    assert await tasks.get_active_tasks() == []
    assert await tasks.get_task_outcome("analysis", video) == {"status": "cancelled", "error_message": None}


async def test_a_video_cancelled_before_is_not_a_task(test_db):
    await insert_video(test_db, "video.mp4", "cancelled")

    assert await TaskService(test_db).get_active_tasks() == []


async def test_a_cancelled_generation_is_stopping_until_the_worker_releases_it(test_db):
    tasks = TaskService(test_db)
    generation = GenerationService(test_db)
    _, job = await running_generation(test_db)

    assert await tasks.cancel_task("generation", job)
    [task] = await tasks.get_active_tasks()

    assert (task["type"], task["id"], task["status"]) == ("generation", job, TaskService.STOPPING)
    assert (await generation.get_job_status(job))["stopping"] is True

    # What the worker does when its job has stopped (the finally of run_generation)
    await generation.update_job_status(job, "cancelled")

    assert await tasks.get_active_tasks() == []
    assert (await generation.get_job_status(job))["stopping"] is False
    assert await tasks.get_task_outcome("generation", job) == {"status": "cancelled", "error_message": None}


async def test_a_generation_cancelled_while_queued_stops_at_once(test_db):
    tasks = TaskService(test_db)
    _, job = await running_generation(test_db, job_status="pending")

    assert await tasks.cancel_task("generation", job)

    assert await tasks.get_active_tasks() == [] and await tasks.get_pending_tasks() == []
    assert (await GenerationService(test_db).get_job_status(job))["stopping"] is False


async def test_an_older_cancelled_job_is_not_stopping_while_a_new_one_generates(test_db):
    video, old_job = await running_generation(test_db, job_status="cancelled")
    cluster = await insert_cluster(test_db, video, cluster_index=1)
    new_job = await insert_job(test_db, video, cluster, "generating")

    tasks = await TaskService(test_db).get_active_tasks()

    assert [(task["id"], task["status"]) for task in tasks] == [(new_job, "generating")]
    assert (await GenerationService(test_db).get_job_status(old_job))["stopping"] is False


# =============================================================================
# HOW A TASK ENDED
# =============================================================================

@pytest.mark.parametrize("job_status, error, outcome", [
    ("completed", None, {"status": "completed", "error_message": None}),
    ("cancelled", None, {"status": "cancelled", "error_message": None}),
    ("error", "Image 2 of 5: gemini: blocked", {"status": "error", "error_message": "Image 2 of 5: gemini: blocked"}),
])
async def test_a_generation_ends_as_its_job_says(test_db, job_status, error, outcome):
    _, job = await running_generation(test_db)
    await GenerationService(test_db).update_job_status(job, job_status, error_message=error)

    assert await TaskService(test_db).get_task_outcome("generation", job) == outcome


async def test_a_task_that_left_nothing_behind_counts_as_cancelled(test_db):
    tasks = TaskService(test_db)
    never_analyzed = await insert_video(test_db, "video.mp4", "pending")

    cancelled = {"status": "cancelled", "error_message": None}
    assert await tasks.get_task_outcome("analysis", never_analyzed) == cancelled
    assert await tasks.get_task_outcome("analysis", 9999) == cancelled
    assert await tasks.get_task_outcome("generation", 9999) == cancelled


async def test_the_outcome_route_answers_a_page_whose_stream_was_cut(client, test_db):
    _, job = await running_generation(test_db)
    await GenerationService(test_db).update_job_status(job, "error", error_message="Gran Sabio did not answer")

    response = await client.get(f"/api/tasks/generation/{job}/outcome")

    assert response.status_code == 200
    assert response.json() == {"status": "error", "error_message": "Gran Sabio did not answer"}
    assert (await client.get("/api/tasks/backup/1/outcome")).status_code == 422


# =============================================================================
# THE TASK STREAM
# =============================================================================

@pytest.mark.parametrize("final_status, expected_event", [("analyzed", "task_completed"), ("error", "task_error")])
async def test_analysis_outcome_is_visible_before_the_worker_releases_it(test_db, worker, streams, monkeypatch,
                                                                      final_status, expected_event):
    video = await insert_video(test_db, "video.mp4", "pending")
    run = await AnalysisService(test_db).claim_analysis(video)
    stream, _ = await streams.tasks()
    releasing = asyncio.Event()
    allow_finish = asyncio.Event()
    release = AnalysisService.release_analysis
    error = "Synthetic transcription failure" if final_status == "error" else None

    async def finish_analysis(self, video_id, **kwargs):
        await self.update_video_status(video_id, final_status, error)

    async def held_release(self, video_id, analysis_run):
        # The analysis has ended and the worker is about to give up the video
        releasing.set()
        await allow_finish.wait()
        await release(self, video_id, analysis_run)

    monkeypatch.setattr(AnalysisService, "run_full_analysis", finish_analysis)
    monkeypatch.setattr(AnalysisService, "release_analysis", held_release)
    running = asyncio.create_task(worker.analyze_video({}, video, run))
    try:
        await asyncio.wait_for(releasing.wait(), 2)
        assert (await AnalysisService(test_db).get_video(video))["analysis_run"] == run
        name, task = await stream.next()
        assert (name, task["error_message"]) == (expected_event, error)
        if final_status == "analyzed":
            cluster = await insert_cluster(test_db, video)
            with pytest.raises(GenerationNotReadyError):
                await GenerationService(test_db).create_generation_job(video, [cluster])
    finally:
        allow_finish.set()
        await asyncio.wait_for(running, 2)
        await stream.events.aclose()


async def test_the_stream_tells_a_completed_analysis(test_db, streams):
    video, run = await running_analysis(test_db)
    stream, tasks = await streams.tasks()
    assert [task["status"] for task in tasks] == ["analyzing_faces"]

    service = AnalysisService(test_db)
    await service.update_video_status(video, "analyzed")
    await service.release_analysis(video, run)

    name, task = await stream.next()
    assert name == "task_completed"
    assert (task["type"], task["id"], task["video_name"], task["status"]) == ("analysis", video, "video.mp4", "completed")


async def test_the_stream_tells_a_failed_analysis_with_its_reason(test_db, streams):
    video, run = await running_analysis(test_db)
    stream, _ = await streams.tasks()

    service = AnalysisService(test_db)
    await service.update_video_status(video, "error", "transcription_error: ElevenLabs: HTTP 401")
    await service.release_analysis(video, run)

    name, task = await stream.next()
    assert name == "task_error"
    assert task["error_message"] == "transcription_error: ElevenLabs: HTTP 401"


async def test_the_stream_keeps_a_cancelled_analysis_until_its_worker_stops(test_db, streams):
    video, run = await running_analysis(test_db)
    stream, _ = await streams.tasks()

    assert await TaskService(test_db).cancel_task("analysis", video)
    name, task = await stream.next()
    assert (name, task["status"]) == ("task_progress", TaskService.STOPPING)

    await AnalysisService(test_db).release_analysis(video, run)
    name, task = await stream.next()
    assert (name, task["status"]) == ("task_cancelled", "cancelled")


async def test_the_stream_tells_an_analysis_cancelled_in_the_queue_as_cancelled(test_db, streams):
    """Its video goes back to 'completed', which used to be told as a completed analysis."""
    video = await insert_video(test_db, "video.mp4", "completed")
    await AnalysisService(test_db).claim_analysis(video)
    stream, tasks = await streams.tasks()
    assert [task["status"] for task in tasks] == ["analyzing"]

    assert await TaskService(test_db).cancel_task("analysis", video)

    name, _ = await stream.next()
    assert name == "task_cancelled"


async def test_the_stream_tells_a_failed_generation_with_its_reason(test_db, streams):
    _, job = await running_generation(test_db)
    stream, _ = await streams.tasks()

    await GenerationService(test_db).update_job_status(job, "error", error_message="Image 1 of 5: openai: safety")

    name, task = await stream.next()
    assert name == "task_error"
    assert (task["type"], task["id"], task["error_message"]) == ("generation", job, "Image 1 of 5: openai: safety")


async def test_the_stream_follows_a_task_from_start_to_end(test_db, streams):
    stream, tasks = await streams.tasks()
    assert tasks == []

    _, job = await running_generation(test_db, job_status="pending")
    assert (await stream.next())[0] == "task_started"

    generation = GenerationService(test_db)
    await generation.update_job_status(job, "generating", 40)
    name, task = await stream.next()
    assert (name, task["status"], task["progress"]) == ("task_progress", "generating", 40)

    await generation.update_job_status(job, "completed", 100)
    assert (await stream.next())[0] == "task_completed"


async def test_the_stream_says_when_the_queue_cannot_run_tasks(test_db, streams):
    stream, _ = await streams.tasks()

    streams.queue_status["status"] = "worker_down"
    assert await stream.next() == ("queue_status", {"status": "worker_down"})

    # Only changes are told
    streams.queue_status["status"] = "ok"
    assert await stream.next() == ("queue_status", {"status": "ok"})


# =============================================================================
# THE STREAM OF ONE GENERATION JOB
# =============================================================================

async def test_a_cancelled_job_keeps_its_stream_open_until_the_worker_stops(test_db, streams):
    """The thumbnail being made when it was cancelled still reaches the page."""
    generation = GenerationService(test_db)
    _, job = await running_generation(test_db)
    await add_thumbnail(test_db, job, 1)
    stream = await streams.generation(job)
    assert (await stream.next())[0] == "thumbnail"
    assert (await stream.next())[1]["status"] == "generating"

    assert await generation.cancel_job(job)
    name, data = await stream.next()
    assert (name, data["status"]) == ("progress", TaskService.STOPPING)
    assert data["current_step"]

    # The worker saves the thumbnail it was making, then lets go of the video
    await add_thumbnail(test_db, job, 2)
    name, data = await stream.next()
    assert (name, data["image_index"]) == ("thumbnail", 2)
    assert (await stream.next())[0] == "progress"

    await generation.update_job_status(job, "cancelled")
    assert (await stream.next())[0] == "cancelled"
    assert await stream.ended()


async def test_a_failed_job_is_told_as_failed_not_as_a_connection_error(test_db, streams):
    """EventSource fires 'error' on every dropped connection: a failure has its own name."""
    _, job = await running_generation(test_db)
    await GenerationService(test_db).update_job_status(job, "error", error_message="Gran Sabio did not answer")
    stream = await streams.generation(job)

    name, data = await stream.next()

    assert name == "failed"
    assert data["error_message"] == "Gran Sabio did not answer"
    assert await stream.ended()


async def test_a_job_that_does_not_exist_is_told_as_failed(test_db, streams):
    stream = await streams.generation(9999)

    name, data = await stream.next()

    assert name == "failed" and data["error"]
    assert await stream.ended()


async def test_the_history_of_a_video_says_why_a_job_failed(client, test_db):
    video, job = await running_generation(test_db)
    await GenerationService(test_db).update_job_status(job, "error", error_message="Image 3 of 5: poe: no image")

    response = await client.get(f"/api/generation/{video}/history")

    [last] = response.json()["jobs"]
    assert (last["id"], last["status"], last["error_message"]) == (job, "error", "Image 3 of 5: poe: no image")

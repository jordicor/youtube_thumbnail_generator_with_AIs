"""Whether queued tasks can run: Redis answers and the worker's heartbeat is
there. A fake Redis and a fake clock; nothing connects anywhere."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services import queue_health


@pytest.fixture
def health(monkeypatch):
    """The module as a freshly started server has it, with Redis and the clock in hand."""
    for name in ("_status", "_checked_at", "_worker_seen_at", "_refresh_task"):
        monkeypatch.setattr(queue_health, name, None)

    redis = SimpleNamespace(exists=AsyncMock(return_value=1))
    monkeypatch.setattr(queue_health.RedisManager, "get_client", AsyncMock(return_value=redis))
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(queue_health, "monotonic", lambda: clock.now)

    async def checked():
        """The state once the check running in the background has ended."""
        queue_health.status()
        await queue_health._refresh_task
        return queue_health.status()

    return SimpleNamespace(redis=redis, clock=clock, checked=checked)


async def test_a_worker_with_its_heartbeat_is_ok(health):
    assert await queue_health.check() == queue_health.OK
    # The key arq writes for the queue the worker listens to
    health.redis.exists.assert_awaited_once_with("arq:queue:health-check")


async def test_no_heartbeat_since_the_server_started_is_a_stopped_worker(health):
    health.redis.exists.return_value = 0

    assert await queue_health.check() == queue_health.WORKER_DOWN


async def test_a_late_heartbeat_is_not_a_stopped_worker(health):
    """A busy worker rewrites its key a moment after Redis dropped it."""
    assert await queue_health.check() == queue_health.OK
    health.redis.exists.return_value = 0

    health.clock.now += queue_health.WORKER_GRACE_SECONDS - 1
    assert await queue_health.check() == queue_health.OK

    health.clock.now += 2
    assert await queue_health.check() == queue_health.WORKER_DOWN


async def test_a_redis_that_refuses_is_down(health):
    health.redis.exists.side_effect = ConnectionError("refused")

    assert await queue_health.check() == queue_health.REDIS_DOWN


async def test_a_redis_that_does_not_answer_is_down(health, monkeypatch):
    async def never_answers(key):
        await asyncio.sleep(5)

    monkeypatch.setattr(queue_health, "REDIS_TIMEOUT_SECONDS", 0.01)
    health.redis.exists.side_effect = never_answers

    assert await queue_health.check() == queue_health.REDIS_DOWN


async def test_the_state_is_unknown_until_the_first_check_ends(health):
    assert queue_health.status() is None

    assert await health.checked() == queue_health.OK


async def test_redis_is_asked_again_only_when_the_last_answer_is_old(health):
    assert await health.checked() == queue_health.OK

    health.clock.now += queue_health.CHECK_SECONDS - 1
    assert queue_health.status() == queue_health.OK
    assert health.redis.exists.await_count == 1

    # Old: the known state is still answered at once while Redis is asked again
    health.redis.exists.side_effect = ConnectionError("refused")
    health.clock.now += 1
    assert queue_health.status() == queue_health.OK
    assert await health.checked() == queue_health.REDIS_DOWN
    assert health.redis.exists.await_count == 2

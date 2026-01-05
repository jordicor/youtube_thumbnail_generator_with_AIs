"""
Redis client connection management.

Provides async and sync Redis clients with connection pooling.
"""

import os
import logging
from typing import Optional
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
import redis as sync_redis

logger = logging.getLogger(__name__)

# Configuration from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))


class RedisManager:
    """
    Manages Redis connections with lazy initialization.

    Usage:
        # Async context manager (preferred)
        async with RedisManager.get_client() as redis:
            await redis.set("key", "value")

        # Or get the pool directly
        pool = await RedisManager.get_pool()
    """

    _pool: Optional[aioredis.ConnectionPool] = None
    _sync_pool: Optional[sync_redis.ConnectionPool] = None

    @classmethod
    async def get_pool(cls) -> aioredis.ConnectionPool:
        """Get or create the async connection pool."""
        if cls._pool is None:
            cls._pool = aioredis.ConnectionPool.from_url(
                REDIS_URL,
                max_connections=REDIS_MAX_CONNECTIONS,
                decode_responses=True,
            )
            logger.info(f"Redis async pool created: {REDIS_URL}")
        return cls._pool

    @classmethod
    async def get_client(cls) -> aioredis.Redis:
        """Get an async Redis client from the pool."""
        pool = await cls.get_pool()
        return aioredis.Redis(connection_pool=pool)

    @classmethod
    def get_sync_pool(cls) -> sync_redis.ConnectionPool:
        """Get or create the sync connection pool (for workers)."""
        if cls._sync_pool is None:
            cls._sync_pool = sync_redis.ConnectionPool.from_url(
                REDIS_URL,
                max_connections=REDIS_MAX_CONNECTIONS,
                decode_responses=True,
            )
            logger.info(f"Redis sync pool created: {REDIS_URL}")
        return cls._sync_pool

    @classmethod
    def get_sync_client(cls) -> sync_redis.Redis:
        """Get a sync Redis client from the pool."""
        pool = cls.get_sync_pool()
        return sync_redis.Redis(connection_pool=pool)

    @classmethod
    async def close(cls):
        """Close all Redis connections."""
        if cls._pool is not None:
            await cls._pool.disconnect()
            cls._pool = None
            logger.info("Redis async pool closed")

        if cls._sync_pool is not None:
            cls._sync_pool.disconnect()
            cls._sync_pool = None
            logger.info("Redis sync pool closed")

    @classmethod
    async def health_check(cls) -> bool:
        """Check if Redis is reachable."""
        try:
            client = await cls.get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False


@asynccontextmanager
async def get_redis():
    """
    Async context manager for Redis client.

    Usage:
        async with get_redis() as redis:
            await redis.set("key", "value")
    """
    client = await RedisManager.get_client()
    try:
        yield client
    finally:
        # Connection is returned to pool automatically
        pass


def get_redis_sync() -> sync_redis.Redis:
    """
    Get a sync Redis client (for use in workers).

    Usage:
        redis = get_redis_sync()
        redis.set("key", "value")
    """
    return RedisManager.get_sync_client()

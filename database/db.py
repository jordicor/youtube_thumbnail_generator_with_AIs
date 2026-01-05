"""
Database Manager

Async SQLite database connection and management.
"""

import aiosqlite
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
import sys

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default database path (will be overridden by config if available)
DEFAULT_DB_PATH = ROOT_DIR / "database" / "thumbnails.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Global connection pool
_db_connection: Optional[aiosqlite.Connection] = None


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

def get_db_path() -> Path:
    """Get database path from config or use default."""
    try:
        from config import DATABASE_PATH
        return Path(DATABASE_PATH)
    except ImportError:
        return DEFAULT_DB_PATH


async def init_db() -> None:
    """
    Initialize the database.

    Creates the database file and runs schema if it doesn't exist.
    """
    global _db_connection

    db_path = get_db_path()

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create connection with timeout for busy handling
    _db_connection = await aiosqlite.connect(str(db_path), timeout=60)

    # Enable foreign keys
    await _db_connection.execute("PRAGMA foreign_keys = ON")

    # Enable WAL mode for better concurrency
    await _db_connection.execute("PRAGMA journal_mode = WAL")

    # Set busy timeout to wait up to 60 seconds for locks
    await _db_connection.execute("PRAGMA busy_timeout = 60000")

    # Run schema
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema_sql = f.read()

        # Use executescript for full SQL schema (handles triggers correctly)
        try:
            await _db_connection.executescript(schema_sql)
        except Exception as e:
            # Tables may already exist
            if 'already exists' not in str(e).lower():
                print(f"Schema warning: {e}")

    # Run migrations for existing databases
    await _run_migrations(_db_connection)

    # Reset any videos stuck in analysis states from previous server session
    await _reset_orphaned_analysis_states(_db_connection)

    print(f"Database initialized: {db_path}")


async def _reset_orphaned_analysis_states(db: aiosqlite.Connection) -> None:
    """
    Reset videos stuck in analysis states after server restart.

    When the server is stopped during video analysis, videos remain in
    intermediate states (analyzing, analyzing_scenes, etc.) even though
    no analysis process is running. This function resets them to 'pending'
    so they can be re-analyzed.
    """
    # States that indicate an analysis was in progress
    orphaned_states = (
        'analyzing',
        'analyzing_scenes',
        'analyzing_faces',
        'clustering',
        'transcribing'
    )

    # Build placeholders for IN clause
    placeholders = ','.join('?' * len(orphaned_states))

    # Count affected videos first
    async with db.execute(
        f"SELECT COUNT(*) FROM videos WHERE status IN ({placeholders})",
        orphaned_states
    ) as cursor:
        row = await cursor.fetchone()
        count = row[0] if row else 0

    if count > 0:
        # Reset to pending with an error message explaining what happened
        await db.execute(
            f"""
            UPDATE videos
            SET status = 'pending',
                error_message = 'Analysis interrupted by server restart. Please re-analyze.'
            WHERE status IN ({placeholders})
            """,
            orphaned_states
        )
        await db.commit()
        print(f"Startup cleanup: Reset {count} video(s) from interrupted analysis states to 'pending'")


async def _run_migrations(db: aiosqlite.Connection) -> None:
    """
    Run database migrations for existing databases.
    Adds new columns/tables that may not exist in older schemas.
    """
    # Migration 1: Add directory_id column to videos if not exists
    try:
        # Check if column exists
        async with db.execute("PRAGMA table_info(videos)") as cursor:
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]

        if 'directory_id' not in column_names:
            print("Migration: Adding directory_id column to videos table...")
            await db.execute("ALTER TABLE videos ADD COLUMN directory_id INTEGER REFERENCES directories(id) ON DELETE SET NULL")
            await db.commit()
            print("Migration: directory_id column added successfully")
    except Exception as e:
        print(f"Migration warning (directory_id): {e}")

    # Migration 2: Create directories table if not exists (already in schema, but just in case)
    try:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS directories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                name TEXT,
                last_scanned_at TIMESTAMP,
                video_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_directories_path ON directories(path)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_videos_directory ON videos(directory_id)")
        await db.commit()
    except Exception as e:
        print(f"Migration warning (directories table): {e}")

    # Migration 3: Add reference fields to cluster_frames
    try:
        async with db.execute("PRAGMA table_info(cluster_frames)") as cursor:
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]

        if 'is_reference' not in column_names:
            print("Migration: Adding is_reference column to cluster_frames...")
            await db.execute("ALTER TABLE cluster_frames ADD COLUMN is_reference INTEGER DEFAULT 0")
            await db.commit()
            print("Migration: is_reference column added successfully")

        if 'reference_order' not in column_names:
            print("Migration: Adding reference_order column to cluster_frames...")
            await db.execute("ALTER TABLE cluster_frames ADD COLUMN reference_order INTEGER DEFAULT NULL")
            await db.commit()
            print("Migration: reference_order column added successfully")

            # Initialize top 10 frames per cluster as references
            print("Migration: Initializing default reference frames...")
            await db.execute("""
                WITH ranked_frames AS (
                    SELECT
                        id,
                        cluster_id,
                        ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY quality_score DESC) as rank
                    FROM cluster_frames
                )
                UPDATE cluster_frames
                SET
                    is_reference = 1,
                    reference_order = (
                        SELECT rank FROM ranked_frames
                        WHERE ranked_frames.id = cluster_frames.id
                    )
                WHERE id IN (
                    SELECT id FROM ranked_frames WHERE rank <= 10
                )
            """)
            await db.commit()
            print("Migration: Default reference frames initialized")

        # Create index for reference queries
        await db.execute("CREATE INDEX IF NOT EXISTS idx_cluster_frames_reference ON cluster_frames(cluster_id, is_reference)")
        await db.commit()
    except Exception as e:
        print(f"Migration warning (reference fields): {e}")

    # Migration 4: Add description column to clusters
    try:
        async with db.execute("PRAGMA table_info(clusters)") as cursor:
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]

        if 'description' not in column_names:
            print("Migration: Adding description column to clusters table...")
            await db.execute("ALTER TABLE clusters ADD COLUMN description TEXT")
            await db.commit()
            print("Migration: description column added successfully")
    except Exception as e:
        print(f"Migration warning (description): {e}")

    # Migration 5: Add generated_titles table for persisting AI-generated titles
    try:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS generated_titles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                title_text TEXT NOT NULL,
                style TEXT,
                language TEXT DEFAULT 'es',
                provider TEXT,
                model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_generated_titles_video ON generated_titles(video_id)")
        await db.commit()
    except Exception as e:
        print(f"Migration warning (generated_titles table): {e}")

    # Migration 6: Add generated_descriptions table for persisting AI-generated descriptions
    try:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS generated_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                description_text TEXT NOT NULL,
                style TEXT,
                language TEXT DEFAULT 'es',
                length TEXT DEFAULT 'medium',
                provider TEXT,
                model TEXT,
                include_timestamps BOOLEAN DEFAULT 0,
                include_hashtags BOOLEAN DEFAULT 0,
                include_emojis BOOLEAN DEFAULT 0,
                include_social_links BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_generated_descriptions_video ON generated_descriptions(video_id)")
        await db.commit()
    except Exception as e:
        print(f"Migration warning (generated_descriptions table): {e}")

    # Migration 7: Add scene_index column to cluster_frames for scene-based grouping
    try:
        async with db.execute("PRAGMA table_info(cluster_frames)") as cursor:
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]

        if 'scene_index' not in column_names:
            print("Migration: Adding scene_index column to cluster_frames...")
            await db.execute("ALTER TABLE cluster_frames ADD COLUMN scene_index INTEGER DEFAULT NULL")
            await db.commit()
            print("Migration: scene_index column added successfully")

            # Populate existing records by parsing frame_path (scene_XXX_frame_YYYYYY.jpg)
            print("Migration: Populating scene_index from existing frame paths...")
            # Extract scene number from paths like '.../scene_000_frame_000007.jpg'
            # SQLite INSTR finds position, SUBSTR extracts 3 digits after 'scene_'
            await db.execute("""
                UPDATE cluster_frames
                SET scene_index = CAST(
                    SUBSTR(frame_path,
                        INSTR(frame_path, 'scene_') + 6,
                        3)
                    AS INTEGER)
                WHERE frame_path LIKE '%scene_%_frame_%'
                  AND scene_index IS NULL
            """)
            await db.commit()
            print("Migration: scene_index populated successfully")

        # Create index for scene-based queries
        await db.execute("CREATE INDEX IF NOT EXISTS idx_cluster_frames_scene ON cluster_frames(cluster_id, scene_index)")
        await db.commit()
    except Exception as e:
        print(f"Migration warning (scene_index): {e}")

    # Migration 8: Add cluster_type, parent_cluster_id, scene_index columns to clusters
    # This enables two view modes: 'person' (unified) and 'person_scene' (by scene)
    try:
        async with db.execute("PRAGMA table_info(clusters)") as cursor:
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]

        if 'cluster_type' not in column_names:
            print("Migration: Adding cluster_type column to clusters...")
            await db.execute("ALTER TABLE clusters ADD COLUMN cluster_type TEXT DEFAULT 'person'")
            await db.commit()
            print("Migration: cluster_type column added successfully")

            # Mark all existing clusters as 'person' type
            await db.execute("UPDATE clusters SET cluster_type = 'person' WHERE cluster_type IS NULL")
            await db.commit()

        if 'parent_cluster_id' not in column_names:
            print("Migration: Adding parent_cluster_id column to clusters...")
            await db.execute("ALTER TABLE clusters ADD COLUMN parent_cluster_id INTEGER DEFAULT NULL")
            await db.commit()
            print("Migration: parent_cluster_id column added successfully")

        # Check if clusters table already has scene_index (it's a new column for clusters, not cluster_frames)
        if 'scene_index' not in column_names:
            print("Migration: Adding scene_index column to clusters...")
            await db.execute("ALTER TABLE clusters ADD COLUMN scene_index INTEGER DEFAULT NULL")
            await db.commit()
            print("Migration: scene_index column added to clusters successfully")

        # Create indices for efficient querying
        await db.execute("CREATE INDEX IF NOT EXISTS idx_clusters_type ON clusters(video_id, cluster_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_clusters_parent ON clusters(parent_cluster_id)")
        await db.commit()
    except Exception as e:
        print(f"Migration warning (cluster_type/parent): {e}")


async def close_db() -> None:
    """Close database connection."""
    global _db_connection

    if _db_connection:
        await _db_connection.close()
        _db_connection = None


# =============================================================================
# CONNECTION CONTEXT MANAGER
# =============================================================================

@asynccontextmanager
async def get_db():
    """
    Get database connection as async context manager.

    Usage:
        async with get_db() as db:
            await db.execute("SELECT * FROM videos")
    """
    global _db_connection

    # If no global connection, create a new one
    if _db_connection is None:
        db_path = get_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = await aiosqlite.connect(str(db_path), timeout=60)
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA busy_timeout = 60000")

        try:
            yield conn
        finally:
            await conn.close()
    else:
        # Use global connection
        yield _db_connection


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def execute_query(query: str, params: tuple = ()) -> list:
    """
    Execute a query and return all results.

    Args:
        query: SQL query string
        params: Query parameters

    Returns:
        List of row dictionaries
    """
    async with get_db() as db:
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]


async def execute_one(query: str, params: tuple = ()) -> Optional[dict]:
    """
    Execute a query and return first result.

    Args:
        query: SQL query string
        params: Query parameters

    Returns:
        Row dictionary or None
    """
    async with get_db() as db:
        async with db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None


async def execute_insert(query: str, params: tuple = ()) -> int:
    """
    Execute an insert query and return the last row ID.

    Args:
        query: SQL INSERT query
        params: Query parameters

    Returns:
        ID of inserted row
    """
    async with get_db() as db:
        cursor = await db.execute(query, params)
        await db.commit()
        return cursor.lastrowid


async def execute_update(query: str, params: tuple = ()) -> int:
    """
    Execute an update/delete query and return affected row count.

    Args:
        query: SQL UPDATE/DELETE query
        params: Query parameters

    Returns:
        Number of affected rows
    """
    async with get_db() as db:
        cursor = await db.execute(query, params)
        await db.commit()
        return cursor.rowcount


# =============================================================================
# VIDEO CRUD
# =============================================================================

async def get_video(video_id: int) -> Optional[dict]:
    """Get a video by ID."""
    return await execute_one(
        "SELECT * FROM videos WHERE id = ?",
        (video_id,)
    )


async def get_video_by_path(filepath: str) -> Optional[dict]:
    """Get a video by file path."""
    return await execute_one(
        "SELECT * FROM videos WHERE filepath = ?",
        (filepath,)
    )


async def create_video(filename: str, filepath: str, duration: float = 0, directory_id: int = None) -> int:
    """Create a new video record."""
    return await execute_insert(
        """
        INSERT INTO videos (filename, filepath, duration_seconds, status, directory_id)
        VALUES (?, ?, ?, 'pending', ?)
        """,
        (filename, filepath, duration, directory_id)
    )


async def update_video_status(video_id: int, status: str, error: str = None) -> None:
    """Update video status."""
    if error:
        await execute_update(
            "UPDATE videos SET status = ?, error_message = ? WHERE id = ?",
            (status, error, video_id)
        )
    else:
        await execute_update(
            "UPDATE videos SET status = ? WHERE id = ?",
            (status, video_id)
        )


async def list_videos(status: str = None, limit: int = 50, offset: int = 0) -> list:
    """List videos with optional status filter."""
    if status:
        return await execute_query(
            """
            SELECT * FROM videos
            WHERE status = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (status, limit, offset)
        )
    else:
        return await execute_query(
            """
            SELECT * FROM videos
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        )


# =============================================================================
# CLUSTER CRUD
# =============================================================================

async def get_cluster(cluster_id: int) -> Optional[dict]:
    """Get a cluster by ID."""
    return await execute_one(
        "SELECT * FROM clusters WHERE id = ?",
        (cluster_id,)
    )


async def get_clusters_for_video(video_id: int) -> list:
    """Get all clusters for a video."""
    return await execute_query(
        """
        SELECT * FROM clusters
        WHERE video_id = ?
        ORDER BY cluster_index
        """,
        (video_id,)
    )


async def create_cluster(
    video_id: int,
    cluster_index: int,
    num_frames: int,
    representative_frame: str,
    embedding_centroid: bytes = None
) -> int:
    """Create a new cluster."""
    return await execute_insert(
        """
        INSERT INTO clusters
        (video_id, cluster_index, num_frames, representative_frame, embedding_centroid)
        VALUES (?, ?, ?, ?, ?)
        """,
        (video_id, cluster_index, num_frames, representative_frame, embedding_centroid)
    )


async def delete_clusters_for_video(video_id: int) -> int:
    """Delete all clusters for a video."""
    return await execute_update(
        "DELETE FROM clusters WHERE video_id = ?",
        (video_id,)
    )


# =============================================================================
# GENERATION JOB CRUD
# =============================================================================

async def get_job(job_id: int) -> Optional[dict]:
    """Get a generation job by ID."""
    return await execute_one(
        "SELECT * FROM generation_jobs WHERE id = ?",
        (job_id,)
    )


async def create_job(
    video_id: int,
    cluster_id: int,
    num_prompts: int = 5,
    num_variations: int = 1,
    preferred_expression: str = None
) -> int:
    """Create a new generation job."""
    return await execute_insert(
        """
        INSERT INTO generation_jobs
        (video_id, cluster_id, num_prompts, num_variations, preferred_expression, status)
        VALUES (?, ?, ?, ?, ?, 'pending')
        """,
        (video_id, cluster_id, num_prompts, num_variations, preferred_expression)
    )


async def update_job_status(
    job_id: int,
    status: str,
    progress: int = 0,
    error: str = None
) -> None:
    """Update job status and progress."""
    if error:
        await execute_update(
            """
            UPDATE generation_jobs
            SET status = ?, progress = ?, error_message = ?
            WHERE id = ?
            """,
            (status, progress, error, job_id)
        )
    else:
        await execute_update(
            """
            UPDATE generation_jobs
            SET status = ?, progress = ?
            WHERE id = ?
            """,
            (status, progress, job_id)
        )


async def complete_job(job_id: int) -> None:
    """Mark a job as completed."""
    await execute_update(
        """
        UPDATE generation_jobs
        SET status = 'completed', progress = 100, completed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (job_id,)
    )


# =============================================================================
# THUMBNAIL CRUD
# =============================================================================

async def get_thumbnail(thumbnail_id: int) -> Optional[dict]:
    """Get a thumbnail by ID."""
    return await execute_one(
        "SELECT * FROM thumbnails WHERE id = ?",
        (thumbnail_id,)
    )


async def create_thumbnail(
    job_id: int,
    prompt_index: int,
    variation_index: int,
    filepath: str,
    prompt_text: str = None,
    suggested_title: str = None,
    text_overlay: str = None
) -> int:
    """Create a new thumbnail record."""
    return await execute_insert(
        """
        INSERT INTO thumbnails
        (job_id, prompt_index, variation_index, filepath, prompt_text, suggested_title, text_overlay)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (job_id, prompt_index, variation_index, filepath, prompt_text, suggested_title, text_overlay)
    )


async def get_thumbnails_for_job(job_id: int) -> list:
    """Get all thumbnails for a job."""
    return await execute_query(
        """
        SELECT * FROM thumbnails
        WHERE job_id = ?
        ORDER BY prompt_index, variation_index
        """,
        (job_id,)
    )


async def get_thumbnails_for_video(video_id: int) -> list:
    """Get all thumbnails for a video."""
    return await execute_query(
        """
        SELECT t.* FROM thumbnails t
        JOIN generation_jobs j ON t.job_id = j.id
        WHERE j.video_id = ?
        ORDER BY t.created_at DESC
        """,
        (video_id,)
    )


# =============================================================================
# DIRECTORY CRUD
# =============================================================================

async def get_directory(directory_id: int) -> Optional[dict]:
    """Get a directory by ID."""
    return await execute_one(
        "SELECT * FROM directories WHERE id = ?",
        (directory_id,)
    )


async def get_directory_by_path(path: str) -> Optional[dict]:
    """Get a directory by path."""
    return await execute_one(
        "SELECT * FROM directories WHERE path = ?",
        (path,)
    )


async def list_directories() -> list:
    """List all saved directories, ordered by last used."""
    return await execute_query(
        """
        SELECT * FROM directories
        ORDER BY
            CASE WHEN last_scanned_at IS NOT NULL THEN last_scanned_at ELSE created_at END DESC
        """
    )


async def create_directory(path: str, name: str = None) -> int:
    """Create a new directory record."""
    return await execute_insert(
        """
        INSERT INTO directories (path, name)
        VALUES (?, ?)
        """,
        (path, name)
    )


async def update_directory_scanned(directory_id: int, video_count: int) -> None:
    """Update directory after scanning."""
    await execute_update(
        """
        UPDATE directories
        SET last_scanned_at = CURRENT_TIMESTAMP, video_count = ?
        WHERE id = ?
        """,
        (video_count, directory_id)
    )


async def update_directory_name(directory_id: int, name: str) -> None:
    """Update directory name/alias."""
    await execute_update(
        "UPDATE directories SET name = ? WHERE id = ?",
        (name, directory_id)
    )


async def delete_directory(directory_id: int) -> int:
    """Delete a directory (videos will have directory_id set to NULL)."""
    return await execute_update(
        "DELETE FROM directories WHERE id = ?",
        (directory_id,)
    )


async def get_videos_for_directory(directory_id: int, status: str = None, limit: int = 100, offset: int = 0) -> list:
    """Get all videos for a specific directory."""
    if status:
        return await execute_query(
            """
            SELECT * FROM videos
            WHERE directory_id = ? AND status = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (directory_id, status, limit, offset)
        )
    else:
        return await execute_query(
            """
            SELECT * FROM videos
            WHERE directory_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (directory_id, limit, offset)
        )


async def count_videos_for_directory(directory_id: int) -> int:
    """Count videos in a directory."""
    result = await execute_one(
        "SELECT COUNT(*) as count FROM videos WHERE directory_id = ?",
        (directory_id,)
    )
    return result['count'] if result else 0


# =============================================================================
# GENERATED TITLES CRUD
# =============================================================================

async def get_titles_for_video(video_id: int) -> list:
    """Get all generated titles for a video, ordered by creation date (newest first)."""
    return await execute_query(
        """
        SELECT * FROM generated_titles
        WHERE video_id = ?
        ORDER BY created_at DESC
        """,
        (video_id,)
    )


async def create_title(
    video_id: int,
    title_text: str,
    style: str = None,
    language: str = 'es',
    provider: str = None,
    model: str = None
) -> int:
    """Create a new generated title record."""
    return await execute_insert(
        """
        INSERT INTO generated_titles
        (video_id, title_text, style, language, provider, model)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (video_id, title_text, style, language, provider, model)
    )


async def create_titles_batch(
    video_id: int,
    titles: list,
    style: str = None,
    language: str = 'es',
    provider: str = None,
    model: str = None
) -> list:
    """Create multiple title records in a single transaction. Returns list of IDs."""
    async with get_db() as db:
        ids = []
        for title_text in titles:
            cursor = await db.execute(
                """
                INSERT INTO generated_titles
                (video_id, title_text, style, language, provider, model)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (video_id, title_text, style, language, provider, model)
            )
            ids.append(cursor.lastrowid)
        await db.commit()
        return ids


async def delete_title(title_id: int) -> int:
    """Delete a generated title by ID."""
    return await execute_update(
        "DELETE FROM generated_titles WHERE id = ?",
        (title_id,)
    )


async def delete_titles_for_video(video_id: int) -> int:
    """Delete all generated titles for a video."""
    return await execute_update(
        "DELETE FROM generated_titles WHERE video_id = ?",
        (video_id,)
    )


# =============================================================================
# GENERATED DESCRIPTIONS CRUD
# =============================================================================

async def get_descriptions_for_video(video_id: int) -> list:
    """Get all generated descriptions for a video, ordered by creation date (newest first)."""
    return await execute_query(
        """
        SELECT * FROM generated_descriptions
        WHERE video_id = ?
        ORDER BY created_at DESC
        """,
        (video_id,)
    )


async def create_description(
    video_id: int,
    description_text: str,
    style: str = None,
    language: str = 'es',
    length: str = 'medium',
    provider: str = None,
    model: str = None,
    include_timestamps: bool = False,
    include_hashtags: bool = False,
    include_emojis: bool = False,
    include_social_links: bool = False
) -> int:
    """Create a new generated description record."""
    return await execute_insert(
        """
        INSERT INTO generated_descriptions
        (video_id, description_text, style, language, length, provider, model,
         include_timestamps, include_hashtags, include_emojis, include_social_links)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (video_id, description_text, style, language, length, provider, model,
         include_timestamps, include_hashtags, include_emojis, include_social_links)
    )


async def create_descriptions_batch(
    video_id: int,
    descriptions: list,
    style: str = None,
    language: str = 'es',
    length: str = 'medium',
    provider: str = None,
    model: str = None,
    include_timestamps: bool = False,
    include_hashtags: bool = False,
    include_emojis: bool = False,
    include_social_links: bool = False
) -> list:
    """Create multiple description records in a single transaction. Returns list of IDs."""
    async with get_db() as db:
        ids = []
        for description_text in descriptions:
            cursor = await db.execute(
                """
                INSERT INTO generated_descriptions
                (video_id, description_text, style, language, length, provider, model,
                 include_timestamps, include_hashtags, include_emojis, include_social_links)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (video_id, description_text, style, language, length, provider, model,
                 include_timestamps, include_hashtags, include_emojis, include_social_links)
            )
            ids.append(cursor.lastrowid)
        await db.commit()
        return ids


async def delete_description(description_id: int) -> int:
    """Delete a generated description by ID."""
    return await execute_update(
        "DELETE FROM generated_descriptions WHERE id = ?",
        (description_id,)
    )


async def delete_descriptions_for_video(video_id: int) -> int:
    """Delete all generated descriptions for a video."""
    return await execute_update(
        "DELETE FROM generated_descriptions WHERE video_id = ?",
        (video_id,)
    )

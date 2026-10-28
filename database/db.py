"""
Database Manager

Async SQLite database connection and management.
"""

import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

import aiosqlite
import orjson

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
            # executescript implicitly commits and resets PRAGMAs; re-enable them
            await _db_connection.execute("PRAGMA foreign_keys = ON")
            await _db_connection.execute("PRAGMA busy_timeout = 60000")
        except Exception as e:
            # Only tolerate "already exists" errors; everything else is fatal
            if 'already exists' not in str(e).lower():
                raise

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
    # States that indicate an analysis was in progress -> reset to 'pending'
    analysis_orphaned_states = (
        'analyzing',
        'analyzing_scenes',
        'analyzing_faces',
        'clustering',
        'transcribing',
        'cancelled',
    )

    # Videos stuck in 'generating' already completed analysis -> reset to 'analyzed'
    generation_orphaned_states = ('generating',)

    # Reset analysis-orphaned videos to 'pending'
    placeholders = ','.join('?' * len(analysis_orphaned_states))
    async with db.execute(
        f"SELECT COUNT(*) FROM videos WHERE status IN ({placeholders})",
        analysis_orphaned_states
    ) as cursor:
        row = await cursor.fetchone()
        analysis_count = row[0] if row else 0

    if analysis_count > 0:
        await db.execute(
            f"""
            UPDATE videos
            SET status = 'pending',
                error_message = 'Analysis interrupted by server restart. Please re-analyze.'
            WHERE status IN ({placeholders})
            """,
            analysis_orphaned_states
        )
        await db.commit()
        print(f"Startup cleanup: Reset {analysis_count} video(s) from interrupted analysis states to 'pending'")

    # Reset generation-orphaned videos to 'analyzed'
    placeholders = ','.join('?' * len(generation_orphaned_states))
    async with db.execute(
        f"SELECT COUNT(*) FROM videos WHERE status IN ({placeholders})",
        generation_orphaned_states
    ) as cursor:
        row = await cursor.fetchone()
        generation_count = row[0] if row else 0

    if generation_count > 0:
        await db.execute(
            f"""
            UPDATE videos
            SET status = 'analyzed',
                error_message = 'Generation interrupted by server restart.'
            WHERE status IN ({placeholders})
            """,
            generation_orphaned_states
        )
        await db.commit()
        print(f"Startup cleanup: Reset {generation_count} video(s) from interrupted generation state to 'analyzed'")


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

    # Migration 9: Architecture V2 - Create video_frames and cluster_frame_assignments tables
    # This enables independent views where frames exist separately from cluster assignments.
    try:
        # Check if video_frames table exists
        async with db.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='video_frames'
        """) as cursor:
            video_frames_exists = await cursor.fetchone() is not None

        if not video_frames_exists:
            print("Migration: Creating video_frames table for Architecture V2...")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS video_frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    frame_path TEXT NOT NULL,
                    quality_score REAL,
                    expression TEXT,
                    scene_index INTEGER,
                    face_embedding BLOB,
                    similarity_to_centroid REAL,
                    source_cluster_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_video_frames_video ON video_frames(video_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_video_frames_scene ON video_frames(video_id, scene_index)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_video_frames_quality ON video_frames(video_id, quality_score DESC)")
            await db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_video_frames_path ON video_frames(video_id, frame_path)")
            await db.commit()
            print("Migration: video_frames table created successfully")

        # Check if cluster_frame_assignments table exists
        async with db.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='cluster_frame_assignments'
        """) as cursor:
            assignments_exists = await cursor.fetchone() is not None

        if not assignments_exists:
            print("Migration: Creating cluster_frame_assignments table for Architecture V2...")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cluster_frame_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id INTEGER NOT NULL,
                    frame_id INTEGER NOT NULL,
                    is_reference INTEGER DEFAULT 0,
                    reference_order INTEGER,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE,
                    FOREIGN KEY (frame_id) REFERENCES video_frames(id) ON DELETE CASCADE,
                    UNIQUE(cluster_id, frame_id)
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_assignments_cluster ON cluster_frame_assignments(cluster_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_assignments_frame ON cluster_frame_assignments(frame_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_assignments_reference ON cluster_frame_assignments(cluster_id, is_reference)")
            await db.commit()
            print("Migration: cluster_frame_assignments table created successfully")

        # Add view_mode and representative_frame_id columns to clusters
        async with db.execute("PRAGMA table_info(clusters)") as cursor:
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]

        if 'view_mode' not in column_names:
            print("Migration: Adding view_mode column to clusters...")
            await db.execute("ALTER TABLE clusters ADD COLUMN view_mode TEXT")
            await db.commit()
            # Copy existing cluster_type to view_mode
            await db.execute("UPDATE clusters SET view_mode = cluster_type WHERE view_mode IS NULL")
            await db.execute("UPDATE clusters SET view_mode = 'person' WHERE view_mode IS NULL")
            await db.commit()
            print("Migration: view_mode column added and populated")

        if 'representative_frame_id' not in column_names:
            print("Migration: Adding representative_frame_id column to clusters...")
            await db.execute("ALTER TABLE clusters ADD COLUMN representative_frame_id INTEGER")
            await db.commit()
            print("Migration: representative_frame_id column added")

        # Create index for view_mode queries
        await db.execute("CREATE INDEX IF NOT EXISTS idx_clusters_view_mode ON clusters(video_id, view_mode)")
        await db.commit()

    except Exception as e:
        print(f"Migration warning (Architecture V2 tables): {e}")

    # Migrations 10-13 fail fast: the code requires their schema and path
    # format, so a skipped migration would only surface later as confusing
    # runtime errors.

    # Migration 12: Add face-swap columns to thumbnails
    async with db.execute("PRAGMA table_info(thumbnails)") as cursor:
        columns = [col[1] for col in await cursor.fetchall()]

    if 'raw_path' not in columns:
        print("Migration: Adding raw_path column to thumbnails...")
        await db.execute("ALTER TABLE thumbnails ADD COLUMN raw_path TEXT")
        await db.commit()

    if 'face_swap_applied' not in columns:
        print("Migration: Adding face_swap_applied column to thumbnails...")
        await db.execute("ALTER TABLE thumbnails ADD COLUMN face_swap_applied INTEGER DEFAULT 0")
        await db.commit()

    if 'face_swap_metadata' not in columns:
        print("Migration: Adding face_swap_metadata column to thumbnails...")
        await db.execute("ALTER TABLE thumbnails ADD COLUMN face_swap_metadata TEXT")
        await db.commit()

    # Migration 11: Replace single cluster_id with generation_job_clusters table.
    # IDEMPOTENT — works on three starting states:
    #   (a) Brand-new DB: schema.sql already created the new table; this just
    #       creates indices.
    #   (b) Legacy DB with cluster_id column AND new table not yet existing:
    #       create new table, backfill from legacy, then drop the column.
    #   (c) Legacy DB with cluster_id column but new table already created
    #       (e.g. schema.sql ran before this migration on a legacy DB): backfill
    #       missing rows then drop the column.
    # Ensure the new table exists (idempotent CREATE)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS generation_job_clusters (
            job_id INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            person_id TEXT NOT NULL,
            cluster_order INTEGER NOT NULL,
            PRIMARY KEY (job_id, cluster_id),
            FOREIGN KEY (job_id) REFERENCES generation_jobs(id) ON DELETE CASCADE,
            FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
        )
    """)
    await db.commit()

    # Detect whether the legacy column is still present
    async with db.execute("PRAGMA table_info(generation_jobs)") as cursor:
        columns = [col[1] for col in await cursor.fetchall()]
    has_legacy_cluster_id = 'cluster_id' in columns

    if has_legacy_cluster_id:
        print("Migration: Migrating generation_jobs.cluster_id (legacy) -> generation_job_clusters...")

        # (1) Backfill: INSERT OR IGNORE so re-runs don't duplicate. Each
        # legacy job becomes a single person_01 entry.
        await db.execute("""
            INSERT OR IGNORE INTO generation_job_clusters
                (job_id, cluster_id, person_id, cluster_order)
            SELECT id, cluster_id, 'person_01', 0
            FROM generation_jobs
            WHERE cluster_id IS NOT NULL
        """)
        await db.commit()

        # (2) Recreate generation_jobs without cluster_id. Standard SQLite
        # table-swap procedure (https://www.sqlite.org/lang_altertable.html).
        # CRITICAL: disable FK enforcement around the swap so the cascade
        # from generation_job_clusters → generation_jobs(id) does NOT wipe
        # the data we just backfilled when we drop the old table.
        await db.execute("PRAGMA foreign_keys = OFF")
        try:
            await db.execute("BEGIN IMMEDIATE")
            await db.execute("""
                CREATE TABLE generation_jobs_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    num_images INTEGER DEFAULT 5,
                    preferred_expression TEXT,
                    status TEXT DEFAULT 'pending',
                    progress INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
                )
            """)
            await db.execute("""
                INSERT INTO generation_jobs_new
                    (id, video_id, num_images, preferred_expression, status,
                     progress, error_message, created_at, completed_at)
                SELECT id, video_id, num_images, preferred_expression, status,
                       progress, error_message, created_at, completed_at
                FROM generation_jobs
            """)
            await db.execute("DROP TABLE generation_jobs")
            await db.execute("ALTER TABLE generation_jobs_new RENAME TO generation_jobs")
            await db.execute("COMMIT")
            print("Migration: generation_jobs recreated without legacy cluster_id column")
        except Exception:
            await db.execute("ROLLBACK")
            raise
        finally:
            # Always restore FK enforcement, even if the swap failed
            await db.execute("PRAGMA foreign_keys = ON")

    # Indices for fast lookup (idempotent)
    await db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_video ON generation_jobs(video_id)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON generation_jobs(status)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_job_clusters_job ON generation_job_clusters(job_id)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_job_clusters_cluster ON generation_job_clusters(cluster_id)")
    await db.commit()

    # Migration 10: Add physical_description columns to clusters
    # Caches LLM-generated facial description per cluster for deterministic
    # face_groups population across generations.
    async with db.execute("PRAGMA table_info(clusters)") as cursor:
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]

    if 'physical_description' not in column_names:
        print("Migration: Adding physical_description column to clusters...")
        await db.execute("ALTER TABLE clusters ADD COLUMN physical_description TEXT")
        await db.commit()
        print("Migration: physical_description column added successfully")

    if 'physical_description_updated_at' not in column_names:
        print("Migration: Adding physical_description_updated_at column to clusters...")
        await db.execute("ALTER TABLE clusters ADD COLUMN physical_description_updated_at TIMESTAMP")
        await db.commit()
        print("Migration: physical_description_updated_at column added successfully")

    # Migration 13: Store paths under OUTPUT_DIR relative to it, so the project
    # and its output folder can move without breaking stored references.
    await _migrate_output_paths_to_relative(db)

    # Migration 14: Face sheet of each person (views chosen by hand, on/off)
    async with db.execute("PRAGMA table_info(clusters)") as cursor:
        column_names = [col[1] for col in await cursor.fetchall()]
    if 'reference_format' not in column_names and 'face_sheet' not in column_names:
        print("Migration: Adding face sheet columns to clusters...")
        await db.execute("ALTER TABLE clusters ADD COLUMN face_sheet TEXT")
        await db.execute("ALTER TABLE clusters ADD COLUMN face_sheet_enabled INTEGER NOT NULL DEFAULT 0")
        await db.commit()
        print("Migration: face sheet columns added successfully")

    # Migration 15: Face match of each thumbnail (how much the people look like themselves)
    async with db.execute("PRAGMA table_info(thumbnails)") as cursor:
        column_names = [col[1] for col in await cursor.fetchall()]
    if 'face_match_status' not in column_names:
        print("Migration: Adding face match columns to thumbnails...")
        await db.execute("ALTER TABLE thumbnails ADD COLUMN face_match_status TEXT")
        await db.execute("ALTER TABLE thumbnails ADD COLUMN face_match_score REAL")
        await db.execute("ALTER TABLE thumbnails ADD COLUMN face_match_people TEXT")
        await db.commit()
        print("Migration: face match columns added successfully")

    # Body and environment references are independent of face-photo selections.
    async with db.execute("PRAGMA table_info(clusters)") as cursor:
        reference_columns = {row[1] for row in await cursor.fetchall()}
    for name in ("body_reference", "environment_reference"):
        if "reference_format" not in reference_columns and name not in reference_columns:
            await db.execute(f"ALTER TABLE clusters ADD COLUMN {name} TEXT")
    await db.commit()

    # Migration 16: Index the frame a cluster is represented by. Deleting a
    # video_frames row sets that column to NULL in the clusters pointing at it;
    # without an index SQLite scans every cluster for each frame deleted.
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_clusters_representative_frame ON clusters(representative_frame_id)"
    )
    await db.commit()

    # Migration 17: Moving a directory rewrites its videos' paths, which must not
    # bump their updated_at (the library's activity order)
    async with db.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = 'update_videos_timestamp'"
    ) as cursor:
        row = await cursor.fetchone()
    if row is None or 'WHEN NEW.filepath IS OLD.filepath' not in row[0]:
        print("Migration: Making the videos updated_at trigger skip path changes...")
        await db.execute("DROP TRIGGER IF EXISTS update_videos_timestamp")
        await db.execute("""
            CREATE TRIGGER update_videos_timestamp
            AFTER UPDATE ON videos
            WHEN NEW.filepath IS OLD.filepath
            BEGIN
                UPDATE videos SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        """)
        await db.commit()
        print("Migration: videos updated_at trigger updated successfully")

    # Migration 18: Alternative paths of each directory (the same folder reached
    # from other machines), used when its path does not answer
    async with db.execute("PRAGMA table_info(directories)") as cursor:
        column_names = [col[1] for col in await cursor.fetchall()]
    if 'alternative_paths' not in column_names:
        print("Migration: Adding alternative_paths column to directories...")
        await db.execute("ALTER TABLE directories ADD COLUMN alternative_paths TEXT NOT NULL DEFAULT '[]'")
        await db.commit()
        print("Migration: alternative_paths column added successfully")

    # Migration 19: App settings (AI engines, values every video starts with)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            data TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.commit()

    # Migration 20: Restoration is independent from swapping and starts off.
    await db.execute("""
        UPDATE app_settings
        SET data = json_set(data, '$.generation.face_restoration', json('false'))
        WHERE json_type(data, '$.generation.face_restoration') IS NULL
    """)
    await db.commit()

    # Migration 21: People whose face never moves are pictures (paintings, posters)
    async with db.execute("PRAGMA table_info(clusters)") as cursor:
        column_names = [col[1] for col in await cursor.fetchall()]
    if 'is_picture' not in column_names:
        print("Migration: Adding is_picture column to clusters...")
        await db.execute("ALTER TABLE clusters ADD COLUMN is_picture INTEGER NOT NULL DEFAULT 0")
        await db.commit()
        print("Migration: is_picture column added successfully")

    # Migration 22: Whether the video's thumbnails use its setting photos ('photos')
    # or let the AI choose the background ('ai'); NULL until the user picks
    async with db.execute("PRAGMA table_info(videos)") as cursor:
        column_names = [col[1] for col in await cursor.fetchall()]
    if 'environment_source' not in column_names:
        print("Migration: Adding environment_source column to videos...")
        await db.execute(
            "ALTER TABLE videos ADD COLUMN environment_source TEXT CHECK(environment_source IN ('photos', 'ai'))"
        )
        await db.commit()
        print("Migration: environment_source column added successfully")

    # Migration 23: Add the local image connection without changing saved engines.
    from services.comfyui_service import Connection
    await db.execute("""
        UPDATE app_settings SET data = json_set(data, '$.comfyui', json(?))
        WHERE json_type(data, '$.comfyui') IS NULL
    """, (Connection().model_dump_json(),))
    await db.commit()

    from database.reference_migration import migrate_visual_references
    await migrate_visual_references(db)


# Columns holding files under OUTPUT_DIR, stored relative to it (POSIX separators)
_OUTPUT_PATH_COLUMNS = (
    ('video_frames', 'frame_path'),
    ('cluster_frames', 'frame_path'),
    ('clusters', 'representative_frame'),
    ('thumbnails', 'filepath'),
    ('thumbnails', 'raw_path'),
)
# Subdirectories of a video's output folder that stored paths point into
_VIDEO_OUTPUT_SUBDIRS = {'frames', 'clusters', 'thumbnails'}


def _is_absolute_stored_path(value: str) -> bool:
    return '\\' in value or value.startswith('/') or value[1:2] == ':'


def _legacy_output_relative(value: str) -> str:
    """Convert an absolute path from any former output location to OUTPUT_DIR-relative form."""
    import config

    path = PureWindowsPath(value) if ('\\' in value or value[1:2] == ':') else PurePosixPath(value)
    output_dir = Path(config.OUTPUT_DIR)
    try:
        return path.relative_to(type(path)(str(output_dir))).as_posix()
    except ValueError:
        pass

    # Moved project: find the former output folder, i.e. the component followed
    # by <video folder>/<frames|clusters|thumbnails>/...
    output_names = {output_dir.name.lower(), 'output'}
    parts = path.parts
    for i, part in enumerate(parts[:-3]):
        if part.lower() in output_names and parts[i + 2].lower() in _VIDEO_OUTPUT_SUBDIRS:
            return '/'.join(parts[i + 1:])
    raise ValueError(f"Cannot locate the output folder in stored path: {value}")


def _relative_cache_path(value: str) -> str:
    return _legacy_output_relative(value) if _is_absolute_stored_path(value) else value


def _convert_faces_cache(data) -> None:
    faces = data if isinstance(data, list) else data.get('all_faces', [])
    for face in faces:
        if face.get('frame_path'):
            face['frame_path'] = _relative_cache_path(face['frame_path'])
    if isinstance(data, dict) and 'processed_frames' in data:
        data['processed_frames'] = [_relative_cache_path(p) for p in data['processed_frames']]


def _convert_clustering_cache(data) -> None:
    for cluster in data.get('clusters', []):
        if cluster.get('representative_frame'):
            cluster['representative_frame'] = _relative_cache_path(cluster['representative_frame'])
        cluster['top_frames'] = [_relative_cache_path(p) for p in cluster.get('top_frames', [])]


def _convert_clusters_metadata(data) -> None:
    for cluster in data.get('clusters', []):
        if cluster.get('representative'):
            cluster['representative'] = _relative_cache_path(cluster['representative'])


def _migrate_output_json_caches() -> int:
    """Rewrite absolute paths in per-video JSON caches relative to OUTPUT_DIR."""
    import config

    output_dir = Path(config.OUTPUT_DIR)
    if not output_dir.is_dir():
        return 0

    converted = 0
    for video_dir in output_dir.iterdir():
        if not video_dir.is_dir():
            continue
        for cache_file, convert in (
            (video_dir / 'faces.json', _convert_faces_cache),
            (video_dir / 'clustering_result.json', _convert_clustering_cache),
            (video_dir / 'clusters' / 'clusters.json', _convert_clusters_metadata),
        ):
            if not cache_file.exists():
                continue
            original = cache_file.read_bytes()
            data = orjson.loads(original)
            convert(data)
            updated = orjson.dumps(data, option=orjson.OPT_INDENT_2)
            if updated == original:
                continue
            # Atomic write: a crash leaves either the old or the new file
            temp_fd, temp_path = tempfile.mkstemp(dir=str(cache_file.parent), suffix='.tmp')
            try:
                with os.fdopen(temp_fd, 'wb') as f:
                    f.write(updated)
                os.replace(temp_path, cache_file)
            except BaseException:
                os.unlink(temp_path)
                raise
            converted += 1
    return converted


async def _migrate_output_paths_to_relative(db: aiosqlite.Connection) -> None:
    legacy = {}
    for table, column in _OUTPUT_PATH_COLUMNS:
        async with db.execute(
            f"SELECT rowid, {column} FROM {table} WHERE {column} IS NOT NULL "
            f"AND (instr({column}, char(92)) > 0 OR substr({column}, 1, 1) = '/' "
            f"OR substr({column}, 2, 1) = ':')"
        ) as cursor:
            rows = await cursor.fetchall()
        if rows:
            legacy[(table, column)] = rows
    if not legacy:
        return

    total = sum(len(rows) for rows in legacy.values())
    print(f"Migration: Converting {total} stored output paths to OUTPUT_DIR-relative form...")
    # JSON caches first: if they fail, the database keeps its absolute paths
    # and the whole migration runs again on the next start.
    converted_files = _migrate_output_json_caches()
    # Convert every value before writing, so an unknown layout changes nothing.
    updates = {
        key: [(_legacy_output_relative(value), rowid) for rowid, value in rows]
        for key, rows in legacy.items()
    }
    for (table, column), pairs in updates.items():
        await db.executemany(f"UPDATE {table} SET {column} = ? WHERE rowid = ?", pairs)
    await db.commit()
    print(f"Migration: {total} database paths and {converted_files} JSON caches converted")


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
    # A connection belongs to one operation. Sharing the initialization
    # connection lets concurrent requests commit each other's transactions.
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db_path), timeout=60)
    try:
        await conn.execute("PRAGMA foreign_keys = ON")
        # init_db configures WAL once. Switching journal modes here races with
        # other requests opening/reading a database that has not been initialized.
        await conn.execute("PRAGMA busy_timeout = 60000")
        yield conn
    finally:
        # Closing rolls back an uncommitted operation, including cancellation.
        await conn.close()


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
    cluster_ids: list,
    num_images: int = 5,
    preferred_expression: str = None
) -> int:
    """Create a new generation job referencing one or more clusters.

    Each cluster in `cluster_ids` is mapped to a sequential person_id
    (person_01, person_02, ...) by its position in the list.
    """
    if not cluster_ids:
        raise ValueError("cluster_ids must contain at least one cluster")

    async with get_db() as db:
        cursor = await db.execute(
            """
            INSERT INTO generation_jobs
            (video_id, num_images, preferred_expression, status)
            VALUES (?, ?, ?, 'pending')
            """,
            (video_id, num_images, preferred_expression)
        )
        job_id = cursor.lastrowid

        for order, cluster_id in enumerate(cluster_ids):
            person_id = f"person_{order + 1:02d}"
            await db.execute(
                """
                INSERT INTO generation_job_clusters
                (job_id, cluster_id, person_id, cluster_order)
                VALUES (?, ?, ?, ?)
                """,
                (job_id, cluster_id, person_id, order)
            )

        await db.commit()
        return job_id


async def get_job_clusters(job_id: int) -> list:
    """Return clusters linked to a job, ordered by cluster_order.

    Each row contains: cluster_id, person_id, cluster_order.
    """
    return await execute_query(
        """
        SELECT cluster_id, person_id, cluster_order
        FROM generation_job_clusters
        WHERE job_id = ?
        ORDER BY cluster_order ASC
        """,
        (job_id,)
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

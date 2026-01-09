-- YouTube Thumbnail Generator - Database Schema
-- =============================================
-- SQLite database for managing videos, clusters, and generation jobs

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- =============================================================================
-- DIRECTORIES TABLE
-- =============================================================================
-- Saved video directories for multi-directory support

CREATE TABLE IF NOT EXISTS directories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    name TEXT,                        -- Optional user-friendly alias
    last_scanned_at TIMESTAMP,        -- When directory was last scanned
    video_count INTEGER DEFAULT 0,    -- Cached count of videos
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- VIDEOS TABLE
-- =============================================================================
-- Registered videos from the videos directory

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    directory_id INTEGER,             -- Source directory (nullable for backwards compat)
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL UNIQUE,
    duration_seconds REAL,
    status TEXT DEFAULT 'pending',
    -- Status values:
    --   pending     - Not yet analyzed
    --   analyzing   - Analysis in progress
    --   analyzed    - Analysis complete, ready for generation
    --   generating  - Thumbnail generation in progress
    --   completed   - Thumbnails generated
    --   error       - An error occurred
    error_message TEXT,
    is_hidden BOOLEAN DEFAULT FALSE,  -- Hidden videos don't appear in normal lists
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (directory_id) REFERENCES directories(id) ON DELETE SET NULL
);

-- =============================================================================
-- CLUSTERS TABLE
-- =============================================================================
-- Face clusters detected in each video
-- Each cluster represents a person or "character" appearance
--
-- Cluster types:
--   'person'       - Groups faces by facial similarity (DBSCAN result)
--   'person_scene' - Subdivides a person cluster by video scene
--
-- The person_scene clusters have a parent_cluster_id pointing to their person cluster.
-- This allows two views: unified by person, or split by person+scene.

CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    cluster_index INTEGER NOT NULL,  -- 0, 1, 2... (ordered by size)
    label TEXT,                       -- Optional user-assigned name
    description TEXT,                 -- Optional notes/comments about this cluster
    num_frames INTEGER NOT NULL,      -- Number of frames in this cluster
    representative_frame TEXT NOT NULL,  -- Path to best representative frame
    embedding_centroid BLOB,          -- Centroid embedding (512 floats, serialized)
    cluster_type TEXT DEFAULT 'person',  -- 'person' or 'person_scene'
    parent_cluster_id INTEGER DEFAULT NULL,  -- FK to parent cluster (for person_scene)
    scene_index INTEGER DEFAULT NULL,  -- Scene number (only for person_scene clusters)
    view_mode TEXT,                   -- 'person' or 'person_scene' (V2 architecture)
    representative_frame_id INTEGER REFERENCES video_frames(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_cluster_id) REFERENCES clusters(id) ON DELETE CASCADE,
    UNIQUE(video_id, cluster_index, view_mode)
);

-- =============================================================================
-- CLUSTER FRAMES TABLE
-- =============================================================================
-- Individual frames belonging to each cluster
-- Allows viewing more examples of each detected person

CREATE TABLE IF NOT EXISTS cluster_frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL,
    frame_path TEXT NOT NULL,
    quality_score REAL,
    expression TEXT,  -- mouth_closed, smiling, neutral, mouth_open
    similarity_score REAL,  -- Similarity to cluster centroid
    is_reference INTEGER DEFAULT 0,  -- 1 if this frame is selected as AI reference
    reference_order INTEGER DEFAULT NULL,  -- Order in reference list (1-10), NULL if not reference
    scene_index INTEGER DEFAULT NULL,  -- Scene number from video (0, 1, 2...) for grouping by scene
    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
);

-- =============================================================================
-- GENERATION JOBS TABLE
-- =============================================================================
-- Thumbnail generation jobs

CREATE TABLE IF NOT EXISTS generation_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,     -- Selected cluster for face reference
    num_images INTEGER DEFAULT 5,    -- Number of images to generate
    preferred_expression TEXT,       -- smiling, mouth_closed, neutral
    status TEXT DEFAULT 'pending',
    -- Status values:
    --   pending      - Job created, waiting to start
    --   transcribing - Transcribing audio
    --   prompting    - Generating prompts with LLM
    --   generating   - Generating thumbnail images
    --   completed    - All thumbnails generated
    --   cancelled    - Job was cancelled
    --   error        - An error occurred
    progress INTEGER DEFAULT 0,      -- 0-100 percentage
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
);

-- =============================================================================
-- THUMBNAILS TABLE
-- =============================================================================
-- Generated thumbnail images

CREATE TABLE IF NOT EXISTS thumbnails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    image_index INTEGER NOT NULL,     -- 1, 2, 3... (sequential image number)
    filepath TEXT NOT NULL,           -- Path to generated image
    prompt_text TEXT,                 -- The image prompt used
    suggested_title TEXT,             -- Suggested video title
    text_overlay TEXT,                -- Suggested text overlay
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES generation_jobs(id) ON DELETE CASCADE
);

-- =============================================================================
-- GENERATED TITLES TABLE
-- =============================================================================
-- AI-generated titles for videos (persisted across sessions)

CREATE TABLE IF NOT EXISTS generated_titles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    title_text TEXT NOT NULL,
    style TEXT,                       -- neutral, seo, clickbait, custom
    language TEXT DEFAULT 'es',       -- es, en, fr, it, de, pt
    provider TEXT,                    -- anthropic, openai, google, xai
    model TEXT,                       -- Model used for generation
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

-- =============================================================================
-- GENERATED DESCRIPTIONS TABLE
-- =============================================================================
-- AI-generated descriptions for videos (persisted across sessions)

CREATE TABLE IF NOT EXISTS generated_descriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    description_text TEXT NOT NULL,
    style TEXT,                       -- informative, seo, minimal, custom
    language TEXT DEFAULT 'es',       -- es, en, fr, it, de, pt
    length TEXT DEFAULT 'medium',     -- short, medium, long, very_long
    provider TEXT,                    -- anthropic, openai, google, xai
    model TEXT,                       -- Model used for generation
    include_timestamps BOOLEAN DEFAULT 0,
    include_hashtags BOOLEAN DEFAULT 0,
    include_emojis BOOLEAN DEFAULT 0,
    include_social_links BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

-- =============================================================================
-- INDICES FOR PERFORMANCE
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_directories_path ON directories(path);
CREATE INDEX IF NOT EXISTS idx_videos_directory ON videos(directory_id);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_filepath ON videos(filepath);
CREATE INDEX IF NOT EXISTS idx_videos_hidden ON videos(is_hidden);
CREATE INDEX IF NOT EXISTS idx_clusters_video ON clusters(video_id);
CREATE INDEX IF NOT EXISTS idx_clusters_type ON clusters(video_id, cluster_type);
CREATE INDEX IF NOT EXISTS idx_clusters_parent ON clusters(parent_cluster_id);
CREATE INDEX IF NOT EXISTS idx_cluster_frames_cluster ON cluster_frames(cluster_id);
CREATE INDEX IF NOT EXISTS idx_cluster_frames_reference ON cluster_frames(cluster_id, is_reference);
CREATE INDEX IF NOT EXISTS idx_cluster_frames_scene ON cluster_frames(cluster_id, scene_index);
CREATE INDEX IF NOT EXISTS idx_jobs_video ON generation_jobs(video_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON generation_jobs(status);
CREATE INDEX IF NOT EXISTS idx_thumbnails_job ON thumbnails(job_id);
CREATE INDEX IF NOT EXISTS idx_generated_titles_video ON generated_titles(video_id);
CREATE INDEX IF NOT EXISTS idx_generated_descriptions_video ON generated_descriptions(video_id);

-- =============================================================================
-- TRIGGERS FOR UPDATED_AT
-- =============================================================================

CREATE TRIGGER IF NOT EXISTS update_videos_timestamp
AFTER UPDATE ON videos
BEGIN
    UPDATE videos SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

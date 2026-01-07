-- Migration 003: Add generated_descriptions table
-- Created: 2026-01-02
-- Description: Persistent storage for AI-generated video descriptions

-- =============================================================================
-- GENERATED DESCRIPTIONS TABLE
-- =============================================================================
-- AI-generated descriptions for videos (persisted across sessions)
-- Mirrors the structure of generated_titles for consistency

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

-- Index for fast lookups by video
CREATE INDEX IF NOT EXISTS idx_generated_descriptions_video ON generated_descriptions(video_id);

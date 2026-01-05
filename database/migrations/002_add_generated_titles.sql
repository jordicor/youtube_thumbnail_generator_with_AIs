-- Migration: Add generated_titles table
-- Date: 2026-01-02
-- Description: Adds table to persist AI-generated titles across sessions

-- Create the generated_titles table
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

-- Create index for faster video-based queries
CREATE INDEX IF NOT EXISTS idx_generated_titles_video ON generated_titles(video_id);

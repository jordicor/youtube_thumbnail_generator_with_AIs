-- Migration: Add reference fields to cluster_frames
-- Date: 2025-01-XX
-- Description: Adds is_reference and reference_order columns for AI reference frame selection

-- Add new columns (SQLite doesn't support IF NOT EXISTS for columns, so we use a workaround)
-- These will fail silently if columns already exist when run through the migration script

ALTER TABLE cluster_frames ADD COLUMN is_reference INTEGER DEFAULT 0;
ALTER TABLE cluster_frames ADD COLUMN reference_order INTEGER DEFAULT NULL;

-- Create index for faster reference queries
CREATE INDEX IF NOT EXISTS idx_cluster_frames_reference ON cluster_frames(cluster_id, is_reference);

-- Initialize: Set top 10 frames per cluster as references based on quality_score
-- This ensures existing data works correctly after migration
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
);

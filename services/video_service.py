"""
Video Service

Business logic for video management operations.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, List
import aiosqlite


class VideoService:
    """Service for video-related operations."""

    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    async def list_videos(
        self,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 50
    ) -> List[dict]:
        """List all registered videos."""
        query = "SELECT * FROM videos"
        params = []

        if status:
            query += " WHERE status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, skip])

        async with self.db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    async def count_videos(self, status: Optional[str] = None) -> int:
        """Count videos with optional status filter."""
        query = "SELECT COUNT(*) FROM videos"
        params = []

        if status:
            query += " WHERE status = ?"
            params.append(status)

        async with self.db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def get_video(self, video_id: int) -> Optional[dict]:
        """Get a single video by ID."""
        query = "SELECT * FROM videos WHERE id = ?"

        async with self.db.execute(query, [video_id]) as cursor:
            row = await cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None

    async def delete_video(self, video_id: int) -> bool:
        """Delete a video and all associated data."""
        # Check if exists
        video = await self.get_video(video_id)
        if not video:
            return False

        # Delete from database (cascade will handle related records)
        await self.db.execute("DELETE FROM videos WHERE id = ?", [video_id])
        await self.db.commit()

        return True

    async def hide_video(self, video_id: int) -> bool:
        """Hide a video from normal lists."""
        video = await self.get_video(video_id)
        if not video:
            return False

        await self.db.execute(
            "UPDATE videos SET is_hidden = TRUE WHERE id = ?",
            [video_id]
        )
        await self.db.commit()
        return True

    async def show_video(self, video_id: int) -> bool:
        """Show a hidden video again."""
        video = await self.get_video(video_id)
        if not video:
            return False

        await self.db.execute(
            "UPDATE videos SET is_hidden = FALSE WHERE id = ?",
            [video_id]
        )
        await self.db.commit()
        return True

    async def bulk_hide_videos(self, video_ids: List[int]) -> int:
        """Hide multiple videos at once. Returns count of hidden videos."""
        if not video_ids:
            return 0

        placeholders = ','.join(['?' for _ in video_ids])
        cursor = await self.db.execute(
            f"UPDATE videos SET is_hidden = TRUE WHERE id IN ({placeholders})",
            video_ids
        )
        await self.db.commit()
        return cursor.rowcount

    async def reanalyze_video(self, video_id: int) -> bool:
        """
        Delete all analysis data for a video and reset to pending.
        Deletes: clusters, cluster_frames, generation_jobs, thumbnails (DB and files).
        Keeps: video record, preview image.
        """
        video = await self.get_video(video_id)
        if not video:
            return False

        import sys
        import shutil
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import OUTPUT_DIR
        from utils import sanitize_filename

        video_name = Path(video['filepath']).stem
        safe_name = sanitize_filename(video_name)
        output_dir = Path(OUTPUT_DIR) / safe_name

        # Delete from database (cascade handles cluster_frames and thumbnails)
        # Delete clusters (cascade deletes cluster_frames)
        await self.db.execute(
            "DELETE FROM clusters WHERE video_id = ?",
            [video_id]
        )

        # Delete generation_jobs (cascade deletes thumbnails)
        await self.db.execute(
            "DELETE FROM generation_jobs WHERE video_id = ?",
            [video_id]
        )

        # Delete video_frames so stale frame data does not persist
        await self.db.execute(
            "DELETE FROM video_frames WHERE video_id = ?",
            [video_id]
        )

        # Reset video status to pending
        await self.db.execute(
            "UPDATE videos SET status = 'pending', error_message = NULL WHERE id = ?",
            [video_id]
        )
        await self.db.commit()

        # Delete files from disk (keep preview.jpg)
        dirs_to_delete = ['clusters', 'frames', 'thumbnails']
        for dir_name in dirs_to_delete:
            dir_path = output_dir / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)

        return True

    async def get_video_thumbnail(self, video_id: int) -> Optional[str]:
        """
        Get a thumbnail preview for a video.

        The search runs in a thread: it may read the video from a slow network
        share, and on the event loop that would hold up every other request.
        """
        video = await self.get_video(video_id)
        if not video:
            return None

        return await asyncio.to_thread(_find_video_preview, video['filepath'])


def _find_video_preview(filepath: str) -> Optional[str]:
    """Path of a preview frame of the video, extracting its first frame if there is none."""
    from config import OUTPUT_DIR
    from utils import extract_first_frame, sanitize_filename

    # Check if we have a cached thumbnail
    video_name = Path(filepath).stem
    safe_name = sanitize_filename(video_name)
    output_dir = Path(OUTPUT_DIR) / safe_name

    # Try to find a representative frame from clusters first
    clusters_dir = output_dir / "clusters"
    if clusters_dir.exists():
        # Look for cluster_0 representative (main person)
        cluster_0 = clusters_dir / "cluster_0"
        if cluster_0.exists():
            rep_frame = cluster_0 / "representative.jpg"
            if rep_frame.exists():
                return str(rep_frame)
            # Fallback to any frame in cluster
            frame = next(cluster_0.glob("*.jpg"), None)
            if frame:
                return str(frame)

    # Try regular frames (any one: a video has thousands)
    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        frame = next(frames_dir.glob("*.jpg"), None)
        if frame:
            return str(frame)

    # Try preview frame (first frame extracted on registration)
    preview_path = output_dir / "preview.jpg"
    if preview_path.exists():
        return str(preview_path)

    # Generate preview on-demand if it doesn't exist. os.path.isfile, unlike
    # Path.exists, answers False for a network share that is not answering here.
    video_path = Path(filepath)
    if os.path.isfile(video_path):
        result = extract_first_frame(video_path, preview_path)
        if result:
            return str(result)

    return None

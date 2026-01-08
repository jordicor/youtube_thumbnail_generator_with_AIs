"""
Video Service

Business logic for video management operations.
"""

from pathlib import Path
from typing import Optional, List
import aiosqlite
import asyncio


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

    async def scan_videos_directory(self) -> dict:
        """Scan videos directory and register new videos."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import VIDEOS_DIR, VIDEO_EXTENSIONS, EXCLUDE_FOLDERS, OUTPUT_DIR
        from utils import find_videos, get_video_info, sanitize_filename, extract_first_frame

        videos = find_videos(Path(VIDEOS_DIR), VIDEO_EXTENSIONS, EXCLUDE_FOLDERS)

        new_count = 0
        existing_count = 0

        for video_path in videos:
            # Check if already registered
            query = "SELECT id FROM videos WHERE filepath = ?"
            async with self.db.execute(query, [str(video_path)]) as cursor:
                existing = await cursor.fetchone()

            if existing:
                existing_count += 1
                continue

            # Get video info
            info = get_video_info(video_path)

            # Insert new video
            insert_query = """
                INSERT INTO videos (filename, filepath, duration_seconds, status)
                VALUES (?, ?, ?, 'pending')
            """
            await self.db.execute(insert_query, [
                video_path.name,
                str(video_path),
                info.get('duration', 0)
            ])
            new_count += 1

            # Extract first frame for preview (run in thread to avoid blocking)
            video_name = video_path.stem
            safe_name = sanitize_filename(video_name)
            preview_path = Path(OUTPUT_DIR) / safe_name / "preview.jpg"
            await asyncio.to_thread(extract_first_frame, video_path, preview_path)

        await self.db.commit()

        return {
            "scanned": len(videos),
            "new": new_count,
            "existing": existing_count
        }

    async def upload_video(self, file) -> dict:
        """Handle video file upload."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import VIDEOS_DIR, OUTPUT_DIR
        from utils import get_video_info, sanitize_filename, extract_first_frame
        import aiofiles

        # Security: sanitize filename to prevent path traversal
        safe_filename = sanitize_filename(file.filename)
        if not safe_filename:
            safe_filename = "uploaded_video"

        # Save file with sanitized filename
        dest_path = Path(VIDEOS_DIR) / safe_filename
        async with aiofiles.open(dest_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Get video info
        info = get_video_info(dest_path)

        # Insert into database
        query = """
            INSERT INTO videos (filename, filepath, duration_seconds, status)
            VALUES (?, ?, ?, 'pending')
        """
        cursor = await self.db.execute(query, [
            safe_filename,
            str(dest_path),
            info.get('duration', 0)
        ])
        await self.db.commit()

        # Extract first frame for preview (run in thread to avoid blocking)
        video_name = dest_path.stem
        safe_name = sanitize_filename(video_name)
        preview_path = Path(OUTPUT_DIR) / safe_name / "preview.jpg"
        await asyncio.to_thread(extract_first_frame, dest_path, preview_path)

        return {
            "id": cursor.lastrowid,
            "filename": file.filename,
            "status": "pending"
        }

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
        await self.db.execute(
            f"UPDATE videos SET is_hidden = TRUE WHERE id IN ({placeholders})",
            video_ids
        )
        await self.db.commit()
        return len(video_ids)

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
        """Get a thumbnail preview for a video."""
        video = await self.get_video(video_id)
        if not video:
            return None

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import OUTPUT_DIR
        from utils import sanitize_filename

        # Check if we have a cached thumbnail
        video_name = Path(video['filepath']).stem
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
                frames = list(cluster_0.glob("*.jpg"))
                if frames:
                    return str(frames[0])

        # Try regular frames
        frames_dir = output_dir / "frames"
        if frames_dir.exists():
            frames = list(frames_dir.glob("*.jpg"))
            if frames:
                return str(frames[0])

        # Try preview frame (first frame extracted on registration)
        preview_path = output_dir / "preview.jpg"
        if preview_path.exists():
            return str(preview_path)

        # Generate preview on-demand if it doesn't exist
        from utils import extract_first_frame
        video_path = Path(video['filepath'])
        if video_path.exists():
            result = extract_first_frame(video_path, preview_path)
            if result:
                return str(result)

        return None

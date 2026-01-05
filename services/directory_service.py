"""
Directory Service

Business logic for directory management operations.
Handles multi-directory support for video sources.
"""

from pathlib import Path
from typing import Optional, List
from datetime import datetime
import aiosqlite


class DirectoryService:
    """Service for directory-related operations."""

    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    async def list_directories(self) -> List[dict]:
        """List all saved directories, ordered by most recently used."""
        query = """
            SELECT * FROM directories
            ORDER BY
                CASE WHEN last_scanned_at IS NOT NULL THEN last_scanned_at ELSE created_at END DESC
        """
        async with self.db.execute(query) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            directories = [dict(zip(columns, row)) for row in rows]

            # Add time_ago for UI display
            for d in directories:
                d['time_ago'] = self._format_time_ago(d.get('last_scanned_at'))

            return directories

    async def get_directory(self, directory_id: int) -> Optional[dict]:
        """Get a directory by ID."""
        query = "SELECT * FROM directories WHERE id = ?"
        async with self.db.execute(query, [directory_id]) as cursor:
            row = await cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                d = dict(zip(columns, row))
                d['time_ago'] = self._format_time_ago(d.get('last_scanned_at'))
                return d
            return None

    async def get_directory_by_path(self, path: str) -> Optional[dict]:
        """Get a directory by path."""
        # Normalize path for comparison
        normalized_path = str(Path(path).resolve())

        query = "SELECT * FROM directories WHERE path = ?"
        async with self.db.execute(query, [normalized_path]) as cursor:
            row = await cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None

    async def add_directory(self, path: str, name: str = None) -> dict:
        """
        Add a new directory.

        Args:
            path: Path to the directory
            name: Optional friendly name/alias

        Returns:
            The created directory record

        Raises:
            ValueError: If path doesn't exist or is not a directory
            ValueError: If directory already exists in database
        """
        # Validate path exists and is a directory
        path_obj = Path(path)
        if not path_obj.exists():
            raise ValueError(f"Path does not exist: {path}")
        if not path_obj.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        # Normalize path
        normalized_path = str(path_obj.resolve())

        # Check if already exists
        existing = await self.get_directory_by_path(normalized_path)
        if existing:
            raise ValueError(f"Directory already added: {normalized_path}")

        # Use folder name as default alias if not provided
        if not name:
            name = path_obj.name

        # Insert
        query = """
            INSERT INTO directories (path, name)
            VALUES (?, ?)
        """
        cursor = await self.db.execute(query, [normalized_path, name])
        await self.db.commit()

        return await self.get_directory(cursor.lastrowid)

    async def update_directory(self, directory_id: int, name: str) -> Optional[dict]:
        """Update directory name/alias."""
        query = "UPDATE directories SET name = ? WHERE id = ?"
        await self.db.execute(query, [name, directory_id])
        await self.db.commit()
        return await self.get_directory(directory_id)

    async def delete_directory(self, directory_id: int) -> bool:
        """
        Delete a directory from the database.
        Videos will have their directory_id set to NULL (ON DELETE SET NULL).
        """
        directory = await self.get_directory(directory_id)
        if not directory:
            return False

        query = "DELETE FROM directories WHERE id = ?"
        await self.db.execute(query, [directory_id])
        await self.db.commit()
        return True

    async def scan_directory(self, directory_id: int) -> dict:
        """
        Scan a directory for videos and register new ones.

        Returns:
            Dict with scan results: {scanned, new, existing}
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import VIDEO_EXTENSIONS, EXCLUDE_FOLDERS
        from utils import find_videos, get_video_info

        directory = await self.get_directory(directory_id)
        if not directory:
            raise ValueError(f"Directory not found: {directory_id}")

        dir_path = Path(directory['path'])
        if not dir_path.exists():
            raise ValueError(f"Directory path no longer exists: {directory['path']}")

        # Find all video files
        videos = find_videos(dir_path, VIDEO_EXTENSIONS, EXCLUDE_FOLDERS)

        new_count = 0
        existing_count = 0

        for video_path in videos:
            # Check if already registered
            query = "SELECT id FROM videos WHERE filepath = ?"
            async with self.db.execute(query, [str(video_path)]) as cursor:
                existing = await cursor.fetchone()

            if existing:
                # Update directory_id if not set (for backwards compatibility)
                update_query = """
                    UPDATE videos SET directory_id = ?
                    WHERE id = ? AND directory_id IS NULL
                """
                await self.db.execute(update_query, [directory_id, existing[0]])
                existing_count += 1
                continue

            # Get video info
            info = get_video_info(video_path)

            # Insert new video with directory_id
            insert_query = """
                INSERT INTO videos (filename, filepath, duration_seconds, status, directory_id)
                VALUES (?, ?, ?, 'pending', ?)
            """
            await self.db.execute(insert_query, [
                video_path.name,
                str(video_path),
                info.get('duration', 0),
                directory_id
            ])
            new_count += 1

        await self.db.commit()

        # Update directory scan time and video count
        total_videos = await self._count_videos_in_directory(directory_id)
        update_query = """
            UPDATE directories
            SET last_scanned_at = CURRENT_TIMESTAMP, video_count = ?
            WHERE id = ?
        """
        await self.db.execute(update_query, [total_videos, directory_id])
        await self.db.commit()

        return {
            "scanned": len(videos),
            "new": new_count,
            "existing": existing_count,
            "total": total_videos
        }

    async def get_videos(
        self,
        directory_id: int,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 50
    ) -> List[dict]:
        """Get videos for a specific directory."""
        query = "SELECT * FROM videos WHERE directory_id = ?"
        params = [directory_id]

        # Handle hidden filter
        if status == 'hidden':
            # Show only hidden videos
            query += " AND is_hidden = TRUE"
        else:
            # By default, exclude hidden videos
            query += " AND (is_hidden = FALSE OR is_hidden IS NULL)"

            if status:
                # "analyzed" filter includes both "analyzed" and "completed" states
                if status == 'analyzed':
                    query += " AND status IN ('analyzed', 'completed')"
                else:
                    query += " AND status = ?"
                    params.append(status)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, skip])

        async with self.db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    async def count_videos(self, directory_id: int, status: Optional[str] = None) -> int:
        """Count videos in a directory."""
        return await self._count_videos_in_directory(directory_id, status)

    async def _count_videos_in_directory(self, directory_id: int, status: str = None) -> int:
        """Internal method to count videos."""
        query = "SELECT COUNT(*) FROM videos WHERE directory_id = ?"
        params = [directory_id]

        # Handle hidden filter
        if status == 'hidden':
            query += " AND is_hidden = TRUE"
        else:
            # By default, exclude hidden videos
            query += " AND (is_hidden = FALSE OR is_hidden IS NULL)"

            if status:
                # "analyzed" filter includes both "analyzed" and "completed" states
                if status == 'analyzed':
                    query += " AND status IN ('analyzed', 'completed')"
                else:
                    query += " AND status = ?"
                    params.append(status)

        async with self.db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def count_hidden_videos(self, directory_id: int) -> int:
        """Count hidden videos in a directory."""
        query = "SELECT COUNT(*) FROM videos WHERE directory_id = ? AND is_hidden = TRUE"
        async with self.db.execute(query, [directory_id]) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def count_visible_videos(self, directory_id: int) -> int:
        """Count visible (non-hidden) videos in a directory."""
        query = "SELECT COUNT(*) FROM videos WHERE directory_id = ? AND (is_hidden = FALSE OR is_hidden IS NULL)"
        async with self.db.execute(query, [directory_id]) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    def _format_time_ago(self, timestamp_str: Optional[str]) -> str:
        """Format timestamp as human-readable 'time ago' string."""
        if not timestamp_str:
            return "Nunca"

        try:
            # Parse SQLite timestamp
            if 'T' in timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

            now = datetime.now()
            diff = now - timestamp

            seconds = diff.total_seconds()

            if seconds < 60:
                return "Hace un momento"
            elif seconds < 3600:
                minutes = int(seconds / 60)
                return f"Hace {minutes} min"
            elif seconds < 86400:
                hours = int(seconds / 3600)
                return f"Hace {hours} hora{'s' if hours > 1 else ''}"
            elif seconds < 604800:
                days = int(seconds / 86400)
                return f"Hace {days} dia{'s' if days > 1 else ''}"
            else:
                weeks = int(seconds / 604800)
                return f"Hace {weeks} semana{'s' if weeks > 1 else ''}"
        except Exception:
            return "Desconocido"

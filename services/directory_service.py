"""
Directory Service

Business logic for directory management operations.
Handles multi-directory support for video sources.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timezone
import aiosqlite
import orjson

from i18n.i18n import translate as t

logger = logging.getLogger(__name__)


# Library filters: every visible video falls in exactly one status group.
VIDEO_STATUS_GROUPS = {
    'pending': ('pending', 'cancelled'),
    'processing': ('analyzing', 'analyzing_scenes', 'analyzing_faces', 'clustering', 'transcribing', 'generating'),
    'analyzed': ('analyzed',),
    'completed': ('completed',),
    'error': ('error',),
}
_GROUP_BY_STATUS = {status: group for group, statuses in VIDEO_STATUS_GROUPS.items() for status in statuses}

# 'all' is every visible video; 'hidden' is every hidden video, whatever its status
LIBRARY_FILTERS = ('all', *VIDEO_STATUS_GROUPS, 'hidden')

LIBRARY_SORTS = {
    'recent': "v.created_at DESC, v.id DESC",
    'name': "v.filename COLLATE NOCASE ASC, v.id ASC",
    'duration': "v.duration_seconds IS NULL, v.duration_seconds DESC, v.id DESC",
    'activity': "v.updated_at DESC, v.id DESC",
}


# Parallel checks of video files: on a network share each one waits for a round trip
FILE_CHECK_WORKERS = 16


class MissingVideosError(Exception):
    """Videos of a directory that are not in the path it would move to."""

    def __init__(self, missing: List[str], hidden: int, total: int):
        super().__init__(f"{len(missing)} of {total} videos are not in the new path")
        self.missing = missing  # Paths relative to the directory, visible videos first
        self.hidden = hidden    # How many of the missing videos are hidden
        self.total = total


def _absolute_path(path: str) -> str:
    """Absolute form of a directory path as typed.

    Path.resolve() is not used: on Windows it turns a mapped drive (S:\\...)
    into its network share (\\\\server\\share\\...).
    """
    return os.path.abspath(path)


def _existing_directory(path: str) -> str:
    """Absolute path of an existing directory (ValueError if it is not one)."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    return _absolute_path(path)


def _files_exist(paths: List[str]) -> List[bool]:
    """Whether each path is an existing file."""
    with ThreadPoolExecutor(max_workers=FILE_CHECK_WORKERS) as pool:
        return list(pool.map(os.path.isfile, paths))


def _first_answering_path(directory: dict) -> Optional[str]:
    """The directory's path if it answers here, else its first alternative that does."""
    paths = [directory['path'], *directory['alternative_paths']]
    return next((path for path in paths if os.path.isdir(path)), None)


def _directory_from_row(columns: List[str], row) -> dict:
    """Directory record of a directories row, with its alternative paths as a list."""
    directory = dict(zip(columns, row))
    directory['alternative_paths'] = orjson.loads(directory['alternative_paths'])
    return directory


def _escape_like(text: str) -> str:
    """Escape LIKE wildcards so the text matches literally (ESCAPE '\\')."""
    return text.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


def _percent_encode_punctuation(text: str) -> str:
    """Encode ASCII punctuation the way downloaded filenames store it ("!" -> "%21")."""
    return ''.join(
        c if c.isalnum() or c in ' .-_' or not c.isascii() else f"%{ord(c):02X}"
        for c in text
    )


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
            directories = [_directory_from_row(columns, row) for row in rows]

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
                d = _directory_from_row(columns, row)
                d['time_ago'] = self._format_time_ago(d.get('last_scanned_at'))
                return d
            return None

    async def _check_paths_free(self, paths: List[str], directory_id: Optional[int] = None) -> None:
        """
        Raise ValueError if another directory uses one of the paths, as its
        path or as an alternative: both could end up with the same path.
        """
        query = "SELECT path, alternative_paths FROM directories WHERE id IS NOT ?"
        async with self.db.execute(query, [directory_id]) as cursor:
            rows = await cursor.fetchall()
        taken = {used for path, alternatives in rows for used in (path, *orjson.loads(alternatives))}
        for path in paths:
            if path in taken:
                raise ValueError(f"Path already used by another directory: {path}")

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
            ValueError: If another directory uses the path
        """
        normalized_path = _existing_directory(path)
        await self._check_paths_free([normalized_path])

        # Use folder name as default alias if not provided
        if not name:
            name = Path(normalized_path).name

        # Insert
        query = """
            INSERT INTO directories (path, name)
            VALUES (?, ?)
        """
        cursor = await self.db.execute(query, [normalized_path, name])
        await self.db.commit()

        return await self.get_directory(cursor.lastrowid)

    async def update_directory(
        self,
        directory_id: int,
        name: str,
        path: str,
        alternative_paths: List[str],
        allow_missing_videos: bool = False
    ) -> Optional[dict]:
        """
        Update a directory's name/alias, path and alternative paths.

        A new path is for the same folder reached another way (another drive
        letter, share or mount point): every video of the directory gets its
        path rewritten under it. Nothing is moved on disk, and the videos keep
        their analysis and thumbnails, whose output folder only depends on
        the video file name.

        Alternative paths are the same folder from other machines, so they need
        not exist here: the directory switches to them when its path does not
        answer (see switch_to_answering_path).

        Args:
            directory_id: The directory ID
            name: New name/alias (blank: the folder name)
            path: New path of the directory (may be the current one)
            alternative_paths: Absolute paths, by preference
            allow_missing_videos: Move the directory even if some of its
                videos are not found in the new path

        Returns:
            The updated directory record, or None if it does not exist

        Raises:
            ValueError: If the path is not an existing directory, an alternative
                is not absolute, a path is listed twice, another directory uses
                one of the paths, or a video's path under one of them belongs
                to another video
            MissingVideosError: If some videos are not in the new path and
                allow_missing_videos is False
        """
        directory = await self.get_directory(directory_id)
        if not directory:
            return None

        new_path = _absolute_path(path)
        name = name.strip() or Path(new_path).name
        alternatives = []
        for alternative in alternative_paths:
            if not Path(alternative).is_absolute():
                raise ValueError(f"Alternative path is not absolute: {alternative}")
            alternative = _absolute_path(alternative)
            if alternative in (new_path, *alternatives):
                raise ValueError(f"Path listed twice: {alternative}")
            alternatives.append(alternative)
        await self._check_paths_free([new_path, *alternatives], directory_id)

        # Only a new path must exist: a directory can be renamed while it is unreachable
        moves = []
        if new_path != directory['path']:
            _existing_directory(new_path)
            moves = await self._video_moves(directory, new_path)
            if not allow_missing_videos:
                await self._check_videos_present(moves)
        # The directory may switch to any alternative later: its videos must fit there
        for alternative in alternatives:
            await self._video_moves(directory, alternative)

        await self._save_directory(directory_id, name, new_path, alternatives, moves)
        return await self.get_directory(directory_id)

    async def switch_unanswering_directories(self) -> Dict[int, str]:
        """
        Point every directory whose path does not answer here at its first
        alternative path that does (see _switch). Directories without
        alternatives are not checked.

        Returns:
            {directory_id: path it switched from} of the directories that switched
        """
        directories = [d for d in await self.list_directories() if d['alternative_paths']]
        # A path that does not answer can keep its check waiting: check them all at once
        answering = await asyncio.gather(*(asyncio.to_thread(_first_answering_path, d) for d in directories))

        switched = {}
        # One at a time: every switch is a transaction of this connection
        for directory, path in zip(directories, answering):
            if await self._switch(directory, path):
                switched[directory['id']] = directory['path']
        return switched

    async def switch_to_answering_path(self, directory_id: int) -> Optional[str]:
        """
        switch_unanswering_directories() for one directory.

        Returns:
            The path it switched from, or None if it kept its path
        """
        directory = await self.get_directory(directory_id)
        if not directory or not directory['alternative_paths']:
            return None
        path = await asyncio.to_thread(_first_answering_path, directory)
        return directory['path'] if await self._switch(directory, path) else None

    async def switch_video_directory(self, video_id: int) -> Optional[str]:
        """switch_to_answering_path() for the directory of a video, before reading its file."""
        async with self.db.execute("SELECT directory_id FROM videos WHERE id = ?", [video_id]) as cursor:
            row = await cursor.fetchone()
        if not row or row[0] is None:
            return None
        return await self.switch_to_answering_path(row[0])

    async def _switch(self, directory: dict, answering: Optional[str]) -> bool:
        """
        Move a directory to the one of its paths that answers here, unless it
        is its path already or none answers. The old path takes that
        alternative's place in the list, so the directory switches back on a
        machine where only the old one answers. Videos missing in the new path
        move anyway: nobody is there to confirm, and they keep their analysis.

        Returns:
            True if the directory moved
        """
        if answering is None or answering == directory['path']:
            return False
        moves = await self._video_moves(directory, answering)
        alternatives = [directory['path'] if path == answering else path for path in directory['alternative_paths']]
        await self._save_directory(directory['id'], directory['name'], answering, alternatives, moves)
        logger.info(f"Directory {directory['id']} did not answer at {directory['path']}; now at {answering}")
        return True

    async def _save_directory(
        self, directory_id: int, name: str, path: str, alternatives: List[str], moves: List[tuple]
    ) -> None:
        """Store a directory's name, path and alternatives and its videos' new paths, all at once."""
        try:
            for video_id, filepath, _, _ in moves:
                await self.db.execute("UPDATE videos SET filepath = ? WHERE id = ?", [filepath, video_id])
            await self.db.execute(
                "UPDATE directories SET name = ?, path = ?, alternative_paths = ? WHERE id = ?",
                [name, path, orjson.dumps(alternatives).decode(), directory_id]
            )
            await self.db.commit()
        except BaseException:
            await self.db.rollback()
            raise

    async def _check_videos_present(self, moves: List[tuple]) -> None:
        """Raise MissingVideosError if some moved videos are not in their new path."""
        found = await asyncio.to_thread(_files_exist, [filepath for _, filepath, _, _ in moves])
        missing = sorted(
            ((relative, hidden) for (_, _, relative, hidden), ok in zip(moves, found) if not ok),
            key=lambda video: video[1]  # Visible videos first
        )
        if missing:
            raise MissingVideosError(
                [relative for relative, _ in missing],
                sum(hidden for _, hidden in missing),
                len(moves)
            )

    async def _video_moves(self, directory: dict, new_path: str) -> List[tuple]:
        """
        New path of every video of a directory moving to new_path.

        Returns:
            (video_id, new filepath, path relative to the directory, hidden) of each video

        Raises:
            ValueError: If a video is not inside the directory, or its new path
                belongs to another video
        """
        old_root = Path(directory['path'])
        new_root = Path(new_path)

        query = """
            SELECT id, filepath, COALESCE(is_hidden, FALSE) FROM videos
            WHERE directory_id = ? ORDER BY filepath
        """
        async with self.db.execute(query, [directory['id']]) as cursor:
            videos = await cursor.fetchall()

        moves = []
        for video_id, filepath, hidden in videos:
            try:
                relative = Path(filepath).relative_to(old_root)
            except ValueError:
                raise ValueError(f"Video {video_id} is not inside its directory: {filepath}") from None
            new_filepath = str(new_root / relative)

            query = "SELECT id FROM videos WHERE filepath = ? AND id != ?"
            async with self.db.execute(query, [new_filepath, video_id]) as cursor:
                clash = await cursor.fetchone()
            if clash:
                raise ValueError(f"Video {clash[0]} already has the new path of video {video_id}: {new_filepath}")

            moves.append((video_id, new_filepath, str(relative), bool(hidden)))
        return moves

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

        await self.switch_to_answering_path(directory_id)
        directory = await self.get_directory(directory_id)
        if not directory:
            raise ValueError(f"Directory not found: {directory_id}")

        # Walking the folders and probing each new video (ffprobe, up to 30 s
        # each, often on a network share) run in a thread: on the event loop
        # they would leave every other page and stream without an answer
        dir_path = Path(directory['path'])
        if not await asyncio.to_thread(dir_path.exists):
            raise ValueError(f"Directory path no longer exists: {directory['path']}")

        # Find all video files
        videos = await asyncio.to_thread(find_videos, dir_path, VIDEO_EXTENSIONS, EXCLUDE_FOLDERS)

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
            info = await asyncio.to_thread(get_video_info, video_path)

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
        total_videos = await self.count_visible_videos(directory_id)
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

    async def list_library_videos(
        self,
        directory_id: int,
        filter_name: str = 'all',
        search: Optional[str] = None,
        sort: str = 'recent',
        skip: int = 0,
        limit: int = 50
    ) -> List[dict]:
        """
        Get one page of a directory's videos for the library view.

        Args:
            directory_id: The directory ID
            filter_name: One of LIBRARY_FILTERS
            search: Text the filename must contain (case-insensitive for ASCII)
            sort: One of LIBRARY_SORTS
            skip: Offset for pagination
            limit: Page size

        Returns:
            Video rows plus thumbnail_count and person_count.
        """
        where, params = self._library_where(directory_id, filter_name, search)
        query = f"""
            SELECT v.*,
                   (SELECT COUNT(*) FROM thumbnails t
                    JOIN generation_jobs j ON j.id = t.job_id
                    WHERE j.video_id = v.id) AS thumbnail_count,
                   (SELECT COUNT(*) FROM clusters c
                    WHERE c.video_id = v.id AND c.view_mode = 'person') AS person_count
            FROM videos v
            WHERE {where}
            ORDER BY {LIBRARY_SORTS[sort]}
            LIMIT ? OFFSET ?
        """
        async with self.db.execute(query, params + [limit, skip]) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    async def count_library_videos(self, directory_id: int, search: Optional[str] = None) -> dict:
        """
        Count a directory's videos for every library filter.

        Every visible video belongs to exactly one status group, so the group
        counts add up to 'all'. Hidden videos only count in 'hidden'.
        """
        where, params = self._library_where(directory_id, None, search)
        query = f"""
            SELECT v.status, COALESCE(v.is_hidden, FALSE) AS hidden, COUNT(*)
            FROM videos v
            WHERE {where}
            GROUP BY v.status, hidden
        """
        counts = dict.fromkeys(LIBRARY_FILTERS, 0)
        async with self.db.execute(query, params) as cursor:
            for status, hidden, count in await cursor.fetchall():
                if hidden:
                    counts['hidden'] += count
                    continue
                if status not in _GROUP_BY_STATUS:
                    raise ValueError(f"Video status {status!r} belongs to no library group")
                counts['all'] += count
                counts[_GROUP_BY_STATUS[status]] += count
        return counts

    def _library_where(self, directory_id: int, filter_name: Optional[str], search: Optional[str]):
        """Build the WHERE clause of the library queries (None filter: every video)."""
        clauses = ["v.directory_id = ?"]
        params = [directory_id]

        if filter_name == 'hidden':
            clauses.append("v.is_hidden = TRUE")
        elif filter_name is not None:
            clauses.append("(v.is_hidden = FALSE OR v.is_hidden IS NULL)")
            if filter_name != 'all':
                statuses = VIDEO_STATUS_GROUPS[filter_name]
                clauses.append(f"v.status IN ({','.join('?' * len(statuses))})")
                params.extend(statuses)

        if search:
            # Downloaded filenames keep punctuation percent-encoded ("%21" for "!"),
            # so match the typed text as is and in that encoded form.
            clauses.append("(v.filename LIKE ? ESCAPE '\\' OR v.filename LIKE ? ESCAPE '\\')")
            params.append(f"%{_escape_like(search)}%")
            params.append(f"%{_escape_like(_percent_encode_punctuation(search))}%")

        return " AND ".join(clauses), params

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
            return t('time_service.never')

        try:
            # Parse SQLite timestamp
            if 'T' in timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                # Naive timestamps from SQLite are treated as UTC
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

            # Use tz-aware now so subtraction works regardless of timestamp tz-info
            now = datetime.now(timezone.utc)
            diff = now - timestamp

            seconds = diff.total_seconds()

            if seconds < 60:
                return t('time_service.just_now')
            elif seconds < 3600:
                minutes = int(seconds / 60)
                return t('time_service.minutes_ago', count=minutes)
            elif seconds < 86400:
                hours = int(seconds / 3600)
                if hours == 1:
                    return t('time_service.hour_ago')
                return t('time_service.hours_ago', count=hours)
            elif seconds < 604800:
                days = int(seconds / 86400)
                if days == 1:
                    return t('time_service.day_ago')
                return t('time_service.days_ago', count=days)
            else:
                weeks = int(seconds / 604800)
                if weeks == 1:
                    return t('time_service.week_ago')
                return t('time_service.weeks_ago', count=weeks)
        except Exception:
            return t('time_service.unknown')

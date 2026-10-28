"""
Person Service

Business logic of the person editor. A person is a cluster of one view
('person' or 'person_scene'). Its frames are named by their filename inside
output/{video}/frames/, the folder every view draws from.
"""

import asyncio
import re
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

import aiosqlite
import orjson
from PIL import Image

from i18n import t
from services.analysis_service import AnalysisService
from services.face_index import (
    FaceIndex,
    face_reference_view,
    portrait_crop_box,
    load_face_index,
    match_faces,
    person_centroid,
    stored_centroid,
)
from utils import ReferenceImageError, to_output_relative
from services.face_selection import select_face_references
from services.reference_service import ReferenceService, VisualReference

FRAME_NAME_PATTERN = re.compile(r'scene_(\d+)_frame_(\d+)')

PERSON_SUMMARY_FIELDS = (
    'id', 'cluster_index', 'label', 'view_mode', 'scene_index', 'parent_cluster_index',
    'parent_label', 'num_frames', 'reference_count', 'representative_frame_id'
)


class PersonEditError(ReferenceImageError):
    """An edit the person editor cannot apply; the message is shown to the user."""


def _frame_name(stored_path: str) -> str:
    return PurePosixPath(stored_path).name


def _summary(cluster: dict) -> dict:
    return {field: cluster[field] for field in PERSON_SUMMARY_FIELDS}


class PersonService:
    """Create people and edit their frames, references, name and notes."""

    def __init__(self, db: aiosqlite.Connection):
        self.db = db
        self.analysis = AnalysisService(db)

    # =========================================================================
    # READ
    # =========================================================================

    async def get_person(self, video_id: int, person_id: int) -> Optional[dict]:
        """A person with its member frames and its ordered references."""
        async with self.db.execute(
            "SELECT c.view_mode, c.reference_format, v.reference_revision FROM clusters c JOIN videos v ON v.id=c.video_id WHERE c.id = ? AND c.video_id = ?", [person_id, video_id]
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        clusters = await self.analysis.get_clusters(video_id, row[0])
        cluster = next(c for c in clusters if c["id"] == person_id)
        async with self.db.execute("""
            SELECT vf.frame_path FROM cluster_frame_assignments cfa
            JOIN video_frames vf ON vf.id=cfa.frame_id WHERE cfa.cluster_id=? ORDER BY vf.frame_path
        """, [person_id]) as cursor:
            members = [_frame_name(row[0]) for row in await cursor.fetchall()]
        references = await ReferenceService(self.db).list(video_id, person_id)
        return {**_summary(cluster), "description": cluster["description"], "members": members,
                "references": references,
                "reference_format": row[1], "revision": row[2]}

    async def body_candidates(self, video_id: int, person_id: int, analyze: bool = False) -> Optional[dict]:
        """Body crops of the person found in its frames, best first.

        Without analyze only an existing body analysis is read; with it the frames
        are analyzed first when they have none (local, can take a while).
        """
        from services.body_references import ensure_body_index, load_body_index, person_body_candidates
        person = await self.get_person(video_id, person_id)
        if person is None:
            return None
        output_dir = await self._output_dir(video_id)
        index = await self._face_index(output_dir)
        matches = match_faces(index, await self._centroid(index, person), person['members'])
        data = await asyncio.to_thread(ensure_body_index if analyze else load_body_index, output_dir)
        from services.face_selection import same_look_frames
        primary = next((ref["source"] for ref in person["references"] if ref["role"] == "face"
                        and ref["primary"] and ref["source_type"] == "frame"), None)
        allowed = same_look_frames(matches, index, primary)
        return person_body_candidates(data, {name: match for name, match in matches.items() if name in allowed})

    async def automatic_body(self, video_id: int, person_id: int) -> Optional[list[dict]]:
        """Clear body references with complementary orientations, without filling a quota."""
        from services.body_references import select_body_references
        body = await self.body_candidates(video_id, person_id, analyze=True)
        if body is None:
            return None
        output_dir = await self._output_dir(video_id)
        candidates = await asyncio.to_thread(select_body_references, output_dir / 'frames', body['candidates'])
        if not candidates:
            raise PersonEditError(t('api.errors.no_color_body'))
        return [VisualReference(
            role="body", person_id=person_id, source_type="frame", source=candidate["filename"],
            crop=candidate["box"], coverage=candidate["coverage"], posture=candidate["posture"],
            view=candidate["view"], identity_confirmed=True,
        ).model_dump() for candidate in candidates]

    async def list_frames(
        self,
        video_id: int,
        view_mode: str,
        person_id: Optional[int] = None
    ) -> Optional[dict]:
        """Every frame of the video with the people of the view that hold it.

        With a person, each frame carries the face closest to that person and
        its similarity; the view is the person's own.
        """
        video = await self.analysis.get_video(video_id)
        if not video:
            return None
        output_dir = self.analysis._get_video_output_dir(video)

        person = None
        if person_id is not None:
            person = await self.get_person(video_id, person_id)
            if person is None:
                return None
            view_mode = person['view_mode']

        names = self._frame_files(output_dir)
        async with self.db.execute(
            "SELECT frame_path, quality_score, expression FROM video_frames WHERE video_id = ?",
            [video_id]
        ) as cursor:
            stored = {_frame_name(path): (quality, expression) for path, quality, expression in await cursor.fetchall()}

        holders: Dict[str, List[int]] = {}
        async with self.db.execute("""
            SELECT vf.frame_path, c.id
            FROM cluster_frame_assignments cfa
            JOIN video_frames vf ON vf.id = cfa.frame_id
            JOIN clusters c ON c.id = cfa.cluster_id
            WHERE c.video_id = ? AND c.view_mode = ?
            ORDER BY c.cluster_index
        """, [video_id, view_mode]) as cursor:
            for path, cluster_id in await cursor.fetchall():
                holders.setdefault(_frame_name(path), []).append(cluster_id)

        index = await self._face_index(output_dir)
        centroid = await self._centroid(index, person) if person else None
        matches = match_faces(index, centroid, names)

        frames = []
        for name in names:
            scene_match = FRAME_NAME_PATTERN.search(name)
            match = matches.get(name)
            quality, expression = stored.get(name, (None, None))
            frames.append({
                'filename': name,
                'scene_index': int(scene_match.group(1)) if scene_match else None,
                'frame_number': int(scene_match.group(2)) if scene_match else None,
                'quality_score': match.quality_score if match else quality,
                'expression': match.expression if match else expression,
                'face_box': match.box if match else None,
                'similarity': match.similarity if match else None,
                'yaw': match.yaw if match else None,
                'face_count': index.spans.get(name, (0, 0))[1] - index.spans.get(name, (0, 0))[0],
                'people': holders.get(name, []),
            })

        clusters = await self.analysis.get_clusters(video_id, view_mode)
        result = {
            'view_mode': view_mode,
            'frames': frames,
            'people': [_summary(cluster) for cluster in clusters],
        }
        return result

    async def match_frames(self, video_id: int, view_mode: str, members: List[str]) -> Optional[dict]:
        """Face of every frame closest to the new person of a view these frames would form.

        In each frame the person is a face the people of the view do not have yet.
        """
        video = await self.analysis.get_video(video_id)
        if not video:
            return None
        output_dir = self.analysis._get_video_output_dir(video)
        names = self._frame_files(output_dir)
        self._check_frames(members, set(names), allow_empty=True)

        index = await self._face_index(output_dir)
        claimed = await self.analysis.view_claimed_faces(index, video_id, view_mode)
        matches = match_faces(index, person_centroid(index, members, claimed=claimed), names)
        return {
            name: {
                'face_box': match.box,
                'similarity': match.similarity,
                'quality_score': match.quality_score,
                'expression': match.expression,
            }
            for name, match in matches.items()
        }

    async def automatic_references(self, video_id, view_mode, person_id, members):
        """Suggest a clear primary and genuinely different optional views, never fill a quota."""
        output_dir = await self._output_dir(video_id)
        self._check_frames(members, set(self._frame_files(output_dir)), allow_empty=False)
        index = await self._face_index(output_dir)
        if person_id is None:
            claimed = await self.analysis.view_claimed_faces(index, video_id, view_mode)
            centroid = person_centroid(index, members, claimed=claimed)
        else:
            person = await self.get_person(video_id, person_id)
            if person is None:
                return None
            centroid = await self._centroid(index, person)
        matches = match_faces(index, centroid, members)
        names = await asyncio.to_thread(select_face_references, output_dir / "frames", matches, index)
        if not names:
            raise PersonEditError(t('api.errors.no_clear_face'))
        picked = []
        for name in names:
            match = matches[name]
            crop = None
            if person_id is None:
                row = match.row
                others = [index.boxes[i].tolist() for i in range(*index.spans[name]) if i != row]
                with Image.open(output_dir / "frames" / name) as original:
                    crop = portrait_crop_box(index.boxes[row].tolist(), others, *original.size)
            picked.append(VisualReference(
                role="face", person_id=person_id or -1, source_type="frame", source=name,
                primary=not picked, view=face_reference_view(match.yaw), expression=match.expression,
                crop=crop, identity_confirmed=person_id is None,
            ).model_dump())
        return picked

    async def get_frame_path(self, video_id: int, filename: str) -> Optional[Path]:
        """Path of a frame image of the video, None if it does not exist."""
        video = await self.analysis.get_video(video_id)
        if not video or Path(filename).name != filename:
            return None
        frame_path = self.analysis._get_video_output_dir(video) / 'frames' / filename
        return frame_path if frame_path.is_file() else None

    async def render_avatar(self, video_id: int, person_id: int) -> Optional[bytes]:
        """Exact primary face crop, including uploaded originals."""
        person = await self.get_person(video_id, person_id)
        if person is None:
            return None
        primary = next((r for r in person["references"] if r["role"] == "face" and r["primary"]), None)
        if primary:
            return await ReferenceService(self.db).crop_bytes(video_id, primary)
        return None

    async def create_person(self, video_id, view_mode, label, description, members,
                            references, reference_format, revision):
        video = await self.analysis.get_video(video_id)
        if video is None:
            return None
        output = self.analysis._get_video_output_dir(video)
        self._check_frames(members, set(self._frame_files(output)), allow_empty=True)
        await self.db.execute("SAVEPOINT create_person_references")
        try:
            person_id = await self._insert_person(video_id, output, view_mode, members, [], label, description)
            references = [dict(ref, person_id=person_id) for ref in references]
            await self._save_reference_edit(video_id, person_id, label, description, references,
                                            reference_format, revision)
        except BaseException:
            await self.db.execute("ROLLBACK TO create_person_references")
            raise
        finally:
            await self.db.execute("RELEASE create_person_references")
        await self.db.commit()
        await self._sync_caches(video_id, output)
        return await self.get_person(video_id, person_id)

    async def update_person(self, video_id, person_id, label, description, references,
                            reference_format, revision):
        person = await self.get_person(video_id, person_id)
        if person is None:
            return None
        await self.db.execute("SAVEPOINT edit_person_references")
        try:
            await self._save_reference_edit(video_id, person_id, label, description, references,
                                            reference_format, revision)
        except BaseException:
            await self.db.execute("ROLLBACK TO edit_person_references")
            raise
        finally:
            await self.db.execute("RELEASE edit_person_references")
        await self.db.commit()
        await self._sync_caches(video_id, await self._output_dir(video_id))
        return await self.get_person(video_id, person_id)

    async def _save_reference_edit(self, video_id, person_id, label, description, references,
                                   reference_format, revision):
        service = ReferenceService(self.db)
        if any(r["role"] == "environment" or r["person_id"] != person_id for r in references):
            raise PersonEditError("Only edit the face and body references of this person.")
        cleaned = await service.keep_automatic(
            video_id, await service.validate(video_id, references, [{"id": person_id, "label": label}], None))
        if reference_format not in ("photos", "collage"):
            raise PersonEditError("Unknown face presentation format.")
        # Materialize/check every crop before changing persisted choices.
        for ref in cleaned:
            await service.crop_bytes(video_id, ref)
        cursor = await self.db.execute(
            "UPDATE videos SET reference_revision=reference_revision+1 WHERE id=? AND reference_revision=?",
            [video_id, revision],
        )
        if cursor.rowcount != 1:
            raise PersonEditError(t('api.errors.references_changed'))
        await self.db.execute("DELETE FROM visual_references WHERE video_id=? AND person_id=?", [video_id, person_id])
        await service.insert(video_id, cleaned)
        await self.db.execute("UPDATE clusters SET label=?, description=?, reference_format=? WHERE id=?",
                              [_clean(label), _clean(description), reference_format, person_id])
        await self._refresh_person(person_id)

    async def add_frames(self, video_id: int, person_id: int, filenames: List[str]) -> Optional[dict]:
        """Add frames of the video to a person (frames it already has stay as they are)."""
        person = await self.get_person(video_id, person_id)
        if person is None:
            return None
        output_dir = await self._output_dir(video_id)
        self._check_frames(filenames, set(self._frame_files(output_dir)), allow_empty=False)

        await self._assign(person_id, await self._frame_ids(video_id, output_dir, filenames))
        await self._refresh_person(person_id)
        await self.analysis.fit_person_centroid(person_id, output_dir)
        await self.db.commit()
        await self._sync_caches(video_id, output_dir)
        return await self.get_person(video_id, person_id)

    async def remove_frames(self, video_id: int, person_id: int, filenames: List[str]) -> Optional[dict]:
        """Take frames out of a person; the frames stay in the video."""
        person = await self.get_person(video_id, person_id)
        if person is None:
            return None
        self._check_removal(filenames, person)
        output_dir = await self._output_dir(video_id)

        await self._unassign(person_id, await self._frame_ids(video_id, output_dir, filenames))
        await self._drop_frame_references(video_id, person_id, filenames)
        await self._refresh_person(person_id)
        await self.analysis.fit_person_centroid(person_id, output_dir)
        await self.db.commit()
        await self._sync_caches(video_id, output_dir)
        return await self.get_person(video_id, person_id)

    async def move_frames(
        self,
        video_id: int,
        person_id: int,
        filenames: List[str],
        target_id: Optional[int]
    ) -> Optional[dict]:
        """Move frames of a person to another person of its view, or to a new one."""
        person = await self.get_person(video_id, person_id)
        if person is None:
            return None
        self._check_removal(filenames, person)
        output_dir = await self._output_dir(video_id)
        if target_id is not None:
            target = await self.get_person(video_id, target_id)
            if target is None or target_id == person_id or target['view_mode'] != person['view_mode']:
                raise PersonEditError(t('api.errors.invalid_move_target'))

        # The frames leave first: a new person takes, in each of them, the face
        # the people of the view no longer claim
        await self._unassign(person_id, await self._frame_ids(video_id, output_dir, filenames))
        await self._drop_frame_references(video_id, person_id, filenames)
        await self._refresh_person(person_id)
        await self.analysis.fit_person_centroid(person_id, output_dir)

        if target_id is None:
            target_id = await self._insert_person(
                video_id, output_dir, person['view_mode'], filenames, [], None, None
            )
        else:
            await self._assign(target_id, await self._frame_ids(video_id, output_dir, filenames))
            await self._refresh_person(target_id)
            await self.analysis.fit_person_centroid(target_id, output_dir)
        await self.db.commit()
        await self._sync_caches(video_id, output_dir)
        return {
            'person': await self.get_person(video_id, person_id),
            'target': await self.get_person(video_id, target_id),
        }

    async def delete_frames(self, video_id: int, filenames: List[str]) -> Optional[dict]:
        """Delete frames from disk and from every person of both views.

        People left without frames are deleted too.
        """
        video = await self.analysis.get_video(video_id)
        if not video:
            return None
        output_dir = self.analysis._get_video_output_dir(video)
        self._check_frames(filenames, set(self._frame_files(output_dir)), allow_empty=False)

        refs = await ReferenceService(self.db).list(video_id)
        if any(ref["source_type"] == "frame" and ref["source"] in filenames for ref in refs):
            raise PersonEditError(t('api.errors.frames_in_use'))
        stored_paths = [self._stored_path(output_dir, name) for name in filenames]
        placeholders = ','.join('?' * len(stored_paths))
        async with self.db.execute(f"""
            SELECT DISTINCT cfa.cluster_id
            FROM cluster_frame_assignments cfa
            JOIN video_frames vf ON vf.id = cfa.frame_id
            WHERE vf.video_id = ? AND vf.frame_path IN ({placeholders})
            ORDER BY cfa.cluster_id
        """, [video_id, *stored_paths]) as cursor:
            affected = [row[0] for row in await cursor.fetchall()]

        await self.db.execute(
            f"DELETE FROM video_frames WHERE video_id = ? AND frame_path IN ({placeholders})",
            [video_id, *stored_paths]
        )

        emptied = []
        for cluster_id in affected:
            async with self.db.execute(
                "SELECT COUNT(*) FROM cluster_frame_assignments WHERE cluster_id = ?", [cluster_id]
            ) as cursor:
                if (await cursor.fetchone())[0] == 0:
                    emptied.append(cluster_id)
                    continue
            await self._refresh_person(cluster_id)
            await self.analysis.fit_person_centroid(cluster_id, output_dir)

        views = []
        if emptied:
            emptied_placeholders = ','.join('?' * len(emptied))
            async with self.db.execute(
                f"SELECT DISTINCT view_mode FROM clusters WHERE id IN ({emptied_placeholders})", emptied
            ) as cursor:
                views = [row[0] for row in await cursor.fetchall()]
            await self.db.execute(f"DELETE FROM clusters WHERE id IN ({emptied_placeholders})", emptied)
        # Indices of the remaining 'person' clusters before reindexing, to rename their folders
        old_index_by_id = await self.analysis._get_cluster_index_map(video_id, 'person')
        for view_mode in views:
            await self.analysis._reindex_clusters_v2(video_id, view_mode)
        await self.db.commit()

        for name in filenames:
            (output_dir / 'frames' / name).unlink()

        if 'person' in views:
            # Also rewrites the JSON caches
            await self.analysis._sync_cluster_folders(video_id, output_dir, old_index_by_id)
        else:
            await self._sync_caches(video_id, output_dir)
        return {'deleted': len(filenames), 'deleted_people': emptied}

    # =========================================================================
    # HELPERS
    # =========================================================================

    async def _output_dir(self, video_id: int) -> Path:
        return self.analysis._get_video_output_dir(await self.analysis.get_video(video_id))

    @staticmethod
    async def _face_index(output_dir: Path) -> FaceIndex:
        # Parsing a large faces.json takes a moment the first time
        return await asyncio.to_thread(load_face_index, output_dir / 'faces.json')

    async def _centroid(self, index: FaceIndex, person: dict):
        """The person's stored face embedding (None when none of their frames shows a face)."""
        return stored_centroid(await self.analysis.centroid_blob(person['id']), index, person['members'], person['id'])

    @staticmethod
    def _frame_files(output_dir: Path) -> List[str]:
        frames_dir = output_dir / 'frames'
        return sorted(path.name for path in frames_dir.glob('*.jpg')) if frames_dir.is_dir() else []

    @staticmethod
    def _stored_path(output_dir: Path, name: str) -> str:
        return to_output_relative(output_dir / 'frames' / name)

    @staticmethod
    def _check_frames(filenames: List[str], available: set, allow_empty: bool) -> None:
        if not filenames and not allow_empty:
            raise PersonEditError(t('api.errors.min_one_frame_required'))
        if len(set(filenames)) != len(filenames):
            raise PersonEditError(t('api.errors.duplicate_frames'))
        missing = [name for name in filenames if name not in available]
        if missing:
            raise PersonEditError(t('api.errors.frame_not_found') + f": {missing[0]}")

    @staticmethod
    def _check_removal(filenames: List[str], person: dict) -> None:
        if not filenames:
            raise PersonEditError(t('api.errors.min_one_frame_required'))
        if len(set(filenames)) != len(filenames):
            raise PersonEditError(t('api.errors.duplicate_frames'))
        if not set(filenames) <= set(person['members']):
            raise PersonEditError(t('api.errors.frame_not_in_person'))
        if len(filenames) == len(person['members']):
            raise PersonEditError(t('api.errors.person_needs_frames'))

    async def _frame_ids(self, video_id: int, output_dir: Path, filenames: List[str]) -> List[int]:
        """Database IDs of frames, registering the frames the database lacks."""
        best_faces = match_faces(await self._face_index(output_dir), None, filenames)
        ids = []
        for name in filenames:
            stored_path = self._stored_path(output_dir, name)
            async with self.db.execute(
                "SELECT id FROM video_frames WHERE video_id = ? AND frame_path = ?", [video_id, stored_path]
            ) as cursor:
                row = await cursor.fetchone()
            if row:
                ids.append(row[0])
                continue
            # Frames whose faces clustering left out (or without faces) are not registered yet
            face = best_faces.get(name)
            scene_match = FRAME_NAME_PATTERN.search(name)
            cursor = await self.db.execute("""
                INSERT INTO video_frames (video_id, frame_path, quality_score, expression, scene_index)
                VALUES (?, ?, ?, ?, ?)
            """, [video_id, stored_path, face.quality_score if face else 0,
                  face.expression if face else 'unknown', int(scene_match.group(1)) if scene_match else None])
            ids.append(cursor.lastrowid)
        return ids

    async def _insert_person(
        self,
        video_id: int,
        output_dir: Path,
        view_mode: str,
        members: List[str],
        references: List[str],
        label: Optional[str],
        description: Optional[str]
    ) -> int:
        async with self.db.execute(
            "SELECT COALESCE(MAX(cluster_index), -1) + 1 FROM clusters WHERE video_id = ? AND view_mode = ?",
            [video_id, view_mode]
        ) as cursor:
            next_index = (await cursor.fetchone())[0]

        # The representative is set by _refresh_person once the frames are in
        cursor = await self.db.execute("""
            INSERT INTO clusters (video_id, cluster_index, label, description, view_mode,
                                  num_frames, representative_frame)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [video_id, next_index, _clean(label), _clean(description), view_mode, len(members),
              self._stored_path(output_dir, members[0]) if members else ''])
        person_id = cursor.lastrowid

        await self._assign(person_id, await self._frame_ids(video_id, output_dir, members))
        await ReferenceService(self.db).replace_person_faces(video_id, person_id, references)
        await self._refresh_person(person_id)
        if members:
            await self.analysis.fit_person_centroid(person_id, output_dir)
        return person_id

    async def _drop_frame_references(self, video_id: int, person_id: int, filenames: List[str]) -> None:
        """Frames that leave a person stop being its photos; a lost primary passes to the next face."""
        refs = await ReferenceService(self.db).list(video_id, person_id)
        gone = [ref for ref in refs if ref["source_type"] == "frame" and ref["source"] in filenames]
        if not gone:
            return
        await self.db.executemany("DELETE FROM visual_references WHERE id = ?", [(ref["id"],) for ref in gone])
        faces = [ref for ref in refs if ref["role"] == "face" and ref not in gone]
        if any(ref["primary"] for ref in gone) and faces:
            await self.db.execute(
                "UPDATE visual_references SET data = json_set(data, '$.primary', json('true')) WHERE id = ?",
                [faces[0]["id"]]
            )
        # An editor holding the old photos must reload before saving them
        await self.db.execute("UPDATE videos SET reference_revision = reference_revision + 1 WHERE id = ?", [video_id])

    async def _assign(self, person_id: int, frame_ids: List[int]) -> None:
        await self.db.executemany(
            "INSERT OR IGNORE INTO cluster_frame_assignments (cluster_id, frame_id) VALUES (?, ?)",
            [(person_id, frame_id) for frame_id in frame_ids]
        )

    async def _unassign(self, person_id: int, frame_ids: List[int]) -> None:
        await self.db.executemany(
            "DELETE FROM cluster_frame_assignments WHERE cluster_id = ? AND frame_id = ?",
            [(person_id, frame_id) for frame_id in frame_ids]
        )

    async def _refresh_person(self, person_id):
        """Keep the listing representative synchronized with the unified primary face."""
        async with self.db.execute("""
            SELECT vf.id, vf.frame_path, COUNT(*) OVER ()
            FROM cluster_frame_assignments cfa JOIN video_frames vf ON vf.id=cfa.frame_id
            WHERE cfa.cluster_id=? ORDER BY vf.quality_score DESC, vf.id LIMIT 1
        """, [person_id]) as cursor:
            row = await cursor.fetchone()
        frame_id, path, count = row if row else (None, "", 0)
        async with self.db.execute("""
            SELECT data FROM visual_references WHERE person_id=? AND role='face'
              AND json_extract(data, '$.primary')=1 LIMIT 1
        """, [person_id]) as cursor:
            selected = await cursor.fetchone()
        if selected:
            ref = orjson.loads(selected[0])
            if ref["source_type"] == "frame":
                async with self.db.execute("""
                    SELECT vf.id, vf.frame_path FROM video_frames vf JOIN clusters c ON c.video_id=vf.video_id
                    WHERE c.id=? AND (vf.frame_path=? OR vf.frame_path LIKE ?) LIMIT 1
                """, [person_id, ref["source"], "%/frames/" + ref["source"]]) as cursor:
                    primary = await cursor.fetchone()
                if primary:
                    frame_id, path = primary
        await self.db.execute(
            "UPDATE clusters SET representative_frame_id=?, representative_frame=?, num_frames=? WHERE id=?",
            [frame_id, path, count, person_id],
        )

    async def _sync_caches(self, video_id: int, output_dir: Path) -> None:
        """Rewrite the JSON caches of the analysis, which mirror the 'person' view."""
        (output_dir / 'clusters').mkdir(parents=True, exist_ok=True)
        clusters = await self.analysis.get_clusters(video_id, 'person')
        await self.analysis._update_clusters_json(output_dir, clusters)
        await self.analysis._update_clustering_result_json(output_dir, clusters)


def _clean(text: Optional[str]) -> Optional[str]:
    """Stored form of a name or notes: trimmed, None when empty."""
    if text is None:
        return None
    return text.strip() or None

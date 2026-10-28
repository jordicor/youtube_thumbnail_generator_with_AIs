"""
Analysis Service

Business logic for video analysis (scenes, faces, clustering).
"""

import asyncio
from pathlib import Path, PurePosixPath
from typing import Optional, List, Dict, Callable, Set
import aiosqlite
import numpy as np
import orjson
import shutil
import re
import logging

from config import OUTPUT_DIR
from i18n import t
from services.face_index import (
    MIN_IDENTITY_SIMILARITY,
    FaceIndex,
    centroid_from_blob,
    centroid_list,
    centroid_to_blob,
    claimed_faces,
    face_reference_view,
    load_face_index,
    match_faces,
    person_centroid,
    stored_centroid,
)
from services.face_selection import select_face_references
from utils import VideoOutput, resolve_output_path, resolve_row_paths, to_output_relative
from scene_detection import process_video_scenes
from services.visual_context import setting_frame
from face_extraction import process_faces
from transcription import transcribe_video

logger = logging.getLogger(__name__)


class AnalysisCancelledError(Exception):
    """Raised when analysis is cancelled by user."""
    pass


def extract_scene_index_from_path(frame_path: str) -> Optional[int]:
    """
    Extract scene index from frame filename.

    Frame paths follow the pattern: .../scene_XXX_frame_YYYYYY.jpg
    Returns the scene number (XXX) as an integer, or None if not found.
    """
    if not frame_path:
        return None
    match = re.search(r'scene_(\d{3})', str(frame_path))
    return int(match.group(1)) if match else None


class AnalysisService:
    """Service for video analysis operations."""

    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    async def get_video(self, video_id: int) -> Optional[dict]:
        """Get video by ID."""
        query = "SELECT * FROM videos WHERE id = ?"
        async with self.db.execute(query, [video_id]) as cursor:
            row = await cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None

    async def update_video_status(
        self,
        video_id: int,
        status: str,
        error_message: Optional[str] = None
    ):
        """Update video status."""
        if error_message:
            query = "UPDATE videos SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
            await self.db.execute(query, [status, error_message, video_id])
        else:
            query = "UPDATE videos SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
            await self.db.execute(query, [status, video_id])
        await self.db.commit()

    async def run_full_analysis(
        self,
        video_id: int,
        force_scenes: bool = False,
        force_faces: bool = False,
        force_clustering: bool = False,
        force_transcription: bool = False,
        clustering_eps: float = 0.5,
        clustering_min_samples: int = 3,
        cancellation_check: Optional[Callable[[], bool]] = None,
        refresh_references: bool = False,
        expected_reference_revision: Optional[int] = None,
    ):
        """
        Run full analysis pipeline for a video.

        Steps:
        1. Scene detection
        2. Face extraction
        3. Face clustering
        4. Audio transcription (ElevenLabs)
        """
        try:
            if refresh_references:
                if any((force_scenes, force_faces, force_clustering, force_transcription)):
                    raise ValueError("Reference refresh cannot be combined with forced analysis stages.")
                await self.refresh_visual_references(video_id, cancellation_check, expected_reference_revision)
                return
            # Get video info
            video = await self.get_video(video_id)
            if not video:
                return

            video_path = Path(video['filepath'])
            output = VideoOutput(video_path, Path(OUTPUT_DIR))
            output.setup()

            # Step 1: Scene detection
            await self.update_video_status(video_id, 'analyzing_scenes')

            if force_scenes:
                output.scenes_file.unlink(missing_ok=True)
                shutil.rmtree(output.frames_dir, ignore_errors=True)
                output.frames_dir.mkdir(exist_ok=True)

            try:
                loop = asyncio.get_event_loop()
                scene_result, extracted_frames = await loop.run_in_executor(
                    None, process_video_scenes, video_path, output
                )
                # Frame extraction raises when it cannot write a frame or reads none
                if not scene_result:
                    await self.update_video_status(video_id, 'error', 'scene_detection_error: No scenes detected')
                    return
            except Exception as e:
                await self.update_video_status(video_id, 'error', f'scene_detection_error: {str(e)}')
                return

            # Check for cancellation after scene detection
            if cancellation_check and cancellation_check():
                raise AnalysisCancelledError(f"Video {video_id} cancelled after scene detection")

            # Step 2: Face extraction
            await self.update_video_status(video_id, 'analyzing_faces')

            if force_faces:
                output.faces_file.unlink(missing_ok=True)

            try:
                face_result = await loop.run_in_executor(
                    None, process_faces, extracted_frames, output
                )
            except Exception as e:
                await self.update_video_status(video_id, 'error', f'face_extraction_error: {str(e)}')
                return

            # Check for cancellation after face extraction
            if cancellation_check and cancellation_check():
                raise AnalysisCancelledError(f"Video {video_id} cancelled after face extraction")

            if face_result and face_result.all_faces:
                from services.portrait_quality import enrich_portraits
                try:
                    await asyncio.to_thread(enrich_portraits, output.output_dir, video_path, cancellation_check)
                except InterruptedError as error:
                    raise AnalysisCancelledError(str(error)) from error

            # Body landmarks reuse indexed faces and extracted frames.
            if face_result and face_result.all_faces:
                from services.body_references import ensure_body_index
                try:
                    await asyncio.to_thread(ensure_body_index, output.output_dir, cancellation_check)
                except InterruptedError as error:
                    raise AnalysisCancelledError(str(error)) from error

            # Step 3: Face clustering (if faces found)
            if face_result and face_result.all_faces:
                await self.update_video_status(video_id, 'clustering')
                try:
                    await self._run_clustering(
                        video_id,
                        output,
                        force_clustering,
                        clustering_eps,
                        clustering_min_samples
                    )
                except Exception as e:
                    await self.update_video_status(video_id, 'error', f'clustering_error: {str(e)}')
                    return

                # Check for cancellation after clustering
                if cancellation_check and cancellation_check():
                    raise AnalysisCancelledError(f"Video {video_id} cancelled after clustering")

            # Step 4: Audio transcription
            await self.update_video_status(video_id, 'transcribing')

            if force_transcription:
                output.transcription_file.unlink(missing_ok=True)
                # Also remove JSON version if exists
                json_transcription = output.transcription_file.with_suffix('.json')
                json_transcription.unlink(missing_ok=True)

            try:
                transcription = await loop.run_in_executor(
                    None, transcribe_video, video_path, output
                )
                if not transcription:
                    await self.update_video_status(video_id, 'error', 'transcription_error: Failed to transcribe audio')
                    return
            except Exception as e:
                await self.update_video_status(video_id, 'error', f'transcription_error: {str(e)}')
                return

            # Mark as analyzed
            await self.update_video_status(video_id, 'analyzed')

        except AnalysisCancelledError as e:
            # Handle cancellation gracefully
            logger.info(f"Analysis for video {video_id} cancelled: {e}")
            await self.update_video_status(video_id, 'cancelled')
            # Don't raise - return to let worker handle SSE events
            return

        except Exception as e:
            await self.update_video_status(video_id, 'error', str(e))
            raise

    async def refresh_visual_references(self, video_id, cancellation_check=None, expected_reference_revision=None):
        """Refresh the automatic visual choices without reclustering or transcribing.

        People keep their names, notes and frames, except that a person showing
        two characters (the same face with clearly different hair, a wig or a
        disguise) has the other character moved to a new person. Choices the
        user made (automatic false) stay.
        """
        from services.body_references import ensure_body_index
        from services.portrait_quality import enrich_portraits
        from services.reference_service import ReferenceService

        video = await self.get_video(video_id)
        if video is None:
            raise ValueError("Video not found.")
        output = VideoOutput(Path(video["filepath"]), Path(OUTPUT_DIR))
        scenes = orjson.loads(output.scenes_file.read_bytes())["scenes"]
        frames = [output.frames_dir / f"scene_{s['index']:03d}_frame_{number:06d}.jpg"
                  for s in scenes for number in s["frame_indices"]]
        if not frames or any(not path.is_file() or not path.stat().st_size for path in frames):
            raise ValueError("Reference refresh requires complete extracted scene frames.")
        revision = video["reference_revision"]
        if expected_reference_revision is not None and revision != expected_reference_revision:
            raise ValueError("References changed after this refresh was queued; no choices were overwritten.")
        await self.update_video_status(video_id, "analyzing_faces")
        await asyncio.to_thread(process_faces, frames, output)
        try:
            await asyncio.to_thread(enrich_portraits, output.output_dir, video["filepath"], cancellation_check)
            body_index = await asyncio.to_thread(ensure_body_index, output.output_dir, cancellation_check)
        except InterruptedError as error:
            raise AnalysisCancelledError(str(error)) from error
        if cancellation_check and cancellation_check():
            raise AnalysisCancelledError("Reference refresh cancelled.")
        index = await asyncio.to_thread(load_face_index, output.faces_file)
        refs = ReferenceService(self.db)
        existing = await refs.list(video_id)
        async with self.db.execute("""
            SELECT c.id,c.view_mode,c.embedding_centroid,vf.frame_path FROM clusters c
            JOIN cluster_frame_assignments a ON a.cluster_id=c.id
            JOIN video_frames vf ON vf.id=a.frame_id WHERE c.video_id=? ORDER BY c.id,vf.id
        """, [video_id]) as cursor:
            rows = await cursor.fetchall()
        groups = {}
        for person_id, view_mode, blob, path in rows:
            groups.setdefault(person_id, (view_mode, blob, []))[2].append(Path(path).name)
        if not groups:
            raise ValueError("Reference refresh requires existing people with indexed frames. Run normal analysis first.")
        automatic_setting = all(ref.get("automatic") for ref in existing if ref["role"] == "environment")

        plan = await asyncio.to_thread(
            self._plan_reference_refresh, index, body_index, output.frames_dir, groups, existing,
            automatic_setting)
        if cancellation_check and cancellation_check():
            raise AnalysisCancelledError("Reference refresh cancelled before saving.")
        await self.db.execute("BEGIN IMMEDIATE")
        try:
            # A video with thumbnails stays completed, as generation leaves it
            cursor = await self.db.execute("""
                UPDATE videos SET reference_revision=reference_revision+1,error_message=NULL,
                                  updated_at=CURRENT_TIMESTAMP,
                                  status=CASE WHEN EXISTS (SELECT 1 FROM generation_jobs done
                                      WHERE done.video_id=videos.id AND done.status='completed')
                                    THEN 'completed' ELSE 'analyzed' END
                WHERE id=? AND reference_revision=? AND status!='cancelled'
            """, [video_id, revision])
            if cursor.rowcount != 1:
                raise ValueError("References changed or analysis was cancelled; no choices were overwritten.")
            people = await self._apply_character_splits(video_id, output, plan["people"])
            await self.db.execute("DELETE FROM visual_references WHERE video_id=? AND role!='environment'", [video_id])
            await refs.insert(video_id, [dict(ref, person_id=people[ref["person_id"]]) for ref in plan["references"]])
            if plan["setting"] is not None:
                await self.db.execute("DELETE FROM visual_references WHERE video_id=? AND role='environment'", [video_id])
                await refs.insert(video_id, [plan["setting"]])
            for key, person in plan["people"].items():
                primary = next((ref["source"] for ref in plan["references"]
                                if ref["person_id"] == key and ref["primary"] and ref["source_type"] == "frame"), None)
                relative = to_output_relative(output.frames_dir / primary) if primary else ""
                await self.db.execute("""
                    UPDATE clusters SET representative_frame=?, is_picture=?,
                      representative_frame_id=(SELECT id FROM video_frames WHERE video_id=? AND frame_path=?)
                    WHERE id=?
                """, [relative, person["is_picture"], video_id, relative, people[key]])
            await self.db.commit()
        except BaseException:
            await self.db.rollback()
            raise
        if any(person["new"] for person in plan["people"].values()):
            clusters = await self.get_clusters(video_id, "person")
            (output.output_dir / "clusters").mkdir(parents=True, exist_ok=True)
            await self._update_clusters_json(output.output_dir, clusters)
            await self._update_clustering_result_json(output.output_dir, clusters)

    @staticmethod
    def _plan_reference_refresh(index, body_index, frames_dir, groups, existing, automatic_setting):
        """What a reference refresh writes: people (with characters split off), references and setting.

        People are keyed by their id; a character split off a person gets the
        key ("new", parent id, number) until it is inserted.
        """
        from services.body_references import person_body_candidates, select_body_references
        from services.face_selection import character_groups, is_picture, look_vector, same_look_frames
        from services.reference_service import VisualReference

        people = {}
        for person_id, (view_mode, blob, members) in groups.items():
            centroid = stored_centroid(blob, index, members, person_id)
            matches = match_faces(index, centroid, members)
            own = {name: match for name, match in matches.items()
                   if match.similarity is not None and match.similarity >= MIN_IDENTITY_SIMILARITY}
            characters = (character_groups({name: look_vector(index.portraits[match.row] or {})
                                            for name, match in own.items()})
                          if view_mode == "person" else [list(own)])
            if len(characters) > 1:
                # The person keeps the character of its primary photo (named, chosen)
                primary = next((ref["source"] for ref in existing if ref["person_id"] == person_id
                                and ref["role"] == "face" and ref["primary"]), None)
                kept = next((group for group in characters if primary in group), characters[0])
                moved = [group for group in characters if group is not kept]
                leaving = {name for group in moved for name in group}
                remaining = [name for name in members if name not in leaving]
                people[person_id] = {"new": False, "members": remaining,
                                     "centroid": person_centroid(index, remaining, centroid)}
                for number, group in enumerate(moved, 1):
                    mean = index.embeddings[[own[name].row for name in group]].mean(axis=0)
                    people[("new", person_id, number)] = {
                        "new": True, "parent": person_id, "members": sorted(group),
                        "centroid": person_centroid(index, sorted(group), mean / np.linalg.norm(mean))}
            else:
                people[person_id] = {"new": False, "members": members, "centroid": centroid}

        references = []
        for key, person in people.items():
            matches = match_faces(index, person["centroid"], person["members"])
            person["is_picture"] = is_picture([
                (name, match.box) for name, match in matches.items()
                if match.similarity is not None and match.similarity >= MIN_IDENTITY_SIMILARITY])
            members = set(person["members"])
            current = [ref for ref in existing if not person["new"] and ref["person_id"] == key
                       and (ref["source_type"] != "frame" or ref["source"] in members)]
            # A person whose photos the user chose or changed keeps them all
            if any(ref["role"] == "face" and not ref.get("automatic") for ref in current):
                faces = [ref for ref in current if ref["role"] == "face"]
                if faces and not any(ref["primary"] for ref in faces):
                    faces[0] = dict(faces[0], primary=True)
            else:
                faces = [VisualReference(role="face", person_id=-1, source_type="frame", source=name,
                                         primary=position == 0, view=face_reference_view(matches[name].yaw),
                                         expression=matches[name].expression, automatic=True).model_dump()
                         for position, name in enumerate(select_face_references(frames_dir, matches, index))]
            references += [dict(ref, person_id=key) for ref in faces]
            manual_bodies = [ref for ref in current if ref["role"] == "body" and not ref.get("automatic")]
            references += manual_bodies
            primary = next((ref["source"] for ref in faces if ref["primary"] and ref["source_type"] == "frame"), None)
            allowed = same_look_frames(matches, index, primary)
            candidates = person_body_candidates(body_index, {name: match for name, match in matches.items()
                                                             if name in allowed})["candidates"]
            occupied = {"front" if ref["view"] == "three_quarter" else ref["view"] for ref in manual_bodies}
            for candidate in select_body_references(frames_dir, candidates):
                if ("front" if candidate["view"] == "three_quarter" else candidate["view"]) in occupied:
                    continue
                references.append(dict(VisualReference(
                    role="body", person_id=-1, source_type="frame", source=candidate["filename"],
                    crop=candidate["box"], coverage=candidate["coverage"], posture=candidate["posture"],
                    view=candidate["view"], identity_confirmed=True, automatic=True).model_dump(), person_id=key))
        setting = None
        if automatic_setting:
            setting = VisualReference(role="environment", source_type="frame",
                                      source=setting_frame(frames_dir), automatic=True).model_dump()
        return {"people": people, "references": references, "setting": setting}

    async def _apply_character_splits(self, video_id, output, people) -> Dict:
        """Insert the characters a refresh split off and move their frames; the database id of every key."""
        ids = {key: key for key, person in people.items() if not person["new"]}
        for key, person in people.items():
            if not person["new"]:
                continue
            async with self.db.execute(
                "SELECT COALESCE(MAX(cluster_index), -1) + 1 FROM clusters WHERE video_id=? AND view_mode='person'",
                [video_id]
            ) as cursor:
                next_index = (await cursor.fetchone())[0]
            cursor = await self.db.execute("""
                INSERT INTO clusters (video_id, cluster_index, view_mode, num_frames, representative_frame,
                                      embedding_centroid)
                VALUES (?, ?, 'person', ?, '', ?)
            """, [video_id, next_index, len(person["members"]), centroid_to_blob(person["centroid"])])
            ids[key] = cursor.lastrowid
            paths = [to_output_relative(output.frames_dir / name) for name in person["members"]]
            placeholders = ",".join("?" * len(paths))
            await self.db.execute(f"""
                DELETE FROM cluster_frame_assignments WHERE cluster_id=? AND frame_id IN
                  (SELECT id FROM video_frames WHERE video_id=? AND frame_path IN ({placeholders}))
            """, [person["parent"], video_id, *paths])
            await self.db.execute(f"""
                INSERT INTO cluster_frame_assignments (cluster_id, frame_id)
                SELECT ?, id FROM video_frames WHERE video_id=? AND frame_path IN ({placeholders})
            """, [ids[key], video_id, *paths])
        for key, person in people.items():
            if person["new"] or any(other["new"] and other["parent"] == key for other in people.values()):
                await self.db.execute("UPDATE clusters SET num_frames=?, embedding_centroid=? WHERE id=?", [
                    len(person["members"]),
                    None if person["centroid"] is None else centroid_to_blob(person["centroid"]), ids[key]])
        return ids

    async def _run_clustering(
        self,
        video_id: int,
        output,
        force: bool,
        eps: float,
        min_samples: int
    ):
        """
        Run face clustering and save results to database.

        V2 Architecture:
        1. Insert all frames into video_frames (single source of truth)
        2. Create clusters with view_mode ('person' and 'person_scene')
        3. Create cluster_frame_assignments instead of cluster_frames
        """
        from face_clustering import run_clustering_pipeline

        # Check if faces.json exists
        if not output.faces_file.exists():
            return

        # Check for existing clustering result
        clustering_result_path = output.output_dir / "clustering_result.json"
        result = None
        if clustering_result_path.exists() and not force:
            result = orjson.loads(clustering_result_path.read_bytes())
            stat = output.faces_file.stat()
            if result.get("faces_fingerprint") != [stat.st_mtime_ns, stat.st_size]:
                result = None
        if result is not None:
            # Paths in the cache are stored relative to OUTPUT_DIR
            for cluster in result.get('clusters', []):
                if 'centroid' not in cluster or 'is_picture' not in cluster:
                    raise ValueError(
                        f"{clustering_result_path} lacks the face embedding or picture flag of a cluster; "
                        "reanalyze the video forcing the clustering"
                    )
                if cluster.get('representative_frame'):
                    cluster['representative_frame'] = str(resolve_output_path(cluster['representative_frame']))
                for frame in cluster['frames']:
                    frame['frame_path'] = str(resolve_output_path(frame['frame_path']))
        else:
            result = run_clustering_pipeline(
                video_output_dir=output.output_dir,
                eps=eps,
                min_samples=min_samples,
                save_representatives=True
            )

        if result.get('error'):
            raise ValueError(result['error'])

        # Face photos of each person and scene, picked before the write
        # transaction because it opens images
        index = await asyncio.to_thread(load_face_index, output.faces_file)
        references = await asyncio.to_thread(
            self._initial_references, index, output.frames_dir, result.get('clusters', [])
        )
        environment_frame = await asyncio.to_thread(setting_frame, output.frames_dir)

        from services.body_references import load_body_index
        body_index = await asyncio.to_thread(load_body_index, output.output_dir)

        # V2: Delete existing data and insert new data in a single transaction
        await self.db.execute("BEGIN IMMEDIATE")
        await self.db.execute("""
            DELETE FROM cluster_frame_assignments
            WHERE cluster_id IN (SELECT id FROM clusters WHERE video_id = ?)
        """, [video_id])
        await self.db.execute("DELETE FROM clusters WHERE video_id = ?", [video_id])
        await self.db.execute("DELETE FROM video_frames WHERE video_id = ?", [video_id])

        clusters_data = result.get('clusters', [])

        # 1. First pass: Insert ALL unique frames into video_frames
        frame_rows = {}  # frame_path -> row to insert
        for cluster in clusters_data:
            all_frames = cluster.get('frames', [])
            if not all_frames:
                all_frames = [{'frame_path': p, 'quality_score': 0, 'expression': 'unknown'}
                              for p in cluster.get('top_frames', [])]

            for frame in all_frames:
                if isinstance(frame, dict):
                    path = frame.get('frame_path', '')
                    quality = frame.get('quality_score', 0)
                    expression = frame.get('expression', 'unknown')
                else:
                    path = frame
                    quality = 0
                    expression = 'unknown'

                if path and path not in frame_rows:
                    frame_rows[path] = (video_id, to_output_relative(path), quality, expression,
                                        extract_scene_index_from_path(path))

        # One statement for all rows (each execute is a round trip to the
        # aiosqlite thread), then the ids by their unique (video_id, frame_path)
        await self.db.executemany("""
            INSERT INTO video_frames (video_id, frame_path, quality_score, expression, scene_index)
            VALUES (?, ?, ?, ?, ?)
        """, list(frame_rows.values()))
        async with self.db.execute(
            "SELECT frame_path, id FROM video_frames WHERE video_id = ?", [video_id]
        ) as cursor:
            frame_ids = dict(await cursor.fetchall())
        frame_id_map = {path: frame_ids[row[1]] for path, row in frame_rows.items()}  # frame_path -> frame_id

        # 2. Create clusters for BOTH view_modes
        person_scene_index = 0  # Running index for person_scene clusters

        for cluster in clusters_data:
            cluster_idx = cluster.get('cluster_index', 0)
            face_references = references[(cluster_idx, None)]
            representative_path = str(output.frames_dir / face_references[0]["source"]) if face_references else ""

            frames_with_quality = [
                {**frame, 'frame_id': frame_id_map.get(frame['frame_path'])}
                for frame in self._cluster_frames(cluster)
            ]
            if not frames_with_quality:
                continue

            # Listing and editor show the primary face photo
            rep_frame_id = frame_id_map.get(representative_path)

            # The person's face: the mean of the faces DBSCAN grouped (shared
            # with their person_scene clusters). Their frames can show others too.
            centroid = cluster['centroid']
            centroid_blob = None if centroid is None else centroid_to_blob(
                np.asarray(centroid, dtype=np.float32) / np.linalg.norm(centroid)
            )

            # === Create 'person' cluster ===
            cursor = await self.db.execute("""
                INSERT INTO clusters (video_id, cluster_index, view_mode, representative_frame_id,
                                      num_frames, representative_frame, embedding_centroid, is_picture)
                VALUES (?, ?, 'person', ?, ?, ?, ?, ?)
            """, [video_id, cluster_idx, rep_frame_id, len(frames_with_quality),
                  to_output_relative(representative_path) if representative_path else "", centroid_blob,
                  cluster['is_picture']])
            person_cluster_id = cursor.lastrowid

            # Assign all frames to person cluster
            await self._insert_assignments(person_cluster_id, frames_with_quality, references[(cluster_idx, None)])

            # === Create 'person_scene' clusters ===
            frames_by_scene = self._frames_by_scene(frames_with_quality)
            for scene_idx in sorted(frames_by_scene.keys()):
                scene_frames = frames_by_scene[scene_idx]
                scene_references = references[(cluster_idx, scene_idx)]
                scene_rep_path = str(output.frames_dir / scene_references[0]["source"]) if scene_references else ""
                scene_rep_id = frame_id_map.get(scene_rep_path)

                # Create person_scene cluster
                cursor = await self.db.execute("""
                    INSERT INTO clusters (video_id, cluster_index, view_mode, scene_index,
                                          representative_frame_id, num_frames, representative_frame,
                                          embedding_centroid, is_picture)
                    VALUES (?, ?, 'person_scene', ?, ?, ?, ?, ?, ?)
                """, [video_id, person_scene_index, scene_idx if scene_idx != -1 else None,
                      scene_rep_id, len(scene_frames), to_output_relative(scene_rep_path) if scene_rep_path else "",
                      centroid_blob, cluster['is_picture']])
                scene_cluster_id = cursor.lastrowid
                person_scene_index += 1

                # Assign frames to person_scene cluster
                await self._insert_assignments(scene_cluster_id, scene_frames, references[(cluster_idx, scene_idx)])

        from services.reference_service import ReferenceService
        await ReferenceService(self.db).initialize_context(video_id, index, body_index, environment_frame)
        await self.db.execute("UPDATE videos SET reference_revision=reference_revision+1 WHERE id=?", [video_id])
        await self.db.commit()

    @staticmethod
    def _cluster_frames(cluster: dict) -> List[dict]:
        """A clustering result's frames of one person, once each, best first.

        Each one: frame_path, quality_score, scene_index.
        """
        all_frames = cluster.get('frames', []) or [
            {'frame_path': p, 'quality_score': 0} for p in cluster.get('top_frames', [])
        ]
        frames, seen_paths = [], set()
        for frame in all_frames:
            path = frame.get('frame_path', '') if isinstance(frame, dict) else frame
            if path and path not in seen_paths:
                seen_paths.add(path)
                frames.append({
                    'frame_path': path,
                    'quality_score': frame.get('quality_score', 0) if isinstance(frame, dict) else 0,
                    'scene_index': extract_scene_index_from_path(path),
                })
        frames.sort(key=lambda f: f['quality_score'], reverse=True)
        return frames

    @staticmethod
    def _frames_by_scene(frames: List[dict]) -> Dict[int, List[dict]]:
        """Frames grouped by scene index (-1: unknown), keeping their order."""
        by_scene: Dict[int, List[dict]] = {}
        for frame in frames:
            scene_idx = frame['scene_index'] if frame['scene_index'] is not None else -1
            by_scene.setdefault(scene_idx, []).append(frame)
        return by_scene

    @classmethod
    def _initial_references(
        cls,
        index: FaceIndex,
        frames_dir: Path,
        clusters: List[dict],
    ) -> Dict[tuple, List[dict]]:
        """Up to three face photos per person and scene, with observed metadata."""
        references = {}
        for cluster in clusters:
            frames = cls._cluster_frames(cluster)
            centroid = None if cluster['centroid'] is None else np.asarray(cluster['centroid'], dtype=np.float32)
            paths = {Path(frame['frame_path']).name: frame['frame_path'] for frame in frames}
            matches = match_faces(index, centroid, list(paths))

            def pick(group: List[dict]) -> List[dict]:
                names = [Path(frame["frame_path"]).name for frame in group]
                selected = select_face_references(frames_dir, {name: matches[name] for name in names if name in matches}, index)
                return [{"source": name, "primary": position == 0, "automatic": True,
                         "view": face_reference_view(matches[name].yaw), "expression": matches[name].expression}
                        for position, name in enumerate(selected)]

            cluster_idx = cluster.get('cluster_index', 0)
            references[(cluster_idx, None)] = pick(frames)
            for scene_idx, scene_frames in cls._frames_by_scene(frames).items():
                references[(cluster_idx, scene_idx)] = pick(scene_frames)
        return references

    async def _insert_assignments(self, cluster_id: int, frames: List[dict], references: List[dict]) -> None:
        """Assign membership and the initial ordered face references to a new cluster."""
        await self.db.executemany(
            "INSERT INTO cluster_frame_assignments (cluster_id, frame_id) VALUES (?, ?)",
            [(cluster_id, frame["frame_id"]) for frame in frames],
        )
        async with self.db.execute("SELECT video_id FROM clusters WHERE id=?", [cluster_id]) as cursor:
            video_id = (await cursor.fetchone())[0]
        from services.reference_service import ReferenceService, VisualReference
        await ReferenceService(self.db).insert(video_id, [
            VisualReference(role="face", person_id=cluster_id, source_type="frame", **ref).model_dump()
            for ref in references
        ])

    # =========================================================================
    # PERSON FACE EMBEDDING
    # =========================================================================

    async def centroid_blob(self, cluster_id: int) -> Optional[bytes]:
        """Stored face embedding of a person (NULL when none of their frames shows a face)."""
        async with self.db.execute(
            "SELECT embedding_centroid FROM clusters WHERE id = ?", [cluster_id]
        ) as cursor:
            return (await cursor.fetchone())[0]

    async def _members_by_person(self, query: str, params: list) -> Dict[int, List[str]]:
        """Frame filenames of each cluster a query (cluster id, frame path) returns."""
        members: Dict[int, List[str]] = {}
        async with self.db.execute(query, params) as cursor:
            for cluster_id, frame_path in await cursor.fetchall():
                members.setdefault(cluster_id, []).append(PurePosixPath(frame_path).name)
        return members

    async def view_claimed_faces(
        self,
        index: FaceIndex,
        video_id: int,
        view_mode: str,
        exclude_id: Optional[int] = None
    ) -> Set[int]:
        """Faces (index rows) the people of a view have in their frames, but exclude_id's."""
        members = await self._members_by_person("""
            SELECT c.id, vf.frame_path
            FROM clusters c
            JOIN cluster_frame_assignments cfa ON cfa.cluster_id = c.id
            JOIN video_frames vf ON vf.id = cfa.frame_id
            WHERE c.video_id = ? AND c.view_mode = ? AND c.id IS NOT ?
        """, [video_id, view_mode, exclude_id])
        return claimed_faces(index, [
            (stored_centroid(await self.centroid_blob(cluster_id), index, names, cluster_id), names)
            for cluster_id, names in members.items()
        ])

    async def fit_person_centroid(
        self,
        cluster_id: int,
        output_dir: Path,
        seed: Optional[np.ndarray] = None
    ) -> None:
        """Fit a person's stored face embedding to their current frames.

        The fit starts from seed, by default the person's embedding before the
        edit. A person without one (new, or without faces until now) starts from
        the faces no other person of the view claims. Not committed.
        """
        async with self.db.execute(
            "SELECT video_id, view_mode, embedding_centroid FROM clusters WHERE id = ?", [cluster_id]
        ) as cursor:
            video_id, view_mode, blob = await cursor.fetchone()
        members = (await self._members_by_person("""
            SELECT cfa.cluster_id, vf.frame_path
            FROM cluster_frame_assignments cfa
            JOIN video_frames vf ON vf.id = cfa.frame_id
            WHERE cfa.cluster_id = ?
        """, [cluster_id])).get(cluster_id, [])

        # Parsing a large faces.json takes a moment the first time
        index = await asyncio.to_thread(load_face_index, output_dir / 'faces.json')
        if seed is None and blob is not None:
            seed = centroid_from_blob(blob)
        claimed = set() if seed is not None else await self.view_claimed_faces(index, video_id, view_mode, cluster_id)
        centroid = person_centroid(index, members, seed, claimed)
        await self.db.execute(
            "UPDATE clusters SET embedding_centroid = ? WHERE id = ?",
            [None if centroid is None else centroid_to_blob(centroid), cluster_id]
        )

    async def get_analysis_status(self, video_id: int) -> Optional[dict]:
        """Get current analysis status for a video."""
        video = await self.get_video(video_id)
        if not video:
            return None

        # Get cluster count
        async with self.db.execute(
            "SELECT COUNT(*) FROM clusters WHERE video_id = ?",
            [video_id]
        ) as cursor:
            row = await cursor.fetchone()
            cluster_count = row[0] if row else 0

        # Calculate progress based on status
        progress_map = {
            'pending': 0,
            'analyzing': 10,
            'analyzing_scenes': 15,
            'analyzing_faces': 40,
            'clustering': 65,
            'transcribing': 85,
            'analyzed': 100,
            'generating': 100,
            'completed': 100,
            'error': 0
        }

        return {
            'video_id': video_id,
            'status': video['status'],
            'progress': progress_map.get(video['status'], 0),
            'current_step': video['status'],
            'error_message': video.get('error_message'),
            'clusters': cluster_count
        }

    async def get_clusters(self, video_id: int, view_mode: str = "person") -> List[dict]:
        """
        Get clusters for a video in a specific view mode.

        V2 Architecture: Uses view_mode instead of cluster_type.
        Frame counts are calculated from cluster_frame_assignments.

        Args:
            video_id: The video ID
            view_mode: 'person' or 'person_scene' (default: 'person')

        Returns:
            List of cluster dictionaries with frame counts from assignments.
            person_scene clusters also carry the person cluster that currently
            shares most of their frames (parent_cluster_index, parent_label),
            or None when no person cluster holds any of them.
            face_photo_expressions counts the selected face references by expression.
            People whose face never moves (is_picture: a painting, a poster) come last.
        """
        # Frame counts come from cluster_frame_assignments, looked up once per
        # cluster: joined to the assignments, the parent subquery below would run
        # once per frame. Falls back to representative_frame if
        # representative_frame_id is not set.
        # The two views are independent, so the parent person of a person_scene
        # cluster is derived from the current 'person' view; the 'person' view
        # has no parent and skips it.
        if view_mode == 'person_scene':
            parent_columns = "p.cluster_index as parent_cluster_index, p.label as parent_label"
            parent_join = """
            LEFT JOIN clusters p ON p.id = (
                SELECT pc.id
                FROM cluster_frame_assignments sa
                JOIN cluster_frame_assignments pa ON pa.frame_id = sa.frame_id
                JOIN clusters pc ON pc.id = pa.cluster_id
                WHERE sa.cluster_id = c.id
                  AND pc.video_id = c.video_id
                  AND pc.view_mode = 'person'
                GROUP BY pc.id
                ORDER BY COUNT(*) DESC, pc.cluster_index
                LIMIT 1
            )"""
        else:
            parent_columns = "NULL as parent_cluster_index, NULL as parent_label"
            parent_join = ""
        query = f"""
            SELECT
                c.id,
                c.cluster_index,
                c.label,
                c.description,
                c.view_mode,
                c.scene_index,
                c.representative_frame_id,
                COALESCE(vf.frame_path, c.representative_frame) as representative_frame,
                (SELECT COUNT(*) FROM cluster_frame_assignments
                 WHERE cluster_id = c.id) as num_frames,
                (SELECT COUNT(*) FROM visual_references
                 WHERE person_id = c.id AND role = 'face') as reference_count,
                c.reference_format,
                c.is_picture,
                {parent_columns}
            FROM clusters c
            LEFT JOIN video_frames vf ON vf.id = c.representative_frame_id{parent_join}
            WHERE c.video_id = ? AND c.view_mode = ?
            ORDER BY c.is_picture, c.cluster_index
        """

        async with self.db.execute(query, [video_id, view_mode]) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            clusters = [resolve_row_paths(dict(zip(columns, row)), 'representative_frame') for row in rows]

        expressions_query = """
            SELECT r.person_id, COALESCE(json_extract(r.data, '$.expression'), vf.expression), COUNT(*)
            FROM visual_references r
            JOIN clusters c ON c.id=r.person_id
            LEFT JOIN video_frames vf ON vf.video_id=r.video_id AND json_extract(r.data, '$.source_type')='frame'
                AND (vf.frame_path=json_extract(r.data, '$.source') OR
                     vf.frame_path LIKE '%/frames/' || json_extract(r.data, '$.source'))
            WHERE c.video_id=? AND c.view_mode=? AND r.role='face'
            GROUP BY r.person_id, COALESCE(json_extract(r.data, '$.expression'), vf.expression)
        """
        expressions = {cluster['id']: {} for cluster in clusters}
        async with self.db.execute(expressions_query, [video_id, view_mode]) as cursor:
            async for cluster_id, expression, count in cursor:
                expressions[cluster_id][expression] = count
        for cluster in clusters:
            cluster['face_photo_expressions'] = expressions[cluster['id']]
        return clusters

    async def delete_cluster(
        self,
        video_id: int,
        cluster_index: int,
        view_mode: str = 'person'
    ) -> bool:
        """
        Delete a cluster.

        V2 Architecture: Frames are NOT deleted, only cluster_frame_assignments (via CASCADE).
        Each view is independent - deleting from 'person' doesn't affect 'person_scene'.

        NOTE: Frames in video_frames and on disk are NEVER deleted,
        allowing users to reuse them for manual cluster creation later.

        Returns True if successful, False if cluster not found.
        """
        # Get video and cluster info
        video = await self.get_video(video_id)
        if not video:
            return False

        cluster = await self._get_cluster_by_index_v2(video_id, cluster_index, view_mode)
        if not cluster:
            return False

        output_dir = self._get_video_output_dir(video)

        # Cluster folders (clusters/cluster_X/) mirror the 'person' view only
        cluster_folder = output_dir / "clusters" / f"cluster_{cluster_index}"
        if view_mode == 'person' and cluster_folder.exists():
            shutil.rmtree(cluster_folder)

        # Delete cluster (assignments cascade automatically)
        await self.db.execute("""
            DELETE FROM clusters
            WHERE video_id = ? AND view_mode = ? AND cluster_index = ?
        """, [video_id, view_mode, cluster_index])

        # Capture old indices BEFORE reindex so _sync_cluster_folders can map folders correctly
        old_index_by_id = await self._get_cluster_index_map(video_id, view_mode)

        # Reindex remaining clusters in this view_mode
        await self._reindex_clusters_v2(video_id, view_mode)
        await self.db.commit()

        # Sync physical cluster folders and update JSON files
        if view_mode == 'person':
            await self._sync_cluster_folders(video_id, output_dir, old_index_by_id)

        return True

    async def _get_cluster_index_map(self, video_id: int, view_mode: str) -> Dict[int, int]:
        """Get mapping of cluster_id -> cluster_index for all clusters in a view_mode.

        Must be called BEFORE _reindex_clusters_v2 to capture old indices.
        """
        async with self.db.execute("""
            SELECT id, cluster_index FROM clusters
            WHERE video_id = ? AND view_mode = ?
        """, [video_id, view_mode]) as cursor:
            rows = await cursor.fetchall()
            return {row[0]: row[1] for row in rows}

    async def merge_clusters(
        self,
        video_id: int,
        cluster_indices: List[int],
        target_index: int,
        view_mode: str = 'person'
    ) -> Optional[dict]:
        """
        Merge multiple clusters into a target cluster.

        V2 Architecture: Moves assignments, not physical frames.

        Args:
            video_id: The video ID
            cluster_indices: List of cluster indices to merge
            target_index: Which cluster index to keep as the "main" one
            view_mode: 'person' or 'person_scene'

        Returns:
            The merged cluster info, or None if failed.
        """
        if len(cluster_indices) < 2:
            return {'success': False, 'error': t('api.errors.min_clusters_required')}

        if target_index not in cluster_indices:
            target_index = cluster_indices[0]

        # Get video info for output_dir
        video = await self.get_video(video_id)
        if not video:
            return {'success': False, 'error': t('api.errors.video_not_found')}

        output_dir = self._get_video_output_dir(video)
        clusters_dir = output_dir / "clusters"

        # Get target cluster
        target = await self._get_cluster_by_index_v2(video_id, target_index, view_mode)
        if not target:
            return {'success': False, 'error': t('api.errors.cluster_not_found')}

        target_id = target['id']
        source_indices = [i for i in cluster_indices if i != target_index]

        if not source_indices:
            return {'success': False, 'error': t('api.errors.no_source_clusters')}

        # Get source cluster IDs
        source_ids = []
        for idx in source_indices:
            source = await self._get_cluster_by_index_v2(video_id, idx, view_mode)
            if source:
                source_ids.append(source['id'])

        if not source_ids:
            return {'success': False, 'error': t('api.errors.source_clusters_not_found')}

        # The merged person is the largest one's face (the others join them)
        merged_ids = [target_id] + source_ids
        async with self.db.execute(f"""
            SELECT c.embedding_centroid
            FROM clusters c
            WHERE c.id IN ({','.join('?' * len(merged_ids))})
            ORDER BY (SELECT COUNT(*) FROM cluster_frame_assignments WHERE cluster_id = c.id) DESC, c.id
            LIMIT 1
        """, merged_ids) as cursor:
            largest_blob = (await cursor.fetchone())[0]

        # Move assignments to target (ignore duplicates)
        placeholders = ','.join('?' * len(source_ids))
        await self.db.execute(f"""
            INSERT OR IGNORE INTO cluster_frame_assignments
                (cluster_id, frame_id)
            SELECT ?, frame_id
            FROM cluster_frame_assignments
            WHERE cluster_id IN ({placeholders})
        """, [target_id] + source_ids)

        # Find best representative frame (highest quality_score)
        async with self.db.execute("""
            SELECT vf.id, vf.frame_path
            FROM video_frames vf
            JOIN cluster_frame_assignments cfa ON cfa.frame_id = vf.id
            WHERE cfa.cluster_id = ?
            ORDER BY vf.quality_score DESC
            LIMIT 1
        """, [target_id]) as cursor:
            row = await cursor.fetchone()
            if row:
                best_rep_id, best_rep_path = row
                await self.db.execute("""
                    UPDATE clusters SET representative_frame_id = ?, representative_frame = ? WHERE id = ?
                """, [best_rep_id, best_rep_path, target_id])

        # Preserve manual crops/uploads on merge; keep one primary and deduplicate sources.
        from services.reference_service import ReferenceService
        reference_service = ReferenceService(self.db)
        all_refs = await reference_service.list(video_id)
        merged_refs = [r for r in all_refs if r["person_id"] in merged_ids]
        merged_refs.sort(key=lambda r: (r["person_id"] != target_id, not r["primary"]))
        kept, seen = [], set()
        primary_found = False
        for ref in merged_refs:
            key = (ref["role"], ref["source_type"], ref["source"])
            if key in seen:
                continue
            seen.add(key)
            ref["person_id"] = target_id
            ref["primary"] = ref["role"] == "face" and not primary_found
            primary_found = primary_found or ref["primary"]
            kept.append(ref)
        await self.db.execute(
            f"DELETE FROM visual_references WHERE person_id IN ({','.join('?' * len(merged_ids))})", merged_ids
        )
        await reference_service.insert(video_id, kept)
        await self.db.execute("UPDATE videos SET reference_revision=reference_revision+1 WHERE id=?", [video_id])

        # Delete source clusters (assignments cascade)
        await self.db.execute(f"""
            DELETE FROM clusters
            WHERE id IN ({placeholders})
        """, source_ids)
        await self.fit_person_centroid(
            target_id, output_dir, None if largest_blob is None else centroid_from_blob(largest_blob)
        )

        # Merge physical cluster folders, which mirror the 'person' view only
        if view_mode == 'person':
            target_folder = clusters_dir / f"cluster_{target_index}"
            target_folder.mkdir(parents=True, exist_ok=True)

            for source_idx in source_indices:
                source_folder = clusters_dir / f"cluster_{source_idx}"
                if source_folder.exists():
                    shutil.rmtree(source_folder)

        # Capture old indices BEFORE reindex so _sync_cluster_folders can map folders correctly
        old_index_by_id = await self._get_cluster_index_map(video_id, view_mode)

        # Reindex clusters
        await self._reindex_clusters_v2(video_id, view_mode)

        # Keep the merged references and get the target's new index.
        async with self.db.execute("""
            SELECT cluster_index FROM clusters WHERE id = ?
        """, [target_id]) as cursor:
            row = await cursor.fetchone()
            new_target_index = row[0] if row else 0

        await self.db.commit()

        # Sync physical cluster folders and update JSON files
        if view_mode == 'person':
            await self._sync_cluster_folders(video_id, output_dir, old_index_by_id)

        return {
            'success': True,
            'merged_count': len(source_ids),
            'target_cluster_index': new_target_index
        }

    # =========================================================================
    # HELPER METHODS FOR FILE OPERATIONS
    # =========================================================================

    def _get_video_output_dir(self, video: dict) -> Path:
        """Get the output directory for a video."""
        from utils import sanitize_filename

        video_name = Path(video['filepath']).stem
        safe_name = sanitize_filename(video_name)
        return Path(OUTPUT_DIR) / safe_name

    async def _sync_cluster_folders(
        self,
        video_id: int,
        output_dir: Path,
        old_index_by_id: Optional[Dict[int, int]] = None
    ):
        """
        Synchronize physical cluster folders with database state.
        - Rename folders to match new indices (cluster_0, cluster_1, ...)
        - Update clusters.json and clustering_result.json

        Args:
            video_id: The video ID
            output_dir: Output directory for the video
            old_index_by_id: Optional mapping of cluster_id -> old_cluster_index.
                             Used after reindexing to find original folder names.
        """
        clusters_dir = output_dir / "clusters"
        if not clusters_dir.exists():
            clusters_dir.mkdir(parents=True, exist_ok=True)

        # Get current clusters from DB
        clusters = await self.get_clusters(video_id)

        # Build mapping from old folder names to new indices
        # First, rename all existing cluster folders to temp names to avoid collisions
        existing_folders = sorted(clusters_dir.glob("cluster_*"))
        temp_mapping = {}

        for folder in existing_folders:
            if folder.is_dir():
                temp_name = folder.parent / f"_temp_{folder.name}"
                folder.rename(temp_name)
                temp_mapping[folder.name] = temp_name

        # Now rename temp folders to correct indices based on DB order
        # We need to figure out which temp folder corresponds to which cluster
        # The simplest approach: rename based on position (cluster_0 -> first DB cluster, etc.)
        temp_folders = sorted(clusters_dir.glob("_temp_cluster_*"))

        for i, cluster in enumerate(clusters):
            new_folder_name = f"cluster_{i}"
            new_folder_path = clusters_dir / new_folder_name

            # Find corresponding temp folder if it exists
            # Use old_index_by_id if provided (after reindex), otherwise use current index
            if old_index_by_id and cluster['id'] in old_index_by_id:
                old_index = old_index_by_id[cluster['id']]
            else:
                old_index = cluster['cluster_index']

            old_folder_name = f"cluster_{old_index}"
            temp_folder = temp_mapping.get(old_folder_name)

            if temp_folder and temp_folder.exists():
                temp_folder.rename(new_folder_path)
            else:
                # Create new folder if it doesn't exist
                new_folder_path.mkdir(exist_ok=True)

                # Copy representative frame
                rep_frame = cluster.get('representative_frame')
                if rep_frame and Path(rep_frame).exists():
                    shutil.copy2(rep_frame, new_folder_path / "representative.jpg")

        # Clean up any remaining temp folders
        for temp_folder in clusters_dir.glob("_temp_cluster_*"):
            if temp_folder.is_dir():
                shutil.rmtree(temp_folder)

        # Update clusters.json
        await self._update_clusters_json(output_dir, clusters)

        # Update clustering_result.json
        await self._update_clustering_result_json(output_dir, clusters)

    async def _update_clusters_json(self, output_dir: Path, clusters: List[dict]):
        """Update clusters/clusters.json metadata file."""
        clusters_json_path = output_dir / "clusters" / "clusters.json"

        metadata = {
            'num_clusters': len(clusters),
            'clusters': [
                {
                    'index': c['cluster_index'],
                    'num_frames': c['num_frames'],
                    'representative': to_output_relative(
                        output_dir / "clusters" / f"cluster_{c['cluster_index']}" / "representative.jpg"
                    ),
                    'label': c.get('label')
                }
                for c in clusters
            ]
        }

        with open(clusters_json_path, 'wb') as f:
            f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

    async def _update_clustering_result_json(self, output_dir: Path, clusters: List[dict]):
        """Update clustering_result.json to reflect current state.

        V2 Architecture: Uses cluster_frame_assignments JOIN video_frames.
        """
        result_path = output_dir / "clustering_result.json"

        # Load existing file to preserve some metadata
        existing_data = {}
        if result_path.exists():
            try:
                with open(result_path, 'rb') as f:
                    existing_data = orjson.loads(f.read())
            except Exception:
                pass

        # Build updated result
        result = {
            'num_clusters': len(clusters),
            'total_faces': existing_data.get('total_faces', sum(c.get('num_frames', 0) for c in clusters)),
            'num_outliers': existing_data.get('num_outliers', 0),
            'parameters': existing_data.get('parameters', {'eps': 0.5, 'min_samples': 3}),
            'clusters': []
        }

        # Get frames for each cluster to build top_frames list
        for cluster in clusters:
            cluster_id = cluster['id']

            # Get top frames from DB (V2: via assignments)
            async with self.db.execute("""
                SELECT vf.frame_path, vf.quality_score, vf.expression
                FROM video_frames vf
                JOIN cluster_frame_assignments cfa ON cfa.frame_id = vf.id
                WHERE cfa.cluster_id = ?
                ORDER BY vf.quality_score DESC
                LIMIT 10
            """, [cluster_id]) as cursor:
                rows = await cursor.fetchall()
                top_frames = [row[0] for row in rows]

            # Count expressions (V2: via assignments)
            async with self.db.execute("""
                SELECT vf.expression, COUNT(*) as count
                FROM video_frames vf
                JOIN cluster_frame_assignments cfa ON cfa.frame_id = vf.id
                WHERE cfa.cluster_id = ?
                GROUP BY vf.expression
            """, [cluster_id]) as cursor:
                expr_rows = await cursor.fetchall()
                expression_distribution = {row[0]: row[1] for row in expr_rows if row[0]}

            blob = await self.centroid_blob(cluster_id)
            result['clusters'].append({
                'cluster_index': cluster['cluster_index'],
                'num_frames': cluster.get('num_frames', 0),
                'representative_frame': (
                    to_output_relative(cluster['representative_frame'])
                    if cluster.get('representative_frame') else ''
                ),
                'representative_quality': 0,
                'expression_distribution': expression_distribution,
                'top_frames': top_frames,
                'centroid': None if blob is None else centroid_list(centroid_from_blob(blob)),
            })

        with open(result_path, 'wb') as f:
            f.write(orjson.dumps(result, option=orjson.OPT_INDENT_2))

    # =========================================================================
    # V2 ARCHITECTURE HELPER METHODS
    # =========================================================================

    async def _reindex_clusters_v2(self, video_id: int, view_mode: str) -> None:
        """Reindex cluster_index to be consecutive within a view_mode."""
        async with self.db.execute("""
            SELECT id FROM clusters
            WHERE video_id = ? AND view_mode = ?
            ORDER BY cluster_index
        """, [video_id, view_mode]) as cursor:
            rows = await cursor.fetchall()

        for new_index, (cluster_id,) in enumerate(rows):
            await self.db.execute("""
                UPDATE clusters SET cluster_index = ? WHERE id = ?
            """, [new_index, cluster_id])

    async def _get_cluster_by_index_v2(
        self,
        video_id: int,
        cluster_index: int,
        view_mode: str = 'person'
    ) -> Optional[dict]:
        """Get cluster by index within a specific view_mode."""
        query = """
            SELECT c.id, c.cluster_index, c.label, c.description, c.view_mode,
                   c.representative_frame_id, c.scene_index,
                   COALESCE(vf.frame_path, c.representative_frame) as representative_frame
            FROM clusters c
            LEFT JOIN video_frames vf ON vf.id = c.representative_frame_id
            WHERE c.video_id = ? AND c.view_mode = ? AND c.cluster_index = ?
        """
        async with self.db.execute(query, [video_id, view_mode, cluster_index]) as cursor:
            row = await cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return resolve_row_paths(dict(zip(columns, row)), 'representative_frame')
            return None

    async def reset_reference_frames(
        self,
        video_id: int,
        cluster_index: int,
        view_mode: str = 'person'
    ) -> bool:
        """
        Reset face references to up to three useful views/expressions of the person.

        V2 Architecture: Uses cluster_frame_assignments.

        Returns:
            True if successful
        """
        cluster = await self._get_cluster_by_index_v2(video_id, cluster_index, view_mode)
        if not cluster:
            return False

        from services.person_service import PersonService
        from services.reference_service import ReferenceService
        people = PersonService(self.db)
        person = await people.get_person(video_id, cluster["id"])
        picked = await people.automatic_references(video_id, view_mode, cluster["id"], person["members"])
        await self.db.execute("DELETE FROM visual_references WHERE person_id=? AND role='face'", [cluster["id"]])
        await ReferenceService(self.db).insert(video_id, picked)
        await self.db.execute("UPDATE videos SET reference_revision=reference_revision+1 WHERE id=?", [video_id])
        await self.db.commit()
        return True

    async def cleanup_partial_analysis(self, video_id: int) -> dict:
        """Keep completed artifacts on cancellation.

        A cancelled status does not identify who created a file or whether it
        is incomplete. Stage writers own their temporary files and transactions;
        cancellation must never erase prior transcripts, people, or references.
        """
        video = await self.get_video(video_id)
        return {"cleaned_files": [], "cleaned_db_records": 0,
                "status_was": video.get("status") if video else None}

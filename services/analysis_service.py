"""
Analysis Service

Business logic for video analysis (scenes, faces, clustering).
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Set
import aiosqlite
import json
import shutil
import re

from config import MAX_REFERENCE_FRAMES
from i18n import t


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
        clustering_min_samples: int = 3
    ):
        """
        Run full analysis pipeline for a video.

        Steps:
        1. Scene detection
        2. Face extraction
        3. Face clustering
        4. Audio transcription (ElevenLabs)
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from config import OUTPUT_DIR
        from utils import VideoOutput
        from scene_detection import process_video_scenes
        from face_extraction import process_faces
        from transcription import transcribe_video

        try:
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
                import shutil
                shutil.rmtree(output.frames_dir, ignore_errors=True)
                output.frames_dir.mkdir(exist_ok=True)

            try:
                scene_result, extracted_frames = process_video_scenes(video_path, output)
                if not scene_result or not extracted_frames:
                    await self.update_video_status(video_id, 'error', 'scene_detection_error: No scenes or frames extracted')
                    return
            except Exception as e:
                await self.update_video_status(video_id, 'error', f'scene_detection_error: {str(e)}')
                return

            # Step 2: Face extraction
            await self.update_video_status(video_id, 'analyzing_faces')

            if force_faces:
                output.faces_file.unlink(missing_ok=True)

            try:
                face_result = process_faces(extracted_frames, output)
            except Exception as e:
                await self.update_video_status(video_id, 'error', f'face_extraction_error: {str(e)}')
                return

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

            # Step 4: Audio transcription
            await self.update_video_status(video_id, 'transcribing')

            if force_transcription:
                output.transcription_file.unlink(missing_ok=True)
                # Also remove JSON version if exists
                json_transcription = output.transcription_file.with_suffix('.json')
                json_transcription.unlink(missing_ok=True)

            try:
                transcription = transcribe_video(video_path, output)
                if not transcription:
                    await self.update_video_status(video_id, 'error', 'transcription_error: Failed to transcribe audio')
                    return
            except Exception as e:
                await self.update_video_status(video_id, 'error', f'transcription_error: {str(e)}')
                return

            # Mark as analyzed
            await self.update_video_status(video_id, 'analyzed')

        except Exception as e:
            await self.update_video_status(video_id, 'error', str(e))
            raise

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
        if clustering_result_path.exists() and not force:
            with open(clustering_result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
        else:
            result = run_clustering_pipeline(
                video_output_dir=output.output_dir,
                eps=eps,
                min_samples=min_samples,
                save_representatives=True
            )

        if result.get('error'):
            return

        # V2: Delete existing data
        await self.db.execute("""
            DELETE FROM cluster_frame_assignments
            WHERE cluster_id IN (SELECT id FROM clusters WHERE video_id = ?)
        """, [video_id])
        await self.db.execute("DELETE FROM clusters WHERE video_id = ?", [video_id])
        await self.db.execute("DELETE FROM video_frames WHERE video_id = ?", [video_id])

        clusters_data = result.get('clusters', [])

        # 1. First pass: Insert ALL unique frames into video_frames
        frame_id_map = {}  # frame_path -> frame_id
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

                if path and path not in frame_id_map:
                    scene_idx = extract_scene_index_from_path(path)
                    cursor = await self.db.execute("""
                        INSERT INTO video_frames (video_id, frame_path, quality_score, expression, scene_index)
                        VALUES (?, ?, ?, ?, ?)
                    """, [video_id, path, quality, expression, scene_idx])
                    frame_id_map[path] = cursor.lastrowid

        # 2. Create clusters for BOTH view_modes
        person_scene_index = 0  # Running index for person_scene clusters

        for cluster in clusters_data:
            cluster_idx = cluster.get('cluster_index', 0)
            representative_path = cluster.get('representative_frame', '')

            # Build frames list with quality info
            all_frames = cluster.get('frames', [])
            if not all_frames:
                all_frames = [{'frame_path': p, 'quality_score': 0, 'expression': 'unknown'}
                              for p in cluster.get('top_frames', [])]

            frames_with_quality = []
            seen_paths = set()
            for frame in all_frames:
                if isinstance(frame, dict):
                    path = frame.get('frame_path', '')
                    quality = frame.get('quality_score', 0)
                else:
                    path = frame
                    quality = 0

                if path and path not in seen_paths:
                    seen_paths.add(path)
                    scene_idx = extract_scene_index_from_path(path)
                    frames_with_quality.append({
                        'frame_path': path,
                        'frame_id': frame_id_map.get(path),
                        'quality_score': quality,
                        'scene_index': scene_idx
                    })

            # Sort by quality descending
            frames_with_quality.sort(key=lambda f: f['quality_score'], reverse=True)

            if not frames_with_quality:
                continue

            # Get representative frame ID
            rep_frame_id = frame_id_map.get(representative_path) or frames_with_quality[0]['frame_id']

            # === Create 'person' cluster ===
            cursor = await self.db.execute("""
                INSERT INTO clusters (video_id, cluster_index, view_mode, representative_frame_id,
                                      num_frames, representative_frame)
                VALUES (?, ?, 'person', ?, ?, ?)
            """, [video_id, cluster_idx, rep_frame_id, len(frames_with_quality), representative_path])
            person_cluster_id = cursor.lastrowid

            # Assign all frames to person cluster
            for idx, frame in enumerate(frames_with_quality):
                is_ref = idx < MAX_REFERENCE_FRAMES
                ref_order = idx + 1 if is_ref else None
                await self.db.execute("""
                    INSERT INTO cluster_frame_assignments (cluster_id, frame_id, is_reference, reference_order)
                    VALUES (?, ?, ?, ?)
                """, [person_cluster_id, frame['frame_id'], 1 if is_ref else 0, ref_order])

            # === Create 'person_scene' clusters ===
            # Group frames by scene_index
            frames_by_scene = {}
            for frame in frames_with_quality:
                scene_idx = frame.get('scene_index')
                if scene_idx is None:
                    scene_idx = -1
                if scene_idx not in frames_by_scene:
                    frames_by_scene[scene_idx] = []
                frames_by_scene[scene_idx].append(frame)

            for scene_idx in sorted(frames_by_scene.keys()):
                scene_frames = frames_by_scene[scene_idx]
                if not scene_frames:
                    continue

                # Sort by quality
                scene_frames.sort(key=lambda f: f['quality_score'], reverse=True)
                scene_rep_id = scene_frames[0]['frame_id']
                scene_rep_path = scene_frames[0]['frame_path']

                # Create person_scene cluster
                cursor = await self.db.execute("""
                    INSERT INTO clusters (video_id, cluster_index, view_mode, scene_index,
                                          representative_frame_id, num_frames, representative_frame)
                    VALUES (?, ?, 'person_scene', ?, ?, ?, ?)
                """, [video_id, person_scene_index, scene_idx if scene_idx != -1 else None,
                      scene_rep_id, len(scene_frames), scene_rep_path])
                scene_cluster_id = cursor.lastrowid
                person_scene_index += 1

                # Assign frames to person_scene cluster
                for idx, frame in enumerate(scene_frames):
                    is_ref = idx < MAX_REFERENCE_FRAMES
                    ref_order = idx + 1 if is_ref else None
                    await self.db.execute("""
                        INSERT INTO cluster_frame_assignments (cluster_id, frame_id, is_reference, reference_order)
                        VALUES (?, ?, ?, ?)
                    """, [scene_cluster_id, frame['frame_id'], 1 if is_ref else 0, ref_order])

        await self.db.commit()

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
        """
        # V2: Query clusters with frame counts from cluster_frame_assignments
        # Note: Falls back to representative_frame if representative_frame_id is not set
        query = """
            SELECT
                c.id,
                c.cluster_index,
                c.label,
                c.description,
                c.view_mode,
                c.scene_index,
                c.representative_frame_id,
                COALESCE(vf.frame_path, c.representative_frame) as representative_frame,
                COUNT(cfa.id) as num_frames,
                SUM(CASE WHEN cfa.is_reference = 1 THEN 1 ELSE 0 END) as reference_count
            FROM clusters c
            LEFT JOIN cluster_frame_assignments cfa ON cfa.cluster_id = c.id
            LEFT JOIN video_frames vf ON vf.id = c.representative_frame_id
            WHERE c.video_id = ? AND c.view_mode = ?
            GROUP BY c.id
            ORDER BY c.cluster_index
        """

        async with self.db.execute(query, [video_id, view_mode]) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    async def get_cluster_frames(
        self,
        video_id: int,
        cluster_index: int,
        limit: int = 20,
        view_mode: str = 'person'
    ) -> List[dict]:
        """Get frames for a specific cluster.

        V2 Architecture: Uses cluster_frame_assignments JOIN video_frames.
        """
        cluster = await self._get_cluster_by_index_v2(video_id, cluster_index, view_mode)
        if not cluster:
            return []

        query = """
            SELECT vf.id, vf.frame_path, vf.quality_score, vf.expression, vf.scene_index
            FROM video_frames vf
            JOIN cluster_frame_assignments cfa ON cfa.frame_id = vf.id
            WHERE cfa.cluster_id = ?
            ORDER BY vf.quality_score DESC
            LIMIT ?
        """
        async with self.db.execute(query, [cluster['id'], limit]) as cursor:
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    async def get_cluster_representative(
        self,
        video_id: int,
        cluster_index: int,
        view_mode: str = 'person'
    ) -> Optional[str]:
        """Get representative frame path for a cluster."""
        cluster = await self._get_cluster_by_index_v2(video_id, cluster_index, view_mode)
        if cluster and cluster.get('representative_frame'):
            return cluster['representative_frame']
        return None

    async def get_cluster_representative_by_id(
        self,
        cluster_id: int
    ) -> Optional[str]:
        """Get representative frame path for a cluster by its ID."""
        query = """
            SELECT representative_frame
            FROM clusters
            WHERE id = ?
        """

        async with self.db.execute(query, [cluster_id]) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                return row[0]
            return None

    async def get_cluster_by_index(
        self,
        video_id: int,
        cluster_index: int,
        view_mode: str = 'person'
    ) -> Optional[dict]:
        """Get a single cluster by video_id, cluster_index and view_mode.

        V2 Architecture: Uses view_mode to distinguish between views.
        """
        return await self._get_cluster_by_index_v2(video_id, cluster_index, view_mode)

    async def update_cluster_info(
        self,
        video_id: int,
        cluster_index: int,
        label: Optional[str] = None,
        description: Optional[str] = None,
        view_mode: str = 'person'
    ) -> bool:
        """
        Update cluster label and/or description.

        Args:
            video_id: Video ID
            cluster_index: Cluster index
            label: New label (None = don't change)
            description: New description (None = don't change)
            view_mode: 'person' or 'person_scene'

        Returns:
            True if updated, False if cluster not found
        """
        cluster = await self._get_cluster_by_index_v2(video_id, cluster_index, view_mode)
        if not cluster:
            return False

        # Build update query dynamically based on what's provided
        updates = []
        params = []

        if label is not None:
            label_val = label.strip()[:128] if label.strip() else None
            updates.append("label = ?")
            params.append(label_val)

        if description is not None:
            desc_val = description.strip()[:2000] if description.strip() else None
            updates.append("description = ?")
            params.append(desc_val)

        if not updates:
            return True  # Nothing to update

        params.append(cluster['id'])

        query = f"UPDATE clusters SET {', '.join(updates)} WHERE id = ?"
        await self.db.execute(query, params)
        await self.db.commit()

        return True

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

        # Delete cluster folder (clusters/cluster_X/) if exists
        cluster_folder = output_dir / "clusters" / f"cluster_{cluster_index}"
        if cluster_folder.exists():
            shutil.rmtree(cluster_folder)

        # Delete cluster (assignments cascade automatically)
        await self.db.execute("""
            DELETE FROM clusters
            WHERE video_id = ? AND view_mode = ? AND cluster_index = ?
        """, [video_id, view_mode, cluster_index])

        # Reindex remaining clusters in this view_mode
        await self._reindex_clusters_v2(video_id, view_mode)
        await self.db.commit()

        # Sync physical cluster folders and update JSON files
        await self._sync_cluster_folders(video_id, output_dir)

        return True

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

        # Move assignments to target (ignore duplicates)
        placeholders = ','.join('?' * len(source_ids))
        await self.db.execute(f"""
            INSERT OR IGNORE INTO cluster_frame_assignments
                (cluster_id, frame_id, is_reference, reference_order)
            SELECT ?, frame_id, 0, NULL
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
                    UPDATE clusters SET representative_frame_id = ? WHERE id = ?
                """, [best_rep_id, target_id])

        # Delete source clusters (assignments cascade)
        await self.db.execute(f"""
            DELETE FROM clusters
            WHERE id IN ({placeholders})
        """, source_ids)

        # Merge physical cluster folders
        target_folder = clusters_dir / f"cluster_{target_index}"
        target_folder.mkdir(parents=True, exist_ok=True)

        for source_idx in source_indices:
            source_folder = clusters_dir / f"cluster_{source_idx}"
            if source_folder.exists():
                shutil.rmtree(source_folder)

        # Reindex clusters
        await self._reindex_clusters_v2(video_id, view_mode)

        # Reset references in target to top quality
        # Get new target index after reindex
        async with self.db.execute("""
            SELECT cluster_index FROM clusters WHERE id = ?
        """, [target_id]) as cursor:
            row = await cursor.fetchone()
            new_target_index = row[0] if row else 0

        await self.reset_reference_frames(video_id, new_target_index, view_mode)

        await self.db.commit()

        # Sync physical cluster folders and update JSON files
        await self._sync_cluster_folders(video_id, output_dir)

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
        from config import OUTPUT_DIR
        from utils import sanitize_filename

        video_name = Path(video['filepath']).stem
        safe_name = sanitize_filename(video_name)
        return Path(OUTPUT_DIR) / safe_name

    def _validate_frame_path(self, frame_path: str, video_output_dir: Path) -> bool:
        """
        Validate that a frame path belongs to the video's output directory.
        Prevents path traversal attacks.

        Args:
            frame_path: The frame path to validate
            video_output_dir: The video's output directory (from _get_video_output_dir)

        Returns:
            True if the path is valid and within the video's directory
        """
        try:
            # Resolve to absolute path
            resolved_path = Path(frame_path).resolve()
            video_dir_resolved = video_output_dir.resolve()

            # Check if path is within video's output directory
            return resolved_path.is_relative_to(video_dir_resolved)
        except (ValueError, OSError):
            return False

    async def _get_cluster_frame_paths(self, cluster_id: int) -> Set[str]:
        """Get all frame paths belonging to a cluster.

        V2 Architecture: Uses cluster_frame_assignments JOIN video_frames.
        """
        async with self.db.execute("""
            SELECT vf.frame_path
            FROM video_frames vf
            JOIN cluster_frame_assignments cfa ON cfa.frame_id = vf.id
            WHERE cfa.cluster_id = ?
        """, [cluster_id]) as cursor:
            rows = await cursor.fetchall()
            return {row[0] for row in rows}

    async def _get_other_clusters_frame_paths(self, video_id: int, exclude_cluster_id: int) -> Set[str]:
        """Get frame paths from all clusters EXCEPT the specified one.

        V2 Architecture: Uses cluster_frame_assignments JOIN video_frames.
        """
        async with self.db.execute("""
            SELECT DISTINCT vf.frame_path
            FROM video_frames vf
            JOIN cluster_frame_assignments cfa ON cfa.frame_id = vf.id
            JOIN clusters c ON cfa.cluster_id = c.id
            WHERE c.video_id = ? AND c.id != ?
        """, [video_id, exclude_cluster_id]) as cursor:
            rows = await cursor.fetchall()
            return {row[0] for row in rows}

    async def _remove_faces_from_json(self, output_dir: Path, frame_paths_to_remove: Set[str]):
        """Remove faces belonging to deleted frames from faces.json."""
        faces_file = output_dir / "faces.json"
        if not faces_file.exists():
            return

        try:
            with open(faces_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'all_faces' not in data:
                return

            original_count = len(data['all_faces'])

            # Normalize paths for comparison (handle Windows path separators)
            normalized_to_remove = {str(Path(p)) for p in frame_paths_to_remove}

            # Filter out faces from deleted frames
            data['all_faces'] = [
                face for face in data['all_faces']
                if str(Path(face.get('frame_path', ''))) not in normalized_to_remove
            ]

            # Update counters
            new_count = len(data['all_faces'])
            data['total_frames_analyzed'] = new_count
            data['frames_with_faces'] = new_count

            # Save updated file
            with open(faces_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception:
            # If faces.json is corrupted or has unexpected format, skip
            pass

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
                    'representative': str(output_dir / "clusters" / f"cluster_{c['cluster_index']}" / "representative.jpg"),
                    'label': c.get('label')
                }
                for c in clusters
            ]
        }

        with open(clusters_json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    async def _update_clustering_result_json(self, output_dir: Path, clusters: List[dict]):
        """Update clustering_result.json to reflect current state.

        V2 Architecture: Uses cluster_frame_assignments JOIN video_frames.
        """
        result_path = output_dir / "clustering_result.json"

        # Load existing file to preserve some metadata
        existing_data = {}
        if result_path.exists():
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
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

            result['clusters'].append({
                'cluster_index': cluster['cluster_index'],
                'num_frames': cluster.get('num_frames', 0),
                'representative_frame': cluster.get('representative_frame', ''),
                'representative_quality': 0,
                'expression_distribution': expression_distribution,
                'top_frames': top_frames
            })

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    async def _reindex_clusters(self, video_id: int):
        """Reindex clusters to maintain consecutive indices (0, 1, 2...)."""
        # Get all remaining clusters ordered by current index
        async with self.db.execute("""
            SELECT id, cluster_index FROM clusters
            WHERE video_id = ?
            ORDER BY cluster_index
        """, [video_id]) as cursor:
            rows = await cursor.fetchall()

        # Update indices to be consecutive
        for new_index, (cluster_id, _) in enumerate(rows):
            await self.db.execute(
                "UPDATE clusters SET cluster_index = ? WHERE id = ?",
                [new_index, cluster_id]
            )

    # =========================================================================
    # V2 ARCHITECTURE HELPER METHODS
    # =========================================================================

    async def _get_frame_by_id(self, frame_id: int) -> Optional[dict]:
        """Get a video_frame by its ID."""
        async with self.db.execute("""
            SELECT id, video_id, frame_path, quality_score, expression, scene_index
            FROM video_frames WHERE id = ?
        """, [frame_id]) as cursor:
            row = await cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None

    async def _get_frame_id_by_path(self, video_id: int, frame_path: str) -> Optional[int]:
        """Get frame ID by its path within a video."""
        async with self.db.execute("""
            SELECT id FROM video_frames
            WHERE video_id = ? AND frame_path = ?
        """, [video_id, frame_path]) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

    async def _assign_frame_to_cluster(
        self,
        cluster_id: int,
        frame_id: int,
        is_reference: bool = False,
        reference_order: Optional[int] = None
    ) -> bool:
        """Assign a frame to a cluster. Returns True if created, False if already exists."""
        try:
            await self.db.execute("""
                INSERT INTO cluster_frame_assignments
                    (cluster_id, frame_id, is_reference, reference_order)
                VALUES (?, ?, ?, ?)
            """, [cluster_id, frame_id, 1 if is_reference else 0, reference_order])
            return True
        except Exception:  # UNIQUE constraint violation
            return False

    async def _reorder_references_v2(self, cluster_id: int) -> None:
        """Reorder reference_order to be consecutive (1, 2, 3, ...)."""
        async with self.db.execute("""
            SELECT id FROM cluster_frame_assignments
            WHERE cluster_id = ? AND is_reference = 1
            ORDER BY reference_order NULLS LAST, added_at
        """, [cluster_id]) as cursor:
            rows = await cursor.fetchall()

        for new_order, (assignment_id,) in enumerate(rows, start=1):
            await self.db.execute("""
                UPDATE cluster_frame_assignments
                SET reference_order = ?
                WHERE id = ?
            """, [new_order, assignment_id])

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
                return dict(zip(columns, row))
            return None

    # =========================================================================
    # FRAME MANAGEMENT METHODS
    # =========================================================================

    async def get_all_cluster_frames_from_disk(
        self,
        video_id: int,
        cluster_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Read ALL frames for a cluster directly from disk.
        Does not depend on DB, just lists files in clusters/cluster_X/frames/

        Returns:
            {
                'frames': [{'filename': str, 'path': str, 'size_bytes': int}, ...],
                'total': int
            }
        """
        video = await self.get_video(video_id)
        if not video:
            return None

        output_dir = self._get_video_output_dir(video)
        cluster_frames_dir = output_dir / "clusters" / f"cluster_{cluster_index}" / "frames"

        if not cluster_frames_dir.exists():
            return {'frames': [], 'total': 0}

        frames = []
        for img_path in sorted(cluster_frames_dir.glob("*.jpg")):
            frames.append({
                'filename': img_path.name,
                'path': str(img_path),
                'size_bytes': img_path.stat().st_size
            })

        return {
            'frames': frames,
            'total': len(frames)
        }

    async def get_cluster_frame_image_path(
        self,
        video_id: int,
        cluster_index: int,
        filename: str
    ) -> Optional[Path]:
        """
        Get the full path to a specific frame image on disk.
        Used by the API endpoint to serve images.

        Searches in order:
        1. Main frames/ directory (virtual references - new system)
        2. Legacy cluster_X/frames/ directory (backwards compatibility)

        Returns:
            Path to image file, or None if not found
        """
        video = await self.get_video(video_id)
        if not video:
            return None

        output_dir = self._get_video_output_dir(video)

        # First try the main frames/ directory (virtual references system)
        main_frame_path = output_dir / "frames" / filename
        if main_frame_path.exists():
            return main_frame_path

        # Fallback to legacy cluster-specific frames directory
        legacy_frame_path = output_dir / "clusters" / f"cluster_{cluster_index}" / "frames" / filename
        if legacy_frame_path.exists():
            return legacy_frame_path

        return None

    async def delete_cluster_frames_from_disk(
        self,
        video_id: int,
        cluster_index: int,
        filenames: List[str]
    ) -> Dict[str, Any]:
        """
        Delete frame files directly from disk (clusters/cluster_X/frames/).
        Does NOT modify DB - this is for the file explorer view.

        Returns:
            {'deleted': int, 'errors': [...]}
        """
        video = await self.get_video(video_id)
        if not video:
            return None

        output_dir = self._get_video_output_dir(video)
        cluster_frames_dir = output_dir / "clusters" / f"cluster_{cluster_index}" / "frames"

        deleted = 0
        errors = []

        for filename in filenames:
            # Security: prevent path traversal
            if ".." in filename or "/" in filename or "\\" in filename:
                errors.append(f"Invalid filename: {filename}")
                continue

            file_path = cluster_frames_dir / filename
            try:
                if file_path.exists():
                    file_path.unlink()
                    deleted += 1
                else:
                    errors.append(f"File not found: {filename}")
            except Exception as e:
                errors.append(f"Error deleting {filename}: {e}")

        return {
            'deleted': deleted,
            'errors': errors,
            'remaining': len(list(cluster_frames_dir.glob("*.jpg"))) if cluster_frames_dir.exists() else 0
        }

    async def get_all_cluster_frames(
        self,
        video_id: int,
        cluster_index: int,
        view_mode: str = 'person'
    ) -> Dict[str, Any]:
        """
        Get all frames for a cluster, split into references and library.

        V2 Architecture: Each view_mode is independent. No parent-child aggregation.

        Returns:
            {
                'reference_frames': [...],  # Frames marked as AI reference
                'library_frames': [...],    # All other frames
                'total_frames': int,
                'reference_count': int,
                'is_custom_selection': bool  # True if user customized references
            }
        """
        cluster = await self._get_cluster_by_index_v2(video_id, cluster_index, view_mode)
        if not cluster:
            return None

        cluster_id = cluster['id']

        # Get reference frames (ordered by reference_order)
        async with self.db.execute("""
            SELECT vf.id, vf.frame_path, vf.quality_score, vf.expression, vf.scene_index,
                   cfa.is_reference, cfa.reference_order
            FROM video_frames vf
            JOIN cluster_frame_assignments cfa ON cfa.frame_id = vf.id
            WHERE cfa.cluster_id = ? AND cfa.is_reference = 1
            ORDER BY cfa.reference_order ASC
        """, [cluster_id]) as cursor:
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            reference_frames = [dict(zip(columns, row)) for row in rows]

        # Get library frames (non-reference, ordered by quality)
        async with self.db.execute("""
            SELECT vf.id, vf.frame_path, vf.quality_score, vf.expression, vf.scene_index,
                   cfa.is_reference, cfa.reference_order
            FROM video_frames vf
            JOIN cluster_frame_assignments cfa ON cfa.frame_id = vf.id
            WHERE cfa.cluster_id = ? AND cfa.is_reference = 0
            ORDER BY vf.quality_score DESC
        """, [cluster_id]) as cursor:
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            library_frames = [dict(zip(columns, row)) for row in rows]

        # Check if it's a custom selection
        is_custom = False
        if reference_frames:
            all_frames_by_quality = sorted(
                reference_frames + library_frames,
                key=lambda f: f['quality_score'] or 0,
                reverse=True
            )
            top_n_ids = {f['id'] for f in all_frames_by_quality[:len(reference_frames)]}
            current_ref_ids = {f['id'] for f in reference_frames}
            is_custom = top_n_ids != current_ref_ids

        return {
            'reference_frames': reference_frames,
            'library_frames': library_frames,
            'total_frames': len(reference_frames) + len(library_frames),
            'reference_count': len(reference_frames),
            'is_custom_selection': is_custom
        }

    async def update_reference_frames(
        self,
        video_id: int,
        cluster_index: int,
        frame_ids: List[int],
        view_mode: str = 'person'
    ) -> bool:
        """
        Update which frames are marked as references for AI generation.

        V2 Architecture: Updates cluster_frame_assignments table.

        Args:
            video_id: Video ID
            cluster_index: Cluster index
            frame_ids: List of video_frame IDs to set as references (in order)
            view_mode: 'person' or 'person_scene'

        Returns:
            True if successful
        """
        cluster = await self._get_cluster_by_index_v2(video_id, cluster_index, view_mode)
        if not cluster:
            return False

        cluster_id = cluster['id']

        # Validate limit
        if len(frame_ids) > MAX_REFERENCE_FRAMES:
            frame_ids = frame_ids[:MAX_REFERENCE_FRAMES]

        # Clear all current references for this cluster
        await self.db.execute("""
            UPDATE cluster_frame_assignments
            SET is_reference = 0, reference_order = NULL
            WHERE cluster_id = ?
        """, [cluster_id])

        # Set new references with order
        for order, frame_id in enumerate(frame_ids, start=1):
            await self.db.execute("""
                UPDATE cluster_frame_assignments
                SET is_reference = 1, reference_order = ?
                WHERE cluster_id = ? AND frame_id = ?
            """, [order, cluster_id, frame_id])

        await self.db.commit()
        return True

    async def reset_reference_frames(
        self,
        video_id: int,
        cluster_index: int,
        view_mode: str = 'person'
    ) -> bool:
        """
        Reset references to top N frames by quality.

        V2 Architecture: Uses cluster_frame_assignments.

        Returns:
            True if successful
        """
        cluster = await self._get_cluster_by_index_v2(video_id, cluster_index, view_mode)
        if not cluster:
            return False

        cluster_id = cluster['id']

        # Clear all references
        await self.db.execute("""
            UPDATE cluster_frame_assignments
            SET is_reference = 0, reference_order = NULL
            WHERE cluster_id = ?
        """, [cluster_id])

        # Get top frames by quality
        async with self.db.execute(f"""
            SELECT cfa.id, cfa.frame_id
            FROM cluster_frame_assignments cfa
            JOIN video_frames vf ON vf.id = cfa.frame_id
            WHERE cfa.cluster_id = ?
            ORDER BY vf.quality_score DESC
            LIMIT {MAX_REFERENCE_FRAMES}
        """, [cluster_id]) as cursor:
            top_assignments = await cursor.fetchall()

        # Mark as references
        for order, (assignment_id, _) in enumerate(top_assignments, start=1):
            await self.db.execute("""
                UPDATE cluster_frame_assignments
                SET is_reference = 1, reference_order = ?
                WHERE id = ?
            """, [order, assignment_id])

        await self.db.commit()
        return True

    async def add_frames_to_references(
        self,
        video_id: int,
        cluster_index: int,
        frame_ids: List[int],
        view_mode: str = 'person'
    ) -> Dict[str, Any]:
        """
        Add frames to references (up to MAX_REFERENCE_FRAMES).

        V2 Architecture: Uses cluster_frame_assignments.

        Args:
            frame_ids: List of video_frame IDs to add as references

        Returns:
            {'added': int, 'skipped': int, 'total': int}
        """
        cluster = await self._get_cluster_by_index_v2(video_id, cluster_index, view_mode)
        if not cluster:
            return {'added': 0, 'skipped': 0, 'error': t('api.errors.cluster_not_found')}

        cluster_id = cluster['id']

        # Get current reference count
        async with self.db.execute("""
            SELECT COUNT(*) FROM cluster_frame_assignments
            WHERE cluster_id = ? AND is_reference = 1
        """, [cluster_id]) as cursor:
            current_count = (await cursor.fetchone())[0]

        # Get current max reference_order
        async with self.db.execute("""
            SELECT COALESCE(MAX(reference_order), 0) FROM cluster_frame_assignments
            WHERE cluster_id = ? AND is_reference = 1
        """, [cluster_id]) as cursor:
            max_order = (await cursor.fetchone())[0]

        added = 0
        skipped = 0

        for frame_id in frame_ids:
            if current_count + added >= MAX_REFERENCE_FRAMES:
                skipped += len(frame_ids) - (added + skipped)
                break

            # Check if already a reference
            async with self.db.execute("""
                SELECT is_reference FROM cluster_frame_assignments
                WHERE cluster_id = ? AND frame_id = ?
            """, [cluster_id, frame_id]) as cursor:
                row = await cursor.fetchone()

            if not row:
                skipped += 1  # Frame not assigned to this cluster
                continue

            if row[0] == 1:
                skipped += 1  # Already a reference
                continue

            # Add as reference
            max_order += 1
            await self.db.execute("""
                UPDATE cluster_frame_assignments
                SET is_reference = 1, reference_order = ?
                WHERE cluster_id = ? AND frame_id = ?
            """, [max_order, cluster_id, frame_id])
            added += 1

        await self.db.commit()
        return {'added': added, 'skipped': skipped, 'total': current_count + added}

    async def remove_frame_from_references(
        self,
        video_id: int,
        cluster_index: int,
        frame_id: int,
        view_mode: str = 'person'
    ) -> bool:
        """
        Remove a single frame from references.

        V2 Architecture: Uses cluster_frame_assignments.
        """
        cluster = await self._get_cluster_by_index_v2(video_id, cluster_index, view_mode)
        if not cluster:
            return False

        cluster_id = cluster['id']

        # Remove from references
        await self.db.execute("""
            UPDATE cluster_frame_assignments
            SET is_reference = 0, reference_order = NULL
            WHERE cluster_id = ? AND frame_id = ?
        """, [cluster_id, frame_id])

        # Reorder remaining references
        await self._reorder_references_v2(cluster_id)

        await self.db.commit()
        return True

    async def delete_cluster_frames(
        self,
        video_id: int,
        cluster_index: int,
        frame_ids: List[int],
        view_mode: str = 'person',
        delete_permanently: bool = False
    ) -> Dict[str, Any]:
        """
        Remove frames from a cluster.

        V2 Architecture: Distinguishes between removing assignment vs deleting permanently.

        Args:
            frame_ids: List of video_frame IDs
            view_mode: 'person' or 'person_scene'
            delete_permanently: If False (default), only removes the assignment.
                               If True, deletes from video_frames and disk.

        Returns:
            {'deleted': int, 'errors': [...], 'remaining_frames': int}
        """
        cluster = await self._get_cluster_by_index_v2(video_id, cluster_index, view_mode)
        if not cluster:
            return {'success': False, 'error': t('api.errors.cluster_not_found')}

        cluster_id = cluster['id']
        video = await self.get_video(video_id)
        output_dir = self._get_video_output_dir(video)

        deleted = 0
        errors = []

        for frame_id in frame_ids:
            if delete_permanently:
                # Get frame info for file deletion
                frame = await self._get_frame_by_id(frame_id)
                if frame:
                    # Delete from video_frames (cascades to all assignments)
                    await self.db.execute("DELETE FROM video_frames WHERE id = ?", [frame_id])

                    # Delete physical file
                    try:
                        frame_path = Path(frame['frame_path'])
                        if frame_path.exists():
                            frame_path.unlink()
                    except Exception as e:
                        errors.append(f"Failed to delete file {frame['frame_path']}: {e}")

                    deleted += 1
                else:
                    errors.append(f"Frame {frame_id} not found")
            else:
                # Just remove from this cluster
                await self.db.execute("""
                    DELETE FROM cluster_frame_assignments
                    WHERE cluster_id = ? AND frame_id = ?
                """, [cluster_id, frame_id])
                deleted += 1

        # Get remaining count
        async with self.db.execute("""
            SELECT COUNT(*) FROM cluster_frame_assignments WHERE cluster_id = ?
        """, [cluster_id]) as cursor:
            new_count = (await cursor.fetchone())[0]

        await self.db.commit()

        return {
            'success': True,
            'deleted': deleted,
            'errors': errors,
            'remaining_frames': new_count
        }

    async def get_reference_frames_for_generation(
        self,
        cluster_id: int,
        limit: int = 10
    ) -> List[dict]:
        """
        Get reference frames for AI generation.
        Used by GenerationService.

        V2 Architecture: Uses cluster_frame_assignments JOIN video_frames.

        Returns frames marked as references, ordered by reference_order.
        If no references are marked, falls back to top by quality_score.
        """
        # Try to get explicitly marked references first
        async with self.db.execute("""
            SELECT vf.frame_path, vf.quality_score, vf.expression
            FROM video_frames vf
            JOIN cluster_frame_assignments cfa ON cfa.frame_id = vf.id
            WHERE cfa.cluster_id = ? AND cfa.is_reference = 1
            ORDER BY cfa.reference_order ASC
            LIMIT ?
        """, [cluster_id, limit]) as cursor:
            rows = await cursor.fetchall()
            if rows:
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]

        # Fallback: top by quality
        async with self.db.execute("""
            SELECT vf.frame_path, vf.quality_score, vf.expression
            FROM video_frames vf
            JOIN cluster_frame_assignments cfa ON cfa.frame_id = vf.id
            WHERE cfa.cluster_id = ?
            ORDER BY vf.quality_score DESC
            LIMIT ?
        """, [cluster_id, limit]) as cursor:
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    # =========================================================================
    # ALL VIDEO FRAMES (for manual cluster creation)
    # =========================================================================

    async def get_all_video_frames(
        self,
        video_id: int,
        include_assigned: bool = True,
        view_mode: str = 'person'
    ) -> Optional[Dict[str, Any]]:
        """
        Get all frames from the video's frames/ directory.
        Used for manual cluster creation.

        V2 Architecture: Uses video_frames and cluster_frame_assignments.

        Args:
            video_id: Video ID
            include_assigned: If True, include frames already in clusters
            view_mode: 'person' or 'person_scene' for assignment lookup

        Returns:
            {
                'frames': [{'filename': str, 'path': str, 'quality_score': float,
                           'expression': str, 'cluster_id': int or None}, ...],
                'total': int,
                'assigned_count': int
            }
        """
        video = await self.get_video(video_id)
        if not video:
            return None

        output_dir = self._get_video_output_dir(video)
        frames_dir = output_dir / "frames"

        if not frames_dir.exists():
            return {'frames': [], 'total': 0, 'assigned_count': 0}

        # Get frames from video_frames table
        async with self.db.execute("""
            SELECT id, frame_path, quality_score, expression
            FROM video_frames
            WHERE video_id = ?
        """, [video_id]) as cursor:
            rows = await cursor.fetchall()
            db_frames = {row[1]: {'id': row[0], 'quality_score': row[2] or 0, 'expression': row[3] or 'unknown'}
                        for row in rows}

        # Get assigned frames (frames already in any cluster for the given view_mode)
        assigned_frames = {}
        async with self.db.execute("""
            SELECT vf.frame_path, c.id as cluster_id, c.cluster_index
            FROM video_frames vf
            JOIN cluster_frame_assignments cfa ON cfa.frame_id = vf.id
            JOIN clusters c ON cfa.cluster_id = c.id
            WHERE c.video_id = ? AND c.view_mode = ?
        """, [video_id, view_mode]) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                assigned_frames[row[0]] = {'cluster_id': row[1], 'cluster_index': row[2]}

        # List all frames from disk
        frames = []
        for img_path in sorted(frames_dir.glob("*.jpg")):
            full_path = str(img_path)

            # Get frame data from DB or use defaults
            frame_info = db_frames.get(full_path, {'quality_score': 0, 'expression': 'unknown'})

            # Get assignment info
            assignment = assigned_frames.get(full_path)

            # Skip assigned frames if requested
            if not include_assigned and assignment:
                continue

            frames.append({
                'filename': img_path.name,
                'path': full_path,
                'quality_score': frame_info.get('quality_score', 0),
                'expression': frame_info.get('expression', 'unknown'),
                'cluster_id': assignment['cluster_id'] if assignment else None,
                'cluster_index': assignment['cluster_index'] if assignment else None
            })

        # Sort by quality descending
        frames.sort(key=lambda f: f['quality_score'], reverse=True)

        return {
            'frames': frames,
            'total': len(frames),
            'assigned_count': len([f for f in frames if f['cluster_id'] is not None])
        }

    async def create_manual_cluster(
        self,
        video_id: int,
        frame_paths: List[str],
        label: Optional[str] = None,
        reference_frame_paths: Optional[List[str]] = None,
        description: Optional[str] = None,
        view_mode: str = 'person'
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new cluster manually from selected frames.

        V2 Architecture: Cluster is created in the specified view_mode.
        Uses video_frames and cluster_frame_assignments tables.

        Args:
            video_id: Video ID
            frame_paths: List of frame paths to include in cluster
            label: Optional name for the cluster
            reference_frame_paths: Optional list of frame paths to use as references
            description: Optional notes/comments about the cluster
            view_mode: 'person' or 'person_scene'

        Returns:
            Created cluster info or None if failed
        """
        if not frame_paths:
            return {'success': False, 'error': t('api.errors.no_frames_provided')}

        video = await self.get_video(video_id)
        if not video:
            return {'success': False, 'error': t('api.errors.video_not_found')}

        output_dir = self._get_video_output_dir(video)
        clusters_dir = output_dir / "clusters"
        clusters_dir.mkdir(parents=True, exist_ok=True)

        # Get next cluster index for this view_mode
        async with self.db.execute("""
            SELECT COALESCE(MAX(cluster_index), -1) + 1
            FROM clusters
            WHERE video_id = ? AND view_mode = ?
        """, [video_id, view_mode]) as cursor:
            next_index = (await cursor.fetchone())[0]

        # Get frame IDs from paths, ensure frames exist in video_frames
        frame_ids = []
        for path in frame_paths:
            # Security: validate path belongs to this video's directory
            if not self._validate_frame_path(path, output_dir):
                continue  # Skip invalid paths

            frame_id = await self._get_frame_id_by_path(video_id, path)
            if frame_id:
                frame_ids.append(frame_id)
            else:
                # Frame doesn't exist in video_frames, insert it
                scene_idx = extract_scene_index_from_path(path)
                cursor = await self.db.execute("""
                    INSERT INTO video_frames (video_id, frame_path, quality_score, expression, scene_index)
                    VALUES (?, ?, 50, 'unknown', ?)
                """, [video_id, path, scene_idx])
                frame_ids.append(cursor.lastrowid)

        if not frame_ids:
            return {'success': False, 'error': t('api.errors.no_valid_frames')}

        # Get frames with quality for sorting
        frames_data = []
        for fid in frame_ids:
            frame = await self._get_frame_by_id(fid)
            if frame:
                frames_data.append(frame)

        # Sort by quality
        frames_data.sort(key=lambda f: f.get('quality_score') or 0, reverse=True)

        # Determine representative frame (first one = highest quality)
        representative_frame_id = frames_data[0]['id'] if frames_data else frame_ids[0]

        # Truncate label and description
        label_val = label.strip()[:128] if label and label.strip() else None
        desc_val = description.strip()[:2000] if description and description.strip() else None

        # Create cluster
        num_frames = len(frame_ids)  # Use frame_ids count (always accurate, even if metadata fetch fails)
        representative_frame_path = frames_data[0]['frame_path'] if frames_data else frame_paths[0]

        cursor = await self.db.execute("""
            INSERT INTO clusters
                (video_id, cluster_index, label, description, view_mode, representative_frame_id, num_frames, representative_frame)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [video_id, next_index, label_val, desc_val, view_mode, representative_frame_id, num_frames, representative_frame_path])

        cluster_id = cursor.lastrowid

        # Determine which frames are references
        reference_ids = set()
        if reference_frame_paths:
            for path in reference_frame_paths[:MAX_REFERENCE_FRAMES]:
                # Security: validate path belongs to this video's directory
                if not self._validate_frame_path(path, output_dir):
                    continue
                fid = await self._get_frame_id_by_path(video_id, path)
                if fid:
                    reference_ids.add(fid)
        else:
            # Default: top N by quality
            reference_ids = {f['id'] for f in frames_data[:MAX_REFERENCE_FRAMES]}

        # Create assignments
        ref_order = 1
        for frame in frames_data:
            fid = frame['id']
            is_ref = fid in reference_ids
            order = ref_order if is_ref else None
            if is_ref:
                ref_order += 1

            await self._assign_frame_to_cluster(cluster_id, fid, is_ref, order)

        await self.db.commit()

        # Update JSON files
        clusters = await self.get_clusters(video_id, view_mode)
        await self._update_clusters_json(output_dir, clusters)
        await self._update_clustering_result_json(output_dir, clusters)

        return {
            'success': True,
            'cluster_id': cluster_id,
            'cluster_index': next_index,
            'num_frames': len(frame_ids),
            'num_references': len(reference_ids),
            'label': label
        }

    async def add_frames_to_cluster(
        self,
        video_id: int,
        cluster_index: int,
        frame_paths: List[str],
        view_mode: str = 'person'
    ) -> Optional[Dict[str, Any]]:
        """
        Add frames to an existing cluster.

        V2 Architecture: Creates assignments without copying files.

        Args:
            video_id: Video ID
            cluster_index: Cluster index to add frames to
            frame_paths: List of frame paths to add
            view_mode: 'person' or 'person_scene'

        Returns:
            {'added': int, 'skipped': int, 'errors': [...]}
        """
        cluster = await self._get_cluster_by_index_v2(video_id, cluster_index, view_mode)
        if not cluster:
            return {'success': False, 'error': t('api.errors.cluster_not_found')}

        video = await self.get_video(video_id)
        output_dir = self._get_video_output_dir(video)
        cluster_id = cluster['id']

        added = 0
        skipped = 0
        errors = []

        for fp in frame_paths:
            # Security: validate path belongs to this video's directory
            if not self._validate_frame_path(fp, output_dir):
                errors.append({'path': fp, 'error': 'Invalid path'})
                continue

            # Get or create frame in video_frames
            frame_id = await self._get_frame_id_by_path(video_id, fp)
            if not frame_id:
                # Insert into video_frames
                scene_idx = extract_scene_index_from_path(fp)
                cursor = await self.db.execute("""
                    INSERT INTO video_frames (video_id, frame_path, quality_score, expression, scene_index)
                    VALUES (?, ?, 50, 'unknown', ?)
                """, [video_id, fp, scene_idx])
                frame_id = cursor.lastrowid

            # Try to create assignment
            success = await self._assign_frame_to_cluster(cluster_id, frame_id)
            if success:
                added += 1
            else:
                skipped += 1  # Already in cluster

        await self.db.commit()

        # Get updated frame count
        async with self.db.execute("""
            SELECT COUNT(*) FROM cluster_frame_assignments WHERE cluster_id = ?
        """, [cluster_id]) as cursor:
            new_count = (await cursor.fetchone())[0]

        # Update JSON files
        clusters = await self.get_clusters(video_id, view_mode)
        await self._update_clusters_json(output_dir, clusters)
        await self._update_clustering_result_json(output_dir, clusters)

        return {
            'success': True,
            'added': added,
            'skipped': skipped,
            'errors': errors,
            'total_frames': new_count
        }

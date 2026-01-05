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
        """Run face clustering and save results to database."""
        from face_clustering import (
            run_clustering_pipeline,
            load_faces_with_embeddings,
            select_best_frames_from_cluster
        )

        # Check if faces.json exists
        if not output.faces_file.exists():
            return

        # Check for existing clustering result
        clustering_result_path = output.output_dir / "clustering_result.json"
        if clustering_result_path.exists() and not force:
            # Load existing result
            with open(clustering_result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
        else:
            # Run clustering pipeline
            result = run_clustering_pipeline(
                video_output_dir=output.output_dir,
                eps=eps,
                min_samples=min_samples,
                save_representatives=True
            )

        if result.get('error'):
            return

        # Delete existing clusters for this video
        await self.db.execute(
            "DELETE FROM cluster_frames WHERE cluster_id IN (SELECT id FROM clusters WHERE video_id = ?)",
            [video_id]
        )
        await self.db.execute(
            "DELETE FROM clusters WHERE video_id = ?",
            [video_id]
        )

        # Save clusters to database with dual-type structure:
        # - 'person' clusters: group by facial similarity (DBSCAN result)
        # - 'person_scene' clusters: subdivide each person by scene
        clusters = result.get('clusters', [])

        # Track next available cluster_index for person_scene clusters
        next_scene_cluster_idx = len(clusters)

        for cluster in clusters:
            cluster_idx = cluster.get('cluster_index', 0)
            representative = cluster.get('representative_frame', '')

            # Get centroid as bytes if available
            centroid_bytes = None
            if 'centroid' in cluster:
                import numpy as np
                centroid_array = np.array(cluster['centroid'], dtype=np.float32)
                centroid_bytes = centroid_array.tobytes()

            # Get ALL frames from the cluster
            all_frames = cluster.get('frames', [])

            if not all_frames:
                # Fallback: loaded from JSON, only has top_frames paths
                top_frames_paths = cluster.get('top_frames', [])
                all_frames = [
                    {'frame_path': p, 'quality_score': 0, 'expression': 'unknown'}
                    for p in top_frames_paths
                ]

            # Build frames list with quality info and scene_index
            frames_with_quality = []
            seen_paths = set()

            for frame in all_frames:
                if isinstance(frame, dict):
                    frame_path = frame.get('frame_path', '')
                    quality = frame.get('quality_score', 0)
                    expression = frame.get('expression', 'unknown')
                else:
                    frame_path = frame
                    quality = 0
                    expression = 'unknown'

                if frame_path and frame_path not in seen_paths:
                    seen_paths.add(frame_path)
                    scene_idx = extract_scene_index_from_path(frame_path)
                    frames_with_quality.append({
                        'frame_path': frame_path,
                        'quality_score': quality,
                        'expression': expression,
                        'scene_index': scene_idx
                    })

            # Sort by quality descending
            frames_with_quality.sort(key=lambda f: f['quality_score'], reverse=True)

            # Insert the 'person' cluster (parent) - num_frames will be calculated from children
            # But we store the total count here for backwards compatibility
            total_frames = len(frames_with_quality)
            cursor = await self.db.execute("""
                INSERT INTO clusters (video_id, cluster_index, num_frames, representative_frame, embedding_centroid, cluster_type)
                VALUES (?, ?, ?, ?, ?, 'person')
            """, [video_id, cluster_idx, total_frames, representative, centroid_bytes])

            person_cluster_id = cursor.lastrowid

            # Group frames by scene_index
            frames_by_scene = {}
            for frame in frames_with_quality:
                scene_idx = frame.get('scene_index')
                if scene_idx is None:
                    scene_idx = -1  # Unknown scene
                if scene_idx not in frames_by_scene:
                    frames_by_scene[scene_idx] = []
                frames_by_scene[scene_idx].append(frame)

            # Create 'person_scene' clusters for each scene
            for scene_idx in sorted(frames_by_scene.keys()):
                scene_frames = frames_by_scene[scene_idx]
                if not scene_frames:
                    continue

                # Sort scene frames by quality
                scene_frames.sort(key=lambda f: f['quality_score'], reverse=True)

                # Best frame in this scene as representative
                scene_representative = scene_frames[0]['frame_path']

                # Insert person_scene cluster
                cursor = await self.db.execute("""
                    INSERT INTO clusters (video_id, cluster_index, num_frames, representative_frame,
                                          cluster_type, parent_cluster_id, scene_index)
                    VALUES (?, ?, ?, ?, 'person_scene', ?, ?)
                """, [video_id, next_scene_cluster_idx, len(scene_frames), scene_representative,
                      person_cluster_id, scene_idx if scene_idx != -1 else None])

                scene_cluster_id = cursor.lastrowid
                next_scene_cluster_idx += 1

                # Insert frames into this person_scene cluster
                for idx, frame in enumerate(scene_frames):
                    is_reference = 1 if idx < MAX_REFERENCE_FRAMES else 0
                    reference_order = idx + 1 if idx < MAX_REFERENCE_FRAMES else None

                    await self.db.execute("""
                        INSERT INTO cluster_frames (cluster_id, frame_path, quality_score, expression, is_reference, reference_order, scene_index)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, [scene_cluster_id, frame['frame_path'], frame['quality_score'],
                          frame['expression'], is_reference, reference_order, frame.get('scene_index')])

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

    async def get_clusters(self, video_id: int, cluster_type: str = "person") -> List[dict]:
        """
        Get clusters for a video, filtered by type.

        Args:
            video_id: The video ID
            cluster_type: 'person' (unified by face) or 'person_scene' (split by scene)

        Returns:
            List of cluster dictionaries. For 'person' clusters, num_frames is
            aggregated from child 'person_scene' clusters if they exist.
        """
        if cluster_type == "person":
            # Get person clusters with aggregated frame count from children (if any)
            # If a person cluster has children, sum their frames; otherwise use its own count
            query = """
                SELECT
                    c.id,
                    c.cluster_index,
                    c.label,
                    c.description,
                    COALESCE(
                        (SELECT SUM(child.num_frames)
                         FROM clusters child
                         WHERE child.parent_cluster_id = c.id),
                        c.num_frames
                    ) as num_frames,
                    c.representative_frame,
                    c.cluster_type,
                    c.scene_index
                FROM clusters c
                WHERE c.video_id = ? AND (c.cluster_type = 'person' OR c.cluster_type IS NULL)
                ORDER BY c.cluster_index
            """
        else:
            # Get person_scene clusters with parent info for display
            query = """
                SELECT
                    c.id,
                    c.cluster_index,
                    c.label,
                    c.description,
                    c.num_frames,
                    c.representative_frame,
                    c.cluster_type,
                    c.scene_index,
                    c.parent_cluster_id,
                    p.cluster_index as parent_cluster_index,
                    p.label as parent_label
                FROM clusters c
                LEFT JOIN clusters p ON c.parent_cluster_id = p.id
                WHERE c.video_id = ? AND c.cluster_type = 'person_scene'
                ORDER BY p.cluster_index, c.scene_index
            """

        async with self.db.execute(query, [video_id]) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    async def get_cluster_frames(
        self,
        video_id: int,
        cluster_index: int,
        limit: int = 20
    ) -> List[dict]:
        """Get frames for a specific cluster."""
        # First get cluster ID
        query = "SELECT id FROM clusters WHERE video_id = ? AND cluster_index = ?"
        async with self.db.execute(query, [video_id, cluster_index]) as cursor:
            row = await cursor.fetchone()
            if not row:
                return []
            cluster_id = row[0]

        # Get frames
        query = """
            SELECT frame_path, quality_score, expression
            FROM cluster_frames
            WHERE cluster_id = ?
            ORDER BY quality_score DESC
            LIMIT ?
        """

        async with self.db.execute(query, [cluster_id, limit]) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    async def get_cluster_representative(
        self,
        video_id: int,
        cluster_index: int
    ) -> Optional[str]:
        """Get representative frame path for a cluster."""
        query = """
            SELECT representative_frame
            FROM clusters
            WHERE video_id = ? AND cluster_index = ?
        """

        async with self.db.execute(query, [video_id, cluster_index]) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                return row[0]
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
        cluster_index: int
    ) -> Optional[dict]:
        """Get a single cluster by video_id and cluster_index."""
        query = """
            SELECT id, cluster_index, label, description, num_frames, representative_frame,
                   embedding_centroid, cluster_type, parent_cluster_id, scene_index
            FROM clusters
            WHERE video_id = ? AND cluster_index = ?
        """
        async with self.db.execute(query, [video_id, cluster_index]) as cursor:
            row = await cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None

    async def update_cluster_info(
        self,
        video_id: int,
        cluster_index: int,
        label: Optional[str] = None,
        description: Optional[str] = None
    ) -> bool:
        """
        Update cluster label and/or description.

        Args:
            video_id: Video ID
            cluster_index: Cluster index
            label: New label (None = don't change)
            description: New description (None = don't change)

        Returns:
            True if updated, False if cluster not found
        """
        cluster = await self.get_cluster_by_index(video_id, cluster_index)
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

    async def delete_cluster(self, video_id: int, cluster_index: int) -> bool:
        """
        Delete a cluster and reindex remaining clusters.

        This method:
        1. Deletes the cluster folder from clusters/cluster_X/
        2. Updates clustering_result.json
        3. Renames remaining cluster folders to maintain sequence
        4. Deletes from database

        NOTE: Frames in the main frames/ directory are NEVER deleted,
        allowing users to reuse them for manual cluster creation later.

        Returns True if successful, False if cluster not found.
        """
        # Get video and cluster info
        video = await self.get_video(video_id)
        if not video:
            return False

        cluster = await self.get_cluster_by_index(video_id, cluster_index)
        if not cluster:
            return False

        output_dir = self._get_video_output_dir(video)
        cluster_id = cluster['id']

        # Delete cluster folder (clusters/cluster_X/)
        cluster_folder = output_dir / "clusters" / f"cluster_{cluster_index}"
        if cluster_folder.exists():
            shutil.rmtree(cluster_folder)

        # Delete from database (CASCADE will delete cluster_frames)
        await self.db.execute(
            "DELETE FROM clusters WHERE video_id = ? AND cluster_index = ?",
            [video_id, cluster_index]
        )

        # Reindex remaining clusters in DB
        await self._reindex_clusters(video_id)
        await self.db.commit()

        # Sync physical cluster folders and update JSON files
        await self._sync_cluster_folders(video_id, output_dir)

        return True

    async def merge_clusters(
        self,
        video_id: int,
        cluster_indices: List[int],
        target_index: int
    ) -> Optional[dict]:
        """
        Merge multiple clusters into one.

        This method:
        1. Moves all frames from secondary clusters to target cluster in DB
        2. Merges cluster preview folders
        3. Updates clustering_result.json
        4. Renames remaining cluster folders to maintain sequence

        Args:
            video_id: The video ID
            cluster_indices: List of cluster indices to merge
            target_index: Which cluster index to keep as the "main" one

        Returns:
            The merged cluster info, or None if failed.
        """
        if len(cluster_indices) < 2:
            return None

        if target_index not in cluster_indices:
            target_index = cluster_indices[0]

        # Get video info for output_dir
        video = await self.get_video(video_id)
        if not video:
            return None

        output_dir = self._get_video_output_dir(video)
        clusters_dir = output_dir / "clusters"

        # Get all clusters to merge
        clusters_to_merge = []
        for idx in cluster_indices:
            cluster = await self.get_cluster_by_index(video_id, idx)
            if cluster:
                clusters_to_merge.append(cluster)

        if len(clusters_to_merge) < 2:
            return None

        # Find the target cluster
        target_cluster = next(
            (c for c in clusters_to_merge if c['cluster_index'] == target_index),
            clusters_to_merge[0]
        )
        target_id = target_cluster['id']
        other_clusters = [c for c in clusters_to_merge if c['id'] != target_id]
        other_cluster_ids = [c['id'] for c in other_clusters]

        # Move all frames from other clusters to target cluster in DB
        for other_id in other_cluster_ids:
            await self.db.execute(
                "UPDATE cluster_frames SET cluster_id = ? WHERE cluster_id = ?",
                [target_id, other_id]
            )

        # Calculate new num_frames
        total_frames = sum(c['num_frames'] for c in clusters_to_merge)

        # Find best representative frame (highest quality_score)
        async with self.db.execute("""
            SELECT frame_path FROM cluster_frames
            WHERE cluster_id = ?
            ORDER BY quality_score DESC
            LIMIT 1
        """, [target_id]) as cursor:
            row = await cursor.fetchone()
            best_representative = row[0] if row else target_cluster['representative_frame']

        # Update target cluster in DB
        await self.db.execute("""
            UPDATE clusters
            SET num_frames = ?, representative_frame = ?
            WHERE id = ?
        """, [total_frames, best_representative, target_id])

        # Merge physical cluster folders
        target_folder = clusters_dir / f"cluster_{target_index}"
        target_folder.mkdir(parents=True, exist_ok=True)

        for other_cluster in other_clusters:
            other_idx = other_cluster['cluster_index']
            other_folder = clusters_dir / f"cluster_{other_idx}"

            if other_folder.exists():
                # Move preview frames to target folder with unique names
                for img_file in other_folder.glob("frame_*.jpg"):
                    new_name = f"merged_{other_idx}_{img_file.name}"
                    dest = target_folder / new_name
                    shutil.move(str(img_file), str(dest))

                # Remove the emptied folder
                shutil.rmtree(other_folder)

        # Update representative.jpg in target folder
        if best_representative and Path(best_representative).exists():
            rep_dest = target_folder / "representative.jpg"
            shutil.copy2(best_representative, rep_dest)

        # Delete other clusters from DB (frames already moved)
        for other_id in other_cluster_ids:
            await self.db.execute("DELETE FROM clusters WHERE id = ?", [other_id])

        # Capture old indices BEFORE reindexing (for folder mapping)
        all_remaining_clusters = await self.get_clusters(video_id)
        old_index_by_id = {c['id']: c['cluster_index'] for c in all_remaining_clusters}

        # Reindex remaining clusters in DB
        await self._reindex_clusters(video_id)
        await self.db.commit()

        # Sync physical cluster folders and update JSON files
        await self._sync_cluster_folders(video_id, output_dir, old_index_by_id)

        # Return updated cluster info
        clusters = await self.get_clusters(video_id)
        return clusters[0] if clusters else None

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

    async def _get_cluster_frame_paths(self, cluster_id: int) -> Set[str]:
        """Get all frame paths belonging to a cluster."""
        frame_paths = set()
        async with self.db.execute(
            "SELECT frame_path FROM cluster_frames WHERE cluster_id = ?",
            [cluster_id]
        ) as cursor:
            rows = await cursor.fetchall()
            frame_paths = {row[0] for row in rows}
        return frame_paths

    async def _get_other_clusters_frame_paths(self, video_id: int, exclude_cluster_id: int) -> Set[str]:
        """Get frame paths from all clusters EXCEPT the specified one."""
        frame_paths = set()
        async with self.db.execute("""
            SELECT cf.frame_path
            FROM cluster_frames cf
            JOIN clusters c ON cf.cluster_id = c.id
            WHERE c.video_id = ? AND c.id != ?
        """, [video_id, exclude_cluster_id]) as cursor:
            rows = await cursor.fetchall()
            frame_paths = {row[0] for row in rows}
        return frame_paths

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
        """Update clustering_result.json to reflect current state."""
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
            'total_faces': existing_data.get('total_faces', sum(c['num_frames'] for c in clusters)),
            'num_outliers': existing_data.get('num_outliers', 0),
            'parameters': existing_data.get('parameters', {'eps': 0.5, 'min_samples': 3}),
            'clusters': []
        }

        # Get frames for each cluster to build top_frames list
        for cluster in clusters:
            cluster_id = cluster['id']

            # Get top frames from DB
            async with self.db.execute("""
                SELECT frame_path, quality_score, expression
                FROM cluster_frames
                WHERE cluster_id = ?
                ORDER BY quality_score DESC
                LIMIT 10
            """, [cluster_id]) as cursor:
                rows = await cursor.fetchall()
                top_frames = [row[0] for row in rows]

            # Count expressions
            async with self.db.execute("""
                SELECT expression, COUNT(*) as count
                FROM cluster_frames
                WHERE cluster_id = ?
                GROUP BY expression
            """, [cluster_id]) as cursor:
                expr_rows = await cursor.fetchall()
                expression_distribution = {row[0]: row[1] for row in expr_rows if row[0]}

            result['clusters'].append({
                'cluster_index': cluster['cluster_index'],
                'num_frames': cluster['num_frames'],
                'representative_frame': cluster['representative_frame'],
                'representative_quality': 0,  # Not stored in DB currently
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
        cluster_index: int
    ) -> Dict[str, Any]:
        """
        Get all frames for a cluster, split into references and library.

        For 'person' clusters: aggregates frames from all child 'person_scene' clusters.
        For 'person_scene' clusters: returns frames directly.

        Returns:
            {
                'reference_frames': [...],  # Frames marked as AI reference
                'library_frames': [...],    # All other frames
                'total_frames': int,
                'reference_count': int,
                'is_custom_selection': bool  # True if user customized references
            }
        """
        # Get cluster info including type
        cluster = await self.get_cluster_by_index(video_id, cluster_index)
        if not cluster:
            return None

        cluster_id = cluster['id']
        cluster_type = cluster.get('cluster_type', 'person')

        # Determine which cluster IDs to query frames from
        if cluster_type == 'person' or cluster_type is None:
            # For 'person' clusters, get frames from all child 'person_scene' clusters
            async with self.db.execute("""
                SELECT id FROM clusters
                WHERE parent_cluster_id = ?
            """, [cluster_id]) as cursor:
                child_rows = await cursor.fetchall()
                child_ids = [row[0] for row in child_rows]

            # If there are children, use those; otherwise use the cluster itself (backwards compat)
            if child_ids:
                cluster_ids = child_ids
            else:
                cluster_ids = [cluster_id]
        else:
            # For 'person_scene' clusters, just use this cluster
            cluster_ids = [cluster_id]

        # Build placeholders for IN clause
        placeholders = ','.join('?' * len(cluster_ids))

        # Get reference frames (ordered by reference_order)
        async with self.db.execute(f"""
            SELECT id, frame_path, quality_score, expression, is_reference, reference_order, scene_index
            FROM cluster_frames
            WHERE cluster_id IN ({placeholders}) AND is_reference = 1
            ORDER BY reference_order ASC
        """, cluster_ids) as cursor:
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            reference_frames = [dict(zip(columns, row)) for row in rows]

        # Get library frames (non-reference, ordered by quality)
        async with self.db.execute(f"""
            SELECT id, frame_path, quality_score, expression, is_reference, reference_order, scene_index
            FROM cluster_frames
            WHERE cluster_id IN ({placeholders}) AND is_reference = 0
            ORDER BY quality_score DESC
        """, cluster_ids) as cursor:
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            library_frames = [dict(zip(columns, row)) for row in rows]

        # Check if it's a custom selection (reference_order might not match quality order)
        is_custom = False
        if reference_frames:
            # Get the top N by quality and compare
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
        frame_ids: List[int]
    ) -> bool:
        """
        Update which frames are marked as references for AI generation.

        Args:
            video_id: Video ID
            cluster_index: Cluster index
            frame_ids: List of frame IDs to set as references (in order)

        Returns:
            True if successful
        """
        cluster = await self.get_cluster_by_index(video_id, cluster_index)
        if not cluster:
            return False

        cluster_id = cluster['id']

        # Clear all existing references for this cluster
        await self.db.execute("""
            UPDATE cluster_frames
            SET is_reference = 0, reference_order = NULL
            WHERE cluster_id = ?
        """, [cluster_id])

        # Set new references with order
        for order, frame_id in enumerate(frame_ids[:MAX_REFERENCE_FRAMES], start=1):
            await self.db.execute("""
                UPDATE cluster_frames
                SET is_reference = 1, reference_order = ?
                WHERE id = ? AND cluster_id = ?
            """, [order, frame_id, cluster_id])

        await self.db.commit()
        return True

    async def reset_reference_frames(
        self,
        video_id: int,
        cluster_index: int
    ) -> bool:
        """
        Reset reference frames to the top 10 by quality score.

        Returns:
            True if successful
        """
        cluster = await self.get_cluster_by_index(video_id, cluster_index)
        if not cluster:
            return False

        cluster_id = cluster['id']

        # Clear all references
        await self.db.execute("""
            UPDATE cluster_frames
            SET is_reference = 0, reference_order = NULL
            WHERE cluster_id = ?
        """, [cluster_id])

        # Get top N by quality (N = MAX_REFERENCE_FRAMES)
        async with self.db.execute(f"""
            SELECT id FROM cluster_frames
            WHERE cluster_id = ?
            ORDER BY quality_score DESC
            LIMIT {MAX_REFERENCE_FRAMES}
        """, [cluster_id]) as cursor:
            rows = await cursor.fetchall()
            top_frame_ids = [row[0] for row in rows]

        # Set them as references
        for order, frame_id in enumerate(top_frame_ids, start=1):
            await self.db.execute("""
                UPDATE cluster_frames
                SET is_reference = 1, reference_order = ?
                WHERE id = ?
            """, [order, frame_id])

        await self.db.commit()
        return True

    async def add_frames_to_references(
        self,
        video_id: int,
        cluster_index: int,
        frame_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Add frames to references (up to max 10 total).

        Returns:
            {'added': int, 'skipped': int, 'total': int}
        """
        cluster = await self.get_cluster_by_index(video_id, cluster_index)
        if not cluster:
            return None

        cluster_id = cluster['id']

        # Get current reference count
        async with self.db.execute("""
            SELECT COUNT(*) FROM cluster_frames
            WHERE cluster_id = ? AND is_reference = 1
        """, [cluster_id]) as cursor:
            row = await cursor.fetchone()
            current_count = row[0]

        # Get current max order
        async with self.db.execute("""
            SELECT MAX(reference_order) FROM cluster_frames
            WHERE cluster_id = ? AND is_reference = 1
        """, [cluster_id]) as cursor:
            row = await cursor.fetchone()
            max_order = row[0] or 0

        available_slots = MAX_REFERENCE_FRAMES - current_count
        added = 0
        skipped = 0

        for frame_id in frame_ids:
            if added >= available_slots:
                skipped += 1
                continue

            # Check if already a reference
            async with self.db.execute("""
                SELECT is_reference FROM cluster_frames
                WHERE id = ? AND cluster_id = ?
            """, [frame_id, cluster_id]) as cursor:
                row = await cursor.fetchone()
                if not row:
                    skipped += 1
                    continue
                if row[0] == 1:
                    skipped += 1
                    continue

            # Add as reference
            max_order += 1
            await self.db.execute("""
                UPDATE cluster_frames
                SET is_reference = 1, reference_order = ?
                WHERE id = ? AND cluster_id = ?
            """, [max_order, frame_id, cluster_id])
            added += 1

        await self.db.commit()

        return {
            'added': added,
            'skipped': skipped,
            'total': current_count + added
        }

    async def remove_frame_from_references(
        self,
        video_id: int,
        cluster_index: int,
        frame_id: int
    ) -> bool:
        """
        Remove a single frame from references.
        """
        cluster = await self.get_cluster_by_index(video_id, cluster_index)
        if not cluster:
            return False

        cluster_id = cluster['id']

        await self.db.execute("""
            UPDATE cluster_frames
            SET is_reference = 0, reference_order = NULL
            WHERE id = ? AND cluster_id = ?
        """, [frame_id, cluster_id])

        # Reorder remaining references
        async with self.db.execute("""
            SELECT id FROM cluster_frames
            WHERE cluster_id = ? AND is_reference = 1
            ORDER BY reference_order
        """, [cluster_id]) as cursor:
            rows = await cursor.fetchall()

        for new_order, (fid,) in enumerate(rows, start=1):
            await self.db.execute("""
                UPDATE cluster_frames
                SET reference_order = ?
                WHERE id = ?
            """, [new_order, fid])

        await self.db.commit()
        return True

    async def delete_cluster_frames(
        self,
        video_id: int,
        cluster_index: int,
        frame_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Delete frames from a cluster (DB + physical files).

        Returns:
            {'deleted': int, 'errors': [...]}
        """
        cluster = await self.get_cluster_by_index(video_id, cluster_index)
        if not cluster:
            return None

        cluster_id = cluster['id']
        video = await self.get_video(video_id)
        output_dir = self._get_video_output_dir(video)

        deleted = 0
        errors = []

        for frame_id in frame_ids:
            # Get frame info
            async with self.db.execute("""
                SELECT frame_path FROM cluster_frames
                WHERE id = ? AND cluster_id = ?
            """, [frame_id, cluster_id]) as cursor:
                row = await cursor.fetchone()
                if not row:
                    errors.append(f"Frame {frame_id} not found")
                    continue

            frame_path = row[0]

            # Check if frame is used by other clusters
            async with self.db.execute("""
                SELECT COUNT(*) FROM cluster_frames
                WHERE frame_path = ? AND id != ?
            """, [frame_path, frame_id]) as cursor:
                count_row = await cursor.fetchone()
                is_shared = count_row[0] > 0

            # Delete from DB
            await self.db.execute("""
                DELETE FROM cluster_frames WHERE id = ?
            """, [frame_id])

            # Delete physical file only if not shared
            if not is_shared:
                try:
                    frame_file = Path(frame_path)
                    if frame_file.exists():
                        frame_file.unlink()
                except Exception as e:
                    errors.append(f"Could not delete file {frame_path}: {e}")

            deleted += 1

        # Update cluster num_frames
        async with self.db.execute("""
            SELECT COUNT(*) FROM cluster_frames WHERE cluster_id = ?
        """, [cluster_id]) as cursor:
            row = await cursor.fetchone()
            new_count = row[0]

        await self.db.execute("""
            UPDATE clusters SET num_frames = ? WHERE id = ?
        """, [new_count, cluster_id])

        # Remove faces from faces.json
        frames_to_remove = set()
        for frame_id in frame_ids:
            async with self.db.execute("""
                SELECT frame_path FROM cluster_frames WHERE id = ?
            """, [frame_id]) as cursor:
                row = await cursor.fetchone()
                if row:
                    frames_to_remove.add(row[0])

        if frames_to_remove:
            await self._remove_faces_from_json(output_dir, frames_to_remove)

        await self.db.commit()

        return {
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

        Returns frames marked as references, ordered by reference_order.
        If no references are marked, falls back to top by quality_score.
        """
        # Try to get marked references first
        async with self.db.execute("""
            SELECT frame_path, quality_score, expression
            FROM cluster_frames
            WHERE cluster_id = ? AND is_reference = 1
            ORDER BY reference_order ASC
            LIMIT ?
        """, [cluster_id, limit]) as cursor:
            rows = await cursor.fetchall()
            if rows:
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]

        # Fallback: top by quality
        async with self.db.execute("""
            SELECT frame_path, quality_score, expression
            FROM cluster_frames
            WHERE cluster_id = ?
            ORDER BY quality_score DESC
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
        include_assigned: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get all frames from the video's frames/ directory.
        Used for manual cluster creation.

        Args:
            video_id: Video ID
            include_assigned: If True, include frames already in clusters

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

        # Load faces.json to get quality_score and expression for each frame
        faces_data = {}
        faces_file = output_dir / "faces.json"
        if faces_file.exists():
            try:
                with open(faces_file, 'r', encoding='utf-8') as f:
                    faces_json = json.load(f)
                    for face in faces_json.get('all_faces', []):
                        frame_path = face.get('frame_path', '')
                        # Use highest quality face if multiple faces in same frame
                        if frame_path not in faces_data or face.get('quality_score', 0) > faces_data[frame_path].get('quality_score', 0):
                            faces_data[frame_path] = {
                                'quality_score': face.get('quality_score', 0),
                                'expression': face.get('expression', 'unknown')
                            }
            except Exception:
                pass

        # Get assigned frames (frames already in any cluster)
        assigned_frames = {}
        async with self.db.execute("""
            SELECT cf.frame_path, c.id as cluster_id, c.cluster_index
            FROM cluster_frames cf
            JOIN clusters c ON cf.cluster_id = c.id
            WHERE c.video_id = ?
        """, [video_id]) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                assigned_frames[row[0]] = {'cluster_id': row[1], 'cluster_index': row[2]}

        # List all frames from disk
        frames = []
        for img_path in sorted(frames_dir.glob("*.jpg")):
            full_path = str(img_path)

            # Get face data if available
            face_info = faces_data.get(full_path, {})

            # Get assignment info
            assignment = assigned_frames.get(full_path)

            # Skip assigned frames if requested
            if not include_assigned and assignment:
                continue

            frames.append({
                'filename': img_path.name,
                'path': full_path,
                'quality_score': face_info.get('quality_score', 0),
                'expression': face_info.get('expression', 'unknown'),
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
        description: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new cluster manually from selected frames.

        Args:
            video_id: Video ID
            frame_paths: List of frame paths to include in cluster
            label: Optional name for the cluster
            reference_frame_paths: Optional list of frame paths to use as references
                                   (if None, top N by quality will be used)
            description: Optional notes/comments about the cluster

        Returns:
            Created cluster info or None if failed
        """
        if not frame_paths:
            return None

        video = await self.get_video(video_id)
        if not video:
            return None

        output_dir = self._get_video_output_dir(video)
        clusters_dir = output_dir / "clusters"
        clusters_dir.mkdir(parents=True, exist_ok=True)

        # Load faces.json for quality/expression data
        faces_data = {}
        faces_file = output_dir / "faces.json"
        if faces_file.exists():
            try:
                with open(faces_file, 'r', encoding='utf-8') as f:
                    faces_json = json.load(f)
                    for face in faces_json.get('all_faces', []):
                        fp = face.get('frame_path', '')
                        if fp not in faces_data or face.get('quality_score', 0) > faces_data[fp].get('quality_score', 0):
                            faces_data[fp] = {
                                'quality_score': face.get('quality_score', 0),
                                'expression': face.get('expression', 'unknown')
                            }
            except Exception:
                pass

        # Get next cluster index
        async with self.db.execute("""
            SELECT COALESCE(MAX(cluster_index) + 1, 0) FROM clusters WHERE video_id = ?
        """, [video_id]) as cursor:
            row = await cursor.fetchone()
            new_cluster_index = row[0]

        # Build frames with quality info
        frames_with_quality = []
        for fp in frame_paths:
            face_info = faces_data.get(fp, {})
            frames_with_quality.append({
                'frame_path': fp,
                'quality_score': face_info.get('quality_score', 50),  # Default 50 if unknown
                'expression': face_info.get('expression', 'unknown')
            })

        # Sort by quality
        frames_with_quality.sort(key=lambda f: f['quality_score'], reverse=True)

        # Select representative frame (highest quality)
        representative_frame = frames_with_quality[0]['frame_path']

        # Create cluster folder structure
        cluster_folder = clusters_dir / f"cluster_{new_cluster_index}"
        cluster_folder.mkdir(exist_ok=True)
        preview_dir = cluster_folder / "preview"
        preview_dir.mkdir(exist_ok=True)
        cluster_frames_dir = cluster_folder / "frames"
        cluster_frames_dir.mkdir(exist_ok=True)

        # Copy representative to preview
        rep_path = Path(representative_frame)
        if rep_path.exists():
            shutil.copy2(rep_path, preview_dir / "representative.jpg")

        # Copy all frames to cluster frames dir
        for frame in frames_with_quality:
            src = Path(frame['frame_path'])
            if src.exists():
                shutil.copy2(src, cluster_frames_dir / src.name)

        # Copy preview frames (top 5)
        for i, frame in enumerate(frames_with_quality[:5]):
            src = Path(frame['frame_path'])
            if src.exists():
                shutil.copy2(src, preview_dir / f"frame_{i}.jpg")

        # Truncate label and description to max lengths
        label_val = label.strip()[:128] if label and label.strip() else None
        desc_val = description.strip()[:2000] if description and description.strip() else None

        # Insert cluster into database
        cursor = await self.db.execute("""
            INSERT INTO clusters (video_id, cluster_index, label, description, num_frames, representative_frame, embedding_centroid)
            VALUES (?, ?, ?, ?, ?, ?, NULL)
        """, [video_id, new_cluster_index, label_val, desc_val, len(frames_with_quality), representative_frame])

        cluster_id = cursor.lastrowid

        # Determine which frames are references
        if reference_frame_paths:
            # Use specified reference frames
            ref_set = set(reference_frame_paths)
        else:
            # Use top N by quality (N = MAX_REFERENCE_FRAMES)
            ref_set = set(f['frame_path'] for f in frames_with_quality[:MAX_REFERENCE_FRAMES])

        # Insert frames into cluster_frames
        ref_order = 1
        for frame in frames_with_quality:
            is_ref = frame['frame_path'] in ref_set
            order = ref_order if is_ref else None
            if is_ref:
                ref_order += 1
            scene_index = extract_scene_index_from_path(frame['frame_path'])

            await self.db.execute("""
                INSERT INTO cluster_frames (cluster_id, frame_path, quality_score, expression, is_reference, reference_order, scene_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [cluster_id, frame['frame_path'], frame['quality_score'], frame['expression'],
                  1 if is_ref else 0, order, scene_index])

        await self.db.commit()

        # Update JSON files
        clusters = await self.get_clusters(video_id)
        await self._update_clusters_json(output_dir, clusters)
        await self._update_clustering_result_json(output_dir, clusters)

        return {
            'cluster_id': cluster_id,
            'cluster_index': new_cluster_index,
            'num_frames': len(frames_with_quality),
            'label': label,
            'representative_frame': representative_frame
        }

    async def add_frames_to_cluster(
        self,
        video_id: int,
        cluster_index: int,
        frame_paths: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Add frames to an existing cluster.

        Args:
            video_id: Video ID
            cluster_index: Cluster index to add frames to
            frame_paths: List of frame paths to add

        Returns:
            {'added': int, 'skipped': int, 'errors': [...]}
        """
        cluster = await self.get_cluster_by_index(video_id, cluster_index)
        if not cluster:
            return None

        video = await self.get_video(video_id)
        output_dir = self._get_video_output_dir(video)
        cluster_id = cluster['id']

        # Load faces.json for quality/expression data
        faces_data = {}
        faces_file = output_dir / "faces.json"
        if faces_file.exists():
            try:
                with open(faces_file, 'r', encoding='utf-8') as f:
                    faces_json = json.load(f)
                    for face in faces_json.get('all_faces', []):
                        fp = face.get('frame_path', '')
                        if fp not in faces_data or face.get('quality_score', 0) > faces_data[fp].get('quality_score', 0):
                            faces_data[fp] = {
                                'quality_score': face.get('quality_score', 0),
                                'expression': face.get('expression', 'unknown')
                            }
            except Exception:
                pass

        # Get existing frame paths in this cluster
        existing_paths = await self._get_cluster_frame_paths(cluster_id)

        # Copy frames to cluster folder
        cluster_frames_dir = output_dir / "clusters" / f"cluster_{cluster_index}" / "frames"
        cluster_frames_dir.mkdir(parents=True, exist_ok=True)

        added = 0
        skipped = 0
        errors = []

        for fp in frame_paths:
            if fp in existing_paths:
                skipped += 1
                continue

            # Get quality/expression
            face_info = faces_data.get(fp, {})
            quality = face_info.get('quality_score', 50)
            expression = face_info.get('expression', 'unknown')

            # Copy file to cluster folder
            src = Path(fp)
            if src.exists():
                try:
                    shutil.copy2(src, cluster_frames_dir / src.name)
                except Exception as e:
                    errors.append(f"Error copying {fp}: {e}")
                    continue

            # Insert into database
            try:
                scene_index = extract_scene_index_from_path(fp)
                await self.db.execute("""
                    INSERT INTO cluster_frames (cluster_id, frame_path, quality_score, expression, is_reference, reference_order, scene_index)
                    VALUES (?, ?, ?, ?, 0, NULL, ?)
                """, [cluster_id, fp, quality, expression, scene_index])
                added += 1
            except Exception as e:
                errors.append(f"Error inserting {fp}: {e}")

        # Update cluster num_frames
        async with self.db.execute("""
            SELECT COUNT(*) FROM cluster_frames WHERE cluster_id = ?
        """, [cluster_id]) as cursor:
            row = await cursor.fetchone()
            new_count = row[0]

        await self.db.execute("""
            UPDATE clusters SET num_frames = ? WHERE id = ?
        """, [new_count, cluster_id])

        await self.db.commit()

        # Update JSON files
        clusters = await self.get_clusters(video_id)
        await self._update_clusters_json(output_dir, clusters)
        await self._update_clustering_result_json(output_dir, clusters)

        return {
            'added': added,
            'skipped': skipped,
            'errors': errors,
            'total_frames': new_count
        }

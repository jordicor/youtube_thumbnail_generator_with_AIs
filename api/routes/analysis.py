"""
Analysis API Routes

Endpoints for video analysis (scenes, faces, clustering).
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field

from database.db import get_db
from services.analysis_service import AnalysisService
from config import MAX_REFERENCE_FRAMES, OUTPUT_DIR
from job_queue.queue import enqueue_analysis
from i18n.i18n import translate as t


router = APIRouter()


# ============================================================================
# MODELS
# ============================================================================

class AnalysisRequest(BaseModel):
    force_scenes: bool = False
    force_faces: bool = False
    force_clustering: bool = False
    force_transcription: bool = False
    clustering_eps: float = 0.5
    clustering_min_samples: int = 3


class ClusterResponse(BaseModel):
    cluster_index: int
    num_frames: int
    representative_frame: str
    label: Optional[str] = None


class MergeClustersRequest(BaseModel):
    cluster_indices: List[int]  # List of cluster indices to merge
    target_index: int  # Which cluster to keep as the main one
    view_mode: str = Field(default='person', pattern='^(person|person_scene)$')


class UpdateReferencesRequest(BaseModel):
    frame_ids: List[int]  # List of frame IDs to set as references (in order)


class AddReferencesRequest(BaseModel):
    frame_ids: List[int]  # List of frame IDs to add as references


class DeleteFramesRequest(BaseModel):
    frame_ids: List[int]  # List of frame IDs to delete


class DeleteDiskFramesRequest(BaseModel):
    filenames: List[str]  # List of filenames to delete from disk


class CreateClusterRequest(BaseModel):
    frame_paths: List[str]  # List of frame paths to include
    label: Optional[str] = None  # Optional cluster name
    description: Optional[str] = None  # Optional notes/comments
    reference_frame_paths: Optional[List[str]] = None  # Optional specific reference frames
    view_mode: str = Field(default='person', pattern='^(person|person_scene)$')


class UpdateClusterRequest(BaseModel):
    label: Optional[str] = None  # New cluster name (None = don't change)
    description: Optional[str] = None  # New description (None = don't change)


class AddFramesToClusterRequest(BaseModel):
    frame_paths: List[str]  # List of frame paths to add


class AnalysisStatusResponse(BaseModel):
    video_id: int
    status: str
    progress: int
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    clusters: Optional[int] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/{video_id}/start")
async def start_analysis(
    video_id: int,
    request: AnalysisRequest = AnalysisRequest()
):
    """
    Start video analysis (scenes + faces + clustering).
    Analysis runs in background via Redis job queue.
    """
    async with get_db() as db:
        service = AnalysisService(db)

        # Verify video exists
        video = await service.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

        # Check if already analyzing (any analysis-related status)
        analysis_in_progress_states = {'analyzing', 'analyzing_scenes', 'analyzing_faces', 'clustering', 'transcribing'}
        if video['status'] in analysis_in_progress_states:
            raise HTTPException(status_code=400, detail=t('api.errors.analysis_in_progress'))

        # Update status
        await service.update_video_status(video_id, 'analyzing')

    # Enqueue analysis job to Redis
    job_id = await enqueue_analysis(
        video_id=video_id,
        force_scenes=request.force_scenes,
        force_faces=request.force_faces,
        force_clustering=request.force_clustering,
        force_transcription=request.force_transcription,
        clustering_eps=request.clustering_eps,
        clustering_min_samples=request.clustering_min_samples
    )

    if not job_id:
        # Failed to enqueue - revert status
        async with get_db() as db:
            service = AnalysisService(db)
            await service.update_video_status(video_id, 'pending', t('api.errors.failed_enqueue_analysis'))
        raise HTTPException(status_code=500, detail=t('api.errors.failed_enqueue_analysis'))

    return {
        "video_id": video_id,
        "status": "analyzing",
        "message": t('api.messages.analysis_started'),
        "job_id": job_id
    }


@router.get("/{video_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(video_id: int):
    """
    Get current analysis status.
    """
    async with get_db() as db:
        service = AnalysisService(db)
        status = await service.get_analysis_status(video_id)

    if not status:
        raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

    return AnalysisStatusResponse(**status)


@router.get("/{video_id}/clusters")
async def get_clusters(
    video_id: int,
    view_mode: str = Query(default="person", pattern="^(person|person_scene)$")
):
    """
    Get detected clusters for a video, filtered by view mode.

    Args:
        video_id: The video ID
        view_mode: View mode - 'person' (unified by face) or 'person_scene' (split by scene)

    Returns:
        List of clusters with frame counts.
    """
    async with get_db() as db:
        service = AnalysisService(db)
        clusters = await service.get_clusters(video_id, view_mode=view_mode)

    return {"video_id": video_id, "clusters": clusters, "view_mode": view_mode}


@router.get("/{video_id}/clusters/{cluster_index}/frames")
async def get_cluster_frames(
    video_id: int,
    cluster_index: int,
    limit: int = 20,
    view_mode: str = Query(default="person", pattern="^(person|person_scene)$")
):
    """
    Get frames for a specific cluster.
    """
    async with get_db() as db:
        service = AnalysisService(db)
        frames = await service.get_cluster_frames(video_id, cluster_index, limit, view_mode=view_mode)

    return {"cluster_index": cluster_index, "frames": frames, "view_mode": view_mode}


@router.get("/{video_id}/clusters/{cluster_index}/image")
async def get_cluster_representative_image(
    video_id: int,
    cluster_index: int,
    view_mode: str = Query(default="person", pattern="^(person|person_scene)$")
):
    """
    Get representative image for a cluster.
    """
    async with get_db() as db:
        service = AnalysisService(db)
        image_path = await service.get_cluster_representative(video_id, cluster_index, view_mode=view_mode)

    if not image_path:
        raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))

    if not Path(image_path).exists():
        raise HTTPException(status_code=404, detail=t('api.errors.image_not_found'))

    return FileResponse(image_path, media_type="image/jpeg")


@router.get("/{video_id}/clusters/by-id/{cluster_id}/image")
async def get_cluster_image_by_id(video_id: int, cluster_id: int):
    """
    Get representative image for a cluster using its database ID.

    This endpoint uses the immutable cluster ID instead of the mutable index,
    making it safe for browser caching even after cluster reindexing operations.
    """
    async with get_db() as db:
        service = AnalysisService(db)
        image_path = await service.get_cluster_representative_by_id(cluster_id)

    if not image_path:
        raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))

    if not Path(image_path).exists():
        raise HTTPException(status_code=404, detail=t('api.errors.image_not_found'))

    return FileResponse(image_path, media_type="image/jpeg")


@router.delete("/{video_id}/clusters/{cluster_index}")
async def delete_cluster(
    video_id: int,
    cluster_index: int,
    view_mode: str = Query(default="person", pattern="^(person|person_scene)$")
):
    """
    Delete a cluster and its associated frames.
    Remaining clusters will be reindexed to maintain consecutive indices.
    """
    async with get_db() as db:
        service = AnalysisService(db)

        # Verify video exists
        video = await service.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

        # Delete the cluster
        success = await service.delete_cluster(video_id, cluster_index, view_mode=view_mode)

        if not success:
            raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))

    return {"message": t('api.messages.cluster_deleted'), "video_id": video_id, "view_mode": view_mode}


@router.post("/{video_id}/clusters/merge")
async def merge_clusters(video_id: int, request: MergeClustersRequest):
    """
    Merge multiple clusters into one.

    The frames from all clusters will be combined into the target cluster.
    Other clusters will be deleted and indices will be renumbered.
    """
    async with get_db() as db:
        service = AnalysisService(db)

        # Verify video exists
        video = await service.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

        # Validate request
        if len(request.cluster_indices) < 2:
            raise HTTPException(
                status_code=400,
                detail=t('api.errors.min_clusters_required')
            )

        # Merge the clusters
        result = await service.merge_clusters(
            video_id,
            request.cluster_indices,
            request.target_index,
            view_mode=request.view_mode
        )

        if not result:
            raise HTTPException(
                status_code=400,
                detail=t('api.errors.failed_merge_clusters')
            )

        # Get updated clusters for the same view_mode
        clusters = await service.get_clusters(video_id, view_mode=request.view_mode)

    return {
        "message": t('api.messages.clusters_merged'),
        "video_id": video_id,
        "clusters": clusters,
        "view_mode": request.view_mode
    }


# ============================================================================
# FRAME MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/{video_id}/clusters/{cluster_index}/frames/all")
async def get_all_cluster_frames(
    video_id: int,
    cluster_index: int,
    view_mode: str = Query(default="person", pattern="^(person|person_scene)$")
):
    """
    Get all frames for a cluster, split into references and library.

    Returns:
        - reference_frames: Frames marked as AI references (ordered)
        - library_frames: All other frames (ordered by quality)
        - total_frames: Total count
        - reference_count: Number of references
        - is_custom_selection: True if user has customized references
    """
    async with get_db() as db:
        service = AnalysisService(db)

        result = await service.get_all_cluster_frames(video_id, cluster_index, view_mode=view_mode)

        if result is None:
            raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))

    return result


@router.get("/{video_id}/clusters/{cluster_index}/frames/{frame_id}/image")
async def get_frame_image(video_id: int, cluster_index: int, frame_id: int):
    """
    Get the image for a specific frame.

    V2 Architecture: frame_id refers to video_frames.id
    """
    async with get_db() as db:
        # frame_id is video_frames.id
        query = "SELECT frame_path FROM video_frames WHERE id = ?"
        async with db.execute(query, [frame_id]) as cursor:
            row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=t('api.errors.frame_not_found'))
            frame_path = row[0]

    frame_path_resolved = Path(frame_path).resolve()
    output_dir_resolved = Path(OUTPUT_DIR).resolve()

    # Security: validate path is within OUTPUT_DIR
    if not frame_path_resolved.is_relative_to(output_dir_resolved):
        raise HTTPException(status_code=400, detail=t('api.errors.invalid_file_path'))

    if not frame_path_resolved.exists():
        raise HTTPException(status_code=404, detail=t('api.errors.frame_not_found'))

    return FileResponse(str(frame_path_resolved), media_type="image/jpeg")


@router.put("/{video_id}/clusters/{cluster_index}/frames/references")
async def update_reference_frames(
    video_id: int,
    cluster_index: int,
    request: UpdateReferencesRequest,
    view_mode: str = Query(default="person", pattern="^(person|person_scene)$")
):
    """
    Update which frames are marked as references for AI generation.
    Replaces all current references with the provided list.
    Maximum defined by MAX_REFERENCE_FRAMES in config.
    """
    async with get_db() as db:
        service = AnalysisService(db)

        if len(request.frame_ids) > MAX_REFERENCE_FRAMES:
            raise HTTPException(
                status_code=400,
                detail=t('api.errors.max_references', max=MAX_REFERENCE_FRAMES)
            )

        success = await service.update_reference_frames(
            video_id, cluster_index, request.frame_ids, view_mode=view_mode
        )

        if not success:
            raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))

    return {"message": t('api.messages.references_updated'), "count": len(request.frame_ids)}


@router.post("/{video_id}/clusters/{cluster_index}/frames/references/add")
async def add_frames_to_references(
    video_id: int,
    cluster_index: int,
    request: AddReferencesRequest,
    view_mode: str = Query(default="person", pattern="^(person|person_scene)$")
):
    """
    Add frames to references (up to max defined by MAX_REFERENCE_FRAMES).
    """
    async with get_db() as db:
        service = AnalysisService(db)

        result = await service.add_frames_to_references(
            video_id, cluster_index, request.frame_ids, view_mode=view_mode
        )

        if result is None:
            raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))

    return {
        "message": t('api.messages.frames_added_refs', count=result['added']),
        "added": result['added'],
        "skipped": result['skipped'],
        "total": result['total']
    }


@router.delete("/{video_id}/clusters/{cluster_index}/frames/references/{frame_id}")
async def remove_frame_from_references(
    video_id: int,
    cluster_index: int,
    frame_id: int,
    view_mode: str = Query(default="person", pattern="^(person|person_scene)$")
):
    """
    Remove a single frame from references.
    """
    async with get_db() as db:
        service = AnalysisService(db)

        success = await service.remove_frame_from_references(
            video_id, cluster_index, frame_id, view_mode=view_mode
        )

        if not success:
            raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))

    return {"message": t('api.messages.frame_removed_refs')}


@router.post("/{video_id}/clusters/{cluster_index}/frames/references/reset")
async def reset_reference_frames(
    video_id: int,
    cluster_index: int,
    view_mode: str = Query(default="person", pattern="^(person|person_scene)$")
):
    """
    Reset reference frames to the top N by quality score (N = MAX_REFERENCE_FRAMES).
    Discards any custom selection.
    """
    async with get_db() as db:
        service = AnalysisService(db)

        success = await service.reset_reference_frames(video_id, cluster_index, view_mode=view_mode)

        if not success:
            raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))

    return {"message": t('api.messages.references_reset')}


@router.delete("/{video_id}/clusters/{cluster_index}/frames")
async def delete_cluster_frames(
    video_id: int,
    cluster_index: int,
    request: DeleteFramesRequest,
    view_mode: str = Query(default="person", pattern="^(person|person_scene)$")
):
    """
    Delete frames from a cluster permanently.
    This removes both the database records and physical image files.
    """
    async with get_db() as db:
        service = AnalysisService(db)

        result = await service.delete_cluster_frames(
            video_id, cluster_index, request.frame_ids, view_mode=view_mode
        )

        if result is None:
            raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))

    return {
        "message": t('api.messages.frames_deleted', count=result['deleted']),
        "deleted": result['deleted'],
        "errors": result['errors'],
        "remaining_frames": result['remaining_frames']
    }


# ============================================================================
# DISK-BASED FRAME ENDPOINTS (File Explorer)
# ============================================================================

@router.get("/{video_id}/clusters/{cluster_index}/frames/disk")
async def get_cluster_frames_from_disk(video_id: int, cluster_index: int):
    """
    List ALL frames for a cluster by reading directly from disk.
    Returns files from clusters/cluster_X/frames/ directory.

    This is the file explorer view that shows all physical frame files,
    not limited to what's stored in the database.
    """
    async with get_db() as db:
        service = AnalysisService(db)
        result = await service.get_all_cluster_frames_from_disk(video_id, cluster_index)

        if result is None:
            raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))

    return result


@router.get("/{video_id}/clusters/{cluster_index}/frames/disk/{filename}")
async def get_cluster_frame_image_from_disk(
    video_id: int,
    cluster_index: int,
    filename: str
):
    """
    Serve a specific frame image from disk.
    Used by the file explorer UI to display images.
    """
    # Security: prevent path traversal - validate filename has no path components
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail=t('api.errors.invalid_filename'))

    async with get_db() as db:
        service = AnalysisService(db)
        frame_path = await service.get_cluster_frame_image_path(
            video_id, cluster_index, filename
        )

        if frame_path is None:
            raise HTTPException(status_code=404, detail=t('api.errors.frame_not_found'))

    # Security: verify resolved path is within OUTPUT_DIR
    frame_path_resolved = Path(frame_path).resolve()
    output_dir_resolved = Path(OUTPUT_DIR).resolve()

    if not frame_path_resolved.is_relative_to(output_dir_resolved):
        raise HTTPException(status_code=400, detail=t('api.errors.invalid_file_path'))

    return FileResponse(str(frame_path_resolved), media_type="image/jpeg")


@router.delete("/{video_id}/clusters/{cluster_index}/frames/disk")
async def delete_cluster_frames_from_disk(
    video_id: int,
    cluster_index: int,
    request: DeleteDiskFramesRequest
):
    """
    Delete frame files directly from disk (file explorer mode).
    Removes files from clusters/cluster_X/frames/ directory.

    This is a simpler operation that only affects physical files,
    not the database records used for AI reference selection.
    """
    async with get_db() as db:
        service = AnalysisService(db)

        result = await service.delete_cluster_frames_from_disk(
            video_id, cluster_index, request.filenames
        )

        if result is None:
            raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))

    return {
        "message": t('api.messages.files_deleted', count=result['deleted']),
        "deleted": result['deleted'],
        "errors": result['errors'],
        "remaining": result['remaining']
    }


# ============================================================================
# ALL VIDEO FRAMES (for manual cluster creation)
# ============================================================================

@router.get("/{video_id}/frames/all")
async def get_all_video_frames(
    video_id: int,
    include_assigned: bool = True
):
    """
    Get ALL frames from the video's frames/ directory.
    Used for manual cluster creation.

    Args:
        include_assigned: If True, include frames already assigned to clusters

    Returns:
        - frames: List of frame info (filename, path, quality, expression, cluster_id)
        - total: Total count
        - assigned_count: How many are already in clusters
    """
    async with get_db() as db:
        service = AnalysisService(db)

        result = await service.get_all_video_frames(video_id, include_assigned)

        if result is None:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

    return result


@router.get("/{video_id}/frames/{filename}/image")
async def get_video_frame_image(video_id: int, filename: str):
    """
    Serve a specific frame image from the video's frames/ directory.
    Used for displaying frames in the manual cluster creation UI.
    """
    # Security: prevent path traversal - validate filename has no path components
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail=t('api.errors.invalid_filename'))

    async with get_db() as db:
        service = AnalysisService(db)

        video = await service.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

        output_dir = service._get_video_output_dir(video)
        frame_path = (output_dir / "frames" / filename).resolve()
        output_dir_resolved = output_dir.resolve()

        # Security: verify final path is within video's output directory
        if not frame_path.is_relative_to(output_dir_resolved):
            raise HTTPException(status_code=400, detail=t('api.errors.invalid_file_path'))

        if not frame_path.exists():
            raise HTTPException(status_code=404, detail=t('api.errors.frame_not_found'))

    return FileResponse(str(frame_path), media_type="image/jpeg")


# ============================================================================
# MANUAL CLUSTER CREATION
# ============================================================================

@router.post("/{video_id}/clusters/create")
async def create_manual_cluster(
    video_id: int,
    request: CreateClusterRequest
):
    """
    Create a new cluster manually from selected frames.

    This allows users to create custom clusters without relying on
    automatic face detection/clustering.
    """
    async with get_db() as db:
        service = AnalysisService(db)

        # Verify video exists
        video = await service.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

        if not request.frame_paths:
            raise HTTPException(status_code=400, detail=t('api.errors.min_one_frame_required'))

        result = await service.create_manual_cluster(
            video_id,
            request.frame_paths,
            request.label,
            request.reference_frame_paths,
            request.description,
            view_mode=request.view_mode
        )

        if result is None:
            raise HTTPException(status_code=400, detail=t('api.errors.failed_create_cluster'))

    return {
        "message": t('api.messages.cluster_created'),
        "cluster": result,
        "view_mode": request.view_mode
    }


@router.patch("/{video_id}/clusters/{cluster_index}")
async def update_cluster(
    video_id: int,
    cluster_index: int,
    request: UpdateClusterRequest,
    view_mode: str = Query(default="person", pattern="^(person|person_scene)$")
):
    """
    Update cluster label and/or description.
    """
    async with get_db() as db:
        service = AnalysisService(db)

        # Verify video exists
        video = await service.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

        success = await service.update_cluster_info(
            video_id,
            cluster_index,
            request.label,
            request.description,
            view_mode=view_mode
        )

        if not success:
            raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))

    return {"message": t('api.messages.cluster_updated')}


@router.post("/{video_id}/clusters/{cluster_index}/frames/add")
async def add_frames_to_cluster(
    video_id: int,
    cluster_index: int,
    request: AddFramesToClusterRequest,
    view_mode: str = Query(default="person", pattern="^(person|person_scene)$")
):
    """
    Add frames to an existing cluster.

    Frames will be copied to the cluster's folder and added to the database.
    They will NOT be marked as references by default.
    """
    async with get_db() as db:
        service = AnalysisService(db)

        result = await service.add_frames_to_cluster(
            video_id, cluster_index, request.frame_paths, view_mode=view_mode
        )

        if result is None:
            raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))

    return {
        "message": t('api.messages.frames_added_cluster', count=result['added']),
        "added": result['added'],
        "skipped": result['skipped'],
        "errors": result['errors'],
        "total_frames": result['total_frames']
    }

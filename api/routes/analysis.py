"""
Analysis API Routes

Endpoints for video analysis (scenes, faces, clustering).
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import ORJSONResponse
from typing import Optional, List
from pydantic import BaseModel, Field, model_validator

from database.db import get_db
from services.analysis_service import AnalysisService
from services.directory_service import DirectoryService
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
    refresh_references: bool = False
    clustering_eps: float = 0.5
    clustering_min_samples: int = 3

    @model_validator(mode="after")
    def validate_refresh_scope(self):
        if self.refresh_references and any((self.force_scenes, self.force_faces,
                                            self.force_clustering, self.force_transcription)):
            raise ValueError("Reference refresh cannot force other analysis stages.")
        return self


class MergeClustersRequest(BaseModel):
    cluster_indices: List[int]  # List of cluster indices to merge
    target_index: int  # Which cluster to keep as the main one
    view_mode: str = Field(default='person', pattern='^(person|person_scene)$')


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

        # The analysis reads the video file: its directory must answer here
        await DirectoryService(db).switch_video_directory(video_id)

        # Atomic status update to prevent TOCTOU race
        async with db.execute(
            """UPDATE videos SET status = 'analyzing', error_message = NULL
               WHERE id = ? AND status NOT IN ('analyzing', 'analyzing_scenes', 'analyzing_faces', 'clustering', 'transcribing')""",
            [video_id]
        ) as cursor:
            if cursor.rowcount == 0:
                raise HTTPException(status_code=400, detail=t('api.errors.analysis_in_progress'))
        await db.commit()

    # Enqueue analysis job to Redis
    job_id = await enqueue_analysis(
        video_id=video_id,
        force_scenes=request.force_scenes,
        force_faces=request.force_faces,
        force_clustering=request.force_clustering,
        force_transcription=request.force_transcription,
        refresh_references=request.refresh_references,
        expected_reference_revision=video["reference_revision"] if request.refresh_references else None,
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

    # Serialized by orjson directly: FastAPI's jsonable_encoder walks every value
    # of a plain dict, a few hundred clusters take about 10 ms
    return ORJSONResponse({"video_id": video_id, "clusters": clusters, "view_mode": view_mode})


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

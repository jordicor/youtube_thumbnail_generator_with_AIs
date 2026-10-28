"""
Videos API Routes

Endpoints for video management.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

from database.db import get_db
from services.video_service import VideoService
from services.analysis_service import AnalysisService
from services.directory_service import DirectoryService
from job_queue.queue import enqueue_analysis
from i18n.i18n import translate as t


router = APIRouter()


# ============================================================================
# MODELS
# ============================================================================

class VideoResponse(BaseModel):
    id: int
    filename: str
    filepath: str
    duration_seconds: Optional[float] = None
    status: str
    error_message: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True


class VideoListResponse(BaseModel):
    videos: List[VideoResponse]
    total: int


class BulkHideRequest(BaseModel):
    video_ids: List[int]


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("", response_model=VideoListResponse)
async def list_videos(
    status: Optional[str] = None,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500)
):
    """
    List all registered videos.

    - **status**: Filter by status (pending, analyzing, analyzed, etc.)
    - **skip**: Offset for pagination
    - **limit**: Maximum number of results
    """
    async with get_db() as db:
        service = VideoService(db)
        videos = await service.list_videos(status=status, skip=skip, limit=limit)
        total = await service.count_videos(status=status)

    # Convert to response model
    video_responses = []
    for v in videos:
        video_responses.append(VideoResponse(
            id=v['id'],
            filename=v['filename'],
            filepath=v['filepath'],
            duration_seconds=v.get('duration_seconds'),
            status=v['status'],
            error_message=v.get('error_message'),
            created_at=str(v['created_at'])
        ))

    return VideoListResponse(videos=video_responses, total=total)


@router.patch("/bulk/hide")
async def bulk_hide_videos(request: BulkHideRequest):
    """Hide multiple videos at once."""
    async with get_db() as db:
        service = VideoService(db)
        count = await service.bulk_hide_videos(request.video_ids)

    return {"hidden": True, "count": count}


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(video_id: int):
    """Get information about a specific video."""
    async with get_db() as db:
        service = VideoService(db)
        video = await service.get_video(video_id)

    if not video:
        raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

    return VideoResponse(
        id=video['id'],
        filename=video['filename'],
        filepath=video['filepath'],
        duration_seconds=video.get('duration_seconds'),
        status=video['status'],
        error_message=video.get('error_message'),
        created_at=str(video['created_at'])
    )


@router.delete("/{video_id}")
async def delete_video(video_id: int):
    """Delete a video and all its associated data."""
    async with get_db() as db:
        service = VideoService(db)
        success = await service.delete_video(video_id)

    if not success:
        raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

    return {"deleted": True, "video_id": video_id}


@router.get("/{video_id}/thumbnail")
async def get_video_thumbnail(video_id: int):
    """
    Get a video preview frame.
    Extracts a frame from the video.
    """
    async with get_db() as db:
        service = VideoService(db)
        thumbnail_path = await service.get_video_thumbnail(video_id)

    if not thumbnail_path or not Path(thumbnail_path).exists():
        raise HTTPException(status_code=404, detail=t('api.errors.thumbnail_not_available'))

    return FileResponse(thumbnail_path, media_type="image/jpeg")


@router.patch("/{video_id}/hide")
async def hide_video(video_id: int):
    """Hide a video from normal lists."""
    async with get_db() as db:
        service = VideoService(db)
        success = await service.hide_video(video_id)

    if not success:
        raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

    return {"hidden": True, "video_id": video_id}


@router.patch("/{video_id}/show")
async def show_video(video_id: int):
    """Show a hidden video again."""
    async with get_db() as db:
        service = VideoService(db)
        success = await service.show_video(video_id)

    if not success:
        raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

    return {"hidden": False, "video_id": video_id}


@router.post("/{video_id}/reanalyze")
async def reanalyze_video(video_id: int):
    """
    Delete all analysis data and restart analysis.
    Deletes: clusters, frames, jobs, thumbnails.
    Keeps: video record, preview image.
    """
    async with get_db() as db:
        # The analysis reads the video file: its directory must answer here
        await DirectoryService(db).switch_video_directory(video_id)

        service = VideoService(db)
        success = await service.reanalyze_video(video_id)

        if not success:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

        # Update status to analyzing
        analysis_service = AnalysisService(db)
        await analysis_service.update_video_status(video_id, 'analyzing')

    # Enqueue analysis job to Redis (force all steps since this is a reanalysis)
    job_id = await enqueue_analysis(
        video_id=video_id,
        force_scenes=True,
        force_faces=True,
        force_clustering=True,
        force_transcription=True
    )

    if not job_id:
        # Failed to enqueue - revert status
        async with get_db() as db:
            analysis_service = AnalysisService(db)
            await analysis_service.update_video_status(video_id, 'pending', t('api.errors.failed_enqueue_analysis'))
        raise HTTPException(status_code=500, detail=t('api.errors.failed_enqueue_analysis'))

    return {"reanalyzing": True, "video_id": video_id, "job_id": job_id}



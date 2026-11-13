"""
Videos API Routes

Endpoints for video management.
"""

import asyncio
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

from api.dependencies import video_taken_error
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
    """Delete a video's record and everything the database holds of it (409 while it has a task)."""
    async with get_db() as db:
        video = await AnalysisService(db).get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))
        # A task in the worker is reading and writing this video
        if video['analysis_run'] is not None or video['status'] == 'generating':
            raise video_taken_error(video['status'])

        await VideoService(db).delete_video(video_id)

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
    Analyze a video again from scratch.

    Its people (with their names, notes and photos), frames, generation jobs
    and thumbnails are deleted, by the worker and only once the reanalysis
    starts. Nothing is deleted when it cannot be made again: the video file is
    not found (409), the video has a task running (409) or the job cannot be
    queued (500).
    """
    async with get_db() as db:
        # The analysis reads the video file: its directory must answer here
        await DirectoryService(db).switch_video_directory(video_id)

        service = AnalysisService(db)
        video = await service.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

        # A video the owner moved away keeps what it has: it cannot be analyzed again
        if not await asyncio.to_thread(os.path.isfile, video['filepath']):
            raise HTTPException(status_code=409, detail=t('api.errors.video_file_missing'))

        analysis_run = await service.claim_analysis(video_id)
        if analysis_run is None:
            raise video_taken_error((await service.get_video(video_id))['status'])

    job_id = await enqueue_analysis(video_id=video_id, analysis_run=analysis_run, reset=True)

    if not job_id:
        # Nothing was queued or deleted: the video goes back as it was
        async with get_db() as db:
            await AnalysisService(db).unclaim_analysis(video_id, analysis_run)
        raise HTTPException(status_code=500, detail=t('api.errors.failed_enqueue_analysis'))

    return {"reanalyzing": True, "video_id": video_id, "job_id": job_id}



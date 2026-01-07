"""
Videos API Routes

Endpoints for video management.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

from database.db import get_db
from services.video_service import VideoService
from services.analysis_service import AnalysisService
from job_queue.queue import enqueue_analysis
from i18n.i18n import translate as t


router = APIRouter()


# ============================================================================
# VIDEO MIME TYPE VALIDATION
# ============================================================================

# Magic bytes (file signatures) for common video formats
VIDEO_SIGNATURES = {
    # MP4/M4V (ftyp box)
    b'\x00\x00\x00\x14ftyp': 'video/mp4',
    b'\x00\x00\x00\x18ftyp': 'video/mp4',
    b'\x00\x00\x00\x1cftyp': 'video/mp4',
    b'\x00\x00\x00\x20ftyp': 'video/mp4',
    # AVI (RIFF....AVI)
    b'RIFF': 'video/x-msvideo',
    # MKV/WebM (EBML header)
    b'\x1a\x45\xdf\xa3': 'video/x-matroska',
    # MOV (ftyp qt or moov)
    b'moov': 'video/quicktime',
    b'mdat': 'video/quicktime',
    b'free': 'video/quicktime',
}

# Extended check for MP4/MOV variants (check for 'ftyp' anywhere in first 12 bytes)
MP4_BRANDS = [b'isom', b'iso2', b'mp41', b'mp42', b'M4V ', b'M4A ', b'qt  ', b'avc1']


def validate_video_content(content: bytes) -> tuple[bool, str]:
    """
    Validate that file content matches a known video format.
    Returns (is_valid, detected_type_or_error).
    """
    if len(content) < 12:
        return False, "File too small to be a valid video"

    header = content[:32]

    # Check for EBML (MKV/WebM)
    if header[:4] == b'\x1a\x45\xdf\xa3':
        return True, "video/x-matroska"

    # Check for RIFF (AVI)
    if header[:4] == b'RIFF' and len(header) >= 12 and header[8:12] == b'AVI ':
        return True, "video/x-msvideo"

    # Check for MP4/MOV/M4V (ftyp box)
    # ftyp can be at offset 4 with various box sizes
    for offset in [4, 0]:
        if header[offset:offset+4] == b'ftyp':
            # Found ftyp, check brand
            brand = header[offset+4:offset+8]
            if brand in MP4_BRANDS or brand.startswith(b'mp4') or brand.startswith(b'M4'):
                return True, "video/mp4"
            if brand == b'qt  ' or brand.startswith(b'qt'):
                return True, "video/quicktime"
            # Unknown brand but has ftyp - likely still valid video
            return True, "video/mp4"

    # Check for moov/mdat atoms (MOV without ftyp)
    if header[:4] in [b'moov', b'mdat', b'free', b'wide', b'skip']:
        return True, "video/quicktime"

    # Check if it looks like ftyp with different box size
    if b'ftyp' in header[:16]:
        return True, "video/mp4"

    return False, "Unknown file format - not a recognized video type"


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
    skip: int = 0,
    limit: int = 50
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


@router.get("/scan")
async def scan_videos_directory():
    """
    Scan the videos directory and register new videos found.
    """
    async with get_db() as db:
        service = VideoService(db)
        result = await service.scan_videos_directory()

    return result


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


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a new video.
    Validates both file extension and content (magic bytes).
    """
    # Validate extension
    allowed_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v'}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=t('api.errors.invalid_file_extension', extensions=', '.join(sorted(allowed_extensions)))
        )

    # Read first bytes to validate content type (magic bytes)
    first_bytes = await file.read(4096)
    is_valid, result = validate_video_content(first_bytes)

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=t('api.errors.invalid_file_content', error=result, ext=ext)
        )

    # Reset file position for actual upload
    await file.seek(0)

    async with get_db() as db:
        service = VideoService(db)
        video = await service.upload_video(file)

    return {"id": video['id'], "filename": video['filename'], "status": video['status']}


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



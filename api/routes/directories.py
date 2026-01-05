"""
Directories API Routes

Endpoints for multi-directory management.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel

from database.db import get_db
from services.directory_service import DirectoryService


router = APIRouter()


# ============================================================================
# MODELS
# ============================================================================

class DirectoryCreate(BaseModel):
    path: str
    name: Optional[str] = None


class DirectoryUpdate(BaseModel):
    name: str


class DirectoryResponse(BaseModel):
    id: int
    path: str
    name: Optional[str] = None
    last_scanned_at: Optional[str] = None
    video_count: int = 0
    hidden_count: int = 0
    time_ago: str = "Nunca"
    created_at: str

    class Config:
        from_attributes = True


class DirectoryListResponse(BaseModel):
    directories: List[DirectoryResponse]
    total: int


class VideoInDirectoryResponse(BaseModel):
    id: int
    filename: str
    filepath: str
    duration_seconds: Optional[float] = None
    status: str
    error_message: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True


class DirectoryVideosResponse(BaseModel):
    directory: DirectoryResponse
    videos: List[VideoInDirectoryResponse]
    total: int


class ScanResultResponse(BaseModel):
    scanned: int
    new: int
    existing: int
    total: int


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("", response_model=DirectoryListResponse)
async def list_directories():
    """
    List all saved directories, ordered by most recently used.
    """
    async with get_db() as db:
        service = DirectoryService(db)
        directories = await service.list_directories()

        dir_responses = []
        for d in directories:
            hidden_count = await service.count_hidden_videos(d['id'])
            visible_count = await service.count_visible_videos(d['id'])
            dir_responses.append(DirectoryResponse(
                id=d['id'],
                path=d['path'],
                name=d.get('name'),
                last_scanned_at=str(d['last_scanned_at']) if d.get('last_scanned_at') else None,
                video_count=visible_count,
                hidden_count=hidden_count,
                time_ago=d.get('time_ago', 'Nunca'),
                created_at=str(d['created_at'])
            ))

    return DirectoryListResponse(directories=dir_responses, total=len(dir_responses))


@router.post("", response_model=DirectoryResponse)
async def add_directory(data: DirectoryCreate):
    """
    Add a new directory to monitor.

    - **path**: Full path to the directory containing videos
    - **name**: Optional friendly name/alias for the directory
    """
    async with get_db() as db:
        service = DirectoryService(db)
        try:
            directory = await service.add_directory(data.path, data.name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        hidden_count = await service.count_hidden_videos(directory['id'])
        visible_count = await service.count_visible_videos(directory['id'])

    return DirectoryResponse(
        id=directory['id'],
        path=directory['path'],
        name=directory.get('name'),
        last_scanned_at=str(directory['last_scanned_at']) if directory.get('last_scanned_at') else None,
        video_count=visible_count,
        hidden_count=hidden_count,
        time_ago=directory.get('time_ago', 'Nunca'),
        created_at=str(directory['created_at'])
    )


@router.get("/{directory_id}", response_model=DirectoryResponse)
async def get_directory(directory_id: int):
    """Get information about a specific directory."""
    async with get_db() as db:
        service = DirectoryService(db)
        directory = await service.get_directory(directory_id)

        if not directory:
            raise HTTPException(status_code=404, detail="Directory not found")

        hidden_count = await service.count_hidden_videos(directory_id)
        visible_count = await service.count_visible_videos(directory_id)

    return DirectoryResponse(
        id=directory['id'],
        path=directory['path'],
        name=directory.get('name'),
        last_scanned_at=str(directory['last_scanned_at']) if directory.get('last_scanned_at') else None,
        video_count=visible_count,
        hidden_count=hidden_count,
        time_ago=directory.get('time_ago', 'Nunca'),
        created_at=str(directory['created_at'])
    )


@router.put("/{directory_id}", response_model=DirectoryResponse)
async def update_directory(directory_id: int, data: DirectoryUpdate):
    """Update a directory's name/alias."""
    async with get_db() as db:
        service = DirectoryService(db)
        directory = await service.update_directory(directory_id, data.name)

        if not directory:
            raise HTTPException(status_code=404, detail="Directory not found")

        hidden_count = await service.count_hidden_videos(directory_id)
        visible_count = await service.count_visible_videos(directory_id)

    return DirectoryResponse(
        id=directory['id'],
        path=directory['path'],
        name=directory.get('name'),
        last_scanned_at=str(directory['last_scanned_at']) if directory.get('last_scanned_at') else None,
        video_count=visible_count,
        hidden_count=hidden_count,
        time_ago=directory.get('time_ago', 'Nunca'),
        created_at=str(directory['created_at'])
    )


@router.delete("/{directory_id}")
async def delete_directory(directory_id: int):
    """
    Delete a directory from the database.
    Videos from this directory will remain in the database but with no directory association.
    """
    async with get_db() as db:
        service = DirectoryService(db)
        success = await service.delete_directory(directory_id)

    if not success:
        raise HTTPException(status_code=404, detail="Directory not found")

    return {"deleted": True, "directory_id": directory_id}


@router.post("/{directory_id}/scan", response_model=ScanResultResponse)
async def scan_directory(directory_id: int):
    """
    Scan a directory for new videos.
    Only adds new videos, doesn't re-analyze existing ones.
    """
    async with get_db() as db:
        service = DirectoryService(db)
        try:
            result = await service.scan_directory(directory_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return ScanResultResponse(**result)


@router.get("/{directory_id}/videos", response_model=DirectoryVideosResponse)
async def get_directory_videos(
    directory_id: int,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 50
):
    """
    Get all videos in a directory.

    - **status**: Filter by status (pending, analyzing, analyzed, hidden, etc.)
    - **skip**: Offset for pagination
    - **limit**: Maximum number of results
    """
    async with get_db() as db:
        service = DirectoryService(db)
        directory = await service.get_directory(directory_id)

        if not directory:
            raise HTTPException(status_code=404, detail="Directory not found")

        videos = await service.get_videos(directory_id, status=status, skip=skip, limit=limit)
        total = await service.count_videos(directory_id, status=status)
        hidden_count = await service.count_hidden_videos(directory_id)
        visible_count = await service.count_visible_videos(directory_id)

    dir_response = DirectoryResponse(
        id=directory['id'],
        path=directory['path'],
        name=directory.get('name'),
        last_scanned_at=str(directory['last_scanned_at']) if directory.get('last_scanned_at') else None,
        video_count=visible_count,
        hidden_count=hidden_count,
        time_ago=directory.get('time_ago', 'Nunca'),
        created_at=str(directory['created_at'])
    )

    video_responses = []
    for v in videos:
        video_responses.append(VideoInDirectoryResponse(
            id=v['id'],
            filename=v['filename'],
            filepath=v['filepath'],
            duration_seconds=v.get('duration_seconds'),
            status=v['status'],
            error_message=v.get('error_message'),
            created_at=str(v['created_at'])
        ))

    return DirectoryVideosResponse(
        directory=dir_response,
        videos=video_responses,
        total=total
    )

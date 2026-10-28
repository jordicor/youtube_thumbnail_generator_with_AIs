"""
Directories API Routes

Endpoints for multi-directory management.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel

from database.db import get_db
from services.directory_service import DirectoryService, MissingVideosError, LIBRARY_FILTERS, LIBRARY_SORTS
from i18n.i18n import translate as t


router = APIRouter()


# ============================================================================
# MODELS
# ============================================================================

class DirectoryCreate(BaseModel):
    path: str
    name: Optional[str] = None


class DirectoryUpdate(BaseModel):
    name: str
    path: str
    alternative_paths: List[str]
    allow_missing_videos: bool = False


class DirectoryResponse(BaseModel):
    id: int
    path: str
    name: Optional[str] = None
    alternative_paths: List[str] = []
    switched_from: Optional[str] = None  # Path it left in this request because it did not answer
    last_scanned_at: Optional[str] = None
    video_count: int = 0
    hidden_count: int = 0
    time_ago: str = ""
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
    thumbnail_count: int
    person_count: int

    class Config:
        from_attributes = True


class LibraryCounts(BaseModel):
    """Videos per library filter (with the search applied)."""
    all: int
    pending: int
    processing: int
    analyzed: int
    completed: int
    error: int
    hidden: int


class DirectoryVideosResponse(BaseModel):
    directory: DirectoryResponse
    videos: List[VideoInDirectoryResponse]
    total: int
    counts: LibraryCounts


class ScanResultResponse(BaseModel):
    scanned: int
    new: int
    existing: int
    total: int


async def _directory_response(
    service: DirectoryService, directory: dict, switched_from: Optional[str] = None
) -> DirectoryResponse:
    return DirectoryResponse(
        id=directory['id'],
        path=directory['path'],
        name=directory.get('name'),
        alternative_paths=directory['alternative_paths'],
        switched_from=switched_from,
        last_scanned_at=str(directory['last_scanned_at']) if directory.get('last_scanned_at') else None,
        video_count=await service.count_visible_videos(directory['id']),
        hidden_count=await service.count_hidden_videos(directory['id']),
        time_ago=directory.get('time_ago', t('time_service.never')),
        created_at=str(directory['created_at'])
    )


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("", response_model=DirectoryListResponse)
async def list_directories():
    """
    List all saved directories, ordered by most recently used.

    A directory with alternative paths whose path does not answer first
    switches to the first alternative that does (switched_from says so).
    """
    async with get_db() as db:
        service = DirectoryService(db)
        switched = await service.switch_unanswering_directories()
        directories = await service.list_directories()

        dir_responses = [await _directory_response(service, d, switched.get(d['id'])) for d in directories]

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

        return await _directory_response(service, directory)


@router.get("/{directory_id}", response_model=DirectoryResponse)
async def get_directory(directory_id: int):
    """Get information about a specific directory."""
    async with get_db() as db:
        service = DirectoryService(db)
        directory = await service.get_directory(directory_id)

        if not directory:
            raise HTTPException(status_code=404, detail=t('api.errors.directory_not_found'))

        return await _directory_response(service, directory)


@router.put("/{directory_id}", response_model=DirectoryResponse)
async def update_directory(directory_id: int, data: DirectoryUpdate):
    """
    Update a directory's name/alias, path and alternative paths.

    - **name**: Friendly name/alias (blank: the folder name)
    - **path**: Path of the directory. A new one is for the same folder reached
      another way: the directory's videos keep their analysis and thumbnails
    - **alternative_paths**: The same folder from other machines, by preference.
      When the path does not answer, the directory switches to the first that does
    - **allow_missing_videos**: Change the path even if some videos are not in it

    Answers 409 with the missing videos (relative to the directory, visible
    ones first) and how many of them are hidden, when some are not in the new
    path and allow_missing_videos is false.
    """
    async with get_db() as db:
        service = DirectoryService(db)
        try:
            directory = await service.update_directory(
                directory_id, data.name, data.path, data.alternative_paths, data.allow_missing_videos
            )
        except MissingVideosError as e:
            raise HTTPException(
                status_code=409, detail={"missing": e.missing, "hidden": e.hidden, "total": e.total}
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not directory:
            raise HTTPException(status_code=404, detail=t('api.errors.directory_not_found'))

        return await _directory_response(service, directory)


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
        raise HTTPException(status_code=404, detail=t('api.errors.directory_not_found'))

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
    status: str = Query(default="all", pattern=f"^({'|'.join(LIBRARY_FILTERS)})$"),
    search: Optional[str] = Query(default=None, max_length=200),
    sort: str = Query(default="recent", pattern=f"^({'|'.join(LIBRARY_SORTS)})$"),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500)
):
    """
    Get one page of a directory's videos for the library.

    - **status**: Library filter: all, pending (incl. cancelled), processing,
      analyzed, completed (has thumbnails), error or hidden
    - **search**: Text the filename must contain
    - **sort**: recent, name, duration or activity
    - **skip**: Offset for pagination
    - **limit**: Maximum number of results

    The response counts the videos of every filter, with the search applied.
    """
    search = search.strip() if search else None

    async with get_db() as db:
        service = DirectoryService(db)
        directory = await service.get_directory(directory_id)

        if not directory:
            raise HTTPException(status_code=404, detail=t('api.errors.directory_not_found'))

        videos = await service.list_library_videos(
            directory_id, filter_name=status, search=search, sort=sort, skip=skip, limit=limit
        )
        counts = await service.count_library_videos(directory_id, search=search)
        dir_response = await _directory_response(service, directory)

    video_responses = []
    for v in videos:
        video_responses.append(VideoInDirectoryResponse(
            id=v['id'],
            filename=v['filename'],
            filepath=v['filepath'],
            duration_seconds=v.get('duration_seconds'),
            status=v['status'],
            error_message=v.get('error_message'),
            created_at=str(v['created_at']),
            thumbnail_count=v['thumbnail_count'],
            person_count=v['person_count']
        ))

    return DirectoryVideosResponse(
        directory=dir_response,
        videos=video_responses,
        total=counts[status],
        counts=LibraryCounts(**counts)
    )

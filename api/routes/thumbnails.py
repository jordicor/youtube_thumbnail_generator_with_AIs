"""
Thumbnails API Routes

Endpoints for serving and managing generated thumbnails.
"""

import io
import zipfile
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
from typing import Optional
from datetime import datetime

from database.db import get_db
from i18n.i18n import translate as t
from config import OUTPUT_DIR


router = APIRouter()


@router.get("/{thumbnail_id}")
async def get_thumbnail(thumbnail_id: int):
    """
    Get a specific thumbnail image.
    """
    async with get_db() as db:
        query = "SELECT filepath FROM thumbnails WHERE id = ?"
        async with db.execute(query, [thumbnail_id]) as cursor:
            row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=t('api.errors.thumbnail_not_found'))

    filepath = row[0]
    filepath_resolved = Path(filepath).resolve()
    output_dir_resolved = Path(OUTPUT_DIR).resolve()

    # Security: validate path is within OUTPUT_DIR
    if not filepath_resolved.is_relative_to(output_dir_resolved):
        raise HTTPException(status_code=400, detail=t('api.errors.invalid_file_path'))

    if not filepath_resolved.exists():
        raise HTTPException(status_code=404, detail=t('api.errors.thumbnail_file_not_found'))

    return FileResponse(str(filepath_resolved), media_type="image/png")


@router.get("/{thumbnail_id}/info")
async def get_thumbnail_info(thumbnail_id: int):
    """
    Get thumbnail metadata.
    """
    async with get_db() as db:
        query = """
            SELECT t.*, j.video_id
            FROM thumbnails t
            JOIN generation_jobs j ON t.job_id = j.id
            WHERE t.id = ?
        """
        async with db.execute(query, [thumbnail_id]) as cursor:
            row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=t('api.errors.thumbnail_not_found'))

    columns = [description[0] for description in cursor.description]
    thumbnail = dict(zip(columns, row))

    return thumbnail


@router.delete("/{thumbnail_id}")
async def delete_thumbnail(thumbnail_id: int):
    """
    Delete a specific thumbnail.
    """
    async with get_db() as db:
        # Get filepath first
        query = "SELECT filepath FROM thumbnails WHERE id = ?"
        async with db.execute(query, [thumbnail_id]) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=t('api.errors.thumbnail_not_found'))

        filepath = Path(row[0])

        # Delete from database
        await db.execute("DELETE FROM thumbnails WHERE id = ?", [thumbnail_id])
        await db.commit()

        # Delete file if exists
        if filepath.exists():
            filepath.unlink()

    return {"deleted": True, "thumbnail_id": thumbnail_id}


@router.get("/video/{video_id}")
async def get_video_thumbnails(video_id: int, limit: int = 50):
    """
    Get all thumbnails for a video.
    """
    async with get_db() as db:
        query = """
            SELECT t.id, t.filepath, t.image_index,
                   t.suggested_title, t.text_overlay, t.created_at
            FROM thumbnails t
            JOIN generation_jobs j ON t.job_id = j.id
            WHERE j.video_id = ?
            ORDER BY t.created_at DESC
            LIMIT ?
        """
        async with db.execute(query, [video_id, limit]) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            thumbnails = [dict(zip(columns, row)) for row in rows]

    return {"video_id": video_id, "thumbnails": thumbnails}


@router.get("/job/{job_id}/download-all")
async def download_all_thumbnails(job_id: int):
    """
    Download all thumbnails for a job as a ZIP file.
    """
    async with get_db() as db:
        # Get job info for filename
        query = """
            SELECT j.id, v.filename
            FROM generation_jobs j
            JOIN videos v ON j.video_id = v.id
            WHERE j.id = ?
        """
        async with db.execute(query, [job_id]) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=t('api.errors.job_not_found'))

        video_filename = row[1]

        # Get all thumbnails for this job
        query = """
            SELECT filepath, suggested_title, image_index
            FROM thumbnails
            WHERE job_id = ?
            ORDER BY image_index
        """
        async with db.execute(query, [job_id]) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=t('api.errors.no_thumbnails_job'))

    # Create ZIP in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filepath, title, image_idx in rows:
            file_path = Path(filepath)
            if file_path.exists():
                # Create a clean filename
                safe_title = (title or f"thumbnail_{image_idx}").replace('/', '_').replace('\\', '_')[:50]
                filename = f"{image_idx + 1:02d}_{safe_title}{file_path.suffix}"

                zip_file.write(file_path, filename)

    zip_buffer.seek(0)

    # Create download filename
    video_name = Path(video_filename).stem[:30]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_name = f"thumbnails_{video_name}_{timestamp}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}"'
        }
    )


@router.get("/video/{video_id}/download-all")
async def download_all_video_thumbnails(video_id: int):
    """
    Download all thumbnails for a video as a ZIP file.
    """
    async with get_db() as db:
        # Get video info for filename
        query = "SELECT filename FROM videos WHERE id = ?"
        async with db.execute(query, [video_id]) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))

        video_filename = row[0]

        # Get all thumbnails for this video
        query = """
            SELECT t.filepath, t.suggested_title, t.image_index, j.id as job_id
            FROM thumbnails t
            JOIN generation_jobs j ON t.job_id = j.id
            WHERE j.video_id = ?
            ORDER BY j.created_at DESC, t.image_index
        """
        async with db.execute(query, [video_id]) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=t('api.errors.no_thumbnails_video'))

    # Create ZIP in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filepath, title, image_idx, job_id in rows:
            file_path = Path(filepath)
            if file_path.exists():
                # Create a clean filename with job prefix
                safe_title = (title or f"thumbnail_{image_idx}").replace('/', '_').replace('\\', '_')[:50]
                filename = f"job{job_id}/{image_idx + 1:02d}_{safe_title}{file_path.suffix}"

                zip_file.write(file_path, filename)

    zip_buffer.seek(0)

    # Create download filename
    video_name = Path(video_filename).stem[:30]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_name = f"all_thumbnails_{video_name}_{timestamp}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}"'
        }
    )

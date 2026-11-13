"""
Thumbnails API Routes

Endpoints for serving and managing generated thumbnails.
"""

import asyncio
import io
import shutil
import zipfile
import orjson
import re
from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Tuple
from datetime import datetime
from urllib.parse import quote

from database.db import get_db
from i18n.i18n import translate as t
from config import FACE_MATCH_BAD_THRESHOLD, OUTPUT_DIR
from services.face_match import face_match_fields
from services.generation_service import STOPPING_JOB_SQL
from utils import resolve_output_path, resolve_row_paths


router = APIRouter()


# What Windows does not allow in a file name, and control characters
_ZIP_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _zip_title(title: Optional[str], image_index: int) -> str:
    """Part of a thumbnail's name in a ZIP that comes from its title.

    A valid file name on Windows too: titles carry '?', ':' and quotes, and a
    name cannot end with a dot or a space there.
    """
    name = _ZIP_FORBIDDEN.sub('_', title or '')[:50].rstrip('. ')
    return name or f"thumbnail_{image_index}"


def _provider_raw_path(thumbnail: Path) -> Path:
    """Where the image as the provider gave it is kept when KEEP_RAW_THUMBNAILS is on:
    thumbnails/raw/, with "_raw_" after the first part of its name (image_generation.py)."""
    prefix, separator, rest = thumbnail.stem.partition("_")
    name = f"{prefix}_raw_{rest}" if separator else f"raw_{thumbnail.stem}"
    return thumbnail.parent / "raw" / f"{name}{thumbnail.suffix}"


def _attachment(download_name: str) -> str:
    """Content-Disposition of a download whose name may have any character
    (video names do): an ASCII name for old clients and the real one in UTF-8."""
    fallback = ''.join(c if ' ' <= c <= '~' and c not in '"\\' else '_' for c in download_name)
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(download_name, safe='')}"


def _zip_thumbnails(files: List[Tuple[str, str]]) -> bytes:
    """ZIP of (stored thumbnail path, name in the ZIP without extension).

    Files outside OUTPUT_DIR or no longer on disk are left out. Blocking (it
    reads every file): run it in a thread. Files are stored, not deflated:
    PNG and JPEG are compressed already, and deflating them again takes about
    ten times longer for under 1% less.
    """
    zip_buffer = io.BytesIO()
    output_dir_resolved = Path(OUTPUT_DIR).resolve()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_STORED) as zip_file:
        for filepath, name in files:
            file_path = resolve_output_path(filepath).resolve()
            if file_path.is_relative_to(output_dir_resolved) and file_path.exists():
                zip_file.write(file_path, f"{name}{file_path.suffix}")
    return zip_buffer.getvalue()


def _small_copy(image_path: Path, width: int) -> bytes:
    """The image as a JPEG of that width, for the places that show it small."""
    from PIL import Image

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if image.width > width:
            image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
        data = io.BytesIO()
        image.save(data, "JPEG", quality=85)
    return data.getvalue()


@router.get("/{thumbnail_id}")
async def get_thumbnail(
    thumbnail_id: int,
    width: Annotated[Optional[int], Query(ge=64, le=1280)] = None,
    if_none_match: Annotated[Optional[str], Header()] = None,
):
    """
    Get a specific thumbnail image.

    With `width`, a JPEG copy of that width instead of the file: the gallery
    shows a dozen cards at a time and the files are PNGs of 1-3 MB each. The
    copy is made on each request and the browser keeps it while the file does
    not change (a face swap rewrites it): it revalidates with its ETag.
    """
    async with get_db() as db:
        query = "SELECT filepath FROM thumbnails WHERE id = ?"
        async with db.execute(query, [thumbnail_id]) as cursor:
            row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=t('api.errors.thumbnail_not_found'))

    filepath_resolved = resolve_output_path(row[0]).resolve()
    output_dir_resolved = Path(OUTPUT_DIR).resolve()

    # Security: validate path is within OUTPUT_DIR
    if not filepath_resolved.is_relative_to(output_dir_resolved):
        raise HTTPException(status_code=400, detail=t('api.errors.invalid_file_path'))

    if not filepath_resolved.exists():
        raise HTTPException(status_code=404, detail=t('api.errors.thumbnail_file_not_found'))

    if width is not None:
        stat = filepath_resolved.stat()
        headers = {"ETag": f'"{stat.st_mtime_ns:x}-{stat.st_size:x}-{width}"', "Cache-Control": "no-cache"}
        if if_none_match == headers["ETag"]:
            return Response(status_code=304, headers=headers)
        copy = await asyncio.to_thread(_small_copy, filepath_resolved, width)
        return Response(copy, media_type="image/jpeg", headers=headers)

    # The type comes from the file's extension: not every thumbnail is a PNG
    return FileResponse(str(filepath_resolved))


@router.get("/{thumbnail_id}/raw")
async def get_thumbnail_raw(thumbnail_id: int):
    """Serve the pre-face-swap version of a thumbnail (if one was saved).

    Returns 404 if no swap was ever applied (raw_path is NULL).
    Used by the gallery's before/after toggle.
    """
    async with get_db() as db:
        async with db.execute(
            "SELECT raw_path FROM thumbnails WHERE id = ?", [thumbnail_id]
        ) as cursor:
            row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=t('api.errors.thumbnail_not_found'))
    raw_path = row[0]
    if not raw_path:
        raise HTTPException(status_code=404, detail="No raw version available (no swap applied)")

    raw_resolved = resolve_output_path(raw_path).resolve()
    output_dir_resolved = Path(OUTPUT_DIR).resolve()
    if not raw_resolved.is_relative_to(output_dir_resolved):
        raise HTTPException(status_code=400, detail=t('api.errors.invalid_file_path'))
    if not raw_resolved.exists():
        raise HTTPException(status_code=404, detail="Raw thumbnail file missing on disk")

    return FileResponse(str(raw_resolved))


@router.post("/{thumbnail_id}/face_swap")
async def fix_thumbnail_face(thumbnail_id: int, restore_faces: bool = False):
    """Re-run face swap on a single existing thumbnail.

    Restores the raw (pre-swap) version first so the operation is repeatable.
    Uses the same cluster mapping as the original generation job.
    """
    from services.face_swap_pipeline import (
        apply_face_swap_to_thumbnail,
        persist_face_swap_result,
    )
    from services.generation_service import GenerationService

    async with get_db() as db:
        # Load thumbnail + its job
        async with db.execute(
            f"""
                SELECT t.id, t.job_id, t.filepath, t.raw_path,
                       gj.status AS job_status, {STOPPING_JOB_SQL} AS stopping
                FROM thumbnails t JOIN generation_jobs gj ON gj.id = t.job_id
                WHERE t.id = ?
            """,
            [thumbnail_id]
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=t('api.errors.thumbnail_not_found'))
            cols = [d[0] for d in cursor.description]
            thumb = dict(zip(cols, row))

        filepath = resolve_output_path(thumb['filepath'])
        output_dir_resolved = Path(OUTPUT_DIR).resolve()
        if not filepath.resolve().is_relative_to(output_dir_resolved):
            raise HTTPException(status_code=403, detail=t('api.errors.invalid_file_path'))

        # Cancellation keeps ownership until the worker finishes its last write.
        if thumb['stopping'] or thumb['job_status'] not in ('completed', 'cancelled', 'error'):
            raise HTTPException(
                status_code=409,
                detail="Generation job is still running; retry when it finishes"
            )

        # If a swap was applied before, restore the raw version so this call
        # operates on the pristine LLM output (makes "Fix face" idempotent).
        if thumb.get('raw_path'):
            raw_path = resolve_output_path(thumb['raw_path'])
            if not raw_path.exists():
                raise HTTPException(status_code=404, detail="Original thumbnail is missing; refusing to swap an already processed image")
            try:
                shutil.copy2(raw_path, filepath)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Could not restore raw image: {e}")

        if not filepath.exists():
            raise HTTPException(status_code=404, detail=t('api.errors.thumbnail_file_not_found'))

        # Get the cluster mapping from the originating job
        service = GenerationService(db)
        job_clusters = await service.get_job_clusters(thumb['job_id'])
        if not job_clusters:
            raise HTTPException(
                status_code=400,
                detail="Originating job has no clusters; cannot face-swap"
            )

        result = await apply_face_swap_to_thumbnail(db, filepath, job_clusters, restore_faces=restore_faces)
        await persist_face_swap_result(db, thumbnail_id, result)

    return {
        "thumbnail_id": thumbnail_id,
        "success": result.success,
        "face_swap_applied": result.success and result.raw_path is not None,
        "error": result.error,
        "actions": orjson.loads(orjson.dumps(result.to_metadata())).get("actions", []),
    }


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
            thumbnail = resolve_row_paths(dict(zip(columns, row)), 'filepath', 'raw_path')

    return thumbnail


@router.delete("/{thumbnail_id}")
async def delete_thumbnail(thumbnail_id: int):
    """
    Delete a specific thumbnail, with the copies kept of it: the image from
    before the face swap, the one the provider gave and ComfyUI's workflow.

    Refused (409) while its generation job is running or stopping: the job
    may still be writing those same files.
    """
    output_dir = Path(OUTPUT_DIR).resolve()

    async with get_db() as db:
        query = f"""
            SELECT t.filepath, t.raw_path, gj.status, {STOPPING_JOB_SQL} AS stopping
            FROM thumbnails t JOIN generation_jobs gj ON gj.id = t.job_id
            WHERE t.id = ?
        """
        async with db.execute(query, [thumbnail_id]) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=t('api.errors.thumbnail_not_found'))

        filepath = resolve_output_path(row[0])

        # Security: validate path is within OUTPUT_DIR
        if not filepath.resolve().is_relative_to(output_dir):
            raise HTTPException(status_code=403, detail="Invalid filepath")

        if row[3] or row[2] not in ('completed', 'cancelled', 'error'):
            raise HTTPException(status_code=409, detail=t('api.errors.generation_in_progress'))

        files = [filepath, _provider_raw_path(filepath), filepath.with_suffix(".comfyui.json")]
        if row[1]:
            files.append(resolve_output_path(row[1]))

        # Delete from database
        await db.execute("DELETE FROM thumbnails WHERE id = ?", [thumbnail_id])
        await db.commit()

        for file in files:
            if file.resolve().is_relative_to(output_dir) and file.is_file():
                file.unlink()

    return {"deleted": True, "thumbnail_id": thumbnail_id}


# Gallery orders. Every one ends newest first so pages stay stable.
#   recent         newest first
#   failures_last  newest first, faces of another person or not found last
#   match          most alike first, then faces not found, then the unscored
THUMBNAIL_ORDERS = {
    'recent': "t.created_at DESC, t.id DESC",
    'failures_last': """
        CASE WHEN t.face_match_status = 'missing' OR t.face_match_score < :bad THEN 1 ELSE 0 END,
        t.created_at DESC, t.id DESC
    """,
    'match': """
        CASE t.face_match_status WHEN 'matched' THEN 0 WHEN 'missing' THEN 1 ELSE 2 END,
        t.face_match_score DESC, t.created_at DESC, t.id DESC
    """,
}


@router.get("/video/{video_id}")
async def get_video_thumbnails(
    video_id: int,
    limit: int = Query(default=50, ge=1, le=500),
    skip: int = Query(default=0, ge=0),
    sort: Literal['recent', 'failures_last', 'match'] = 'recent',
):
    """
    Get one page of a video's thumbnails in the chosen order, plus the total count.
    """
    async with get_db() as db:
        async with db.execute("""
            SELECT COUNT(*)
            FROM thumbnails t
            JOIN generation_jobs j ON t.job_id = j.id
            WHERE j.video_id = ?
        """, [video_id]) as cursor:
            total = (await cursor.fetchone())[0]

        query = f"""
            SELECT t.id, t.filepath, t.image_index,
                   t.suggested_title, t.text_overlay, t.created_at,
                   t.face_swap_applied, t.raw_path,
                   t.face_match_status, t.face_match_score, t.face_match_people
            FROM thumbnails t
            JOIN generation_jobs j ON t.job_id = j.id
            WHERE j.video_id = :video_id
            ORDER BY {THUMBNAIL_ORDERS[sort]}
            LIMIT :limit OFFSET :skip
        """
        params = {"video_id": video_id, "limit": limit, "skip": skip, "bad": FACE_MATCH_BAD_THRESHOLD}
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            thumbnails = [
                face_match_fields(resolve_row_paths(dict(zip(columns, row)), 'filepath', 'raw_path'))
                for row in rows
            ]

    return {"video_id": video_id, "thumbnails": thumbnails, "total": total}


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

    zip_content = await asyncio.to_thread(_zip_thumbnails, [
        (filepath, f"{image_index:02d}_{_zip_title(title, image_index)}")
        for filepath, title, image_index in rows
    ])

    # Create download filename
    video_name = Path(video_filename).stem[:30]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_name = f"thumbnails_{video_name}_{timestamp}.zip"

    # Sent whole: it is in memory already, and a StreamingResponse over a
    # BytesIO sends it line by line, hundreds of thousands of pieces for PNGs
    return Response(
        content=zip_content,
        media_type="application/zip",
        headers={"Content-Disposition": _attachment(download_name)}
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

    zip_content = await asyncio.to_thread(_zip_thumbnails, [
        (filepath, f"job{job_id}/{image_index:02d}_{_zip_title(title, image_index)}")
        for filepath, title, image_index, job_id in rows
    ])

    # Create download filename
    video_name = Path(video_filename).stem[:30]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_name = f"all_thumbnails_{video_name}_{timestamp}.zip"

    # Sent whole: it is in memory already, and a StreamingResponse over a
    # BytesIO sends it line by line, hundreds of thousands of pieces for PNGs
    return Response(
        content=zip_content,
        media_type="application/zip",
        headers={"Content-Disposition": _attachment(download_name)}
    )

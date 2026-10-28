"""
Face Swap Pipeline (orchestration layer)
========================================
End-to-end orchestration over a single thumbnail:

    detect → match face↔cluster → load best source face →
    InSwapper swap → optional GFPGAN restore → write raw + swapped to disk →
    update the thumbnail row in the database

This is the layer the GenerationService calls after every successful image
generation, and the layer that the "Fix face" endpoint calls when the user
asks to retry an individual thumbnail.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

import orjson

from services.face_swap_service import (
    FaceSwapResult,
    FaceSwapService,
    FaceSwapThresholds,
    face_swap_available,
    face_restoration_available,
)
from services.face_matcher import match_thumbnail_faces
from utils import resolve_output_path, to_output_relative, write_image

logger = logging.getLogger(__name__)


def _raw_path_for(thumb_path: Path) -> Path:
    """Return the path where the pre-swap version is stored, alongside the thumbnail.

    Layout: thumbnails/<file>.png → thumbnails/raw_swap/<file>.png
    """
    return thumb_path.parent / "raw_swap" / thumb_path.name


async def apply_face_swap_to_thumbnail(
    db,
    thumbnail_path: Path,
    job_clusters: list[dict],
    thresholds: Optional[FaceSwapThresholds] = None,
    restore_faces: bool = False,
) -> FaceSwapResult:
    """Process a single thumbnail file in place.

    On success, the processed image OVERWRITES the input path so
    UI code that already points at it keeps working. The original is
    preserved at the raw_path returned in the result.

    Returns a FaceSwapResult describing what happened (used to populate
    the `thumbnails.face_swap_metadata` column).
    """
    thresholds = thresholds or FaceSwapThresholds.from_config()
    available, reasons = face_swap_available()
    if restore_faces:
        _, restoration_missing = face_restoration_available()
        reasons = reasons + restoration_missing
        available = not reasons
    if not available:
        msg = f"face swap is missing {', '.join(reasons)}"
        logger.warning(f"face_swap_pipeline: skipping — {msg}")
        return FaceSwapResult(success=False, output_path=None, raw_path=None, error=msg)

    if not thumbnail_path.exists():
        return FaceSwapResult(
            success=False, output_path=None, raw_path=None,
            error=f"thumbnail not found: {thumbnail_path}"
        )

    try:
        target_image, swap_pairs, actions = await match_thumbnail_faces(
            db, thumbnail_path, job_clusters, thresholds=thresholds
        )
    except Exception as e:
        logger.error(f"face_swap_pipeline: matcher failed for {thumbnail_path}: {e}")
        return FaceSwapResult(
            success=False, output_path=None, raw_path=None, error=str(e)
        )

    if not swap_pairs:
        # Nothing to swap (either no faces, or none crossed the threshold).
        # Persist the actions list so the UI can show "no swap applied + reasons".
        return FaceSwapResult(
            success=True,
            output_path=thumbnail_path,
            raw_path=None,
            actions=actions,
        )

    # Make a raw copy first so we can offer "before / after" in the gallery.
    raw_path = _raw_path_for(thumbnail_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(thumbnail_path, raw_path)
    except Exception as e:
        logger.error(f"face_swap_pipeline: failed to back up raw thumbnail: {e}")
        return FaceSwapResult(
            success=False, output_path=None, raw_path=None,
            error=f"could not preserve raw image: {e}",
            actions=actions,
        )

    # Apply the requested processing, preserving whatever covers the faces
    try:
        service = FaceSwapService.get()
        processed, covered = service.apply(target_image, swap_pairs, restore_faces=restore_faces)
    except Exception as e:
        logger.error(f"face_swap_pipeline: swap/restore failed: {e}")
        # Restore the raw image as the final file so the thumbnail isn't broken.
        try:
            shutil.copy2(raw_path, thumbnail_path)
        except Exception:
            pass
        return FaceSwapResult(
            success=False, output_path=thumbnail_path, raw_path=raw_path,
            error=str(e), actions=actions,
        )

    # The matcher lists the swapped faces' actions in the order of their pairs
    swapped_actions = [a for a in actions if a.decision in ('swap', 'swap_adaptive')]
    for action, share in zip(swapped_actions, covered, strict=True):
        action.covered = share

    # Write the swapped result back to the original path.
    try:
        write_image(thumbnail_path, processed)
    except Exception as e:
        logger.error(f"face_swap_pipeline: failed to write swapped image: {e}")
        return FaceSwapResult(
            success=False, output_path=thumbnail_path, raw_path=raw_path,
            error=str(e), actions=actions,
        )

    return FaceSwapResult(
        success=True,
        output_path=thumbnail_path,
        raw_path=raw_path,
        actions=actions,
        restoration_applied=restore_faces,
    )


async def persist_face_swap_result(
    db,
    thumbnail_id: int,
    result: FaceSwapResult,
) -> None:
    """Write the face-swap metadata back into the `thumbnails` table."""
    metadata_json = orjson.dumps(result.to_metadata()).decode('utf-8')
    swap_applied_int = 1 if (result.success and result.raw_path is not None) else 0
    raw_str = to_output_relative(result.raw_path) if result.raw_path else None
    await db.execute(
        """
        UPDATE thumbnails
        SET raw_path = ?,
            face_swap_applied = ?,
            face_swap_metadata = ?
        WHERE id = ?
        """,
        [raw_str, swap_applied_int, metadata_json, thumbnail_id]
    )
    await db.commit()


async def apply_face_swap_to_job(
    db,
    job_id: int,
    thresholds: Optional[FaceSwapThresholds] = None,
    restore_faces: bool = False,
    cancellation_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """Apply face swap to every thumbnail produced by a job.

    Run after the generation pipeline has saved all thumbnails to disk
    (the progress callback inserts them into the DB as raw images). For
    each thumbnail, attempt a swap; record the outcome. Failures do not
    abort the loop — each thumbnail's metadata captures its own decision.
    A cancelled job stops the loop before the next thumbnail.
    """
    from services.generation_service import GenerationCancelledError, GenerationService
    service = GenerationService(db)

    job_clusters = await service.get_job_clusters(job_id)
    if not job_clusters:
        logger.info(f"face_swap_pipeline: job {job_id} has no clusters, skipping pass")
        return {"processed": 0, "swapped": 0, "skipped": 0, "errors": 0}

    async with db.execute(
        "SELECT id, filepath FROM thumbnails WHERE job_id = ?", [job_id]
    ) as cursor:
        rows = await cursor.fetchall()

    counters = {"processed": 0, "swapped": 0, "skipped": 0, "errors": 0}
    for thumb_id, filepath in rows:
        if cancellation_check and cancellation_check():
            raise GenerationCancelledError(f"Job {job_id} cancelled during face swap")
        counters["processed"] += 1
        try:
            result = await apply_face_swap_to_thumbnail(
                db, resolve_output_path(filepath), job_clusters, thresholds=thresholds, restore_faces=restore_faces
            )
            await persist_face_swap_result(db, thumb_id, result)
            if not result.success:
                counters["errors"] += 1
            elif result.raw_path is None:
                counters["skipped"] += 1
            else:
                counters["swapped"] += 1
        except Exception as e:
            logger.error(f"face_swap_pipeline: thumbnail {thumb_id} failed: {e}")
            counters["errors"] += 1

    logger.info(
        f"face_swap_pipeline job={job_id}: processed={counters['processed']} "
        f"swapped={counters['swapped']} skipped={counters['skipped']} errors={counters['errors']}"
    )
    return counters

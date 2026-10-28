"""
Face Match

How much each person of a generation job looks like themselves in a thumbnail:
the cosine similarity between the person's face embedding and the face found
for them in the image, both from InsightFace (the model of the analysis).

The person's embedding is the one stored for them (clusters.embedding_centroid),
which the analysis and the person editor keep fitted to their current frames.
Thumbnails are scored as generated, before any face swap: a swap pastes the
person's identity back with this same model and would always score high.
"""

import asyncio
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import orjson

from services.face_index import load_face_index, stored_centroid
from services.face_matcher import greedy_assign
from utils import VideoOutput, read_image, resolve_output_path

# Outcome of scoring a thumbnail (thumbnails.face_match_status; NULL: not scored)
MATCHED = 'matched'            # Every person with a face has one in the image
MISSING = 'missing'            # A person with a face has none in the image
NO_REFERENCE = 'no_reference'  # No person of the job has a face to compare with

# Jobs whose thumbnails no stage will change any more
FINISHED_JOB_STATUSES = ('completed', 'cancelled', 'error')


@dataclass(frozen=True)
class PersonReference:
    """A person of a job and their face embedding (None when their frames show no face)."""
    person_id: str
    cluster_id: int
    embedding: Optional[np.ndarray]


@dataclass(frozen=True)
class FaceMatchResult:
    """Face match of one thumbnail."""
    status: str
    score: Optional[float]  # Lowest similarity of the people, only when MATCHED
    people: List[dict]      # [{person_id, cluster_id, similarity}], None when not found


def face_match_band(status: Optional[str], score: Optional[float]) -> Optional[str]:
    """How a face match reads: good, doubtful or bad, else the status itself
    (missing, no_reference, or None when not scored yet)."""
    from config import FACE_MATCH_BAD_THRESHOLD, FACE_MATCH_GOOD_THRESHOLD

    if status != MATCHED:
        return status
    if score >= FACE_MATCH_GOOD_THRESHOLD:
        return 'good'
    if score >= FACE_MATCH_BAD_THRESHOLD:
        return 'doubtful'
    return 'bad'


def face_match_fields(row: dict) -> dict:
    """Replace a thumbnail row's stored face match columns with the fields the API and CLI show."""
    status = row.pop('face_match_status')
    score = row.pop('face_match_score')
    people = row.pop('face_match_people')
    row['face_match'] = None if status is None else {
        'status': status,
        'band': face_match_band(status, score),
        'score': score,
        'people': orjson.loads(people),
    }
    return row


# =============================================================================
# REFERENCES
# =============================================================================

async def load_job_references(db, job_id: int) -> List[PersonReference]:
    """Face embedding of each person of a job, in person_id order."""
    from config import OUTPUT_DIR

    async with db.execute("""
        SELECT gjc.person_id, gjc.cluster_id, v.filepath
        FROM generation_job_clusters gjc
        JOIN generation_jobs j ON j.id = gjc.job_id
        JOIN videos v ON v.id = j.video_id
        WHERE gjc.job_id = ?
        ORDER BY gjc.cluster_order
    """, [job_id]) as cursor:
        people = await cursor.fetchall()
    if not people:
        return []

    faces_file = VideoOutput(Path(people[0][2]), Path(OUTPUT_DIR)).faces_file
    # Parsing a large faces.json takes a moment the first time
    index = await asyncio.to_thread(load_face_index, faces_file)

    references = []
    for person_id, cluster_id, _ in people:
        async with db.execute("""
            SELECT vf.frame_path
            FROM cluster_frame_assignments cfa
            JOIN video_frames vf ON vf.id = cfa.frame_id
            WHERE cfa.cluster_id = ?
        """, [cluster_id]) as cursor:
            members = [PurePosixPath(path).name for (path,) in await cursor.fetchall()]
        if members and not faces_file.exists():
            raise FileNotFoundError(f"Job {job_id}: the video has no face analysis ({faces_file})")
        async with db.execute("SELECT embedding_centroid FROM clusters WHERE id = ?", [cluster_id]) as cursor:
            (blob,) = await cursor.fetchone()
        references.append(PersonReference(person_id, cluster_id, stored_centroid(blob, index, members, cluster_id)))
    return references


# =============================================================================
# SCORING
# =============================================================================

def score_image(image_path: Path, references: List[PersonReference]) -> FaceMatchResult:
    """Face match of an image for these people (blocking: runs the face detector).

    Each person gets at most one face and each face at most one person, most
    similar pairs first, so extras in the background never stand in for them.
    """
    people = [ref for ref in references if ref.embedding is not None]
    if not people:
        return FaceMatchResult(NO_REFERENCE, None, [])

    from face_extraction import get_face_app
    faces = get_face_app().get(read_image(image_path))

    similarities: Dict[int, float] = {}
    if faces:
        matrix = np.stack([face.normed_embedding for face in faces]) @ np.stack([p.embedding for p in people]).T
        for _, person, similarity in greedy_assign(matrix):
            similarities[person] = round(similarity, 3)

    details = [
        {'person_id': ref.person_id, 'cluster_id': ref.cluster_id, 'similarity': similarities.get(i)}
        for i, ref in enumerate(people)
    ]
    if len(similarities) < len(people):
        return FaceMatchResult(MISSING, None, details)
    return FaceMatchResult(MATCHED, min(similarities.values()), details)


async def score_thumbnail(db, thumbnail_id: int, references: List[PersonReference]) -> FaceMatchResult:
    """Score a thumbnail on its image as generated (the pre-swap copy if swapped) and save it."""
    async with db.execute(
        "SELECT filepath, raw_path FROM thumbnails WHERE id = ?", [thumbnail_id]
    ) as cursor:
        filepath, raw_path = await cursor.fetchone()

    result = await asyncio.to_thread(score_image, resolve_output_path(raw_path or filepath), references)
    await db.execute(
        """
        UPDATE thumbnails
        SET face_match_status = ?, face_match_score = ?, face_match_people = ?
        WHERE id = ?
        """,
        [result.status, result.score, orjson.dumps(result.people).decode(), thumbnail_id]
    )
    await db.commit()
    return result


async def score_job_thumbnails(
    db,
    job_id: int,
    cancellation_check: Optional[Callable[[], bool]] = None,
) -> Counter:
    """Score every thumbnail of a job; the generation runs it before the face swap.

    Returns how many thumbnails got each band.
    """
    from services.generation_service import GenerationCancelledError

    references = await load_job_references(db, job_id)
    async with db.execute(
        "SELECT id FROM thumbnails WHERE job_id = ? ORDER BY image_index", [job_id]
    ) as cursor:
        thumbnail_ids = [row[0] for row in await cursor.fetchall()]

    bands = Counter()
    for thumbnail_id in thumbnail_ids:
        if cancellation_check and cancellation_check():
            raise GenerationCancelledError(f"Job {job_id} cancelled during face match scoring")
        result = await score_thumbnail(db, thumbnail_id, references)
        bands[face_match_band(result.status, result.score)] += 1
    return bands


async def score_existing_thumbnails(
    db,
    video_id: Optional[int] = None,
    rescore: bool = False,
    on_scored: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Counter, List[dict]]:
    """Score the thumbnails of finished jobs, only those not scored yet unless rescore.

    Thumbnails whose image is no longer on disk are left unscored and listed.
    on_scored(done, total) is called after each thumbnail, scored or not.

    Returns how many thumbnails got each band, and the ones without image as
    [{thumbnail_id, video_id, path}].
    """
    query = """
        SELECT t.id, t.job_id, j.video_id, COALESCE(t.raw_path, t.filepath)
        FROM thumbnails t
        JOIN generation_jobs j ON j.id = t.job_id
        WHERE j.status IN (?, ?, ?)
    """
    params: list = list(FINISHED_JOB_STATUSES)
    if video_id is not None:
        query += " AND j.video_id = ?"
        params.append(video_id)
    if not rescore:
        query += " AND t.face_match_status IS NULL"
    query += " ORDER BY j.video_id, t.job_id, t.image_index"
    async with db.execute(query, params) as cursor:
        rows = await cursor.fetchall()

    bands = Counter()
    missing_files = []
    references: Dict[int, List[PersonReference]] = {}
    for done, (thumbnail_id, job_id, thumbnail_video_id, image) in enumerate(rows, start=1):
        image_path = resolve_output_path(image)
        if not image_path.exists():
            missing_files.append({'thumbnail_id': thumbnail_id, 'video_id': thumbnail_video_id, 'path': str(image_path)})
        else:
            if job_id not in references:
                references[job_id] = await load_job_references(db, job_id)
            result = await score_thumbnail(db, thumbnail_id, references[job_id])
            bands[face_match_band(result.status, result.score)] += 1
        if on_scored:
            on_scored(done, len(rows))
    return bands, missing_files

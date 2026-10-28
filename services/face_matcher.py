"""
Match generated faces to the selected people and choose an identity-verified
source face for each swap. Reference quality and expression come from the
indexed face, not from the full frame that may contain several people.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from services.face_swap_service import FaceSwapAction, FaceSwapThresholds
from services.face_index import (
    MIN_IDENTITY_SIMILARITY, centroid_from_blob, identity_face_row, load_face_index,
)
from utils import VideoOutput, read_image, resolve_output_path

logger = logging.getLogger(__name__)


# =============================================================================
# COSINE SIMILARITY
# =============================================================================


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, robust to zero norms."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(_normalize(a), _normalize(b)))


# =============================================================================
# IMAGE → FACE OBJECTS
# =============================================================================


def _load_image_bgr(path: Path) -> Optional[np.ndarray]:
    """Load an image as a BGR numpy array (OpenCV convention).

    Returns None if the file is missing or unreadable.
    """
    if not path.exists():
        logger.warning(f"face_matcher: image not found: {path}")
        return None
    try:
        return read_image(path)
    except ValueError as e:
        logger.warning(f"face_matcher: {e}")
        return None


def _cluster_embedding(cluster: dict) -> np.ndarray:
    """Use the stored identity; a different face in the representative frame cannot replace it."""
    blob = cluster.get('embedding_centroid')
    if blob is None:
        raise ValueError(f"Cluster {cluster['id']} has no stored face identity; analyze its faces again")
    embedding = centroid_from_blob(blob)
    if not np.isfinite(embedding).all() or np.linalg.norm(embedding) == 0:
        raise ValueError(f"Cluster {cluster['id']} has an invalid stored face identity")
    return _normalize(embedding)


def _target_expression(face) -> str:
    """Classify the generated face with the same landmark classifier used during analysis."""
    from face_extraction import classify_expression

    landmarks = getattr(face, 'landmark_3d_68', None)
    if landmarks is None or len(landmarks) < 68:
        raise ValueError("The target face has no landmarks for expression matching")
    return classify_expression(np.asarray(landmarks)[:, :2])[0]


@dataclass
class SourceFaceCandidate:
    """An indexed face of the selected identity, with its own quality and expression."""
    frame_path: Path
    quality_score: float
    expression: str
    embedding: np.ndarray
    pixels: bytes


async def list_source_candidates(db, cluster_id: int) -> list[SourceFaceCandidate]:
    from services.reference_service import ReferenceService
    from face_extraction import get_face_app
    import cv2

    async with db.execute("SELECT video_id, embedding_centroid FROM clusters WHERE id=?", [cluster_id]) as cursor:
        record = await cursor.fetchone()
    if record is None:
        raise ValueError("Selected person no longer exists.")
    video_id, blob = record
    service = ReferenceService(db)
    references = [r for r in await service.list(video_id) if r["person_id"] == cluster_id and r["role"] == "face"]
    if not references:
        raise ValueError(f"Cluster {cluster_id} has no selected reference faces")
    index = await asyncio.to_thread(load_face_index, (await service.output_dir(video_id)) / "faces.json")
    centroid = None if blob is None else _cluster_embedding({"id": cluster_id, "embedding_centroid": blob})
    candidates = []
    for ref in references:
        path = await service.source_path(video_id, ref)
        pixels = await service.crop_bytes(video_id, ref, index)
        if ref["source_type"] == "frame":
            row = identity_face_row(index, centroid, path.name)
            quality, expression, embedding = float(index.qualities[row]), index.expressions[row], index.embeddings[row]
        else:
            image = cv2.imdecode(np.frombuffer(pixels, dtype=np.uint8), cv2.IMREAD_COLOR)
            faces = await asyncio.to_thread(get_face_app().get, image)
            if len(faces) != 1:
                raise ValueError("The uploaded face-swap crop must contain exactly one detected face.")
            embedding = faces[0].embedding
            if centroid is not None and cosine_similarity(embedding, centroid) < MIN_IDENTITY_SIMILARITY:
                raise ValueError("The uploaded face-swap source does not match this person.")
            quality, expression = float(faces[0].det_score) * 100, _target_expression(faces[0])
        candidates.append(SourceFaceCandidate(path, quality, expression, embedding, pixels))
    return candidates


async def pick_source_face(db, cluster_id: int, target_expression: Optional[str] = None,
                           min_quality: float = 70.0):
    """Prefer a matching expression among high-quality faces, then the best available photo.

    Quality is measured on a 0-100 scale. Low-quality photos remain behind every
    high-quality candidate, even when their expression matches. An unreadable
    or identity-mismatched chosen photo is an error, never a different donor.
    """
    from face_extraction import get_face_app

    candidates = await list_source_candidates(db, cluster_id)
    candidates.sort(key=lambda candidate: (
        candidate.quality_score < min_quality,
        bool(target_expression) and candidate.quality_score >= min_quality and candidate.expression != target_expression,
        -candidate.quality_score,
        str(candidate.frame_path),
    ))
    candidate = candidates[0]

    def detect_source():
        import cv2
        image = cv2.imdecode(np.frombuffer(candidate.pixels, dtype=np.uint8), cv2.IMREAD_COLOR)
        faces = get_face_app().get(image)
        faces = [face for face in faces if getattr(face, 'embedding', None) is not None]
        if not faces:
            raise ValueError(f"No source face detected in {candidate.frame_path.name}")
        source = max(faces, key=lambda face: cosine_similarity(face.embedding, candidate.embedding))
        similarity = cosine_similarity(source.embedding, candidate.embedding)
        if not np.isfinite(similarity) or similarity < MIN_IDENTITY_SIMILARITY:
            raise ValueError(f"The detected source face no longer matches {candidate.frame_path.name}")
        source.source_frame_path = str(candidate.frame_path)
        return source

    return await asyncio.to_thread(detect_source)


# =============================================================================
# MAIN MATCHING ENTRY POINT
# =============================================================================


def greedy_assign(sim_matrix: np.ndarray) -> list[tuple]:
    """Greedy 1-to-1 assignment over a (faces × clusters) similarity matrix.

    Yields (face_idx, cluster_idx, similarity) tuples from highest to
    lowest similarity. Each face and each cluster appear at most once.
    Returns the list ordered by descending similarity.
    """
    pairs = []
    if sim_matrix.size == 0:
        return pairs
    # Copy because we'll write -inf into used rows/cols
    m = sim_matrix.astype(np.float32).copy()
    n_faces, n_clusters = m.shape
    for _ in range(min(n_faces, n_clusters)):
        idx = np.unravel_index(np.argmax(m), m.shape)
        f_idx, c_idx = int(idx[0]), int(idx[1])
        sim = float(m[f_idx, c_idx])
        if not np.isfinite(sim):
            break
        pairs.append((f_idx, c_idx, sim))
        m[f_idx, :] = -np.inf
        m[:, c_idx] = -np.inf
    return pairs


async def match_thumbnail_faces(
    db,
    thumbnail_path: Path,
    job_clusters: list[dict],
    thresholds: Optional[FaceSwapThresholds] = None,
) -> tuple:
    """Detect → match → select source. Returns the components Phase 2.4 needs.

    Args:
        db: aiosqlite connection.
        thumbnail_path: path to the generated thumbnail to fix.
        job_clusters: rows like those returned by GenerationService.get_job_clusters()
            (must include cluster_id and person_id). Additional cluster fields
            are looked up internally.
        thresholds: optional FaceSwapThresholds, defaults to the dataclass defaults.

    Returns:
        (target_image, swap_pairs, actions) where:
            target_image    is the loaded BGR numpy array (caller passes it to
                            FaceSwapService.apply / saves to disk).
            swap_pairs      is the list of (target_face, source_face) tuples ready
                            to feed FaceSwapService.apply.
            actions         is a list[FaceSwapAction] capturing the decision for
                            every detected face — including those that are
                            skipped — so the orchestrator can persist metadata.
    """
    # Defaults from .env-driven config when not supplied by the caller
    thresholds = thresholds or FaceSwapThresholds.from_config()

    target_image = _load_image_bgr(Path(thumbnail_path))
    if target_image is None:
        raise FileNotFoundError(f"Could not load thumbnail: {thumbnail_path}")

    # 1) Detect target faces
    from face_extraction import get_face_app
    app = get_face_app()
    target_faces = list(app.get(target_image))

    if not target_faces:
        logger.info("face_matcher: no faces detected in thumbnail; nothing to swap")
        return target_image, [], []

    # 2) Build cluster embeddings (one per cluster in the job)
    cluster_records: list[dict] = []
    cluster_embeddings: list[np.ndarray] = []
    for jc in job_clusters:
        async with db.execute(
            "SELECT * FROM clusters WHERE id = ?", [jc['cluster_id']]
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                raise ValueError(f"Selected cluster {jc['cluster_id']} no longer exists")
            columns = [d[0] for d in cursor.description]
            cluster_dict = dict(zip(columns, row))
        cluster_dict['person_id'] = jc.get('person_id')

        cluster_records.append(cluster_dict)
        cluster_embeddings.append(_cluster_embedding(cluster_dict))

    if not cluster_records:
        raise ValueError("No selected face identities for this thumbnail")

    # 2.5) ADAPTIVE FAST-PATH: one cluster + one detected face is unambiguous.
    # We KNOW which cluster the face must come from (the user picked it), so
    # the similarity threshold is irrelevant — bypass it and swap directly.
    # This recovers the most common case (single-creator channel) where the
    # generated face often drifts and we'd otherwise leave it unfixed.
    if len(cluster_records) == 1 and len(target_faces) == 1:
        target_face = target_faces[0]
        cluster = cluster_records[0]
        # Still compute the similarity for the metadata, but ignore it for the decision.
        emb = getattr(target_face, 'embedding', None)
        sim_value = (
            cosine_similarity(emb, cluster_embeddings[0])
            if emb is not None else None
        )
        logger.info(
            f"face_matcher: adaptive fast-path (1 cluster + 1 face) — "
            f"forcing swap regardless of similarity (sim={sim_value})"
        )

        source_face = await pick_source_face(
            db,
            cluster_id=cluster['id'],
            target_expression=_target_expression(target_face),
        )
        bbox = tuple(map(float, target_face.bbox.tolist())) if hasattr(target_face.bbox, 'tolist') else tuple(target_face.bbox)
        action = FaceSwapAction(
            target_face_index=0,
            target_bbox=bbox,
            person_id=cluster.get('person_id'),
            cluster_id=cluster['id'],
            similarity=sim_value,
            decision='swap_adaptive',  # distinct from threshold-driven 'swap' for analytics
            source_frame=getattr(source_face, 'source_frame_path', None) or None,
        )
        return target_image, [(target_face, source_face)], [action]

    # 3) Similarity matrix (multi-cluster or multi-face path)
    sim = np.zeros((len(target_faces), len(cluster_records)), dtype=np.float32)
    for i, f in enumerate(target_faces):
        emb = getattr(f, 'embedding', None)
        if emb is None:
            sim[i, :] = -np.inf
            continue
        for j, ce in enumerate(cluster_embeddings):
            sim[i, j] = cosine_similarity(emb, ce)

    # 4) Greedy assignment
    pairs = greedy_assign(sim)

    # 5) Decide and build swap_pairs + actions
    swap_pairs: list[tuple] = []
    actions: list[FaceSwapAction] = []
    assigned_face_indices = set()

    for f_idx, c_idx, similarity in pairs:
        assigned_face_indices.add(f_idx)
        target_face = target_faces[f_idx]
        cluster = cluster_records[c_idx]
        decision = thresholds.decide(similarity)

        action = FaceSwapAction(
            target_face_index=f_idx,
            target_bbox=tuple(map(float, target_face.bbox.tolist())) if hasattr(target_face.bbox, 'tolist') else tuple(target_face.bbox),
            person_id=cluster.get('person_id'),
            cluster_id=cluster['id'],
            similarity=similarity,
            decision=decision,
            source_frame=None,
        )

        if decision == 'swap':
            source_face = await pick_source_face(
                db,
                cluster_id=cluster['id'],
                target_expression=_target_expression(target_face),
            )
            action.source_frame = source_face.source_frame_path
            swap_pairs.append((target_face, source_face))
        # 'skip_low_match' and 'skip_threshold' just leave swap_pairs untouched

        actions.append(action)

    # 6) Faces that didn't get any cluster (e.g. extras in the scene)
    for i, f in enumerate(target_faces):
        if i in assigned_face_indices:
            continue
        actions.append(FaceSwapAction(
            target_face_index=i,
            target_bbox=tuple(map(float, f.bbox.tolist())) if hasattr(f.bbox, 'tolist') else tuple(f.bbox),
            person_id=None,
            cluster_id=None,
            similarity=None,
            decision='skip_no_source',
            source_frame=None,
        ))

    return target_image, swap_pairs, actions

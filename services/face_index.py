"""
Face Index

Every face detected in a video (box, embedding, quality, expression), read from
the faces.json cache of the analysis. It tells which face of a frame belongs to
a person, so the UI can show face crops instead of whole frames.
"""

import io
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import orjson
from PIL import Image


EMBEDDING_SIZE = 512
# The person editor and face sheets already reject identities below this similarity.
MIN_IDENTITY_SIMILARITY = 0.3

# Square face crop geometry for the technical face-sheet renderer:
# a square FACE_CROP_SCALE times the larger side of the face box, centred on the
# face and raised by FACE_CROP_LIFT of the box height to include the hair.
FACE_CROP_SCALE = 1.8
FACE_CROP_LIFT = 0.12


@dataclass(frozen=True)
class FaceIndex:
    """Faces of a video grouped by frame filename.

    The faces of one frame are contiguous rows: frame `name` owns the rows
    spans[name][0]:spans[name][1] of every array.
    """
    spans: Dict[str, Tuple[int, int]]
    boxes: np.ndarray        # (faces, 4) [x1, y1, x2, y2] in frame pixels
    embeddings: np.ndarray   # (faces, 512) L2-normalized
    qualities: np.ndarray    # (faces,)
    poses: np.ndarray        # (faces, 3) head pitch, yaw and roll in degrees, NaN when unknown
    expressions: Tuple[str, ...]
    portraits: tuple[dict | None, ...]


@dataclass(frozen=True)
class FaceMatch:
    """The face of a frame chosen for a person."""
    row: int  # Exact row in the FaceIndex that produced this match
    box: List[float]
    similarity: Optional[float]  # Cosine similarity to the person, None without one
    quality_score: float
    expression: str
    yaw: Optional[float]  # Head yaw in degrees, None when unknown


_load_lock = threading.Lock()

EMPTY_INDEX = FaceIndex(
    spans={},
    boxes=np.zeros((0, 4), dtype=np.float32),
    embeddings=np.zeros((0, EMBEDDING_SIZE), dtype=np.float32),
    qualities=np.zeros(0, dtype=np.float32),
    poses=np.zeros((0, 3), dtype=np.float32),
    expressions=(),
    portraits=(),
)


def load_face_index(faces_json: Path) -> FaceIndex:
    """Face index of a video, cached while its faces.json stays unchanged.

    A video without faces.json has no detected faces yet.
    """
    if not faces_json.exists():
        return EMPTY_INDEX
    stat = faces_json.stat()
    # A page asks for many faces at once: parse each faces.json only once
    with _load_lock:
        return _load_face_index(str(faces_json), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=2)
def _load_face_index(path: str, mtime_ns: int, size: int) -> FaceIndex:
    data = orjson.loads(Path(path).read_bytes())

    by_frame: Dict[str, list] = {}
    for face in data['all_faces']:
        embedding = face.get('embedding')
        if not embedding or len(embedding) != EMBEDDING_SIZE:
            continue
        by_frame.setdefault(Path(face['frame_path']).name, []).append(face)

    faces = [face for name in sorted(by_frame) for face in by_frame[name]]
    if not faces:
        return EMPTY_INDEX

    spans = {}
    start = 0
    for name in sorted(by_frame):
        spans[name] = (start, start + len(by_frame[name]))
        start += len(by_frame[name])

    embeddings = np.array([face['embedding'] for face in faces], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1

    return FaceIndex(
        spans=spans,
        boxes=np.array([face['bbox'] for face in faces], dtype=np.float32),
        embeddings=embeddings / norms,
        qualities=np.array([face.get('quality_score') or 0 for face in faces], dtype=np.float32),
        # InsightFace pose is [pitch, yaw, roll]
        poses=np.array([face.get('pose') or [np.nan] * 3 for face in faces], dtype=np.float32),
        expressions=tuple(face.get('expression') or 'unknown' for face in faces),
        portraits=tuple(face.get('portrait_quality') for face in faces),
    )


class MissingCentroidError(RuntimeError):
    """A person whose frames show faces has no stored face embedding."""


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def centroid_to_blob(centroid: np.ndarray) -> bytes:
    """Stored form of a person's face embedding (clusters.embedding_centroid)."""
    return np.asarray(centroid, dtype=np.float32).tobytes()


def centroid_list(centroid: np.ndarray) -> List[float]:
    """JSON form of a person's face embedding (clustering_result.json)."""
    return [round(value, 6) for value in np.asarray(centroid, dtype=np.float32).tolist()]


def centroid_from_blob(blob: bytes) -> np.ndarray:
    centroid = np.frombuffer(blob, dtype=np.float32)
    if centroid.size != EMBEDDING_SIZE:
        raise ValueError(f"A stored face embedding has {centroid.size} values, not {EMBEDDING_SIZE}")
    return centroid


def stored_centroid(
    blob: Optional[bytes],
    index: FaceIndex,
    filenames: Iterable[str],
    person_id: int
) -> Optional[np.ndarray]:
    """A person's stored face embedding, None when none of their frames shows a face."""
    if blob is not None:
        return centroid_from_blob(blob)
    if any(name in index.spans for name in filenames):
        raise MissingCentroidError(
            f"Person {person_id} has faces but no stored face embedding; "
            "rerun face analysis for this video to store its embedding"
        )
    return None


def closest_rows(index: FaceIndex, centroid: np.ndarray, spans: List[Tuple[int, int]]) -> List[int]:
    """In each span of rows (a frame), the face most similar to the centroid."""
    scores = index.embeddings @ centroid
    return [start + int(np.argmax(scores[start:end])) for start, end in spans]


def identity_face_row(index: FaceIndex, centroid: np.ndarray, filename: str) -> int:
    """Find an indexed face of the selected identity, rejecting absent or unrelated faces."""
    span = index.spans.get(filename)
    if span is None:
        raise ValueError(f"No indexed face for identity reference {filename}")
    centroid = _normalize(centroid)
    row = closest_rows(index, centroid, [span])[0]
    similarity = float(index.embeddings[row] @ centroid)
    if not np.isfinite(similarity) or similarity < MIN_IDENTITY_SIMILARITY:
        raise ValueError(f"Reference {filename} has no face matching the selected identity")
    return row


def claimed_faces(index: FaceIndex, people: Iterable[Tuple[Optional[np.ndarray], Iterable[str]]]) -> Set[int]:
    """Rows of the faces these people (centroid, frame filenames) have in their frames."""
    claimed = set()
    for centroid, filenames in people:
        if centroid is not None:
            spans = [index.spans[name] for name in filenames if name in index.spans]
            claimed.update(closest_rows(index, centroid, spans))
    return claimed


def person_centroid(
    index: FaceIndex,
    filenames: Iterable[str],
    seed: Optional[np.ndarray] = None,
    claimed: Set[int] = frozenset()
) -> Optional[np.ndarray]:
    """Embedding of the person shown in all these frames, or None without faces.

    A frame can show several people, and two people seen together in every frame
    share all their frames, so the person is followed from a face known to be
    theirs: seed (their stored embedding), else the mean of the faces of these
    frames no other person claims (claimed face rows), or of all of them when
    every face is claimed (the same person split into another one of the view).
    Picking in each frame the face closest to it and averaging the picks, twice,
    fits it to these frames.
    """
    spans = [index.spans[name] for name in filenames if name in index.spans]
    if not spans:
        return None

    if seed is None:
        rows = [row for start, end in spans for row in range(start, end)]
        free = [row for row in rows if row not in claimed]
        seed = index.embeddings[free or rows].mean(axis=0)
    centroid = _normalize(np.asarray(seed, dtype=np.float32))
    for _ in range(2):
        centroid = _normalize(index.embeddings[closest_rows(index, centroid, spans)].mean(axis=0))
    return centroid


def match_faces(
    index: FaceIndex,
    centroid: Optional[np.ndarray],
    filenames: Iterable[str]
) -> Dict[str, FaceMatch]:
    """Face of each frame closest to the person (best quality without a person).

    Frames without detected faces are left out.
    """
    if centroid is not None:
        scores = (index.embeddings @ centroid).tolist()
    else:
        scores = index.qualities.tolist()

    matches = {}
    for name in filenames:
        span = index.spans.get(name)
        if not span:
            continue
        row = max(range(*span), key=scores.__getitem__)
        matches[name] = FaceMatch(
            row=row,
            box=[round(value, 1) for value in index.boxes[row].tolist()],
            similarity=scores[row] if centroid is not None else None,
            quality_score=round(float(index.qualities[row]), 1),
            expression=index.expressions[row],
            yaw=None if np.isnan(index.poses[row, 1]) else round(float(index.poses[row, 1]), 1),
        )
    return matches


def face_crop_box(box: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Square region around a face box, kept inside the frame."""
    x1, y1, x2, y2 = box
    side = min(max(x2 - x1, y2 - y1) * FACE_CROP_SCALE, width, height)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2 - (y2 - y1) * FACE_CROP_LIFT
    left = min(max(center_x - side / 2, 0), width - side)
    top = min(max(center_y - side / 2, 0), height - side)
    return round(left), round(top), round(left + side), round(top + side)


# Head-and-shoulders crop of one person in a frame shared with other people:
# PORTRAIT_WIDTH face widths wide, PORTRAIT_HEIGHT face heights tall, starting
# PORTRAIT_HEADROOM face heights above the face box to keep the hair.
PORTRAIT_WIDTH = 3.0
PORTRAIT_HEIGHT = 3.6
PORTRAIT_HEADROOM = 0.9


def portrait_crop_box(
    box: List[float],
    other_boxes: Iterable[List[float]],
    width: int,
    height: int
) -> Tuple[int, int, int, int]:
    """Head and shoulders of the face in box, kept inside the frame.

    A side stops halfway between this face and a face beside it, so the other
    people stay out of the crop (unless their faces overlap this one).
    """
    x1, y1, x2, y2 = box
    face_width, face_height = x2 - x1, y2 - y1
    center_x = (x1 + x2) / 2
    left = center_x - face_width * PORTRAIT_WIDTH / 2
    right = center_x + face_width * PORTRAIT_WIDTH / 2
    top = y1 - face_height * PORTRAIT_HEADROOM
    bottom = top + face_height * PORTRAIT_HEIGHT
    for other_x1, other_y1, other_x2, other_y2 in other_boxes:
        if other_y2 < top or other_y1 > bottom:
            continue  # Above or below the crop, not beside it
        if (other_x1 + other_x2) / 2 > center_x:
            right = min(right, max((x2 + other_x1) / 2, x2))
        else:
            left = max(left, min((other_x2 + x1) / 2, x1))
    return (round(max(left, 0)), round(max(top, 0)),
            round(min(right, width)), round(min(bottom, height)))


def save_portrait_crop(
    frame_path: Path,
    box: List[float],
    other_boxes: List[List[float]],
    destination: Path
) -> None:
    """Save the head and shoulders of one face of a frame (portrait_crop_box) as a lossless PNG."""
    with Image.open(frame_path) as image:
        crop = image.crop(portrait_crop_box(box, other_boxes, *image.size)).convert('RGB')
    destination.parent.mkdir(parents=True, exist_ok=True)
    crop.save(destination, 'PNG')


# A face whose crop has a mean HSV saturation (0-255) under this is black and
# white: such frames measure 0-3, pale skin in colour 16 or more.
GRAYSCALE_SATURATION = 12
_GRAYSCALE_SAMPLE = 96


def is_grayscale_face(frame_path: Path, box: Optional[List[float]]) -> bool:
    """Whether the face of a frame (the frame's centre without one) is in black and white."""
    return _is_grayscale(str(frame_path), tuple(box) if box else None)


@lru_cache(maxsize=4096)
def _is_grayscale(frame_path: str, box: Optional[Tuple[float, ...]]) -> bool:
    with Image.open(frame_path) as image:
        width, height = image.size
        region = face_crop_box(list(box), width, height) if box else (0, 0, width, height)
        crop = image.crop(region).convert('RGB')
    crop.thumbnail((_GRAYSCALE_SAMPLE, _GRAYSCALE_SAMPLE))
    saturation = np.asarray(crop.convert('HSV'))[:, :, 1]
    return float(saturation.mean()) < GRAYSCALE_SATURATION


def face_reference_view(yaw: Optional[float]) -> Optional[str]:
    """Describe an observed angle without inventing missing pose data.

    InsightFace measures a head turned enough to hide part of the far cheek at
    about 20 degrees, which a person already calls a three-quarter view.
    """
    if yaw is None:
        return None
    return "front" if abs(yaw) < 18 else "three_quarter" if abs(yaw) < 55 else "profile"


def render_face_crop(frame_path: Path, box: Optional[List[float]], size: int) -> bytes:
    """JPEG of the face region of a frame, or of its centre without a face."""
    with Image.open(frame_path) as image:
        width, height = image.size
        if box:
            region = face_crop_box(box, width, height)
        else:
            side = min(width, height)
            region = ((width - side) // 2, (height - side) // 2,
                      (width + side) // 2, (height + side) // 2)
        crop = image.crop(region).convert('RGB')
    crop.thumbnail((size, size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    crop.save(buffer, 'JPEG', quality=88)
    return buffer.getvalue()

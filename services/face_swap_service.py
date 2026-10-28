"""
Face Swap Service
=================
Local post-processing pipeline that replaces faces in a generated thumbnail
with cleaned-up versions of the influencer's real face, using:

  - InsightFace InSwapper-128  → the actual face swap
  - GFPGAN v1.4                → optional restoration, off by default because
                                 it can change facial detail and expression
  - XSeg occluders (1, 2, 3)   → what covers each face (a hand, food, a mask,
                                 glasses, a magnifying glass). The result is
                                 pasted only where the face shows, so whatever
                                 was in front of it stays in front.

Hardware: runs on CPU (slow) or CUDA (instant on RTX 4080/4090). Models
load lazily on first use and are cached as a module-level singleton.

This module exposes only the low-level primitives. Cluster→face matching
and source-frame selection live in `services/face_matcher.py` (Phase 2.3).
"""
from __future__ import annotations

import importlib.util
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
INSWAPPER_PATH = MODELS_DIR / "inswapper" / "inswapper_128.onnx"
GFPGAN_PATH = MODELS_DIR / "gfpgan" / "GFPGANv1.4.pth"
# DeepFaceLab's XSeg as FaceFusion exports it; the three are combined
OCCLUDER_PATHS = tuple(MODELS_DIR / "occluder" / f"xseg_{n}.onnx" for n in (1, 2, 3))

# The occluders read the face aligned to FaceFusion's arcface_128 template,
# the same crop InSwapper works on, at 256x256 (BGR, 0-1)
OCCLUDER_CROP_SIZE = 256
ARCFACE_128_TEMPLATE = np.array([
    [0.36167656, 0.40387734],
    [0.63696719, 0.40235469],
    [0.50019687, 0.56044219],
    [0.38710391, 0.72160547],
    [0.61507734, 0.72034453],
], dtype=np.float32)


# =============================================================================
# RESULTS DATACLASSES
# =============================================================================


@dataclass
class FaceSwapAction:
    """One face swap that was applied (or skipped) by the pipeline.

    Used so the orchestrator can produce a structured face_swap_metadata
    JSON for the database, and so the UI can show "person_01 swap applied
    with similarity 0.78".
    """
    target_face_index: int            # Which detected face in the thumbnail
    target_bbox: tuple                # (x1, y1, x2, y2)
    person_id: Optional[str]          # The cluster's person role, if matched
    cluster_id: Optional[int]
    similarity: Optional[float]       # Cosine similarity vs cluster centroid
    decision: str                     # 'swap' | 'skip_low_match' | 'skip_no_source' | 'skip_threshold'
    source_frame: Optional[str]       # Source frame path used for the swap
    error: Optional[str] = None       # Set when decision involves a failure
    covered: Optional[float] = None   # Share of the swapped face something covered (kept in front)


@dataclass
class FaceSwapResult:
    """End-to-end result of running the swap pipeline on one thumbnail."""
    success: bool
    output_path: Optional[Path]
    raw_path: Optional[Path]
    actions: list = field(default_factory=list)
    error: Optional[str] = None
    restoration_applied: bool = False

    def to_metadata(self) -> dict:
        """Serializable metadata for the `thumbnails.face_swap_metadata` column."""
        return {
            "success": self.success,
            "restoration_applied": self.restoration_applied,
            "error": self.error,
            "actions": [
                {
                    "target_face_index": a.target_face_index,
                    "target_bbox": list(a.target_bbox) if a.target_bbox else None,
                    "person_id": a.person_id,
                    "cluster_id": a.cluster_id,
                    "similarity": (round(a.similarity, 4) if a.similarity is not None else None),
                    "decision": a.decision,
                    "source_frame": a.source_frame,
                    "error": a.error,
                    "covered": (round(a.covered, 3) if a.covered is not None else None),
                }
                for a in self.actions
            ],
        }


# =============================================================================
# THRESHOLD CONFIG (D2 = "por umbral": alta=swap, media=original+warning, baja=error)
# =============================================================================


@dataclass
class FaceSwapThresholds:
    """Cosine-similarity thresholds for the swap decision.

    Empirical bands for buffalo_l embeddings on real photos:
      ≥ swap_threshold  → unmistakable same person  → swap
      ≥ warn_threshold  → likely match, some doubt → keep original + warning
      otherwise         → too low, refuse → recorded as low-similarity skip

    Defaults come from config.py (FACE_SWAP_THRESHOLD_SWAP / WARN), which
    in turn read from the .env so they're tunable without code changes.
    """
    swap_threshold: float = 0.55
    warn_threshold: float = 0.40

    @classmethod
    def from_config(cls) -> "FaceSwapThresholds":
        """Build using the values from config.py (.env-driven)."""
        try:
            from config import FACE_SWAP_THRESHOLD_SWAP, FACE_SWAP_THRESHOLD_WARN
            return cls(
                swap_threshold=FACE_SWAP_THRESHOLD_SWAP,
                warn_threshold=FACE_SWAP_THRESHOLD_WARN,
            )
        except Exception:
            # If config isn't importable for any reason, fall back to defaults
            return cls()

    def decide(self, similarity: float) -> str:
        if similarity >= self.swap_threshold:
            return 'swap'
        if similarity >= self.warn_threshold:
            return 'skip_low_match'
        return 'skip_threshold'


# =============================================================================
# OCCLUSION (whatever covers a face stays in front of its swap)
# =============================================================================


@dataclass
class FaceOcclusion:
    """Where the swap of one face may be pasted."""
    mask: np.ndarray    # float32, image-sized: 1 where the face shows, 0 elsewhere
    covered: float      # share of the face (jaw to brows) something covers


def _crop_border_mask(size: int) -> np.ndarray:
    """1 inside the crop, fading out towards its border (FaceFusion's box mask)."""
    blur = int(size * 0.15)
    border = max(blur // 2, 1)
    mask = np.ones((size, size), np.float32)
    mask[:border, :] = mask[-border:, :] = mask[:, :border] = mask[:, -border:] = 0
    return cv2.GaussianBlur(mask, (0, 0), blur * 0.25)


def _paste_edge(visible: np.ndarray) -> np.ndarray:
    """Grow the covered side a little and soften the edge, as FaceFusion does,
    so no fringe of the swap is left over whatever covers the face."""
    return (cv2.GaussianBlur(visible, (0, 0), 5).clip(0.5, 1) - 0.5) * 2


def paste_visible_faces(original: np.ndarray, processed: np.ndarray, occlusions: Sequence[FaceOcclusion]) -> np.ndarray:
    """`processed` where a swapped face shows, `original` everywhere else."""
    mask = np.maximum.reduce([occlusion.mask for occlusion in occlusions])[:, :, None]
    blended = processed.astype(np.float32) * mask + original.astype(np.float32) * (1 - mask)
    return np.rint(blended).astype(np.uint8)


# =============================================================================
# SERVICE
# =============================================================================


class FaceSwapService:
    """Singleton wrapper around InSwapper + GFPGAN + the XSeg occluders.

    Construct once, reuse across requests. The first call triggers model
    loading (a few seconds on GPU). Subsequent calls are fast.
    """

    _instance: Optional["FaceSwapService"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "FaceSwapService":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._swapper = None
        self._restorer = None
        self._occluders = None
        self._swapper_lock = threading.Lock()
        self._restorer_lock = threading.Lock()
        self._occluders_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model loading (lazy, thread-safe)
    # ------------------------------------------------------------------

    def _load_swapper(self):
        if self._swapper is not None:
            return self._swapper
        with self._swapper_lock:
            if self._swapper is None:
                if not INSWAPPER_PATH.exists():
                    raise FileNotFoundError(
                        f"InSwapper model not found at {INSWAPPER_PATH}. "
                        f"Run: python scripts/download_face_swap_models.py"
                    )
                try:
                    import insightface  # noqa: F401  (used by get_model)
                    from insightface.model_zoo import get_model
                except ImportError as e:
                    raise RuntimeError(
                        "insightface is required for face swap. Run: pip install insightface"
                    ) from e

                logger.info(f"Loading InSwapper model: {INSWAPPER_PATH}")
                self._swapper = get_model(
                    str(INSWAPPER_PATH),
                    download=False,
                    download_zip=False,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                )
                logger.info("InSwapper loaded successfully")
        return self._swapper

    def _load_restorer(self):
        if self._restorer is not None:
            return self._restorer
        with self._restorer_lock:
            if self._restorer is None:
                if not GFPGAN_PATH.exists():
                    raise FileNotFoundError(
                        f"GFPGAN model not found at {GFPGAN_PATH}. "
                        f"Run: python scripts/download_face_swap_models.py"
                    )
                try:
                    from gfpgan import GFPGANer
                except ImportError as e:
                    raise RuntimeError(
                        "gfpgan is required for face restoration. Run: pip install gfpgan"
                    ) from e

                logger.info(f"Loading GFPGAN restorer: {GFPGAN_PATH}")
                self._restorer = GFPGANer(
                    model_path=str(GFPGAN_PATH),
                    upscale=1,            # we only want restoration, not upscaling
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,    # background goes through untouched
                )
                logger.info("GFPGAN loaded successfully")
        return self._restorer

    def _load_occluders(self) -> list:
        if self._occluders is not None:
            return self._occluders
        with self._occluders_lock:
            if self._occluders is None:
                missing = [str(path) for path in OCCLUDER_PATHS if not path.exists()]
                if missing:
                    raise FileNotFoundError(
                        f"Occluder models not found: {', '.join(missing)}. "
                        f"Run: python scripts/download_face_swap_models.py"
                    )
                # torch has to load its cuDNN before onnxruntime's CUDA provider
                # does: the other way round, torch fails to load cudnn_cnn64_9.dll
                import torch  # noqa: F401

                logger.info("Loading XSeg occluders")
                import onnxruntime
                self._occluders = [
                    onnxruntime.InferenceSession(
                        str(path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    )
                    for path in OCCLUDER_PATHS
                ]
                logger.info("XSeg occluders loaded successfully")
        return self._occluders

    # ------------------------------------------------------------------
    # Public primitives
    # ------------------------------------------------------------------

    def detect_faces(self, image: np.ndarray) -> list:
        """Detect faces in an image using the same InsightFace app used by
        the analysis pipeline. Returns a list of insightface Face objects
        (each carries .bbox, .embedding, .pose, etc.)."""
        # Reuse the already-warm app from face_extraction
        from face_extraction import get_face_app
        app = get_face_app()
        return list(app.get(image))

    def swap_one(self, image: np.ndarray, target_face, source_face) -> np.ndarray:
        """Swap a single face in-place. Both target_face and source_face
        must be insightface Face objects (with .embedding). Returns the
        full image with the swap pasted back in."""
        swapper = self._load_swapper()
        return swapper.get(image, target_face, source_face, paste_back=True)

    def restore(self, image: np.ndarray) -> np.ndarray:
        """Run GFPGAN restoration over every face in the image.

        InSwapper-128 produces a 128px-quality crop pasted back into the
        full-resolution image. GFPGAN restores those pixels back to a
        clean, high-res, photoreal face. Without this step the swap looks
        blocky on close-ups.
        """
        restorer = self._load_restorer()
        _, _, restored = restorer.enhance(
            image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        return restored if restored is not None else image

    def occlusion(self, image: np.ndarray, face) -> FaceOcclusion:
        """Where `face` (an insightface Face with .kps and .landmark_3d_68)
        shows in `image`. Whatever any of the three occluders sees covering
        it counts as covered."""
        size = OCCLUDER_CROP_SIZE
        matrix = cv2.estimateAffinePartial2D(
            face.kps, ARCFACE_128_TEMPLATE * size, method=cv2.RANSAC, ransacReprojThreshold=100
        )[0]
        crop = cv2.warpAffine(image, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
        batch = crop[None].astype(np.float32) / 255
        visible = np.minimum.reduce([
            session.run(None, {'input': batch})[0][0, :, :, 0] for session in self._load_occluders()
        ]).clip(0, 1)

        landmarks = cv2.transform(face.landmark_3d_68[None, :, :2].astype(np.float32), matrix)[0]
        face_area = np.zeros((size, size), np.uint8)
        cv2.fillConvexPoly(face_area, cv2.convexHull(landmarks.astype(np.int32)), 1)
        covered = float((visible[face_area.astype(bool)] < 0.5).mean())

        crop_mask = np.minimum(_crop_border_mask(size), _paste_edge(visible))
        mask = cv2.warpAffine(
            crop_mask, cv2.invertAffineTransform(matrix), (image.shape[1], image.shape[0]), borderValue=0
        )
        return FaceOcclusion(mask=mask, covered=covered)

    # ------------------------------------------------------------------
    # High-level pipeline (single image, multiple swaps + restoration)
    # ------------------------------------------------------------------

    def apply(
        self,
        target_image: np.ndarray,
        swap_pairs: Sequence[tuple],
        restore_faces: bool = False,
    ) -> tuple[np.ndarray, list[float]]:
        """Apply N face swaps, a single restoration pass, then put back
        whatever covered each face.

        Args:
            target_image: the thumbnail to operate on, as a BGR numpy array
                (OpenCV convention) so it matches what insightface expects.
            swap_pairs: iterable of (target_face, source_face) insightface
                Face objects. Each pair replaces target_face's region with
                source_face's identity.
            restore_faces: run the optional GFPGAN pass only when requested.
                With False, the restorer is not loaded or required.

        Returns:
            The processed image as a numpy array, and per pair the share of
            its face that something covered. Only the swapped faces change:
            the rest of the image, other faces GFPGAN restores included,
            stays as it was.
        """
        if not swap_pairs:
            return target_image, []
        # What covers each face is read before anything is swapped
        occlusions = [self.occlusion(target_image, target_face) for target_face, _ in swap_pairs]
        out = target_image
        for target_face, source_face in swap_pairs:
            out = self.swap_one(out, target_face, source_face)
        if restore_faces:
            out = self.restore(out)
        return paste_visible_faces(target_image, out, occlusions), [o.covered for o in occlusions]


# =============================================================================
# CONVENIENCE: ARE MODELS INSTALLED?
# =============================================================================


def _missing_requirements(paths, packages) -> tuple[bool, list[str]]:
    """Check availability without importing inference libraries or loading models."""
    missing = [path.relative_to(PROJECT_DIR).as_posix() for path in paths if not path.exists()]
    missing += [f"Python package '{package}'" for package in packages
                if importlib.util.find_spec(package) is None]
    return not missing, missing


def face_swap_available() -> tuple[bool, list[str]]:
    """Only the dependencies needed for the swap and occlusion masks."""
    return _missing_requirements((INSWAPPER_PATH, *OCCLUDER_PATHS), ('insightface',))


def face_restoration_available() -> tuple[bool, list[str]]:
    """The optional restorer's dependencies, checked independently of face swap."""
    return _missing_requirements((GFPGAN_PATH,), ('gfpgan',))

"""Color and lighting groups weighted by the video time each sample represents."""

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import orjson
from sklearn.cluster import AgglomerativeClustering

from utils import read_image


@dataclass(frozen=True)
class VisualContext:
    names: tuple[str, ...]
    groups: np.ndarray
    seconds: np.ndarray
    descriptors: np.ndarray

    def durations(self, names) -> dict[str, float]:
        """Group support within these frames, counting each interval once."""
        members = set(names)
        totals = defaultdict(float)
        for name, group, seconds in zip(self.names, self.groups, self.seconds):
            if name in members:
                totals[int(group)] += float(seconds)
        return {name: totals[int(group)] for name, group in zip(self.names, self.groups) if name in members}

    def main_setting(self) -> list:
        """Frames of the setting that lasts longest in the video."""
        totals = np.bincount(self.groups, weights=self.seconds)
        return [self.names[i] for i in np.flatnonzero(self.groups == int(totals.argmax()))]


def color_descriptor(image: np.ndarray) -> np.ndarray:
    """Lab distributions and a coarse layout, without a learned scene model.

    Cumulative histograms tolerate small color shifts. Lightness has double
    weight; a 2x2 layout helps distinguish settings with similar palettes.
    Manhattan distance measures a normalized color/layout difference (0..1).
    """
    small = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    parts = []
    for channel, weight in enumerate((0.5, 0.25, 0.25)):
        counts, _ = np.histogram(lab[:, :, channel], bins=16, range=(0, 1))
        parts.append(np.cumsum(counts) / counts.sum() * (0.7 * weight / 16))
        layout = lab[:, :, channel].reshape(2, 18, 2, 32).mean(axis=(1, 3)).ravel()
        parts.append(layout * (0.3 * weight / 4))
    return np.concatenate(parts)


def load_visual_context(frames_dir: Path) -> VisualContext:
    scenes_file = frames_dir.parent / "scenes.json"
    if not scenes_file.is_file():
        raise ValueError("Scene timing is missing. Reanalyze the video before automatic selection.")
    files = tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size)
                  for path in sorted(frames_dir.glob("*.jpg")))
    return _load_context(str(frames_dir), scenes_file.stat().st_mtime_ns, files)


@lru_cache(maxsize=2)
def _load_context(directory: str, scene_mtime: int, files: tuple) -> VisualContext:
    frames_dir = Path(directory)
    scenes = orjson.loads((frames_dir.parent / "scenes.json").read_bytes())["scenes"]
    weights = {}
    for scene in scenes:
        indices = np.asarray(scene["frame_indices"], dtype=float)
        if not len(indices) or scene["end_frame"] <= scene["start_frame"] or scene["duration"] <= 0:
            raise ValueError("Invalid scene timing. Reanalyze the video.")
        # Midpoints allocate the entire scene, including its boundaries.
        boundaries = np.r_[scene["start_frame"], (indices[:-1] + indices[1:]) / 2, scene["end_frame"]]
        seconds = np.diff(boundaries) * scene["duration"] / (scene["end_frame"] - scene["start_frame"])
        if np.any(seconds <= 0):
            raise ValueError("Scene samples must be ordered and inside their scene.")
        for number, duration in zip(indices, seconds):
            weights[f"scene_{scene['index']:03d}_frame_{int(number):06d}.jpg"] = float(duration)

    names = tuple(name for name, _, _ in files)
    # Manually captured frames may compete on quality, but add no invented time.
    seconds = np.asarray([weights.get(name, 0.0) for name in names])
    if not names or seconds.sum() <= 0:
        raise ValueError("No timed video samples are available for automatic selection.")
    descriptors = np.asarray([color_descriptor(read_image(frames_dir / name)) for name in names])
    # Complete linkage avoids chains of gradual changes merging unrelated
    # settings. There is no fixed number of settings imposed on a video.
    groups = (AgglomerativeClustering(n_clusters=None, distance_threshold=0.10,
                                     metric="manhattan", linkage="complete").fit_predict(descriptors)
              if len(names) > 1 else np.zeros(1, dtype=int))
    return VisualContext(names, groups, seconds, descriptors)


# Low resolution at which setting photos are compared (width, height)
_SETTING_GRID = (96, 54)
# Share of a frame's free setting that differs from the setting's usual look
# above which something is laid over it (a title, subtitles, a graphic): clean
# frames measure 0-0.01, a short title 0.03
_OVERLAY_SHARE = 0.015


def _people_mask(boxes, width, height):
    """Where people cover the frame on the setting grid: each face, its hair and the body below it."""
    mask = np.zeros(_SETTING_GRID[::-1], dtype=bool)
    scale_x, scale_y = _SETTING_GRID[0] / width, _SETTING_GRID[1] / height
    for x1, y1, x2, y2 in boxes:
        face_width, face_height = x2 - x1, y2 - y1
        left = max(0, int((x1 - 0.9 * face_width) * scale_x))
        right = min(_SETTING_GRID[0], int(np.ceil((x2 + 0.9 * face_width) * scale_x)))
        top = max(0, int((y1 - 0.6 * face_height) * scale_y))
        mask[top:, left:right] = True
    return mask


def setting_frame(frames_dir: Path) -> str:
    """The frame that shows the main setting best.

    From the setting that lasts longest: the frame where people cover least
    of it, without what its other frames lack in the same place (subtitles,
    titles, graphics, someone passing), sharp, in colour and typical of it.
    """
    from services.face_index import load_face_index

    context = load_visual_context(frames_dir)
    index = load_face_index(frames_dir.parent / "faces.json")
    names = context.main_setting()
    images, masks, sharpness, colour = [], [], [], []
    for name in names:
        image = read_image(frames_dir / name)
        height, width = image.shape[:2]
        small = cv2.resize(image, _SETTING_GRID, interpolation=cv2.INTER_AREA).astype(np.float32)
        images.append(small)
        span = index.spans.get(name)
        masks.append(_people_mask(index.boxes[slice(*span)].tolist() if span else [], width, height))
        gray = cv2.cvtColor(cv2.resize(image, (640, round(640 * height / width)), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        saturation = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_BGR2HSV)[:, :, 1]
        colour.append(float(saturation.mean()) >= 12)
    images, masks = np.asarray(images), np.asarray(masks)
    plate = np.median(images, axis=0)
    # Where people usually are, a frame differs from the median by showing the
    # setting behind them: only the rest of the frame can reveal overlays
    usual_people = masks.mean(axis=0) > 0.3
    sharp_reference = float(np.median(sharpness)) or 1.0
    members = [context.names.index(name) for name in names]
    seconds = context.seconds[members]
    distances = np.abs(context.descriptors[members][:, None, :] - context.descriptors[members][None, :, :]).sum(axis=2)
    typical = distances @ seconds / max(float(seconds.sum()), 1e-9)
    typical = typical / (float(typical.max()) or 1.0)

    def overlay(i):
        free = ~(masks[i] | usual_people)
        difference = np.abs(images[i] - plate).max(axis=2)
        return float((difference[free] > 40).mean()) if free.any() else 0.0

    def score(i):
        return (1.0 - float(masks[i].mean()) - 2.0 * overlay(i)
                + 0.3 * min(1.5, sharpness[i] / sharp_reference) / 1.5 - 0.3 * float(typical[i]))

    ranked = sorted(range(len(names)), key=lambda i: (not colour[i], overlay(i) > _OVERLAY_SHARE, -score(i), names[i]))
    return names[ranked[0]]

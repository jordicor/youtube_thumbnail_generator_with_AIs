"""Local body references: visible pixels, pose, and conservative face association."""
import hashlib
import math
import os
import threading
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import orjson

from services.face_index import (
    MIN_IDENTITY_SIMILARITY,
    load_face_index,
)
from utils import read_image

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_SHA256 = "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pose" / "pose_landmarker_full.task"
SEGMENTATION_URL = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
SEGMENTATION_SHA256 = "bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0"
SEGMENTATION_PATH = MODEL_PATH.parents[1] / "body_segmentation" / "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
CACHE_NAME = "body_references.json"
CACHE_VERSION = 6
COVERAGE_ORDER = {"portrait": 0, "torso": 1, "three_quarter": 2, "full_body": 3}
_analysis_lock = threading.Lock()


def download_model():
    """Explicit setup; analysis never downloads weights or substitutes another model."""
    for path, url, checksum in ((MODEL_PATH, MODEL_URL, MODEL_SHA256),
                                (SEGMENTATION_PATH, SEGMENTATION_URL, SEGMENTATION_SHA256)):
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Body model checksum mismatch: {path.name}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f"Downloaded body model checksum mismatch: {path.name}")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
    return MODEL_PATH, SEGMENTATION_PATH


def _fingerprint(output_dir):
    faces = output_dir / "faces.json"
    if not faces.is_file():
        raise ValueError("Analyze faces before analyzing body references.")
    frames = sorted((output_dir / "frames").glob("*.jpg"))
    if not frames:
        raise ValueError("No extracted frames are available for body analysis.")
    inputs = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in [faces, *frames]]
    return hashlib.sha256(orjson.dumps([CACHE_VERSION, MODEL_SHA256, SEGMENTATION_SHA256, inputs])).hexdigest(), frames


def load_body_index(output_dir):
    """Only current measurements may be used for automatic selection."""
    path = output_dir / CACHE_NAME
    if not path.exists():
        return None
    data = orjson.loads(path.read_bytes())
    fingerprint, _ = _fingerprint(output_dir)
    return data if data.get("fingerprint") == fingerprint else None


def _visible(point):
    return (float(point.visibility or 0) >= 0.5 and float(point.presence or 0) >= 0.5
            and 0 <= point.x < 1 and 0 <= point.y < 1)


def _angle(a, b, c):
    u, v = np.asarray(a) - b, np.asarray(c) - b
    length = np.linalg.norm(u) * np.linalg.norm(v)
    if length < 1e-8:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / length, -1, 1))))


def body_view(points, world, width, height):
    """Torso orientation, independent of head yaw. Back views are assigned manually."""
    shoulder = np.array([world[12].x - world[11].x, world[12].z - world[11].z])
    hip = np.array([world[24].x - world[23].x, world[24].z - world[23].z])
    if min(np.linalg.norm(shoulder), np.linalg.norm(hip)) < 0.03:
        return None
    angles = [math.degrees(math.atan2(abs(axis[1]), abs(axis[0]))) for axis in (shoulder, hip)]
    if abs(angles[0] - angles[1]) > 35:
        return None  # A twisted/uncertain torso is not a reliable standard view.
    angle = sum(angles) / 2
    xy = np.asarray([[p.x * width, p.y * height] for p in points])
    axis = (xy[23] + xy[24] - xy[11] - xy[12]) / 2
    shoulder_xy = xy[12] - xy[11]
    projected_width = abs(shoulder_xy[0] * axis[1] - shoulder_xy[1] * axis[0]) / max(1, float(axis @ axis))
    if angle >= 60 or (angle >= 35 and projected_width < 0.35):
        return "profile"
    return "front" if angle <= 30 else "three_quarter"


def _mask_support(point, mask):
    height, width = mask.shape
    x, y = int(point.x * width), int(point.y * height)
    if not (0 <= x < width and 0 <= y < height):
        return 0.0
    return float(mask[max(0, y - 4):min(height, y + 5), max(0, x - 4):min(width, x + 5)].mean())


def pose_description(points, world, mask, face_height=None):
    """Visible coverage needs pixel evidence; inferred joints behind objects do not count."""
    height, width = mask.shape
    visible = {i for i, point in enumerate(points) if _visible(point) and _mask_support(point, mask) >= 0.25}
    torso = np.zeros(mask.shape, dtype=np.uint8)
    polygon = np.asarray([[p.x * width, p.y * height] for p in (points[11], points[12], points[24], points[23])], dtype=np.int32)
    area = cv2.contourArea(polygon)
    cv2.fillConvexPoly(torso, polygon, 1)
    torso = torso.astype(bool)
    torso_visible = min(1.0, float(np.count_nonzero(mask & torso)) / max(1.0, area))
    view = body_view(points, world, width, height)
    xy = np.asarray([[p.x * width, p.y * height] for p in points])
    axis = (xy[23] + xy[24] - xy[11] - xy[12]) / 2
    torso_length = max(1, float(np.linalg.norm(axis)))
    # A predicted knee inside the shirt is not visible leg evidence.
    for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
        thigh = xy[knee] - xy[hip]
        knee_x, knee_y = xy[knee].astype(int)
        inside_torso = 0 <= knee_x < width and 0 <= knee_y < height and torso[knee_y, knee_x]
        if inside_torso or (face_height is not None and np.linalg.norm(thigh) < .75 * face_height):
            visible.difference_update({knee, ankle, ankle + 2, ankle + 4})
            continue
        if face_height is not None and np.linalg.norm(xy[ankle] - xy[knee]) < .6 * face_height:
            visible.difference_update({ankle, ankle + 2, ankle + 4})
        # A transverse thigh needs a visible lower leg: otherwise a hand on a
        # desk can masquerade as a knee despite a confident pose prediction.
        unsupported_bend = thigh @ axis < 0.35 * torso_length ** 2 and ankle not in visible
        if unsupported_bend or np.linalg.norm(thigh) < 0.35 * torso_length or thigh @ axis < -0.1 * torso_length ** 2:
            visible.difference_update({knee, ankle, ankle + 2, ankle + 4})
        elif np.linalg.norm(xy[ankle] - xy[knee]) < 0.35 * torso_length:
            visible.difference_update({ankle, ankle + 2, ankle + 4})
    coverage = "portrait"
    if visible & {11, 12} and torso_visible >= 0.45:
        coverage = "torso"
        if visible & {25, 26}:
            coverage = "three_quarter"
            left = {25, 27} <= visible and bool(visible & {29, 31})
            right = {26, 28} <= visible and bool(visible & {30, 32})
            if (left and right) or (view == "profile" and (left or right)):
                # Feet at an image boundary are clipped, even if their joints are predicted.
                feet = [points[i] for i in visible & {29, 30, 31, 32}]
                if feet and all(0.01 < p.x < 0.99 and 0.01 < p.y < 0.99 for p in feet):
                    coverage = "full_body"
    posture = "unknown"
    if {11, 12, 23, 24, 25, 26, 27, 28} <= visible:
        shoulder = np.mean([[points[i].x * width, points[i].y * height] for i in (11, 12)], axis=0)
        hip = np.mean([[points[i].x * width, points[i].y * height] for i in (23, 24)], axis=0)
        axis = shoulder - hip
        xyz = [[p.x, p.y, p.z] for p in world]
        hips = [_angle(xyz[s], xyz[h], xyz[k]) for s, h, k in ((11, 23, 25), (12, 24, 26))]
        knees = [_angle(xyz[h], xyz[k], xyz[a]) for h, k, a in ((23, 25, 27), (24, 26, 28))]
        if all(a is not None for a in hips + knees):
            if abs(axis[0]) > abs(axis[1]) * 1.5:
                posture = "reclined"
            elif min(hips) > 150 and min(knees) > 150:
                posture = "standing"
            elif max(hips) < 130 and min(knees) > 65 and max(knees) < 145:
                posture = "seated"
    return coverage, posture, view, torso, torso_visible


def segment_bodies(frame, model, device):
    """Separate person pixels from foreground objects using the installed local model."""
    import torch

    tensor = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).to(device, dtype=torch.float32) / 255
    with torch.inference_mode():
        prediction = model([tensor])[0]
    keep = prediction["scores"] >= 0.6
    labels = prediction["labels"][keep].cpu().numpy()
    boxes = prediction["boxes"][keep].cpu().numpy()
    scores = prediction["scores"][keep].cpu().numpy()
    masks = (prediction["masks"][keep, 0] > 0.5).cpu().numpy()
    people = [{"box": box, "score": float(score), "mask": mask}
              for box, score, label, mask in zip(boxes, scores, labels, masks) if label == 1]
    obstacles = np.any(masks[labels != 1], axis=0) if np.any(labels != 1) else np.zeros(frame.shape[:2], dtype=bool)
    return people, obstacles



def body_sharpness(frame, mask, box):
    """Compare visible person detail at a common height, excluding background edges."""
    left, top, right, bottom = box
    crop = frame[top:bottom, left:right]
    size = (max(1, round(512 * crop.shape[1] / crop.shape[0])), 512)
    gray = cv2.cvtColor(cv2.resize(crop, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    visible = cv2.resize(mask[top:bottom, left:right].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    inner = cv2.erode(visible, np.ones((3, 3), np.uint8)).astype(bool)
    return float(cv2.Laplacian(gray, cv2.CV_64F)[inner].var()) if inner.any() else 0.0

def associate_poses(result, face_boxes, frame, segments, obstacles, portraits):
    """Associate a pose and person mask with exactly one indexed face."""
    height, width = frame.shape[:2]
    records = []
    for points, world in zip(result.pose_landmarks, result.pose_world_landmarks):
        anchors = [points[i] for i in (0, 2, 5) if _visible(points[i])]
        matches = [idx for idx, (x1, y1, x2, y2) in enumerate(face_boxes)
                   if len(anchors) >= 2 and all(x1 <= p.x * width <= x2 and y1 <= p.y * height <= y2 for p in anchors)]
        record = {"status": "unmatched", "face_box": None}
        if len(matches) != 1:
            record["status"] = "ambiguous" if matches else "unmatched"
            records.append(record)
            continue
        face_id = matches[0]
        face_box = [round(float(value), 1) for value in face_boxes[face_id]]
        record["face_box"] = face_box
        x1, y1, x2, y2 = face_box
        masks = [segment for segment in segments
                 if (min(x2, segment["box"][2]) - max(x1, segment["box"][0])) > 0
                 and (max(0, min(x2, segment["box"][2]) - max(x1, segment["box"][0]))
                      * max(0, min(y2, segment["box"][3]) - max(y1, segment["box"][1]))) >= 0.8 * (x2 - x1) * (y2 - y1)]
        if not masks:
            record["status"] = "unsegmented"
            records.append(record)
            continue
        # Duplicate instance detections sometimes overlap; retain the strongest head-linked mask.
        mask = max(masks, key=lambda item: item["score"])["mask"] & ~obstacles
        ys, xs = np.where(mask)
        if not len(xs):
            record["status"] = "empty_mask"
            records.append(record)
            continue
        margin = max(12, (x2 - x1) * 0.2)
        box = [int(max(0, min(x1, xs.min()) - margin)), int(max(0, min(y1, ys.min()) - margin)),
               int(min(width, max(x2, xs.max() + 1) + margin)), int(min(height, max(y2, ys.max() + 1) + margin))]
        coverage, posture, view, torso, torso_visible = pose_description(points, world, mask, y2 - y1)
        if coverage == "full_body" and ys.max() >= height - 3:
            coverage = "three_quarter"
        obstruction = float(np.count_nonzero(torso & obstacles & ~mask) / max(1, np.count_nonzero(torso)))
        sharpness = body_sharpness(frame, mask, box)
        # Coverage is supported by pixels, and never overrides a substantial obstruction.
        quality = (40 * COVERAGE_ORDER[coverage] / 3 + 30 * torso_visible
                   + 15 * min(1, sharpness / 150) + 15 * min(1, (box[3] - box[1]) / 700)
                   - 60 * obstruction)
        portrait = portraits[face_id]
        if portrait is None:
            raise ValueError("Analyze faces again before selecting body references.")
        # Reuse existing face measurements as a secondary signal. Side views may
        # have poor portrait scores while still being excellent body evidence.
        quality += 0.05 * portrait["score"]
        # Face occlusion estimates are less reliable on a side view.
        occlusion_tolerance = 0.35 if view == "profile" else 0
        quality -= 20 * max(0, portrait["occlusion"] - occlusion_tolerance)
        quality -= 10 * bool(set(portrait["issues"]) & {"cropped", "closed_eyes", "extreme_expression"})
        quality -= 8 * (ys.min() <= 1)
        record.update(coverage=coverage, posture=posture, view=view, box=box, status="matched",
                      quality=round(float(max(0, quality)), 2), obstruction=round(obstruction, 4),
                      torso_visible=round(torso_visible, 4), sharpness=round(sharpness, 2))
        if coverage == "portrait":
            record["status"] = "insufficient_coverage"
        elif box[3] - box[1] < 320 or box[2] - box[0] < 140:
            record["status"] = "too_small"
        elif sharpness < 20 or "blurred" in portrait["issues"]:
            record["status"] = "blurred"
        for other_id, other in enumerate(face_boxes):
            if other_id != face_id and min(box[2], other[2]) > max(box[0], other[0]) and min(box[3], other[3]) > max(box[1], other[1]):
                record["status"] = "other_person_in_crop"
        records.append(record)
    counts = Counter(tuple(r["face_box"]) for r in records if r["face_box"] is not None)
    for record in records:
        if record["face_box"] and counts[tuple(record["face_box"])] > 1:
            record["status"] = "ambiguous"
    return records


def ensure_body_index(output_dir, cancellation_check=None, force=False):
    """Analyze once locally and cache measurements, not model-dependent photo choices."""
    output_dir = Path(output_dir)
    with _analysis_lock:
        cached = load_body_index(output_dir)
        if cached is not None and not force:
            return cached
        fingerprint, frames = _fingerprint(output_dir)
        for path, checksum in ((MODEL_PATH, MODEL_SHA256), (SEGMENTATION_PATH, SEGMENTATION_SHA256)):
            if not path.is_file():
                raise RuntimeError("Body model missing. Run: python scripts/download_body_model.py")
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise RuntimeError(f"Body model checksum mismatch: {path.name}")
        import mediapipe as mp
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        segmenter = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None).eval()
        segmenter.load_state_dict(torch.load(SEGMENTATION_PATH, map_location="cpu", weights_only=True))
        segmenter.to(device)
        index = load_face_index(output_dir / "faces.json")
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=8,
            min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5)
        analyzed = {}
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            for path in frames:
                if cancellation_check and cancellation_check():
                    raise InterruptedError("Body reference analysis cancelled.")
                span = index.spans.get(path.name)
                if not span:
                    analyzed[path.name] = []
                    continue  # Back views have no automatic identity; they can be assigned by hand.
                frame = read_image(path)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = detector.detect(image)
                if not result.pose_landmarks:
                    analyzed[path.name] = []
                    continue
                segments, obstacles = segment_bodies(frame, segmenter, device)
                analyzed[path.name] = associate_poses(result, index.boxes[slice(*span)], frame, segments, obstacles, index.portraits[slice(*span)])
        if _fingerprint(output_dir)[0] != fingerprint:
            raise RuntimeError("Frames changed during body analysis. Analyze the current frames again.")
        data = {"fingerprint": fingerprint, "frames": analyzed, "version": CACHE_VERSION}
        path = output_dir / CACHE_NAME
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_bytes(orjson.dumps(data))
        os.replace(temporary, path)
        return data


def person_body_candidates(data, matches):
    """Rank visible body information; a coverage label alone cannot win."""
    candidates, excluded = [], Counter()
    if data is None:
        return {"status": "not_analyzed", "candidates": [], "excluded": {}}
    for name, match in matches.items():
        if match.similarity is None or match.similarity < MIN_IDENTITY_SIMILARITY:
            excluded["uncertain_identity"] += 1
            continue
        belonging = [r for r in data["frames"].get(name, []) if r["face_box"] == match.box]
        if not belonging:
            excluded["unmatched"] += 1
        for record in belonging:
            if record["status"] != "matched":
                excluded[record["status"]] += 1
                continue
            # A head-and-shoulders crop cannot establish torso proportions,
            # even when the pose model predicts hips inside the visible shirt.
            if record["box"][3] - match.box[3] < 2 * (match.box[3] - match.box[1]):
                excluded["insufficient_coverage"] += 1
                continue
            candidates.append({"filename": name, **record})
    candidates.sort(key=lambda candidate: (-candidate["quality"], candidate["filename"]))
    return {"status": "ready" if candidates else "unavailable", "candidates": candidates, "excluded": dict(excluded)}


def select_body_references(frames_dir, candidates):
    """Best usable body, then complementary orientations; never invent a back view."""
    from services.face_index import is_grayscale_face
    from services.visual_context import load_visual_context

    if not candidates:
        return []
    context = load_visual_context(frames_dir)
    support = context.durations(context.names)
    longest = max(support.values(), default=1.0)
    ranked = sorted(candidates, key=lambda candidate: (
        -(candidate["quality"] + 3 * support.get(candidate["filename"], 0) / max(1, longest)), candidate["filename"]))
    selected, views = [], set()
    for candidate in ranked:
        view = "front" if candidate["view"] == "three_quarter" else candidate["view"]
        if selected and (view is None or view in views):
            continue
        if is_grayscale_face(frames_dir / candidate["filename"], candidate["face_box"]):
            continue
        selected.append(candidate)
        views.add(view)
        if len(selected) == 3:
            break
    return selected

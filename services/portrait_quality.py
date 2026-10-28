"""Local expression, look (hair, skin, clothes) and short-window portrait measurements."""
from collections import defaultdict
from pathlib import Path
import hashlib
import os
import re
import threading

import cv2
import numpy as np
import orjson

from utils import read_image

VERSION = 3
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "face_landmarker/face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
SEGMENTER_PATH = MODEL_DIR / "face_segmenter/selfie_multiclass_256x256.tflite"
SEGMENTER_SHA256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"
# Classes of the multiclass selfie segmenter
HAIR, FACE_SKIN, CLOTHES = 1, 3, 4
# Faces whose expression is checked against the video frames around them, per
# scene and person: the best ones, and the best looking to each side
TEMPORAL_BEST, TEMPORAL_PER_SIDE = 3, 1
_lock = threading.Lock()


def measure_expression(frame, box, detector):
    """Blendshapes and lip gap of the face in box, None when the landmarker finds no face."""
    import mediapipe as mp

    x, y, right, bottom = box
    width, height = right - x, bottom - y
    crop = frame[max(0, int(y - .2 * height)):min(frame.shape[0], int(bottom + .15 * height)),
                 max(0, int(x - .25 * width)):min(frame.shape[1], int(right + .25 * width))]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    result = detector.detect(image)
    if not result.face_blendshapes:
        return None
    scores = {value.category_name: float(value.score) for value in result.face_blendshapes[0]}
    points = np.asarray([[point.x * crop.shape[1], point.y * crop.shape[0]] for point in result.face_landmarks[0]])
    mouth_width = np.linalg.norm(points[61] - points[291])
    if mouth_width < 2:
        return None
    lip_gap = float(np.linalg.norm(points[13] - points[14]) / mouth_width)
    return {"lip_gap": round(lip_gap, 4), "blendshapes": {key: round(value, 4) for key, value in scores.items()},
            "motion": [lip_gap, scores["mouthPucker"], scores["mouthFunnel"],
                       max(scores["mouthPressLeft"], scores["mouthPressRight"]),
                       max(scores["eyeBlinkLeft"], scores["eyeBlinkRight"]),
                       max(scores["eyeWideLeft"], scores["eyeWideRight"])]}


def head_region(box, width, height):
    """Head, hair and shoulders around a face box, kept inside the frame."""
    x1, y1, x2, y2 = box
    face_width, face_height = x2 - x1, y2 - y1
    center = (x1 + x2) / 2
    return (int(max(0, center - 1.3 * face_width)), int(max(0, y1 - 0.9 * face_height)),
            int(min(width, center + 1.3 * face_width)), int(min(height, y2 + 1.4 * face_height)))


def measure_look(frame, box, segmenter):
    """Median Lab colour (OpenCV scale) of the hair, face skin and clothes around a face.

    Each part is None when too little of it shows. hair_share is the part of
    the head region the hair covers, face_skin the part of the face box that
    is face skin: hands, cream, masks or big glasses over the face lower it.
    """
    import mediapipe as mp

    left, top, right, bottom = head_region(box, frame.shape[1], frame.shape[0])
    crop = frame[top:bottom, left:right]
    if min(crop.shape[:2]) < 16:
        return None
    result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    mask = result.category_mask.numpy_view().reshape(crop.shape[:2])
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    look = {}
    for name, category in (("hair", HAIR), ("skin", FACE_SKIN), ("clothes", CLOTHES)):
        pixels = lab[mask == category]
        look[name] = np.median(pixels, axis=0).round(1).tolist() if len(pixels) >= max(50, 0.01 * mask.size) else None
    look["hair_share"] = round(float(np.mean(mask == HAIR)), 4)
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    face = mask[max(0, y1 - top):max(0, y2 - top), max(0, x1 - left):max(0, x2 - left)]
    look["face_skin"] = round(float(np.mean(face == FACE_SKIN)), 4) if face.size else None
    return look


def _check_model(path, checksum):
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise RuntimeError(f"{path.name} missing or invalid. Run: python scripts/download_portrait_model.py")


def _same_people(faces, threshold=0.45):
    """Group faces of one scene by identity (greedy, on their embeddings)."""
    groups = []
    for face in faces:
        embedding = np.asarray(face["embedding"], dtype=np.float32)
        embedding /= np.linalg.norm(embedding) or 1
        for group in groups:
            if float(group[0] @ embedding) >= threshold:
                group[1].append(face)
                break
        else:
            groups.append((embedding, [face]))
    return [members for _, members in groups]


def _temporal_shortlist(faces):
    """Faces worth checking against nearby frames, with their frame number."""
    from services.face_selection import assess, is_grayscale_look

    by_scene = defaultdict(list)
    for face in faces:
        match = re.fullmatch(r"scene_(\d+)_frame_(\d+)\.jpg", Path(face["frame_path"]).name)
        if match and face.get("embedding") and face["portrait_quality"].get("expression_details"):
            by_scene[int(match[1])].append((face, int(match[2])))
    shortlist = []
    for members in by_scene.values():
        numbers = {id(face): number for face, number in members}
        for group in _same_people([face for face, _ in members]):
            ranked = sorted(group, key=lambda face: _rank(assess(
                face["portrait_quality"], bool(is_grayscale_look(face["portrait_quality"])))))
            chosen = ranked[:TEMPORAL_BEST]
            for side in (-1, 1):
                chosen += [face for face in ranked if side * face["pose"][1] > 12 and face not in chosen][:TEMPORAL_PER_SIDE]
            shortlist += [(face, numbers[id(face)]) for face in chosen]
    return shortlist


def _rank(assessment):
    return assessment.level, -assessment.score


def enrich_portraits(output_dir, video_path, cancellation_check=None):
    """Reuse identity detections; add expression, look and temporal measurements."""
    import mediapipe as mp
    from services.face_selection import expression_label

    output_dir, video_path = Path(output_dir), Path(video_path)
    path = output_dir / "faces.json"
    with _lock:
        original = path.read_bytes()
        data = orjson.loads(original)
        faces = data["all_faces"]
        if data.get("portrait_reference_version") == VERSION:
            return
        _check_model(MODEL_PATH, MODEL_SHA256)
        _check_model(SEGMENTER_PATH, SEGMENTER_SHA256)
        scenes = {s["index"]: s for s in orjson.loads((output_dir / "scenes.json").read_bytes())["scenes"]}
        landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            num_faces=1, output_face_blendshapes=True)
        segmenter_options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(SEGMENTER_PATH)), output_category_mask=True)
        grouped = defaultdict(list)
        for face in faces:
            grouped[Path(face["frame_path"]).name].append(face)
        with mp.tasks.vision.FaceLandmarker.create_from_options(landmarker_options) as detector, \
                mp.tasks.vision.ImageSegmenter.create_from_options(segmenter_options) as segmenter:
            for name, members in grouped.items():
                if cancellation_check and cancellation_check():
                    raise InterruptedError("Portrait analysis cancelled.")
                frame = read_image(output_dir / "frames" / name)
                for face in members:
                    portrait = face["portrait_quality"]
                    # Issues of the face detection; expression problems are judged at selection
                    portrait["issues"] = list(portrait.pop("base_issues", portrait["issues"]))
                    for stale in ("appearance", "reference_score", "temporal_checked", "temporal_change"):
                        portrait.pop(stale, None)
                    details = measure_expression(frame, face["bbox"], detector)
                    portrait["expression_details"] = details
                    portrait["look"] = measure_look(frame, face["bbox"], segmenter)
                    portrait["reference_version"] = VERSION
                    face["expression"] = expression_label(details)
            # Original neighboring frames stay inside the same cut. Only the
            # best candidates incur video seeks and additional inference.
            capture = cv2.VideoCapture(str(video_path))
            try:
                if not capture.isOpened():
                    raise RuntimeError(f"Cannot read neighboring portrait frames from {video_path.name}")
                for face, number in sorted(_temporal_shortlist(faces), key=lambda item: item[1]):
                    if cancellation_check and cancellation_check():
                        raise InterruptedError("Portrait analysis cancelled.")
                    scene = scenes[int(re.search(r"scene_(\d+)", face["frame_path"])[1])]
                    portrait = face["portrait_quality"]
                    offsets = [number + delta for delta in (-3, 3)
                               if scene["start_frame"] <= number + delta < scene["end_frame"]]
                    neighbors = []
                    for offset in offsets:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, offset)
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError(f"Cannot decode portrait neighbor frame {offset}")
                        details = measure_expression(frame, face["bbox"], detector)
                        if details is not None:
                            neighbors.append(details)
                    if len(neighbors) != len(offsets) or not neighbors:
                        continue
                    motion = np.asarray(portrait["expression_details"]["motion"])
                    portrait["temporal_change"] = round(max(float(np.abs(motion - neighbor["motion"]).max())
                                                            for neighbor in neighbors), 4)
            finally:
                capture.release()
        if path.read_bytes() != original:
            raise RuntimeError("Face detections changed during portrait assessment.")
        data["portrait_reference_version"] = VERSION
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_bytes(orjson.dumps(data))
        os.replace(temporary, path)

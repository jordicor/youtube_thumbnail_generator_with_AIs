"""
Face Selection

Chooses the face photos of a person the way a person choosing them would: the
clearest portrait first, then the most different useful views, lowering the bar
step by step instead of giving up. Every face of the person gets a level
(ideal, good, usable, last resort) and a score within it; only a face that is
not the person or cannot be cropped alone is left out.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from config import DEFAULT_FACE_REFERENCE_COUNT
from services.face_index import (
    MIN_IDENTITY_SIMILARITY,
    FaceIndex,
    FaceMatch,
    is_grayscale_face,
    portrait_crop_box,
)

IDEAL, GOOD, USABLE, LAST_RESORT = range(4)

# The extra photos must differ from the chosen ones at least this much (novelty
# points): a few degrees of head turn, another expression or another scene
MIN_NOVELTY = 6.0
# Points an extra photo earns for its level, and how much of its novelty counts:
# a clearly different angle can outweigh one level of quality, but what a poor
# photo shows is worth less
LEVEL_POINTS = {IDEAL: 40.0, GOOD: 28.0, USABLE: 12.0}
NOVELTY_WEIGHT = {IDEAL: 1.0, GOOD: 0.85, USABLE: 0.6}
# Two faces look alike when their hair lightness against the skin and hair colour
# differ less than this (look_vector units)
LOOK_TOLERANCE = 0.25
# Skin lightness (OpenCV Lab, 0-255) under which a face is too dark to tell its look
MIN_LOOK_LIGHTNESS = 60
# A person shows another character (a wig, a disguise) when their hair against
# the skin falls in two groups with an empty stretch between them: the medians
# of the lightness ratio this far apart, and this much between the darker
# group's 90th percentile and the lighter group's 10th; each group with at
# least this many faces and this share of them. Lighting alone spreads the
# ratio without such a gap (at most 0.12 between those percentiles in a survey
# of the library, 0.37 for a black wig over light hair).
CHARACTER_MIN_GAP = 0.35
CHARACTER_MIN_EMPTY = 0.2
CHARACTER_MIN_FACES = 12
CHARACTER_MIN_SHARE = 0.1
# A face that moves less than this share of its width between the frames of a
# scene, in at least this many frames, is a picture (a painting, a poster, a
# photo on a wall): a painting measured 0.001-0.003, about the detector's own
# jitter, while people, even listening still, move more. Being marked only
# lists them last and labelled.
PICTURE_MAX_MOTION = 0.006
PICTURE_MIN_FACES = 6


@dataclass(frozen=True)
class Assessment:
    """How good a face is as a reference: level first (lower is better), then score."""
    level: int
    score: float          # 0-100 within the measurements, higher is better
    notes: tuple          # Problems seen, e.g. ("speaking", "looking_away")
    expression: str       # smiling | mouth_closed | neutral | mouth_open
    grayscale: bool


def _pair(shapes: dict, name: str) -> float:
    return max(shapes[name + "Left"], shapes[name + "Right"])


def _both(shapes: dict, name: str) -> float:
    """Mean of both sides: with the head turned, the far side alone is unreliable."""
    return (shapes[name + "Left"] + shapes[name + "Right"]) / 2


def gaze(shapes: dict) -> tuple:
    """Eye direction relative to the head: (to the person's left, up), each -1..1."""
    left = (shapes["eyeLookOutLeft"] + shapes["eyeLookInRight"]) / 2
    right = (shapes["eyeLookInLeft"] + shapes["eyeLookOutRight"]) / 2
    up = (shapes["eyeLookUpLeft"] + shapes["eyeLookUpRight"]) / 2
    down = (shapes["eyeLookDownLeft"] + shapes["eyeLookDownRight"]) / 2
    return left - right, up - down


def gaze_away(direction, usual=None) -> float:
    """How far the eyes look from where the person usually looks (the camera, mostly)."""
    usual = usual or (0.0, 0.0)
    return max(abs(direction[0] - usual[0]), abs(direction[1] - usual[1]))


def is_grimace(shapes: dict) -> bool:
    """Frowning hard, sneering, scrunching or stretching the mouth: not a usual face.

    A light frown alone is a look of concentration, not a grimace.
    """
    frown = _both(shapes, "browDown")
    return (frown > 0.28 or (frown > 0.18 and (shapes["jawOpen"] > 0.2 or _both(shapes, "eyeSquint") > 0.45))
            or _both(shapes, "noseSneer") > 0.25 or (_both(shapes, "eyeSquint") > 0.6 and shapes["jawOpen"] > 0.25)
            or _both(shapes, "mouthStretch") > 0.3 or _both(shapes, "mouthFrown") > 0.35)


def expression_label(details: Optional[dict]) -> str:
    """Expression of a face from its measurements; a smile needs the rest of the face to agree."""
    if not details:
        return "unknown"
    shapes = details["blendshapes"]
    smile = min(shapes["mouthSmileLeft"], shapes["mouthSmileRight"])
    lopsided = abs(shapes["mouthSmileLeft"] - shapes["mouthSmileRight"])
    if smile >= 0.35 and lopsided < 0.25 and not is_grimace(shapes):
        return "smiling"
    lips = details["lip_gap"]
    return "mouth_closed" if lips <= 0.035 else "neutral" if lips <= 0.08 else "mouth_open"


def look_vector(portrait: dict) -> Optional[np.ndarray]:
    """Hair lightness against the face's skin and hair colour, None without both measured.

    Faces in near darkness (skin lightness under MIN_LOOK_LIGHTNESS) have no
    reliable look: their hair and skin blur into the same shadows.
    """
    look = portrait.get("look")
    if not look or look["hair"] is None or look["skin"] is None or look["skin"][0] < MIN_LOOK_LIGHTNESS:
        return None
    hair, skin = look["hair"], look["skin"]
    return np.array([hair[0] / max(skin[0], 1.0), (hair[1] - 128) / 40, (hair[2] - 128) / 40])


def is_grayscale_look(portrait: dict) -> Optional[bool]:
    """Whether the face is in black and white, from the skin colour measured; None when unknown."""
    look = portrait.get("look")
    if not look or look["skin"] is None:
        return None
    return abs(look["skin"][1] - 128) + abs(look["skin"][2] - 128) < 4


def assess(portrait: dict, grayscale: bool, usual: Optional[dict] = None) -> Assessment:
    """Level and score of a face as a reference photo of its person.

    usual holds the person's medians: how open their eyes are ("eyes"), where
    they look ("gaze") and how much of their face box is skin ("face_skin").
    They tell wide, staring eyes from naturally big ones, a look away from the
    camera from a measuring bias, and a face hidden by cream, a hand or a mask
    from a beard.
    """
    usual = usual or {}
    notes: Dict[str, int] = {}
    penalty = 0.0

    def note(name, severity):
        notes[name] = max(notes.get(name, 0), severity)

    eyes = portrait["eye_openness"]
    sharpness = portrait["sharpness"]
    covered = portrait["occlusion"]
    base = set(portrait["issues"])

    if sharpness < 12:
        note("blurred", LAST_RESORT)
    elif sharpness < 25:
        note("soft", USABLE)
    elif sharpness < 45:
        note("slightly_soft", GOOD)
    penalty += 20 * max(0.0, 1 - sharpness / 90)

    # Hair over the edges of the face counts a little; hands and objects, more
    if covered > 0.5:
        note("covered", LAST_RESORT)
    elif covered > 0.3:
        note("covered", USABLE)
    elif covered > 0.15:
        note("partly_covered", GOOD)
    penalty += 40 * covered

    skin = (portrait.get("look") or {}).get("face_skin")
    if skin is not None and usual.get("face_skin"):
        hidden = max(0.0, 1 - skin / usual["face_skin"])
        if hidden > 0.5:
            note("face_hidden", LAST_RESORT)
        elif hidden > 0.3:
            note("face_hidden", USABLE)
        elif hidden > 0.15:
            note("face_hidden", GOOD)
        penalty += 40 * hidden

    for issue in ("cropped", "poor_exposure", "low_quality", "extreme_pose", "extreme_expression"):
        if issue in base:
            note(issue, USABLE)
            penalty += 10

    details = portrait.get("expression_details")
    if details is None:
        note("expression_unknown", USABLE)
        penalty += 15
        expression = "unknown"
        if eyes < 0.16:
            note("closed_eyes", LAST_RESORT)
    else:
        shapes = details["blendshapes"]
        expression = expression_label(details)
        blink = _pair(shapes, "eyeBlink")
        if eyes < 0.16 or blink > 0.55:
            note("closed_eyes", LAST_RESORT)
        elif eyes < 0.2 or blink > 0.4:
            note("half_closed_eyes", USABLE)
        elif blink > 0.3:
            note("blinking", GOOD)
        penalty += 15 * max(0.0, 0.25 - eyes) / 0.09 + 30 * max(0.0, blink - 0.15)

        frown, sneer = _both(shapes, "browDown"), _both(shapes, "noseSneer")
        if frown > 0.45 or sneer > 0.5:
            note("grimace", LAST_RESORT)
        elif is_grimace(shapes):
            note("grimace", USABLE)
        elif frown > 0.18:
            note("frowning", GOOD)
        penalty += 40 * frown + 30 * sneer + 25 * max(0.0, _both(shapes, "mouthStretch") - 0.1)

        lips = details["lip_gap"]
        if expression == "smiling":
            if lips > 0.3:
                note("laughing", GOOD)
            penalty += 20 * max(0.0, lips - 0.2)
        else:
            if lips > 0.2:
                note("mouth_open", LAST_RESORT)
            elif lips > 0.08:
                note("speaking", USABLE)
            elif lips > 0.035:
                note("lips_parted", GOOD)
            penalty += 60 * max(0.0, lips - 0.03)

        pucker = max(shapes["mouthPucker"], shapes["mouthFunnel"])
        if pucker > 0.5:
            note("pursed_lips", USABLE)
        elif pucker > 0.3:
            note("pursed_lips", GOOD)
        penalty += 30 * max(0.0, pucker - 0.15) + 15 * max(0.0, _pair(shapes, "mouthPress") - 0.3)

        wide = _pair(shapes, "eyeWide")
        widened = eyes / usual["eyes"] if usual.get("eyes") else 1.0
        if wide > 0.5 or widened > 1.3:
            note("staring", USABLE)
        elif wide > 0.35 or widened > 1.15:
            note("staring", GOOD)
        penalty += 30 * max(0.0, wide - 0.15) + 60 * max(0.0, widened - 1.08)

        if shapes["browInnerUp"] > 0.75:
            note("raised_brows", GOOD)
        penalty += 15 * max(0.0, shapes["browInnerUp"] - 0.5)

        if abs(shapes["mouthSmileLeft"] - shapes["mouthSmileRight"]) > 0.3:
            note("lopsided_mouth", GOOD)

        away = gaze_away(gaze(shapes), usual.get("gaze"))
        if away > 0.6:
            note("looking_away", USABLE)
        elif away > 0.4:
            note("looking_away", GOOD)
        penalty += 20 * max(0.0, away - 0.25)

    change = portrait.get("temporal_change")
    if change is not None:
        if change > 0.45:
            note("mid_movement", USABLE)
        elif change > 0.25:
            note("mid_movement", GOOD)
        penalty += 30 * change

    if grayscale:
        note("black_and_white", USABLE)
        penalty += 15

    # The detection's own score (face size, pose, light) separates otherwise equal faces
    penalty += 0.15 * (100 - portrait["score"])
    level = max(notes.values(), default=IDEAL)
    if sum(severity == GOOD for severity in notes.values()) >= 3:
        level = max(level, USABLE)
    return Assessment(level, round(max(0.0, 100 - penalty), 2), tuple(sorted(notes)), expression, grayscale)


def _frontal_penalty(pose) -> float:
    pitch, yaw, roll = (0.0 if np.isnan(value) else abs(float(value)) for value in pose)
    return 0.6 * max(0.0, yaw - 12) + 0.5 * max(0.0, pitch - 12) + 0.5 * max(0.0, roll - 10)


def _expression_group(expression: str) -> str:
    return {"smiling": "smile", "mouth_open": "open"}.get(expression, "closed")


def novelty(first: dict, second: dict) -> float:
    """How much a photo adds next to another: head angle, gaze, expression and scene."""
    pitch_a, yaw_a, roll_a = first["pose"]
    pitch_b, yaw_b, roll_b = second["pose"]
    angle = lambda a, b: 0.0 if np.isnan(a) or np.isnan(b) else abs(float(a) - float(b))
    # Turning the head shows most of a face not seen yet; eyes alone add little
    points = (min(30.0, 1.2 * angle(yaw_a, yaw_b)) + min(10.0, 0.6 * angle(pitch_a, pitch_b))
              + min(6.0, 0.5 * angle(roll_a, roll_b)))
    if first["gaze"] is not None and second["gaze"] is not None:
        points += min(4.0, 6 * float(np.hypot(*np.subtract(first["gaze"], second["gaze"]))))
    if _expression_group(first["assessment"].expression) != _expression_group(second["assessment"].expression):
        points += 8.0
    if first["scene"] != second["scene"]:
        points += 4.0
    return points


def _usable_crop(face: dict, index: FaceIndex) -> bool:
    """Whether the automatic head-and-shoulders crop keeps this face and no other one."""
    box, crop = index.boxes[face["row"]].tolist(), face["crop"]
    return (crop[0] < crop[2] and crop[1] < crop[3]
            and crop[0] <= (box[0] + box[2]) / 2 < crop[2]
            and crop[1] <= (box[1] + box[3]) / 2 < crop[3]
            and not any(min(crop[2], other[2]) > max(crop[0], other[0])
                        and min(crop[3], other[3]) > max(crop[1], other[1]) for other in face["others"]))


def _usual_at(usual: dict, pose) -> dict:
    """The person's usual measurements for a face turned this way: a turned head shows less face skin."""
    yaw = 0.0 if np.isnan(pose[1]) else abs(float(pose[1]))
    if usual["face_skin"] is None:
        return usual
    return dict(usual, face_skin=usual["face_skin"] * max(0.7, 1 - 0.006 * max(0.0, yaw - 20)))


def candidates(frames_dir: Path, matches: Dict[str, FaceMatch], index: FaceIndex) -> List[dict]:
    """Faces of the person with their assessment and automatic crop, best first."""
    from services.portrait_quality import VERSION

    people = {name: match for name, match in matches.items()
              if match.similarity is not None and match.similarity >= MIN_IDENTITY_SIMILARITY}
    if any(index.portraits[match.row] is None or index.portraits[match.row].get("reference_version") != VERSION
           for match in people.values()):
        raise ValueError("Portrait assessment is missing or outdated. Reanalyze the video before automatic selection.")
    portraits = [index.portraits[match.row] for match in people.values()]
    directions = [gaze(p["expression_details"]["blendshapes"]) for p in portraits if p.get("expression_details")]
    skins = [p["look"]["face_skin"] for p in portraits if p.get("look") and p["look"].get("face_skin") is not None]
    usual = {"eyes": float(np.median([p["eye_openness"] for p in portraits])) if portraits else None,
             "gaze": tuple(np.median(directions, axis=0).tolist()) if directions else None,
             "face_skin": float(np.median(skins)) if skins else None}
    faces, size = [], None
    for name, match in people.items():
        portrait = index.portraits[match.row]
        if size is None:
            with Image.open(frames_dir / name) as image:
                size = image.size  # Every frame of a video has the same size
        others = [index.boxes[i].tolist() for i in range(*index.spans[name]) if i != match.row]
        crop = portrait_crop_box(index.boxes[match.row].tolist(), others, *size)
        grayscale = is_grayscale_look(portrait)
        if grayscale is None:
            grayscale = is_grayscale_face(frames_dir / name, match.box)
        details = portrait.get("expression_details")
        faces.append({
            "name": name, "row": match.row, "pose": tuple(index.poses[match.row].tolist()),
            "scene": name.split("_frame_")[0], "look": look_vector(portrait),
            "gaze": gaze(details["blendshapes"]) if details else None,
            "away": gaze_away(gaze(details["blendshapes"]), usual["gaze"]) if details else 0.0,
            "crop": crop, "others": others,
            "assessment": assess(portrait, grayscale, _usual_at(usual, index.poses[match.row])),
        })
    faces.sort(key=lambda face: (face["assessment"].level, -face["assessment"].score, face["name"]))
    return faces


def _main_look(faces: List[dict]) -> Optional[np.ndarray]:
    """The look most of the person's faces share (hair against skin), None when unmeasured."""
    looks = [face["look"] for face in faces if face["look"] is not None]
    if not looks:
        return None
    looks = np.asarray(looks)
    agreement = (np.abs(looks[:, None, :] - looks[None, :, :]).max(axis=2) <= LOOK_TOLERANCE).sum(axis=1)
    return looks[int(agreement.argmax())]


def _other_look(face: dict, main: Optional[np.ndarray]) -> bool:
    return main is not None and face["look"] is not None and np.abs(face["look"] - main).max() > LOOK_TOLERANCE


def select_face_references(frames_dir: Path, matches: Dict[str, FaceMatch], index: FaceIndex) -> List[str]:
    """Up to DEFAULT_FACE_REFERENCE_COUNT face photos of a person, primary first.

    The primary is the clearest frontal portrait of the person's usual look; the
    others add what it lacks (another angle, gaze, tilt or expression) and are
    chosen from the best levels that offer it. A person with any face that can
    be cropped alone always gets one.
    """
    faces = candidates(frames_dir, matches, index)
    if not faces:
        return []
    main = _main_look(faces)

    def primary_key(face):
        assessment = face["assessment"]
        frontal = _frontal_penalty(face["pose"])
        # A head turned past about 22 degrees (or tilted past 25) is a side view, not the main portrait
        level = assessment.level + (frontal > 6) + _other_look(face, main)
        return (min(level, LAST_RESORT), -(assessment.score - frontal - 25 * max(0.0, face["away"] - 0.2)), face["name"])

    selected = []
    for face in sorted(faces, key=primary_key):
        if _usable_crop(face, index):
            selected.append(face)
            break
    if not selected:
        return []

    pool = [face for face in faces if face["assessment"].level <= USABLE and face is not selected[0]]
    while len(selected) < DEFAULT_FACE_REFERENCE_COUNT and pool:
        best, best_points = None, None
        for face in pool:
            added = min(novelty(face, chosen) for chosen in selected)
            if added < MIN_NOVELTY:
                continue
            level = face["assessment"].level
            points = (LEVEL_POINTS[level] + 0.5 * (face["assessment"].score - 70) + NOVELTY_WEIGHT[level] * added
                      - 15 * _other_look(face, main))
            if best_points is None or points > best_points:
                best, best_points = face, points
        if best is None:
            break
        pool.remove(best)
        if _usable_crop(best, index):
            selected.append(best)
    return [face["name"] for face in selected]


def same_look_frames(matches: Dict[str, FaceMatch], index: FaceIndex, anchor: Optional[str]) -> set:
    """Frames where the person looks like in the anchor frame (same hair against the skin)."""
    if anchor not in matches:
        return set()
    reference = look_vector(index.portraits[matches[anchor].row] or {})
    if reference is None:
        return {anchor}
    return {name for name, match in matches.items()
            if (look := look_vector(index.portraits[match.row] or {})) is not None
            and np.abs(look - reference).max() <= LOOK_TOLERANCE}


def character_groups(looks: Dict[str, Optional[np.ndarray]]) -> List[List[str]]:
    """Faces of one identity grouped by clearly different hair, largest group first.

    The same face in a wig or a disguise is another character for a thumbnail.
    One group when the person keeps one look; faces whose hair was not measured
    stay in the largest group. Each group is split again while it shows two.
    """
    measured = sorted((float(vector[0]), name) for name, vector in looks.items() if vector is not None)
    values = np.array([value for value, _ in measured])
    total = len(values)
    best = None
    for cut in range(1, total):
        low, high = values[:cut], values[cut:]
        if min(len(low), len(high)) < max(CHARACTER_MIN_FACES, CHARACTER_MIN_SHARE * total):
            continue
        empty = float(np.percentile(high, 10) - np.percentile(low, 90))
        if best is None or empty > best[0]:
            best = (empty, cut)
    if best is None or best[0] < CHARACTER_MIN_EMPTY:
        return [list(looks)]
    # The stretch found, each face goes with the nearer group (the cut alone
    # could leave a few faces of one group on the other side)
    middle = (np.median(values[:best[1]]) + np.median(values[best[1]:])) / 2
    cut = int(np.searchsorted(values, middle))
    low, high = values[:cut], values[cut:]
    if (min(len(low), len(high)) < max(CHARACTER_MIN_FACES, CHARACTER_MIN_SHARE * total)
            or np.percentile(high, 10) - np.percentile(low, 90) < CHARACTER_MIN_EMPTY
            or np.median(high) - np.median(low) < CHARACTER_MIN_GAP):
        return [list(looks)]
    names = [name for _, name in measured]
    groups = [character_groups({name: looks[name] for name in part})
              for part in (names[:cut], names[cut:])]
    groups = sorted((group for parts in groups for group in parts), key=len, reverse=True)
    groups[0] += [name for name, vector in looks.items() if vector is None]
    return groups


def is_picture(faces: List[tuple]) -> bool:
    """Whether faces (frame filename, box) of one person never move: a picture, not a person.

    Every scene with at least three of them must be still, and they must add
    up to PICTURE_MIN_FACES.
    """
    by_scene: Dict[str, list] = {}
    for name, box in faces:
        by_scene.setdefault(name.split("_frame_")[0], []).append(box)
    still, faces_seen = [], 0
    for boxes in by_scene.values():
        if len(boxes) < 3:
            continue
        faces_seen += len(boxes)
        boxes = np.asarray(boxes, dtype=float)
        centers = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
        width = float(np.mean(boxes[:, 2] - boxes[:, 0])) or 1.0
        still.append(float(centers.std(axis=0).max()) / width < PICTURE_MAX_MOTION)
    return faces_seen >= PICTURE_MIN_FACES and all(still)

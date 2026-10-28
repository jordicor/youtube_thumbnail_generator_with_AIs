"""Face photos chosen like a person would, and the photo of the setting, without model downloads."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import orjson
import pytest
from PIL import Image

from face_extraction import assess_portrait
from services.face_index import load_face_index, match_faces
from services.face_selection import (
    GOOD,
    IDEAL,
    LAST_RESORT,
    USABLE,
    assess,
    expression_label,
    select_face_references,
)
from services.visual_context import load_visual_context, setting_frame
from tests.conftest import portrait


def _landmark_face():
    points = np.zeros((68, 3), dtype=float)
    points[36:42, :2] = [(75, 80), (80, 77), (90, 77), (95, 80), (90, 83), (80, 83)]
    points[42:48, :2] = [(125, 80), (130, 77), (140, 77), (145, 80), (140, 83), (130, 83)]
    points[48, :2], points[54, :2] = (90, 130), (130, 130)
    points[[61, 62, 63], :2] = [(100, 129), (110, 129), (120, 129)]
    points[[65, 66, 67], :2] = [(100, 131), (110, 131), (120, 131)]
    return SimpleNamespace(landmark_3d_68=points, pose=np.zeros(3), bbox=np.array([55, 35, 165, 160]))


@pytest.mark.parametrize("problem", ["occluded", "cropped", "closed_eyes", "extreme_expression"])
def test_detection_notes_bad_portraits(problem):
    face = _landmark_face()
    covered = 0.65 if problem == "occluded" else 0.03
    if problem == "cropped":
        face.bbox[1] = 0
    if problem == "closed_eyes":
        face.landmark_3d_68[[37, 38, 40, 41], 1] = 80
    if problem == "extreme_expression":
        face.landmark_3d_68[[65, 66, 67], 1] = 155
    result = assess_portrait(face, np.random.default_rng(7).integers(60, 200, (240, 220, 3), dtype=np.uint8), 99, covered)
    assert problem in result["issues"]


def test_blurred_face_is_noted_even_with_a_large_frontal_detection():
    result = assess_portrait(_landmark_face(), np.full((240, 220, 3), 150, dtype=np.uint8), 75, 0.02)
    assert "blurred" in result["issues"]


@pytest.mark.parametrize("changes, level, note", [
    ({}, IDEAL, None),
    ({"lip_gap": 0.05}, GOOD, "lips_parted"),
    ({"lip_gap": 0.15}, USABLE, "speaking"),
    ({"lip_gap": 0.3}, LAST_RESORT, "mouth_open"),
    ({"eyeBlinkLeft": 0.7, "eyeBlinkRight": 0.7}, LAST_RESORT, "closed_eyes"),
    ({"sharpness": 30.0}, GOOD, "slightly_soft"),
    ({"sharpness": 8.0}, LAST_RESORT, "blurred"),
    ({"occlusion": 0.2}, GOOD, "partly_covered"),
    ({"occlusion": 0.6}, LAST_RESORT, "covered"),
    ({"browDownLeft": 0.35, "browDownRight": 0.35, "jawOpen": 0.4}, USABLE, "grimace"),
    ({"browDownLeft": 0.22, "browDownRight": 0.2}, GOOD, "frowning"),
    ({"eyeLookUpLeft": 0.8, "eyeLookUpRight": 0.8}, USABLE, "looking_away"),
    ({"temporal_change": 0.5}, USABLE, "mid_movement"),
])
def test_every_problem_lowers_the_level_instead_of_rejecting_the_face(changes, level, note):
    result = assess(portrait(**changes), grayscale=False)
    assert result.level == level
    assert (note in result.notes) if note else not result.notes


def test_several_small_problems_add_up_to_a_lower_level():
    result = assess(portrait(lip_gap=0.05, sharpness=30.0, occlusion=0.2), grayscale=False)
    assert result.level == USABLE


def test_black_and_white_faces_are_kept_for_last():
    assert assess(portrait(), grayscale=True).level == USABLE


def test_a_smile_needs_the_rest_of_the_face_to_agree():
    smile = {"mouthSmileLeft": 0.6, "mouthSmileRight": 0.6}
    assert expression_label(portrait(**smile)["expression_details"]) == "smiling"
    grimace = portrait(**smile, browDownLeft=0.35, browDownRight=0.35, jawOpen=0.4)["expression_details"]
    assert expression_label(grimace) == "mouth_closed"
    # An open-mouthed laugh is still a smile, and not penalized as speaking
    laugh = portrait(lip_gap=0.2, **smile)
    assert expression_label(laugh["expression_details"]) == "smiling"
    assert assess(laugh, grayscale=False).level == IDEAL


def test_staring_and_looking_away_are_judged_against_the_person():
    usual = {"eyes": 0.3, "gaze": (0.3, 0.0)}
    assert "staring" in assess(portrait(eye_openness=0.4), grayscale=False, usual=usual).notes
    # This person's eyes always measure a bit to the left: that is the camera
    assert not assess(portrait(eyeLookOutLeft=0.6, eyeLookInRight=0.6), grayscale=False, usual=usual).notes


def _person(tmp_path, faces):
    """A video with one person whose faces are (portrait, pose) in frames of their own."""
    frames = tmp_path / "frames"
    frames.mkdir()
    direction = np.ones(512) / np.sqrt(512)
    records = []
    for number, (measurements, pose) in enumerate(faces):
        name = f"scene_{number // 2:03d}_frame_{number:06d}.jpg"
        Image.new("RGB", (320, 180), (200, 160, 140)).save(frames / name)
        records.append({"frame_path": name, "bbox": [180, 40, 220, 90], "embedding": direction.tolist(),
                        "quality_score": measurements["score"], "pose": pose, "expression": "neutral",
                        "portrait_quality": measurements})
    (tmp_path / "faces.json").write_bytes(orjson.dumps({"all_faces": records}))
    index = load_face_index(tmp_path / "faces.json")
    return frames, index, match_faces(index, direction, [Path(face["frame_path"]).name for face in records])


def _names(chosen):
    return [int(name[-10:-4]) for name in chosen]


def test_a_person_without_ideal_photos_still_gets_the_best_ones(tmp_path):
    frames, index, matches = _person(tmp_path, [
        (portrait(lip_gap=0.3), [0, 0, 0]),                      # mouth wide open
        (portrait(lip_gap=0.15), [0, 2, 0]),                     # speaking
        (portrait(eyeBlinkLeft=0.8, eyeBlinkRight=0.8), [0, 25, 0]),  # eyes closed
        (portrait(sharpness=30.0, lip_gap=0.05), [0, 30, 0]),    # a bit soft, lips parted
    ])
    chosen = _names(select_face_references(frames, matches, index))
    assert chosen[0] == 1  # Speaking beats a wide open mouth or closed eyes
    assert 3 in chosen     # A turned face that is only a bit worse adds an angle
    assert 2 not in chosen and 0 not in chosen  # Last resorts never fill extra slots


def test_extra_photos_add_angles_first_then_expressions(tmp_path):
    smile = {"mouthSmileLeft": 0.6, "mouthSmileRight": 0.6}
    frames, index, matches = _person(tmp_path, [
        (portrait(score=90), [0, 0, 0]),
        (portrait(score=90), [0, 1, 0]),                         # the same again
        (portrait(score=85, **smile), [0, 2, 0]),                # another expression
        (portrait(score=80, lip_gap=0.05), [0, 28, 0]),          # three-quarter, lips parted
        (portrait(score=60, lip_gap=0.15), [0, 45, 0]),          # more turned, but speaking and small
    ])
    assert _names(select_face_references(frames, matches, index)) == [0, 3, 2]


def test_similar_photos_still_fill_the_slots_when_they_differ_a_little(tmp_path):
    frames, index, matches = _person(tmp_path, [
        (portrait(), [-5, 0, 0]),
        (portrait(), [-5, 0, 0]),       # identical: adds nothing
        (portrait(), [5, 4, 6]),        # looking a little up and tilting the head
        (portrait(eyeLookOutLeft=0.3, eyeLookInRight=0.3), [-12, -6, 0]),
    ])
    chosen = _names(select_face_references(frames, matches, index))
    assert chosen[0] == 0 and sorted(chosen[1:]) == [2, 3]


def test_the_primary_is_frontal_and_looks_at_the_camera(tmp_path):
    frames, index, matches = _person(tmp_path, [
        (portrait(score=95), [0, 35, 0]),
        (portrait(score=90, eyeLookOutLeft=0.7, eyeLookInRight=0.7), [0, 0, 0]),
        (portrait(score=85), [0, 3, 0]),
    ])
    assert _names(select_face_references(frames, matches, index))[0] == 2


def test_the_primary_comes_from_the_usual_look(tmp_path):
    wig = {"hair": [20.0, 128.0, 128.0], "skin": [190.0, 140.0, 150.0], "clothes": None, "hair_share": 0.3}
    frames, index, matches = _person(tmp_path, [
        (portrait(score=99, look=wig), [0, 0, 0]),
        (portrait(score=80), [0, 0, 0]),
        (portrait(score=80), [0, 3, 0]),
    ])
    assert _names(select_face_references(frames, matches, index))[0] == 1


def test_a_face_hidden_by_cream_or_a_hand_is_kept_for_later(tmp_path):
    skin = lambda share: {"hair": [120.0, 140.0, 150.0], "skin": [190.0, 140.0, 150.0],
                          "clothes": None, "hair_share": 0.2, "face_skin": share}
    frames, index, matches = _person(tmp_path, [
        (portrait(score=99, look=skin(0.3)), [0, 0, 0]),  # cream over half the face
        (portrait(score=80, look=skin(0.7)), [0, 2, 0]),
        (portrait(score=80, look=skin(0.68)), [0, 25, 0]),
    ])
    assert _names(select_face_references(frames, matches, index)) == [1, 2]


def test_missing_portrait_measurements_require_reanalysis(tmp_path):
    import os
    frames, index, matches = _person(tmp_path, [(portrait(), [0, 0, 0])])
    cache = orjson.loads((tmp_path / "faces.json").read_bytes())
    cache["all_faces"][0]["portrait_quality"]["reference_version"] = 2
    path = tmp_path / "faces.json"
    previous = path.stat()
    path.write_bytes(orjson.dumps(cache))
    # This tests outdated measurements, not filesystem timestamp precision.
    # Equal-size writes can share a timestamp on volumes with coarse resolution.
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 2_000_000_000))
    index = load_face_index(tmp_path / "faces.json")
    with pytest.raises(ValueError, match="Reanalyze"):
        select_face_references(frames, matches, index)


@pytest.fixture
def setting(tmp_path):
    """A wall shown 120 s with a person in front, and 10 s of another room."""
    frames = tmp_path / "frames"
    frames.mkdir()
    scenes, faces = [], []
    wall = np.zeros((180, 320, 3), dtype=np.uint8)
    wall[:] = (90, 140, 200)
    wall[20:60, 20:300] = (60, 90, 160)  # A shelf: texture so frames are not featureless
    for scene, indices, duration in ((0, [10, 20, 30, 40], 120), (1, [110, 120], 10)):
        scenes.append({"index": scene, "start_frame": scene * 100, "end_frame": scene * 100 + 50,
                       "duration": duration, "frame_indices": indices})
        for number in indices:
            name = f"scene_{scene:03d}_frame_{number:06d}.jpg"
            image = wall.copy() if scene == 0 else np.full((180, 320, 3), (30, 200, 60), dtype=np.uint8)
            box = [130, 40, 190, 110] if number != 30 else [40, 50, 80, 100]  # in frame 30 the person is small
            if number == 20:
                box = [100, 10, 220, 150]  # a close-up hides the wall
            if number == 40:
                cv2.putText(image, "SUBTITLE", (5, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
            cv2.imwrite(str(frames / name), image)
            faces.append({"frame_path": name, "bbox": box, "embedding": (np.ones(512) / np.sqrt(512)).tolist(),
                          "quality_score": 80, "pose": [0, 0, 0], "expression": "neutral",
                          "portrait_quality": portrait()})
    (tmp_path / "scenes.json").write_bytes(orjson.dumps({"scenes": scenes}))
    (tmp_path / "faces.json").write_bytes(orjson.dumps({"all_faces": faces}))
    return frames, load_face_index(tmp_path / "faces.json")


def test_the_setting_photo_shows_the_longest_setting_least_covered(setting):
    frames, index = setting
    context = load_visual_context(frames)
    assert context.seconds.sum() == pytest.approx(130)
    assert set(context.main_setting()) == {f"scene_000_frame_{n:06d}.jpg" for n in (10, 20, 30, 40)}
    assert setting_frame(frames) == "scene_000_frame_000030.jpg"


def test_the_setting_photo_avoids_overlays_the_setting_lacks(setting):
    frames, index = setting
    # Without the small person, the subtitled frame would tie with the others
    (frames / "scene_000_frame_000030.jpg").unlink()
    assert setting_frame(frames) == "scene_000_frame_000010.jpg"


def test_durations_count_each_interval_once(setting):
    frames, _ = setting
    context = load_visual_context(frames)
    names = list(context.names)
    assert context.durations(names * 2) == context.durations(names)

"""Body evidence: visible coverage, identity association, review, and generation."""
import io
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import orjson
import pytest
from PIL import Image

import config
from services import body_references as bodies
from services.face_index import load_face_index, match_faces
from services.generation_service import GenerationService
from services.person_service import PersonService
from tests.test_person_editor import ANA
from tests.test_person_editor import people_video as people_video  # noqa: PLC0414 -- pytest fixture re-export
from utils import ReferenceImageError, VideoOutput


def pose():
    points = [SimpleNamespace(x=.5, y=.14, z=0, visibility=1, presence=1) for _ in range(33)]
    for i, x, y in [
        (11,.40,.30),(12,.60,.30),(13,.38,.40),(14,.62,.40),
        (15,.36,.50),(16,.64,.50),(23,.44,.52),(24,.56,.52),
        (25,.44,.71),(26,.56,.71),(27,.44,.89),(28,.56,.89),
        (29,.44,.91),(30,.56,.91),(31,.44,.94),(32,.56,.94),
    ]:
        points[i].x, points[i].y = x, y
    return points


def result(points, world=None, copies=1):
    return SimpleNamespace(pose_landmarks=[points] * copies, pose_world_landmarks=[world or points] * copies)


def person_mask():
    mask = np.zeros((1000, 800), dtype=bool)
    mask[80:965, 240:560] = True
    return mask


def segmented_pose(points, faces, frame=None, mask=None, copies=1, obstacles=None, portrait=None):
    mask = person_mask() if mask is None else mask
    frame = np.random.default_rng(4).integers(40, 200, (1000, 800, 3), dtype=np.uint8) if frame is None else frame
    obstacles = np.zeros(mask.shape, dtype=bool) if obstacles is None else obstacles
    segments = [{"box": [240, 80, 560, 965], "score": .99, "mask": mask}]
    portrait = portrait or {"score": 80, "eye_openness": .3, "occlusion": .03, "issues": []}
    return bodies.associate_poses(result(points, copies=copies), faces, frame, segments, obstacles, [portrait] * len(faces))


@pytest.mark.parametrize('missing,coverage', [
    ([], 'full_body'), ([27,28,29,30,31,32], 'three_quarter'),
    ([25,26,27,28,29,30,31,32], 'torso'),
])
def test_coverage_requires_visible_joints(missing, coverage):
    points = pose()
    for i in missing: points[i].visibility = .1
    assert bodies.pose_description(points, points, person_mask())[0] == coverage


def test_off_frame_landmarks_do_not_count_as_complete_body():
    points = pose()
    points[27].y = 1.1
    assert bodies.pose_description(points, points, person_mask())[0] == 'three_quarter'


def test_posture_is_not_inferred_from_the_face():
    points = pose()
    assert bodies.pose_description(points, points, person_mask())[1] == 'standing'
    world = deepcopy(points)
    for h, k, a in [(23,25,27),(24,26,28)]:
        world[k].x, world[k].y = world[h].x + .25, world[h].y
        world[a].x, world[a].y = world[k].x, world[k].y + .25
    assert bodies.pose_description(points, world, person_mask())[1] == 'seated'
    points[25].visibility = .1
    assert bodies.pose_description(points, world, person_mask())[1] == 'unknown'


def test_head_face_assignment_is_one_to_one():
    face = [350,100,450,200]
    matched = segmented_pose(pose(), [face])
    assert orjson.loads(orjson.dumps(matched)) == matched
    assert matched[0]['status'] == 'matched'
    assert matched[0]['face_box'] == face
    assert matched[0]['box'][3] > 940
    duplicate_faces = segmented_pose(pose(), [face,face])
    assert duplicate_faces[0]['status'] == 'ambiguous'
    duplicate_poses = segmented_pose(pose(), [face], copies=2)
    assert {r['status'] for r in duplicate_poses} == {'ambiguous'}


def test_a_crop_containing_another_face_is_not_used():
    records = segmented_pose(pose(), [[350,100,450,200],[480,320,540,390]])
    assert records[0]['status'] == 'other_person_in_crop'


def test_missing_model_fails_instead_of_downloading(tmp_path, monkeypatch):
    (tmp_path/'frames').mkdir()
    Image.new('RGB',(50,50)).save(tmp_path/'frames/f.jpg')
    (tmp_path/'faces.json').write_bytes(orjson.dumps({'all_faces':[]}))
    monkeypatch.setattr(bodies,'MODEL_PATH',tmp_path/'missing.task')
    with pytest.raises(RuntimeError,match='download_body_model'):
        bodies.ensure_body_index(tmp_path)


@pytest.fixture
def body_cache(people_video):
    directory = people_video['dir']
    index = load_face_index(directory/'faces.json')
    matches = match_faces(index, ANA, people_video['names'])
    name = people_video['names'][1]
    candidate = {
        'face_box':matches[name].box, 'status':'matched', 'coverage':'full_body',
        'view':'front','posture':'seated','quality':85.0,'box':[10,5,220,175],
    }
    data = {'version':bodies.CACHE_VERSION,'fingerprint':bodies._fingerprint(directory)[0],
            'frames':{name:[candidate]}}
    (directory/bodies.CACHE_NAME).write_bytes(orjson.dumps(data))
    return name,candidate,data


def test_cache_is_invalidated_when_original_pixels_change(people_video, body_cache):
    assert bodies.load_body_index(people_video['dir']) is not None
    Image.new('RGB',(320,180),'red').save(people_video['dir']/'frames'/body_cache[0])
    assert bodies.load_body_index(people_video['dir']) is None


def test_both_prompts_distinguish_body_from_identity_and_setting():
    from gransabio_prompt_generator import _build_reference_mapping_section
    from image_generation import _build_face_source_header
    mapping=[
        {'image_index':1,'person_id':'person_01','reference_role':'face'},
        {'image_index':2,'person_id':'person_01','reference_role':'body','body_coverage':'torso','body_posture':'unknown','view':'profile'},
        {'image_index':3,'person_id':None,'reference_role':'environment'},
    ]
    for text in (_build_face_source_header(mapping),_build_reference_mapping_section(mapping)):
        assert 'BODY REFERENCE' in text and 'person_01' in text and 'torso' in text and 'profile' in text
        assert 'ENVIRONMENT' in text

@pytest.mark.parametrize("change", [
    {"reference_role": "face"},
    {"body_coverage": "full_body"},
    {"view": "back"},
])
def test_prompt_cache_invalidates_changed_reference_meaning(change):
    import base64

    from prompt_generation import _cache_inputs, _cache_keys

    mapping = {"image_index": 1, "cluster_id": 10, "person_id": "person_01",
               "reference_role": "body", "body_coverage": "torso"}

    def skeleton_key(entry):
        image = base64.b64encode(b"same pixels").decode()
        return _cache_keys(_cache_inputs([entry], [image], False, "", 1, None, None, "summary"))[0]

    assert skeleton_key({**mapping, **change}) != skeleton_key(mapping)



def test_high_confidence_joints_behind_an_object_do_not_count_as_legs():
    mask = person_mask()
    mask[540:] = False
    described = bodies.pose_description(pose(), pose(), mask)
    assert described[0] == "torso"


def test_knees_hallucinated_inside_the_shirt_do_not_count_as_visible_legs():
    points = pose()
    for hip, knee in [(23, 25), (24, 26)]:
        points[knee].y = points[hip].y + .02
    assert bodies.pose_description(points, points, person_mask())[0] == "torso"


def test_a_clear_torso_is_useful_when_the_hips_cross_the_image_edge():
    points = pose()
    points[23].y, points[24].y = 1.01, .98
    for i in range(25, 33):
        points[i].y += 1
    mask = person_mask()
    mask[965:, 240:560] = True
    assert bodies.pose_description(points, points, mask)[0] == "torso"


def test_body_crop_keeps_visible_legs_even_with_uncertain_knee_landmarks():
    points = pose()
    for point in points[25:]:
        point.visibility = .1
    candidate = segmented_pose(points, [[350, 100, 450, 200]])[0]
    assert candidate["status"] == "matched"
    assert candidate["coverage"] == "torso"
    assert candidate["box"][3] >= 965


def test_foreground_objects_are_removed_even_when_the_person_mask_overlaps():
    obstacles = np.zeros((1000, 800), dtype=bool)
    obstacles[440:965, 240:560] = True
    candidate = segmented_pose(pose(), [[350, 100, 450, 200]], obstacles=obstacles)[0]
    assert candidate["coverage"] != "full_body"
    assert candidate["obstruction"] > .3
    assert candidate["box"][3] < 500


def test_side_view_does_not_require_both_legs_to_be_separately_visible():
    points, world = pose(), pose()
    for left, right in [(11, 12), (23, 24)]:
        world[left].x, world[right].x = .49, .51
        world[left].z, world[right].z = -.15, .15
    for i in (26, 28, 30, 32):
        points[i].visibility = .1
    coverage, _, view, *_ = bodies.pose_description(points, world, person_mask())
    assert view == "profile" and coverage == "full_body"


def test_selection_prefers_clear_partial_evidence_and_complementary_views(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    candidates = []
    for number, quality, coverage, view in [(1, 20, "full_body", "front"), (2, 85, "torso", "front"),
                                             (3, 90, "three_quarter", "profile"), (4, 82, "torso", "front")]:
        name = f"scene_000_frame_{number:06d}.jpg"
        Image.new("RGB", (320, 180), (80, 140, 200)).save(frames / name)
        candidates.append({"filename": name, "face_box": [80, 20, 140, 80], "quality": quality,
                           "coverage": coverage, "view": view})
    (tmp_path / "scenes.json").write_bytes(orjson.dumps({"scenes": [
        {"index": 0, "start_frame": 0, "end_frame": 5, "duration": 5, "frame_indices": [1, 2, 3, 4]}
    ]}))
    picked = bodies.select_body_references(frames, candidates)
    assert [item["filename"] for item in picked] == [candidates[2]["filename"], candidates[1]["filename"]]
    assert len(bodies.select_body_references(frames, candidates[1:2])) == 1


@pytest.mark.parametrize("lower_leg_visible,expected", [(False, "torso"), (True, "full_body")])
def test_transverse_thighs_need_visible_lower_legs(lower_leg_visible, expected):
    points = pose()
    for hip, knee, ankle in [(23, 25, 27), (24, 26, 28)]:
        points[knee].x = points[hip].x + .12
        points[knee].y = points[hip].y + .01
        points[ankle].x = points[knee].x
        points[ankle].visibility = 1 if lower_leg_visible else .1
    assert bodies.pose_description(points, points, person_mask())[0] == expected


def test_body_quality_reuses_face_occlusion_without_rejecting_a_side_view():
    face = [350, 100, 450, 200]
    clear = segmented_pose(pose(), [face])[0]
    covered = segmented_pose(pose(), [face], portrait={"score": 80, "eye_openness": .3,
                                                     "occlusion": .9, "issues": ["occluded"]})[0]
    side = segmented_pose(pose(), [face], portrait={"score": 50, "eye_openness": .3,
                                                  "occlusion": .18, "issues": ["extreme_pose", "occluded"]})[0]
    assert clear["quality"] > side["quality"] > covered["quality"]
    assert side["status"] == "matched"


def test_body_selection_counts_context_without_a_visible_body(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    scenes, candidates = [], []
    for scene, color, duration in [(0, (40, 25, 20), 5), (1, (200, 170, 140), 1), (2, (200, 170, 140), 80)]:
        name = f"scene_{scene:03d}_frame_{scene * 100 + 1:06d}.jpg"
        Image.new("RGB", (320, 180), color).save(frames / name)
        scenes.append({"index": scene, "start_frame": scene * 100, "end_frame": scene * 100 + 10,
                       "duration": duration, "frame_indices": [scene * 100 + 1]})
        if scene < 2:
            candidates.append({"filename": name, "face_box": [80, 20, 140, 80], "quality": 80, "view": "front"})
    (tmp_path / "scenes.json").write_bytes(orjson.dumps({"scenes": scenes}))
    assert bodies.select_body_references(frames, candidates)[0]["filename"] == candidates[1]["filename"]

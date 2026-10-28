"""Face sheet: picking the views of a person, editing them and sending the sheet."""

import io

import numpy as np
import orjson
import pytest
from PIL import Image

import config
from image_generation import _build_face_source_header
from gransabio_prompt_generator import _build_reference_mapping_section
from services.face_index import FaceMatch, centroid_to_blob
from services.face_sheet import (
    CELL_GAP,
    CELL_SIZE,
    automatic_views,
    render_face_sheet,
    resolve_views,
    view_candidates,
)
from services.generation_service import GenerationService
from tests.conftest import insert_job, insert_video

BOX = [40.0, 30.0, 140.0, 150.0]
DIRECTION = (lambda v: v / np.linalg.norm(v))(np.random.default_rng(3).normal(size=512))

# (yaw, expression, quality) of the one face in each frame
FACES = {
    'f1': (2, 'neutral', 80),
    'f2': (35, 'neutral', 70),
    'f3': (-35, 'neutral', 70),
    'f4': (-70, 'neutral', 60),
    'f5': (5, 'smiling', 90),
    'f6': (0, 'mouth_open', 75),
    'f7': (1, 'neutral', 50),
}


def _match(yaw, expression, quality, box=BOX):
    return FaceMatch(row=0, box=box, similarity=0.9, quality_score=quality, expression=expression, yaw=yaw)


def test_views_are_picked_by_pose_and_expression():
    matches = {key: _match(*values) for key, values in FACES.items()}
    matches['tiny'] = _match(0, 'neutral', 99, box=[0, 0, 30, 30])
    matches['someone_else'] = FaceMatch(row=0, box=BOX, similarity=0.1, quality_score=99, expression='neutral', yaw=0)

    candidates = view_candidates(matches)
    # Too small faces and faces of someone else never qualify
    assert candidates['front'] == ['f1', 'f7']
    assert automatic_views(candidates, lambda name: True) == {
        'front': 'f1', 'three_quarter_a': 'f2', 'three_quarter_b': 'f3',
        'profile': 'f4', 'smiling': 'f5', 'mouth_open': 'f6',
    }


def test_automatic_views_use_each_frame_once():
    # A slightly turned smile fits both "3/4" and "smiling"; "3/4" comes first
    matches = {'turned_smile': _match(23, 'smiling', 90), 'other_smile': _match(3, 'smiling', 60)}
    picks = automatic_views(view_candidates(matches), lambda name: True)
    assert picks['three_quarter_a'] == 'turned_smile'
    assert picks['smiling'] == 'other_smile'


def test_automatic_views_skip_unusable_frames():
    matches = {key: _match(*values) for key, values in FACES.items()}
    picks = automatic_views(view_candidates(matches), lambda name: name != 'f1')
    # f1 is black and white: the next front view, and no view at all for none left
    assert picks['front'] == 'f7'


def test_chosen_views_override_and_empty_the_automatic_ones():
    automatic = {'front': 'f1', 'three_quarter_a': 'f2', 'three_quarter_b': None,
                 'profile': 'f4', 'smiling': 'f5', 'mouth_open': None}
    cells = resolve_views(automatic, {'front': 'f7', 'profile': None}, {'f1', 'f2', 'f4', 'f5', 'f7'})
    assert cells == [('front', 'f7'), ('three_quarter_a', 'f2'), ('smiling', 'f5')]

    with pytest.raises(ValueError):
        resolve_views(automatic, {'front': 'gone'}, {'f1'})


def test_sheet_image_is_a_grid_of_square_cells(tmp_path):
    frame = tmp_path / 'frame.jpg'
    Image.new('RGB', (320, 180), (200, 10, 10)).save(frame)
    sheet = Image.open(io.BytesIO(render_face_sheet([(frame, BOX)] * 5)))
    assert sheet.size == (3 * CELL_SIZE + 4 * CELL_GAP, 2 * CELL_SIZE + 3 * CELL_GAP)
    pair = Image.open(io.BytesIO(render_face_sheet([(frame, BOX), (frame, None)])))
    assert pair.size == (2 * CELL_SIZE + 3 * CELL_GAP, CELL_SIZE + 2 * CELL_GAP)


# =============================================================================
# EDITING
# =============================================================================

def test_prompts_name_the_sheet_of_its_person():
    mapping = [
        {'image_index': 1, 'person_id': 'person_01', 'cluster_id': 1, 'source_frame': 'face_sheet_person_01.png',
         'is_style_reference': False, 'face_sheet_views': 'front view, profile view'},
        {'image_index': 2, 'person_id': 'person_01', 'cluster_id': 1, 'source_frame': 'a.jpg',
         'is_style_reference': False, 'face_sheet_views': None},
    ]
    header = _build_face_source_header(mapping)
    assert '- Image 1: face sheet for person_01: a grid of crops of that one face (front view, profile view)' in header
    assert '- Image 2: face reference for person_01' in header
    assert 'Image 1: FACE SHEET of person_01' in _build_reference_mapping_section(mapping)

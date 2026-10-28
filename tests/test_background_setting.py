"""The background of a video's thumbnails: the choice is remembered, and with
setting photos every idea and image prompt keeps that place."""
import orjson
import pytest

from gransabio_prompt_generator import (
    GranSabioError,
    build_concept_prompt,
    build_skeleton_prompt,
    skeleton_schema,
    validated_json,
)
from tests.conftest import insert_video

FACE = {'image_index': 1, 'person_id': 'person_01', 'cluster_id': 7, 'reference_role': 'face', 'is_primary': True}


def _setting(index, mode, side=None):
    return {'image_index': index, 'person_id': None, 'reference_role': 'environment', 'environment_mode': mode,
            'environment_instruction': 'Preserve the setting.', **({'environment_side': side} if side else {})}


SAME = [FACE, _setting(2, 'same')]
ALTERNATIVES = [FACE, _setting(2, 'alternatives'), _setting(3, 'alternatives')]
SPLIT = [FACE, _setting(2, 'split', 'left'), _setting(3, 'split', 'right')]


async def test_the_background_choice_is_saved_per_video(client, test_db):
    video_id = await insert_video(test_db, 'video.mp4', 'analyzed')
    url = f'/api/people/{video_id}/references'
    # Until the user picks, the page decides from the photos the video has
    assert (await client.get(url)).json()['environment_source'] is None

    saved = await client.put(f'/api/people/{video_id}/environment/source', json={'source': 'ai'})
    assert saved.status_code == 200 and saved.json() == {'environment_source': 'ai'}
    assert (await client.get(url)).json()['environment_source'] == 'ai'
    assert (await client.put(f'/api/people/{video_id}/environment/source', json={'source': 'photos'})).status_code == 200
    assert (await client.get(url)).json()['environment_source'] == 'photos'

    assert (await client.put(f'/api/people/{video_id}/environment/source', json={'source': 'beach'})).status_code == 422
    assert (await client.put('/api/people/999999/environment/source', json={'source': 'ai'})).status_code == 404


def _concept_prompt(mapping):
    return build_concept_prompt('transcript', 'Video', 1, 3, {'setting': 'A plain white wall'}, [],
                                reference_mapping=mapping)


def test_without_setting_photos_the_ideas_choose_the_place():
    for prompt in (_concept_prompt([FACE]), build_skeleton_prompt('t', 'Video', reference_mapping=[FACE])):
        assert 'SETTING (CHOSEN BY THE USER' not in prompt
    assert '"setting"' not in build_skeleton_prompt('t', 'Video', reference_mapping=[FACE])
    assert skeleton_schema([FACE])['required'] == skeleton_schema(None)['required']


def test_with_setting_photos_every_idea_keeps_the_place():
    skeleton = build_skeleton_prompt('t', 'Video', has_reference_image=True, reference_mapping=SAME)
    assert 'Image 2 show the real place' in skeleton and '"setting": "...",' in skeleton
    assert 'They are different places' in build_skeleton_prompt('t', 'Video', reference_mapping=ALTERNATIVES)

    same = _concept_prompt(SAME)
    assert 'DO NOT INVENT ANOTHER PLACE' in same and 'Image 2 show ONE place' in same
    assert 'Only\n  the framing and the lighting change between concepts' in same
    assert 'Images 2, 3 are ALTERNATIVE places' in _concept_prompt(ALTERNATIVES)
    assert 'the left panel happens in the place of Image 2 and the right panel in the place of Image 3' in \
        _concept_prompt(SPLIT)


def test_a_skeleton_without_the_setting_description_is_refused():
    answer = {
        'face_groups': {}, 'characters': {}, 'style_source': None, 'style_analysis': None,
        'camera_and_lighting_base': {'framing': 'f', 'equipment': 'e', 'lighting': 'l', 'composition': 'c'},
        'colors_palette_base': ['#000000'], 'suggested_titles_pool': ['A title'],
    }
    assert validated_json(answer, skeleton_schema([FACE]), 'skeleton') == answer
    with pytest.raises(GranSabioError, match="'setting' is missing"):
        validated_json(answer, skeleton_schema(SAME), 'skeleton')
    assert validated_json({**answer, 'setting': 'A plain white wall'}, skeleton_schema(SAME), 'skeleton')


@pytest.mark.parametrize('mapping, expected', [
    (SAME, 'The real place shown in Image 2 (ENVIRONMENT reference).'),
    (ALTERNATIVES, 'ONE of the real places shown in Images 2, 3'),
    (SPLIT, 'the left panel is the real place shown in Image 2, the right panel the one in Image 3'),
])
def test_the_image_prompt_points_to_the_setting_photos(mapping, expected):
    from image_generation import build_v3_prompt_string
    from prompt_generation import _concept_to_thumbnail_image

    concept = {
        'image_index': 1, 'concept_name': 'Date', 'thumbnail_concept': 'x', 'suggested_title': 'T', 'mood': 'fun',
        'colors': ['#FFFFFF'], 'key_topics': [], 'scene': 'person_01 smiles', 'subject_pose': 'standing',
        'subject_expression': 'smiling', 'subject_action': None, 'visual_elements': None, 'text_in_image': None,
        'background': 'Close framing of the wall, warm side light', 'environment_effects': None,
        'outfit_changes': None, 'camera_and_lighting_override': None,
    }
    skeleton = {'characters': {'person_01': {'belongs_to_face': 'group_A', 'outfit': 'shirt'}},
                'face_groups': {'group_A': {'physical_description': 'Adult', 'characters_with_this_face': ['person_01']}}}
    image = _concept_to_thumbnail_image(concept, skeleton, mapping, 'Video')
    prompt = build_v3_prompt_string(image, has_reference_images=True)
    environment = orjson.loads(prompt[prompt.index('{'):])['scene']['environment']
    assert expected in environment['setting']
    assert environment['framing_and_lighting'] == 'Close framing of the wall, warm side light'

    image.reference_mapping = [FACE]
    prompt = build_v3_prompt_string(image, has_reference_images=True)
    environment = orjson.loads(prompt[prompt.index('{'):])['scene']['environment']
    assert environment['setting'] == 'Close framing of the wall, warm side light'
    assert 'framing_and_lighting' not in environment

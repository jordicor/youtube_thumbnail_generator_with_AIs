"""Gran Sabio answers: a failure says why, and the JSON is checked against its schema."""

from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

import gransabio_prompt_generator as generator
from gransabio_prompt_generator import CONCEPT_SCHEMA, GranSabioError, PromptGenerationConfig, validated_json
from tests.conftest import insert_video


def _concept(**changes) -> dict:
    concept = {
        'image_index': 1, 'concept_name': 'Idea', 'thumbnail_concept': 'A surprise', 'suggested_title': 'Title',
        'mood': 'fun', 'colors': ['#FF0000'], 'key_topics': ['games'], 'scene': 'person_01 laughs',
        'subject_pose': 'standing', 'subject_expression': 'smiling', 'subject_action': None,
        'visual_elements': None, 'text_in_image': None, 'background': 'a room', 'environment_effects': None,
        'outfit_changes': None, 'camera_and_lighting_override': None,
    }
    return {**concept, **changes}


class _Client:
    """Gran Sabio client answering with `answer` (an exception is raised)."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


# =============================================================================
# VALIDATION
# =============================================================================

def test_a_valid_concept_comes_back_as_a_dict():
    assert validated_json(_concept(), CONCEPT_SCHEMA, 'concept 1') == _concept()
    # Fences and prose around the JSON are fine
    fenced = 'Here it is:\n```json\n' + orjson.dumps(_concept()).decode() + '\n```'
    assert validated_json(fenced, CONCEPT_SCHEMA, 'concept 1')['concept_name'] == 'Idea'


def test_a_concept_that_does_not_fit_names_its_fields():
    # The camera came back as text once and the whole job was lost
    with pytest.raises(GranSabioError, match=r'invalid concept 1: \$\.camera_and_lighting_override: Expected'):
        validated_json(_concept(camera_and_lighting_override='low angle'), CONCEPT_SCHEMA, 'concept 1')
    missing = _concept()
    del missing['mood']
    with pytest.raises(GranSabioError, match="Required property 'mood' is missing"):
        validated_json(missing, CONCEPT_SCHEMA, 'concept 1')
    with pytest.raises(GranSabioError, match='looks truncated'):
        validated_json('{"image_index": 1, "concept_name": "Id', CONCEPT_SCHEMA, 'concept 1')


# =============================================================================
# REQUESTS
# =============================================================================

def test_a_failed_request_says_why(monkeypatch):
    monkeypatch.setattr(generator, 'get_gransabio_client', lambda: _Client(RuntimeError('temperature is deprecated')))
    with pytest.raises(GranSabioError, match=r'\(anthropic/some-model\) failed: temperature is deprecated'):
        generator.generate_with_gransabio('prompt', PromptGenerationConfig(model='some-model'))


def test_an_empty_answer_is_an_error(monkeypatch):
    monkeypatch.setattr(generator, 'get_gransabio_client', lambda: _Client({'content': ''}))
    with pytest.raises(GranSabioError, match='returned no content'):
        generator.generate_with_gransabio('prompt', PromptGenerationConfig(model='some-model'))


def test_too_many_reference_images_are_refused_before_uploading(monkeypatch):
    monkeypatch.setattr(generator, 'get_gransabio_client', lambda: _Client({'content': '{}'}))
    monkeypatch.setattr(generator, 'upload_image_to_gransabio', lambda **kwargs: pytest.fail('uploaded'))
    with pytest.raises(GranSabioError, match='21 reference images'):
        generator.generate_with_gransabio('prompt', PromptGenerationConfig(), reference_images_base64=['x'] * 21)


def test_requests_ask_for_a_bounded_answer_without_tools(monkeypatch):
    from services import content_generation_service as content

    client = _Client({'content': _concept()})
    monkeypatch.setattr(generator, 'get_gransabio_client', lambda: client)
    monkeypatch.setattr(content, 'get_gransabio_client', lambda: client)

    generator.generate_with_gransabio('prompt', PromptGenerationConfig(model='some-model'))
    generator.generate_with_gransabio(
        'prompt', PromptGenerationConfig(model='some-model', thinking_enabled=True, thinking_level='high'))
    content.call_gransabio('prompt', content.TITLE_JSON_SCHEMA, model='claude-x', thinking_enabled=True,
                           thinking_level='low')

    plain, thinking, titles = client.calls
    # Without max_tokens Gran Sabio reserves almost all the model's output and
    # small windows have no room left for the prompt
    assert plain['max_tokens'] == generator.ANSWER_MAX_TOKENS
    assert plain['generation_tools_mode'] == 'never'
    assert 'reasoning_effort' not in plain and 'thinking_level' not in plain
    assert thinking['max_tokens'] == generator.ANSWER_MAX_TOKENS + generator.THINKING_EXTRA_TOKENS['high']
    assert thinking['reasoning_effort'] == 'high' and 'thinking_level' not in thinking
    assert titles['reasoning_effort'] == 'low' and titles['generation_tools_mode'] == 'never'
    assert titles['max_tokens'] == generator.ANSWER_MAX_TOKENS + generator.THINKING_EXTRA_TOKENS['low']
    assert 'thinking_budget_tokens' not in titles


def test_a_concept_is_asked_with_its_schema(monkeypatch):
    client = _Client({'content': _concept()})
    monkeypatch.setattr(generator, 'get_gransabio_client', lambda: client)
    concept = generator.generate_concept_gransabio(
        'transcript', 'Title', 1, 5, {'face_groups': {}}, [], PromptGenerationConfig(model='some-model')
    )
    assert concept['concept_name'] == 'Idea'
    assert client.calls[0]['json_schema'] is CONCEPT_SCHEMA and client.calls[0]['content_type'] == 'json'


def test_prompt_generation_stops_with_the_reason(monkeypatch, tmp_path):
    import prompt_generation

    def fail(**kwargs):
        raise GranSabioError('Gran Sabio LLM server is not available at http://example')

    monkeypatch.setattr(generator, 'generate_skeleton_gransabio', fail)
    output = SimpleNamespace(output_dir=tmp_path)
    with pytest.raises(GranSabioError, match='not available'):
        prompt_generation.generate_thumbnail_images('transcript', 'Title', output, num_images=2,
                                                    prompt_config=PromptGenerationConfig())
    assert not (tmp_path / 'prompts' / 'images.json').exists()
    with pytest.raises(ValueError, match='prompt_config is required'):
        prompt_generation.generate_thumbnail_images('transcript', 'Title', output, num_images=2)


# =============================================================================
# TITLES AND DESCRIPTIONS
# =============================================================================

def test_titles_must_be_a_valid_list(monkeypatch):
    from services import content_generation_service as content

    request = content.TitleGenerationRequest(video_title='Video', model='some-model')
    monkeypatch.setattr(content, 'call_gransabio', lambda *args, **kwargs: {'titles': [{'number': 1}]})
    with pytest.raises(GranSabioError, match=r"invalid title list: \$\.titles\[0\]\.title"):
        content.generate_titles(request)
    monkeypatch.setattr(content, 'call_gransabio', lambda *args, **kwargs: {'titles': []})
    with pytest.raises(GranSabioError, match='empty title list'):
        content.generate_titles(request)
    monkeypatch.setattr(content, 'call_gransabio', lambda *args, **kwargs: '{"titles": [{"number": 1, "title": "Hola"}]}')
    assert content.generate_titles(request).titles == ['Hola']


def test_content_requests_say_why_they_failed(monkeypatch):
    from services import content_generation_service as content

    monkeypatch.setattr(content, 'get_gransabio_client', lambda: _Client(RuntimeError('overloaded')))
    with pytest.raises(GranSabioError, match=r'\(openai/gpt-x\) failed: overloaded'):
        content.call_gransabio('prompt', content.TITLE_JSON_SCHEMA, provider='openai', model='gpt-x')


async def test_title_and_description_routes_answer_the_reason(client, test_db, monkeypatch):
    from api.routes import titles

    video_id = await insert_video(test_db, 'video.mp4', 'analyzed')

    async def video_info(video_id):
        return 'video', Path('video.mp4')

    def fail(request):
        raise GranSabioError('Gran Sabio LLM (anthropic/some-model) failed: overloaded')

    monkeypatch.setattr(titles, 'get_video_info', video_info)
    monkeypatch.setattr(titles, 'get_or_create_transcription', lambda path, name: ('transcript', False))
    monkeypatch.setattr(titles, 'generate_titles', fail)
    monkeypatch.setattr(titles, 'generate_descriptions', fail)

    for url in ('/api/titles/generate', '/api/titles/generate-description'):
        response = await client.post(url, json={'video_id': video_id, 'model': 'some-model'})
        assert response.status_code == 502
        assert response.json()['detail'] == 'Gran Sabio LLM (anthropic/some-model) failed: overloaded'

@pytest.mark.parametrize('visible_text', ['ACTO 2: ¡SORPRESA!', None])
def test_requested_text_reaches_the_image_provider(monkeypatch, tmp_path, visible_text):
    from unittest.mock import Mock

    import openai
    from PIL import Image

    import image_generation
    from prompt_generation import _concept_to_thumbnail_image

    text_client = _Client({'content': _concept(text_in_image=visible_text)})
    monkeypatch.setattr(generator, 'get_gransabio_client', lambda: text_client)
    instructions = f'Render these exact words: {visible_text}' if visible_text else 'Do not add any visible text.'
    concept = generator.generate_concept_gransabio(
        'transcript', 'A title that is not image text', 1, 1, {}, [],
        PromptGenerationConfig(custom_instructions=instructions),
    )
    image = _concept_to_thumbnail_image(concept, {}, None, 'Video title')
    assert image.text_overlay == (visible_text or '')

    edit = Mock(return_value=SimpleNamespace(data=[]))
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: SimpleNamespace(images=SimpleNamespace(edit=edit)))
    monkeypatch.setattr(image_generation, 'OPENAI_API_KEY', 'test-key')
    reference = tmp_path / 'reference.png'
    Image.new('RGB', (32, 32), 'white').save(reference)
    # The stand-in answers without an image
    with pytest.raises(image_generation.ImageProviderError, match='no image in the answer'):
        image_generation.generate_with_openai(
            image.image_prompt, reference_images=[reference], output_path=tmp_path / 'result.png', image=image,
        )
    edit.assert_called_once()
    actual_prompt = orjson.loads(edit.call_args.kwargs['prompt'])
    if visible_text is None:
        assert 'text_in_image' not in actual_prompt
    else:
        assert actual_prompt['text_in_image'] == visible_text


# =============================================================================
# OUTFITS
# =============================================================================

PEOPLE = {
    'person_01': {'belongs_to_face': 'group_A', 'outfit': 'grey sweatshirt and black beanie',
                  'identify_in_references': 'Look for: copper hair, black beanie'},
    'person_02': {'belongs_to_face': 'group_B', 'outfit': 'blue shirt', 'identify_in_references': 'Look for: beard'},
}


def _concept_with_outfits(monkeypatch, changes):
    monkeypatch.setattr(generator, 'get_gransabio_client', lambda: _Client({'content': _concept(outfit_changes=changes)}))
    return generator.generate_concept_gransabio(
        'transcript', 'Video', 1, 1, {'characters': PEOPLE}, [], PromptGenerationConfig(),
    )


def test_a_concept_can_dress_someone_differently(monkeypatch):
    from image_generation import build_image_prompt_v3
    from prompt_generation import _concept_to_thumbnail_image

    covered = 'No clothes: a thick white anime-style cloud covers her from chest to thighs'
    concept = _concept_with_outfits(monkeypatch, [{'person_id': 'person_01', 'outfit': covered}])
    image = _concept_to_thumbnail_image(concept, {'characters': PEOPLE}, None, 'Video')
    characters = build_image_prompt_v3(image, has_reference_images=True)['characters']

    assert characters['person_01']['outfit'] == covered
    assert 'not in the clothes of the reference photos' in characters['person_01']['outfit_note']
    assert characters['person_01']['identify_in_references'] == 'Look for: copper hair, black beanie'
    assert characters['person_02'] == PEOPLE['person_02']
    # The skeleton stays the same for the other concepts of the batch
    assert PEOPLE['person_01']['outfit'] == 'grey sweatshirt and black beanie'


def test_without_changes_everyone_keeps_the_skeleton_outfit(monkeypatch):
    from image_generation import build_image_prompt_v3
    from prompt_generation import _concept_to_thumbnail_image

    image = _concept_to_thumbnail_image(_concept_with_outfits(monkeypatch, None), {'characters': PEOPLE}, None, 'Video')
    assert build_image_prompt_v3(image, has_reference_images=True)['characters'] == PEOPLE


@pytest.mark.parametrize('changes, problem', [
    ([{'person_id': 'person_03', 'outfit': 'a cape'}], 'person_03 is not a character of the skeleton'),
    ([{'person_id': 'person_01', 'outfit': 'a cape'}, {'person_id': 'person_01', 'outfit': 'a hat'}],
     'a person is listed twice'),
    ([{'person_id': 'person_01', 'outfit': ' '}], 'an outfit is empty'),
])
def test_an_outfit_change_that_does_not_fit_fails(monkeypatch, changes, problem):
    with pytest.raises(GranSabioError, match=f'invalid concept 1: outfit_changes: {problem}'):
        _concept_with_outfits(monkeypatch, changes)


def test_the_concept_prompt_explains_outfit_changes():
    prompt = generator.build_concept_prompt('transcript', 'Video', 1, 1, {'characters': PEOPLE}, [])
    assert '"outfit_changes": null' in prompt
    assert 'default clothing, not part of their identity' in prompt


@pytest.mark.parametrize('changes', [{'text_overlay': 'A discarded headline'}, {'text_in_image': ''}])
def test_ambiguous_or_empty_text_fields_fail_validation(changes):
    with pytest.raises(GranSabioError, match='invalid concept'):
        validated_json(_concept(**changes), CONCEPT_SCHEMA, 'concept')


def _skeleton(**changes) -> dict:
    skeleton = {
        'face_groups': {}, 'characters': {}, 'style_source': None, 'style_analysis': None,
        'camera_and_lighting_base': {'framing': 'close', 'equipment': '50mm', 'lighting': 'soft', 'composition': 'center'},
        'colors_palette_base': ['#000000'], 'suggested_titles_pool': ['Title'],
    }
    return {**skeleton, **changes}


class _Ideas:
    """Gran Sabio stand-in for prompt_generation: counts skeletons and concepts."""

    def __init__(self, monkeypatch):
        self.skeletons = []
        self.concepts = []
        monkeypatch.setattr(generator, 'generate_skeleton_gransabio', self.skeleton)
        monkeypatch.setattr(generator, 'generate_concept_gransabio', self.concept)

    def skeleton(self, **kwargs):
        self.skeletons.append(kwargs)
        return _skeleton()

    def concept(self, **kwargs):
        self.concepts.append(kwargs)
        return _concept(concept_name=f"Idea {len(self.concepts)}")


def _ideas_kwargs(tmp_path, **changes):
    return {'transcription': 'transcript', 'video_title': 'Video', 'output': SimpleNamespace(output_dir=tmp_path),
            'num_images': 1, 'prompt_config': PromptGenerationConfig(), **changes}


def test_changed_text_contract_regenerates_concepts_but_reuses_identity(monkeypatch, tmp_path):
    from copy import deepcopy

    import prompt_generation as prompts

    config = PromptGenerationConfig(custom_instructions='Render the headline exactly: HELLO!')
    kwargs = _ideas_kwargs(tmp_path, prompt_config=config)
    summary = prompts.get_transcription_summary('transcript', 3000)
    previous_schema = deepcopy(CONCEPT_SCHEMA)
    previous_schema['properties']['text_overlay'] = {'type': 'string'}
    previous_schema['required'].append('text_overlay')
    with monkeypatch.context() as old:
        old.setattr(generator, 'CONCEPT_SCHEMA', previous_schema)
        previous_inputs = prompts._cache_inputs(None, None, False, None, 1, config, None, summary)
    skeleton_hash, previous_hash = prompts._cache_keys(previous_inputs)

    image = prompts._concept_to_thumbnail_image(_concept(), {}, None, 'Video')
    previous_image = {**image.to_dict(), 'text_overlay': 'HELLO!', 'text_in_image': None}
    cache_file = tmp_path / 'prompts' / 'images.json'
    cache_file.parent.mkdir()
    cache_file.write_bytes(orjson.dumps({'batches': [{
        'skeleton_hash': skeleton_hash, 'concepts_hash': previous_hash, 'inputs': previous_inputs,
        'skeleton': _skeleton(), 'images': [previous_image],
    }]}))
    calls = []

    def regenerate(**kwargs):
        calls.append(kwargs)
        return _concept(text_in_image='HELLO!')

    monkeypatch.setattr(generator, 'generate_concept_gransabio', regenerate)
    monkeypatch.setattr(generator, 'generate_skeleton_gransabio', lambda **kwargs: pytest.fail('Identity was regenerated'))
    first = prompts.generate_thumbnail_images(**kwargs)
    second = prompts.generate_thumbnail_images(**kwargs)
    assert len(calls) == 1
    assert first[0].text_in_image == second[0].text_in_image == 'HELLO!'
    assert first[0].text_overlay == second[0].text_overlay == 'HELLO!'


def test_going_back_to_earlier_choices_reuses_their_ideas(monkeypatch, tmp_path):
    import prompt_generation as prompts

    ideas = _Ideas(monkeypatch)
    one = prompts.generate_thumbnail_images(**_ideas_kwargs(tmp_path))
    two = prompts.generate_thumbnail_images(**_ideas_kwargs(tmp_path, num_images=2))
    titled = prompts.generate_thumbnail_images(**_ideas_kwargs(tmp_path, selected_titles=['A starred title']))
    assert len(ideas.skeletons) == 1 and len(ideas.concepts) == 4

    # Every earlier batch is still there: no new ideas
    assert [i.concept_name for i in prompts.generate_thumbnail_images(**_ideas_kwargs(tmp_path))] == \
        [i.concept_name for i in one]
    assert [i.concept_name for i in prompts.generate_thumbnail_images(**_ideas_kwargs(tmp_path, num_images=2))] == \
        [i.concept_name for i in two]
    assert len(ideas.concepts) == 4 and titled[0].concept_name == 'Idea 4'

    # Only the newest batches are kept
    for count in range(3, prompts.CACHE_BATCHES + 3):
        prompts.generate_thumbnail_images(**_ideas_kwargs(tmp_path, num_images=count))
    batches = orjson.loads((tmp_path / 'prompts' / 'images.json').read_bytes())['batches']
    assert len(batches) == prompts.CACHE_BATCHES
    assert [len(batch['images']) for batch in batches[:2]] == [prompts.CACHE_BATCHES + 2, prompts.CACHE_BATCHES + 1]
    concepts = len(ideas.concepts)
    prompts.generate_thumbnail_images(**_ideas_kwargs(tmp_path, selected_titles=['A starred title']))
    assert len(ideas.concepts) == concepts + 1


def test_avoid_repeat_always_brings_new_ideas(monkeypatch, tmp_path):
    import prompt_generation as prompts

    ideas = _Ideas(monkeypatch)
    for history in (['Title: first'], ['Title: second', 'Title: first']):
        config = PromptGenerationConfig(include_history=True, history_prompts=history)
        prompts.generate_thumbnail_images(**_ideas_kwargs(tmp_path, prompt_config=config))
    assert len(ideas.skeletons) == 1 and len(ideas.concepts) == 2


def test_a_miss_names_what_changed(monkeypatch, tmp_path, caplog):
    import logging

    import prompt_generation as prompts

    _Ideas(monkeypatch)
    prompts.generate_thumbnail_images(**_ideas_kwargs(tmp_path))
    with caplog.at_level(logging.INFO, logger=prompts.logger.name):
        prompts.generate_thumbnail_images(**_ideas_kwargs(
            tmp_path, num_images=2, prompt_config=PromptGenerationConfig(model='other-model', facial_expression='smiling')))
        prompts.generate_thumbnail_images(**_ideas_kwargs(
            tmp_path, prompt_config=PromptGenerationConfig(custom_instructions='Bright colors')))
    messages = [record.getMessage() for record in caplog.records]
    assert 'Cache HIT (skeleton): identity/style reused, new concepts because these changed: count, expression, ideas model' in messages
    assert 'Cache MISS: the closest of 2 cached batches differs in: instructions' in messages


def test_setting_photos_are_their_own_input(tmp_path):
    import base64

    import prompt_generation as prompts

    face, setting, other_setting = (base64.b64encode(data).decode() for data in (b'face', b'wall', b'beach'))
    person = {'image_index': 1, 'person_id': 'person_01', 'reference_role': 'face', 'is_primary': True}
    wall = {'image_index': 2, 'reference_role': 'environment', 'environment_mode': 'same',
            'environment_instruction': 'These are views of ONE shared setting.'}
    summary = 'transcript'

    def inputs(mapping, images):
        return prompts._cache_inputs(mapping, images, False, None, 1, PromptGenerationConfig(), None, summary)

    without = inputs([person], [face])
    with_wall = inputs([person, wall], [face, setting])
    with_beach = inputs([person, wall], [face, other_setting])
    changed = {name for name in without if without[name] != with_wall[name]}
    # The setting brings its own rules for the skeleton and the concepts
    assert changed == {'background', 'skeleton_rules', 'concept_rules'}
    assert {name for name in with_wall if with_wall[name] != with_beach[name]} == {'background'}
    assert prompts._cache_keys(without)[0] != prompts._cache_keys(with_wall)[0]
    with pytest.raises(ValueError, match='2 reference mapping entries for 1 reference images'):
        inputs([person, wall], [face])


@pytest.mark.parametrize("use_titles", [False, True])
@pytest.mark.parametrize("endpoint_id", ["", "ollama-main"])
def test_local_requests_keep_generation_repairs_and_preflight_local(monkeypatch, use_titles, endpoint_id):
    from services import content_generation_service as content

    monkeypatch.setattr(generator, 'GRANSABIO_LOCAL_ENDPOINT_IDS', {'ollama': endpoint_id})
    client = _Client({'content': {'ok': True}, 'local_only_attestation': {'local_only_verified': True}})
    monkeypatch.setattr(generator, 'get_gransabio_client', lambda: client)
    monkeypatch.setattr(content, 'get_gransabio_client', lambda: client)
    info = {'key': 'local-vlm', 'capabilities': ['text', 'vision', 'reasoning'], 'output_tokens': 4096}
    if use_titles:
        content.call_gransabio('prompt', content.TITLE_JSON_SCHEMA, provider='ollama', model='local-vlm',
                              model_info=info)
    else:
        generator.generate_with_gransabio('prompt', PromptGenerationConfig(
            provider='ollama', model='local-vlm', model_info=info))
    request, = client.calls
    assert request['execution_policy'] == {'scope': 'local_only'}
    assert request['feedback_memory_enabled'] is False
    assert request['gran_sabio_fallback'] is False
    assert request['long_text_mode'] == 'off'
    assert request['generator_model'] == 'ollama::local-vlm'
    assert request['max_tokens'] == 4096
    assert request['reasoning_effort'] == 'none'
    routes = request['llm_routing']['calls']
    assert set(routes) == {'generation.main', 'generation.json_retry', 'preflight.validate'}
    assert {route['model'] for route in routes.values()} == {'ollama::local-vlm'}
    assert {route.get('endpoint_id') for route in routes.values()} == {endpoint_id or None}


def test_local_execution_requires_the_server_attestation(monkeypatch):
    client = _Client({'content': {'ok': True}})
    monkeypatch.setattr(generator, 'get_gransabio_client', lambda: client)
    info = {'key': 'local', 'capabilities': ['text'], 'output_tokens': 4096}
    with pytest.raises(GranSabioError, match='did not verify local-only'):
        generator.generate_with_gransabio('prompt', PromptGenerationConfig(
            provider='ollama', model='local', model_info=info))


@pytest.mark.parametrize(('capabilities', 'images', 'error'), [
    (['embeddings'], None, 'does not support text generation'),
    (['text'], ['image'], 'local'),
])
def test_local_capability_errors_stop_before_generation_or_upload(monkeypatch, capabilities, images, error):
    client = _Client({'content': {'ok': True}})
    monkeypatch.setattr(generator, 'get_gransabio_client', lambda: client)
    monkeypatch.setattr(generator, 'upload_image_to_gransabio', lambda **kwargs: pytest.fail('uploaded'))
    info = {'key': 'local', 'capabilities': capabilities, 'output_tokens': 4096}
    with pytest.raises(GranSabioError, match=error):
        generator.generate_with_gransabio('prompt', PromptGenerationConfig(
            provider='ollama', model='local', model_info=info), reference_images_base64=images)
    assert not client.calls


def test_local_selection_cannot_fall_back_to_a_cloud_default():
    from pydantic import ValidationError
    from api.routes.generation import GenerationRequest
    from api.routes.titles import GenerateTitlesRequest, GenerateDescriptionsRequest
    from services.settings_service import TextEngine

    for request_type in (GenerateTitlesRequest, GenerateDescriptionsRequest):
        with pytest.raises(ValidationError, match='explicit local text model'):
            request_type(video_id=1, provider='ollama')
    with pytest.raises(ValidationError, match='explicit local text model'):
        GenerationRequest(cluster_ids=[1], prompt_provider='ollama')
    with pytest.raises(ValidationError, match='explicit local text model'):
        TextEngine(provider='ollama', model='', thinking_enabled=False, thinking_level='medium')


async def test_unavailable_local_text_model_stops_before_transcription(client, monkeypatch):
    from api.routes import titles
    from unittest.mock import AsyncMock

    monkeypatch.setattr(titles, 'require_text_model', AsyncMock(side_effect=GranSabioError('Local model missing')))
    monkeypatch.setattr(titles, 'get_or_create_transcription', lambda *args: pytest.fail('transcribed'))
    for route in ('/api/titles/generate', '/api/titles/generate-description'):
        response = await client.post(route, json={'video_id': 1, 'provider': 'ollama', 'model': 'missing'})
        assert response.status_code == 502
        assert response.json()['detail'] == 'Local model missing'


@pytest.mark.parametrize("use_titles", [False, True])
def test_openrouter_keeps_the_selected_model_and_output_budget(monkeypatch, use_titles):
    from services import content_generation_service as content

    client = _Client({'content': {'ok': True}})
    monkeypatch.setattr(generator, 'get_gransabio_client', lambda: client)
    monkeypatch.setattr(content, 'get_gransabio_client', lambda: client)
    info = {'key': 'vendor/cheap-model', 'capabilities': ['text', 'vision'], 'output_tokens': 2048}
    if use_titles:
        content.call_gransabio('prompt', content.TITLE_JSON_SCHEMA, provider='openrouter', model='vendor/cheap-model',
                              model_info=info)
    else:
        generator.generate_with_gransabio('prompt', PromptGenerationConfig(
            provider='openrouter', model='vendor/cheap-model', model_info=info))
    request, = client.calls
    assert request['generator_model'] == 'openrouter::vendor/cheap-model'
    assert request['max_tokens'] == 2048
    assert request['gran_sabio_fallback'] is False
    assert 'execution_policy' not in request
    routes = request['llm_routing']['calls']
    assert set(routes) == {'generation.main', 'generation.json_retry', 'preflight.validate'}
    assert {route['model'] for route in routes.values()} == {'openrouter::vendor/cheap-model'}
    assert all('endpoint_id' not in route for route in routes.values())


def test_openrouter_requires_an_explicit_model_and_supported_thinking():
    from pydantic import ValidationError
    from api.routes.generation import GenerationRequest
    from api.routes.titles import GenerateTitlesRequest, GenerateDescriptionsRequest

    for request_type in (GenerateTitlesRequest, GenerateDescriptionsRequest):
        with pytest.raises(ValidationError, match='explicit OpenRouter model'):
            request_type(video_id=1, provider='openrouter')
    with pytest.raises(ValidationError, match='explicit OpenRouter model'):
        GenerationRequest(cluster_ids=[1], prompt_provider='openrouter')
    info = {'key': 'vendor/text', 'capabilities': ['text'], 'output_tokens': 4096}
    with pytest.raises(GranSabioError, match='does not support thinking'):
        generator.text_generation_options('openrouter', 'vendor/text', True, 'high', info)

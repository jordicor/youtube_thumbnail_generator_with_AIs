"""Image models: one catalog shared by config, API, CLI choices and the settings dialog."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

import config
from api.routes.generation import GenerationRequest
from services.generation_service import GenerationService
from services.settings_service import IMAGE_MODELS
from tests.conftest import insert_cluster, insert_job, insert_video

ROOT = Path(__file__).parent.parent
CATALOGS = {
    'gemini': (config.GEMINI_MODELS, config.GEMINI_MODEL_MAX_REFS, 'gemini_model'),
    'openai': (config.OPENAI_MODELS, config.OPENAI_MODEL_MAX_REFS, 'openai_model'),
    'poe': (config.POE_MODELS, config.POE_MODEL_MAX_REFS, 'poe_model'),
}


def _request_choices(field: str) -> set:
    (literal, _none) = get_args(GenerationRequest.model_fields[field].annotation)
    return set(get_args(literal))


def _labelled_models() -> set:
    """Image models settings.js has a display name for."""
    js = (ROOT / 'static' / 'js' / 'settings.js').read_text(encoding='utf-8')
    block = re.search(r'const IMAGE_MODEL_LABELS = \{(.*?)\};', js, re.DOTALL).group(1)
    return set(re.findall(r"'([^']+)': \[", block))


@pytest.mark.parametrize('provider', sorted(CATALOGS))
def test_every_layer_offers_the_same_models(provider):
    catalog, max_refs, field = CATALOGS[provider]
    assert set(max_refs) == set(catalog)
    assert _request_choices(field) == set(catalog)
    # The settings dialog offers what the settings API lists, with the limits of config
    assert IMAGE_MODELS[provider] == max_refs
    assert set(catalog) <= _labelled_models()


def test_defaults_are_offered_models():
    assert config.GEMINI_IMAGE_MODEL in config.GEMINI_MODELS
    assert config.OPENAI_IMAGE_MODEL in config.OPENAI_MODELS
    assert config.POE_IMAGE_MODEL in config.POE_MODELS


@pytest.mark.parametrize('field, retired', [
    ('gemini_model', 'gemini-3-pro-image-preview'),
    ('openai_model', 'gpt-image-1'),
    ('poe_model', 'nanobananapro'),
])
def test_retired_models_are_rejected(field, retired):
    with pytest.raises(ValidationError):
        GenerationRequest(cluster_ids=[1], **{field: retired})


async def test_a_stale_model_setting_fails_the_job(test_db, monkeypatch, tmp_path):
    for name, module in {
        'transcription': SimpleNamespace(transcribe_video=lambda video_path, output: 'transcript'),
        'prompt_generation': SimpleNamespace(generate_thumbnail_images=lambda **kwargs: ['concept']),
        'image_generation': SimpleNamespace(generate_thumbnails_from_images=lambda **kwargs: []),
        'gransabio_prompt_generator': SimpleNamespace(
            PromptGenerationConfig=lambda **kwargs: kwargs,
            get_prompt_history_for_video=lambda video_id: None,
            require_vision_model=AsyncMock(),
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(config, 'GEMINI_IMAGE_MODEL', 'gemini-3-pro-image-preview')

    video_id = await insert_video(test_db, 'stale.mp4', 'analyzed')
    cluster_id = await insert_cluster(test_db, video_id)
    job_id = await insert_job(test_db, video_id, cluster_id, 'pending')
    service = GenerationService(test_db)

    with pytest.raises(ValueError, match='Unknown gemini image model: gemini-3-pro-image-preview'):
        await service.run_generation_pipeline(job_id)
    job = await service.get_job(job_id)
    assert job['status'] == 'error'
    assert 'gemini-3-pro-image-preview' in job['error_message']


# =============================================================================
# PROVIDER FALLBACK
# =============================================================================

def _providers(monkeypatch, tried, results, check=lambda kwargs: None):
    """Stand-in providers: each records its call and returns its path, or refuses when it has none."""
    import image_generation

    def provider(name):
        def generate(**kwargs):
            tried.append(name)
            check(kwargs)
            if results.get(name) is None:
                raise image_generation.ImageProviderError(f'{name} refused it')
            return results[name]
        return generate

    for name in ('gemini', 'openai', 'replicate', 'poe'):
        monkeypatch.setattr(image_generation, f'generate_with_{name}', provider(name))
    monkeypatch.setattr(image_generation, 'create_composite_thumbnail', provider('composite'))


def _idea(index=1):
    return SimpleNamespace(image_index=index, concept_name='c', suggested_title='t', image_prompt='p',
                           text_overlay=None, colors=None, style_analysis=None)


def _run_failing_gemini(monkeypatch, tmp_path, allow_provider_fallback, tried):
    """Generate one image whose Gemini call fails."""
    import image_generation

    _providers(monkeypatch, tried, {'openai': tmp_path / 'openai.png', 'composite': tmp_path / 'composite.png'})
    return image_generation.generate_thumbnails_from_images(
        images=[_idea()], best_frames=[tmp_path / 'frame.jpg'], output=SimpleNamespace(output_dir=tmp_path),
        image_provider='gemini', gemini_model='gemini-3-pro-image',
        allow_provider_fallback=allow_provider_fallback,
    )


def test_a_failed_provider_stops_with_its_reason_without_fallback(monkeypatch, tmp_path):
    import image_generation

    tried = []
    with pytest.raises(image_generation.ImageProviderError, match=r'^Image 1 of 1: gemini refused it$'):
        _run_failing_gemini(monkeypatch, tmp_path, allow_provider_fallback=False, tried=tried)
    assert tried == ['gemini']


def test_fallback_tries_the_other_providers_when_allowed(monkeypatch, tmp_path):
    tried = []
    results = _run_failing_gemini(monkeypatch, tmp_path, allow_provider_fallback=True, tried=tried)
    assert tried == ['gemini', 'openai']
    assert results == [tmp_path / 'openai.png']


def test_exhausted_fallback_names_every_reason(monkeypatch, tmp_path):
    import image_generation

    tried = []
    _providers(monkeypatch, tried, {})
    with pytest.raises(image_generation.ImageProviderError) as raised:
        image_generation.generate_thumbnails_from_images(
            images=[_idea()], best_frames=[tmp_path / 'frame.jpg'], output=SimpleNamespace(output_dir=tmp_path),
            image_provider='gemini', allow_provider_fallback=True,
        )
    assert tried == ['gemini', 'openai', 'replicate', 'poe', 'composite']
    assert str(raised.value) == ('Image 1 of 1: gemini refused it; openai refused it; '
                                 'replicate refused it; poe refused it; composite refused it')


def test_images_before_a_refused_one_are_kept(monkeypatch, tmp_path):
    import image_generation

    calls = []

    def openai(**kwargs):
        calls.append(kwargs['output_path'])
        if len(calls) == 2:
            raise image_generation.ImageProviderError('OpenAI gpt-image-2: rejected by the safety system')
        return kwargs['output_path']

    monkeypatch.setattr(image_generation, 'generate_with_openai', openai)
    progress = []
    with pytest.raises(image_generation.ImageProviderError,
                       match=r'^Image 2 of 3: OpenAI gpt-image-2: rejected by the safety system$'):
        image_generation.generate_thumbnails_from_images(
            images=[_idea(1), _idea(2), _idea(3)], best_frames=[], output=SimpleNamespace(output_dir=tmp_path),
            image_provider='openai', progress_callback=lambda current, total, info: progress.append(info),
        )
    assert len(calls) == 2
    assert [info['path'] for info in progress] == calls[:1]


def test_openai_and_gemini_refusals_say_why(monkeypatch, tmp_path):
    from unittest.mock import Mock

    import httpx
    import openai
    from google import genai

    import image_generation

    body ={'message': 'Your request was rejected by the safety system. safety_violations=[sexual].',
            'type': 'image_generation_user_error', 'param': None, 'code': 'moderation_blocked'}
    request = httpx.Request('POST', 'https://api.openai.com/v1/images/edits')
    refusal = openai.BadRequestError('Error code: 400', response=httpx.Response(400, request=request), body=body)
    client = SimpleNamespace(images=SimpleNamespace(generate=Mock(side_effect=refusal)))
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: client)
    monkeypatch.setattr(image_generation, 'OPENAI_API_KEY', 'test-key')
    with pytest.raises(image_generation.ImageProviderError) as raised:
        image_generation.generate_with_openai('prompt', output_path=tmp_path / 'openai.png', model='gpt-image-2')
    assert str(raised.value) == ('OpenAI gpt-image-2: Your request was rejected by the safety system. '
                                 'safety_violations=[sexual]. (moderation_blocked)')

    answer = SimpleNamespace(prompt_feedback=None, candidates=[SimpleNamespace(
        finish_reason=SimpleNamespace(name='IMAGE_SAFETY'),
        content=SimpleNamespace(parts=[SimpleNamespace(inline_data=None, text=' I cannot draw that. ')]),
    )])
    call = Mock(return_value=answer)
    monkeypatch.setattr(genai, 'Client', lambda **kwargs: SimpleNamespace(models=SimpleNamespace(generate_content=call)))
    monkeypatch.setattr(image_generation, 'GEMINI_API_KEY', 'test-key')
    with pytest.raises(image_generation.ImageProviderError) as raised:
        image_generation.generate_with_gemini('prompt', output_path=tmp_path / 'gemini.png', model='gemini-3-pro-image')
    assert str(raised.value) == ('Gemini gemini-3-pro-image: no image in the answer '
                                 '(finish reason: IMAGE_SAFETY; I cannot draw that.)')


def test_generation_request_has_fallback_off_by_default():
    assert GenerationRequest(cluster_ids=[1]).allow_provider_fallback is False


# =============================================================================
# REFERENCE FIDELITY
# =============================================================================

def _capture_provider(monkeypatch, provider):
    from unittest.mock import Mock
    import image_generation

    monkeypatch.setattr(image_generation, 'build_v3_prompt_string', lambda image, has_refs: 'identity prompt')
    if provider == 'openai':
        import openai
        client = SimpleNamespace(images=SimpleNamespace(
            edit=Mock(return_value=SimpleNamespace(data=[])),
            generate=Mock(side_effect=AssertionError('Identity requests must never become text-only')),
        ))
        monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: client)
        monkeypatch.setattr(image_generation, 'OPENAI_API_KEY', 'test-key')
        return image_generation.generate_with_openai, client.images.edit, client.images.generate

    from google import genai
    call = Mock(return_value=SimpleNamespace(candidates=[]))
    monkeypatch.setattr(genai, 'Client', lambda **kwargs: SimpleNamespace(models=SimpleNamespace(generate_content=call)))
    monkeypatch.setattr(image_generation, 'GEMINI_API_KEY', 'test-key')
    return image_generation.generate_with_gemini, call, None


@pytest.mark.parametrize('provider', ['openai', 'gemini'])
@pytest.mark.parametrize('file_format', ['PNG', 'JPEG'])
def test_prompt_and_image_providers_receive_the_same_native_pixels(monkeypatch, tmp_path, provider, file_format):
    import base64
    import io
    import numpy as np
    from PIL import Image
    from image_generation import ImageProviderError

    path = tmp_path / ('reference.png' if file_format == 'PNG' else 'reference.jpg')
    pixels = np.random.default_rng(17).integers(0, 256, (800, 1500, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, format=file_format)
    original = Image.open(path).convert('RGB')
    from utils import load_reference_images
    prompt_bytes = load_reference_images([path])[0]
    generate, call, _ = _capture_provider(monkeypatch, provider)

    # The stand-in answers without an image
    with pytest.raises(ImageProviderError, match='no image in the answer'):
        generate('prompt', reference_images=[path], image=SimpleNamespace(has_v2_fields=lambda: True))

    call.assert_called_once()
    if provider == 'openai':
        filename, payload, mime = call.call_args.kwargs['image']
        assert filename.endswith('.png') and mime == 'image/png'
    else:
        part = call.call_args.kwargs['contents'][0].inline_data
        payload, mime = part.data, part.mime_type
        assert mime == 'image/png'
    assert payload == prompt_bytes
    prepared = Image.open(io.BytesIO(payload))
    assert prepared.format == 'PNG'
    assert prepared.size == (1500, 800)
    assert prepared.tobytes() == original.tobytes()


@pytest.mark.parametrize('provider', ['openai', 'gemini'])
@pytest.mark.parametrize('broken', ['missing', 'corrupt'])
def test_a_bad_reference_prevents_any_provider_request(monkeypatch, tmp_path, provider, broken):
    from PIL import Image
    from utils import ReferenceImageError

    good = tmp_path / 'good.png'
    Image.new('RGB', (32, 32), 'red').save(good)
    bad = tmp_path / 'bad.png'
    if broken == 'corrupt':
        bad.write_bytes(b'not an image')
    generate, call, text_call = _capture_provider(monkeypatch, provider)

    with pytest.raises(ReferenceImageError, match='bad.png'):
        generate('prompt', reference_images=[good, bad], image=SimpleNamespace(has_v2_fields=lambda: True))
    call.assert_not_called()
    if text_call is not None:
        text_call.assert_not_called()


def test_reference_limits_fail_instead_of_slicing_or_recompressing(tmp_path):
    from PIL import Image
    from utils import ReferenceImageError, load_reference_images

    path = tmp_path / 'reference.png'
    Image.new('RGB', (32, 32), 'red').save(path)
    with pytest.raises(ReferenceImageError, match='accepts 1 references'):
        load_reference_images([path, path], max_count=1)
    with pytest.raises(ReferenceImageError, match='byte limit'):
        load_reference_images([path], max_bytes=len(path.read_bytes()))


def test_style_description_does_not_mislabel_a_person_photo():
    from image_generation import build_image_prompt_v3
    from prompt_generation import ThumbnailImage

    image = ThumbnailImage(
        image_index=1, concept_name='style', thumbnail_concept='', suggested_title='',
        image_prompt='', text_overlay='', mood='', colors=[], key_topics=[],
        scene='A portrait', subject_pose='Facing the camera',
        style_analysis={'art_type': 'illustration'},
        reference_mapping=[{'image_index': 1, 'person_id': 'person_01', 'is_style_reference': False}],
    )
    prompt = build_image_prompt_v3(image, has_reference_images=True)
    assert 'the identity photos do not define the style' in prompt['style_directive']['instruction']
    assert 'FIRST reference image' not in prompt['style_directive']['instruction']


def test_fallback_skips_models_that_would_drop_identity_references(monkeypatch, tmp_path):
    import image_generation

    tried = []
    def keeps_both_references(kwargs):
        assert len(kwargs['reference_images']) == 2

    _providers(monkeypatch, tried, {'poe': tmp_path / 'result.png'}, keeps_both_references)
    result = image_generation.generate_thumbnails_from_images(
        images=[_idea()], best_frames=[tmp_path / 'one.png', tmp_path / 'two.png'],
        output=SimpleNamespace(output_dir=tmp_path), image_provider='gemini',
        poe_model='flux2pro', allow_provider_fallback=True,
    )
    assert tried == ['gemini', 'openai', 'poe']
    assert result == [tmp_path / 'result.png']

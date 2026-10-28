"""Gran Sabio status and model list checks.

They run on the API's event loop, so they use the async client: a slow or
absent Gran Sabio must not hold the other requests of the page.
"""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import gransabio_prompt_generator as gransabio

MODELS = {
    'anthropic': [{'key': 'claude-test', 'name': 'Claude Test', 'description': 'x' * 100,
                   'capabilities': ['text', 'vision']}],
    'openai': [{'model_id': 'gpt-test', 'name': 'GPT Test', 'capabilities': ['text']}],
    'openrouter': [
        {'key': 'vendor/cheap-vision', 'name': 'Cheap vision', 'capabilities': ['text', 'vision'],
         'output_tokens': 2048, 'pricing': {'input_per_million': 0.05, 'output_per_million': 0.1}},
        {'key': 'vendor/text', 'name': 'Text only', 'capabilities': ['text'], 'output_tokens': 4096},
        {'key': 'vendor/embed', 'name': 'Embedding', 'capabilities': ['embeddings']},
    ],
    'ollama': [
        {'key': 'local', 'name': 'Local', 'capabilities': ['text', 'vision', 'reasoning'], 'output_tokens': 4096},
        {'key': 'embed', 'name': 'Embedding', 'capabilities': ['embeddings']},
    ],
}


@pytest.fixture
async def fake_gransabio(monkeypatch):
    """A local Gran Sabio whose answers each test sets; the generator points at it."""
    behavior = {'health': 'healthy', 'health_delay': 0, 'models_status': 200}

    async def health(request):
        await asyncio.sleep(behavior['health_delay'])
        return web.json_response({'status': behavior['health'], 'active_sessions': 3})

    async def models(request):
        if behavior['models_status'] != 200:
            return web.json_response({'detail': 'boom'}, status=behavior['models_status'])
        return web.json_response(MODELS)

    app = web.Application()
    app.router.add_get('/health', health)
    app.router.add_get('/models', models)
    server = TestServer(app)
    await server.start_server()
    monkeypatch.setattr(gransabio, 'GRANSABIO_LLM_URL', str(server.make_url('')).rstrip('/'))
    yield behavior
    await server.close()


async def test_status_of_a_healthy_server(fake_gransabio):
    status = await gransabio.check_gransabio_status()

    assert status['available'] is True
    assert status['status'] == 'healthy'
    assert status['active_sessions'] == 3
    assert status['providers'] == ['anthropic', 'openai', 'openrouter', 'ollama']


async def test_status_of_an_unhealthy_server(fake_gransabio):
    fake_gransabio['health'] = 'degraded'

    status = await gransabio.check_gransabio_status()

    assert status['available'] is False
    assert status['status'] == 'offline'


async def test_status_when_nothing_listens(monkeypatch, unused_tcp_port):
    monkeypatch.setattr(gransabio, 'GRANSABIO_LLM_URL', f'http://127.0.0.1:{unused_tcp_port}')

    status = await gransabio.check_gransabio_status()

    assert status['available'] is False
    assert status['status'] == 'offline'


async def test_status_gives_up_on_a_server_that_does_not_answer(fake_gransabio, monkeypatch):
    monkeypatch.setattr(gransabio, 'QUICK_REQUEST_TIMEOUT', 0.2)
    fake_gransabio['health_delay'] = 5

    started = asyncio.get_running_loop().time()
    status = await gransabio.check_gransabio_status()

    assert status['status'] == 'offline'
    assert asyncio.get_running_loop().time() - started < 2


async def test_status_reports_a_failing_model_list(fake_gransabio):
    fake_gransabio['models_status'] = 500

    status = await gransabio.check_gransabio_status()

    assert status['available'] is False
    assert status['status'] == 'error'


async def test_model_list_includes_local_generators_but_not_embeddings(fake_gransabio):
    models = await gransabio.get_available_models()

    assert list(models) == ['anthropic', 'openai', 'openrouter', 'ollama']
    assert models['anthropic'] == [{'key': 'claude-test', 'name': 'Claude Test', 'description': 'x' * 60, 'vision': True, 'reasoning': False}]
    assert [model['key'] for model in models['ollama']] == ['local']
    assert models['ollama'][0]['vision'] and models['ollama'][0]['reasoning']
    assert models['openai'] == [{'key': 'gpt-test', 'name': 'GPT Test', 'description': '', 'vision': False, 'reasoning': False}]


async def test_a_model_list_that_does_not_load_is_an_error(fake_gransabio, client):
    fake_gransabio['models_status'] = 500

    with pytest.raises(gransabio.GranSabioError, match="model list could not be loaded"):
        await gransabio.get_available_models()
    # Settings shows the reason instead of a made-up list
    for url in ('/api/generation/gransabio/models', '/api/titles/gransabio/models'):
        response = await client.get(url)
        assert response.status_code == 502
        assert "model list could not be loaded" in response.json()['detail']

    fake_gransabio['models_status'] = 200
    assert (await client.get('/api/generation/gransabio/models')).json()['anthropic'][0]['key'] == 'claude-test'


async def test_a_thumbnail_ideas_model_must_see_images(fake_gransabio):
    await gransabio.require_vision_model('anthropic', 'claude-test')

    with pytest.raises(gransabio.GranSabioError, match='GPT Test'):
        await gransabio.require_vision_model('openai', 'gpt-test')
    with pytest.raises(gransabio.GranSabioError, match='anthropic/claude-gone'):
        await gransabio.require_vision_model('anthropic', 'claude-gone')
    # No model is the provider's default, which this Gran Sabio does not have
    with pytest.raises(gransabio.GranSabioError, match=gransabio.DEFAULT_MODELS['anthropic']):
        await gransabio.require_vision_model('anthropic', None)


async def test_the_vision_check_fails_when_the_model_list_does_not_load(fake_gransabio):
    fake_gransabio['models_status'] = 500

    with pytest.raises(gransabio.GranSabioError, match="model list could not be loaded"):
        await gransabio.require_vision_model('anthropic', 'claude-test')


async def test_providers_for_content_generation(fake_gransabio):
    from services.content_generation_service import get_available_providers

    providers = await get_available_providers()

    assert list(providers) == ['anthropic', 'openai', 'openrouter', 'ollama']
    assert providers['anthropic']['models'][0]['id'] == 'claude-test'


async def test_a_slow_status_check_does_not_hold_other_requests(fake_gransabio, client, sample_video):
    fake_gransabio['health_delay'] = 1

    status_request = asyncio.create_task(client.get('/api/generation/gransabio/status'))
    await asyncio.sleep(0.1)
    gallery = await client.get(f"/api/thumbnails/video/{sample_video['id']}")

    assert gallery.status_code == 200
    assert not status_request.done()
    status = await status_request
    assert status.json()['available'] is True


async def test_openrouter_catalog_preserves_prices_and_checks_vision(fake_gransabio):
    models = await gransabio.get_available_models()
    assert [model['key'] for model in models['openrouter']] == ['vendor/cheap-vision', 'vendor/text']
    assert models['openrouter'][0]['pricing']['input_per_million'] == 0.05
    assert models['openrouter'][0]['vision'] is True
    assert models['openrouter'][1]['vision'] is False
    await gransabio.require_vision_model('openrouter', 'vendor/cheap-vision')
    with pytest.raises(gransabio.GranSabioError, match='Text only'):
        await gransabio.require_vision_model('openrouter', 'vendor/text')
    with pytest.raises(gransabio.GranSabioError, match='does not support text'):
        await gransabio.require_text_model('openrouter', 'vendor/embed')

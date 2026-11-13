"""App settings: the defaults before saving, saving them, and what is refused."""

import copy

import pytest

from services.settings_service import IMAGE_MODELS, default_settings


def _settings(**changes):
    """Default settings as JSON with some sections changed."""
    data = default_settings().model_dump()
    for section, values in changes.items():
        data[section].update(values)
    return data


async def test_defaults_until_saved(client):
    response = await client.get("/api/settings")
    assert response.status_code == 200
    body = response.json()
    assert body["settings"] == default_settings().model_dump()
    assert body["settings"]["generation"]["face_swap"] is False
    assert body["settings"]["generation"]["face_restoration"] is False
    # The face photos start at what the default image model takes
    image = body["settings"]["image"]
    assert body["settings"]["generation"]["face_photos"] == IMAGE_MODELS[image["provider"]][image["model"]]
    assert body["image_models"]["openai"]["gpt-image-2.5-sunburst"] == 16
    assert body["default_image_models"]["replicate"] == "flux-1.1-pro"
    assert isinstance(body["face_swap_missing"], list)


async def test_settings_say_what_face_swap_misses(client, monkeypatch):
    from api.routes import settings
    monkeypatch.setattr(settings, 'face_swap_available', lambda: (False, ["models/gfpgan/GFPGANv1.4.pth"]))

    assert (await client.put("/api/settings", json=_settings(generation={"face_swap": True}))).status_code == 200
    body = (await client.get("/api/settings")).json()
    assert body["face_swap_missing"] == ["models/gfpgan/GFPGANv1.4.pth"]
    # The saved value stays as it is: the pages show it turned off
    assert body["settings"]["generation"]["face_swap"] is True


@pytest.mark.parametrize("model", ["gpt-image-2.5-sunburst", "gpt-image-2.6-sunburst", "gpt-image-2.6-flare"])
async def test_saved_settings_are_read_back(client, model):
    data = _settings(
        thumbnail_ideas={"model": "claude-fable-5", "thinking_enabled": True, "thinking_level": "high"},
        texts={"provider": "openai", "model": "gpt-5.2"},
        image={"provider": "openai", "model": model},
        generation={"num_images": 8, "expression": "smiling", "face_swap": False, "face_photos": 16},
    )
    response = await client.put("/api/settings", json=data)
    assert response.status_code == 200
    assert response.json()["settings"] == data

    response = await client.get("/api/settings")
    assert response.json()["settings"] == data

    # Saving again replaces the single row
    data = _settings(generation={"num_images": 3})
    assert (await client.put("/api/settings", json=data)).status_code == 200
    assert (await client.get("/api/settings")).json()["settings"] == data


@pytest.mark.parametrize("changes", [
    # A model of another provider
    {"image": {"provider": "gemini", "model": "gpt-image-2.5-sunburst"}},
    # More face photos than the model takes
    {"image": {"provider": "poe", "model": "flux2pro"}, "generation": {"face_photos": 9}},
    {"generation": {"face_photos": 0}},
    {"generation": {"num_images": 51}},
    {"generation": {"expression": "angry"}},
    {"texts": {"provider": "mistral"}},
    {"thumbnail_ideas": {"thinking_level": "max"}},
])
async def test_invalid_settings_are_refused(client, changes):
    response = await client.put("/api/settings", json=_settings(**changes))
    assert response.status_code == 422
    # Nothing was saved
    assert (await client.get("/api/settings")).json()["settings"] == default_settings().model_dump()


async def test_incomplete_or_unknown_fields_are_refused(client):
    data = default_settings().model_dump()
    missing = copy.deepcopy(data)
    del missing["generation"]["provider_fallback"]
    assert (await client.put("/api/settings", json=missing)).status_code == 422

    extra = copy.deepcopy(data)
    extra["generation"]["face_swapp"] = True
    assert (await client.put("/api/settings", json=extra)).status_code == 422


async def test_restoration_migration_preserves_existing_preferences(test_db):
    import json
    from database.db import _run_migrations
    from services.settings_service import load_settings, save_settings

    old_settings = _settings(generation={"num_images": 9, "face_swap": True})
    del old_settings["generation"]["face_restoration"]
    await test_db.execute("INSERT INTO app_settings (id, data) VALUES (1, ?)", [json.dumps(old_settings)])
    await _run_migrations(test_db)
    settings = await load_settings(test_db)
    assert settings.generation.face_restoration is False
    assert settings.model_dump() == {**old_settings, "generation": {**old_settings["generation"], "face_restoration": False}}
    settings.generation.face_restoration = True
    await save_settings(test_db, settings)
    await _run_migrations(test_db)
    assert (await load_settings(test_db)).generation.face_restoration is True


async def test_local_text_settings_check_capabilities_before_saving(client, monkeypatch):
    from unittest.mock import AsyncMock
    from api.routes import settings

    monkeypatch.setattr(settings, 'get_available_models', AsyncMock(return_value={'ollama': [
        {'key': 'vision', 'vision': True, 'reasoning': True},
        {'key': 'text', 'vision': False, 'reasoning': False},
    ]}))
    valid = _settings(thumbnail_ideas={'provider': 'ollama', 'model': 'vision'},
                      texts={'provider': 'ollama', 'model': 'text'})
    response = await client.put('/api/settings', json=valid)
    assert response.status_code == 200
    assert (await client.get('/api/settings')).json()['settings'] == valid
    for section, changes in (
        ('thumbnail_ideas', {'model': 'text'}),
        ('texts', {'model': 'embedding'}),
        ('texts', {'thinking_enabled': True}),
    ):
        invalid = copy.deepcopy(valid)
        invalid[section].update(changes)
        assert (await client.put('/api/settings', json=invalid)).status_code == 400
        assert (await client.get('/api/settings')).json()['settings'] == valid


async def test_openrouter_settings_validate_the_chosen_model(client, monkeypatch):
    from unittest.mock import AsyncMock
    from api.routes import settings

    monkeypatch.setattr(settings, 'get_available_models', AsyncMock(return_value={'openrouter': [
        {'key': 'vendor/vision', 'vision': True, 'reasoning': False},
        {'key': 'vendor/text', 'vision': False, 'reasoning': False},
    ]}))
    valid = _settings(thumbnail_ideas={'provider': 'openrouter', 'model': 'vendor/vision'},
                      texts={'provider': 'openrouter', 'model': 'vendor/text'})
    assert (await client.put('/api/settings', json=valid)).status_code == 200
    assert (await client.get('/api/settings')).json()['settings'] == valid
    invalid = copy.deepcopy(valid)
    invalid['thumbnail_ideas']['model'] = 'vendor/text'
    assert (await client.put('/api/settings', json=invalid)).status_code == 400
    invalid['thumbnail_ideas']['model'] = 'vendor/unknown'
    assert (await client.put('/api/settings', json=invalid)).status_code == 400
    assert (await client.get('/api/settings')).json()['settings'] == valid

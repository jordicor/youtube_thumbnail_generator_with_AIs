"""Face swap controls are turned off, saying why, while its models are missing."""

import re

from services import face_swap_service


def test_missing_models_are_named_relative_to_the_project(monkeypatch, tmp_path):
    models = tmp_path / "models"
    (models / "gfpgan").mkdir(parents=True)
    (models / "gfpgan" / "GFPGANv1.4.pth").write_bytes(b"model")
    monkeypatch.setattr(face_swap_service, 'PROJECT_DIR', tmp_path)
    monkeypatch.setattr(face_swap_service, 'INSWAPPER_PATH', models / "inswapper" / "inswapper_128.onnx")
    monkeypatch.setattr(face_swap_service, 'GFPGAN_PATH', models / "gfpgan" / "GFPGANv1.4.pth")
    (models / "occluder").mkdir()
    (models / "occluder" / "xseg_2.onnx").write_bytes(b"model")
    monkeypatch.setattr(face_swap_service, 'OCCLUDER_PATHS',
                        tuple(models / "occluder" / f"xseg_{n}.onnx" for n in (1, 2, 3)))
    monkeypatch.setattr(face_swap_service.importlib.util, 'find_spec', lambda name: None if name == 'gfpgan' else object())

    available, missing = face_swap_service.face_swap_available()

    assert available is False
    assert missing == [
        "models/inswapper/inswapper_128.onnx",
        "models/occluder/xseg_1.onnx",
        "models/occluder/xseg_3.onnx",
    ]
    assert face_swap_service.face_restoration_available() == (False, ["Python package 'gfpgan'"])
    (models / "inswapper").mkdir()
    (models / "inswapper" / "inswapper_128.onnx").write_bytes(b"model")
    for path in face_swap_service.OCCLUDER_PATHS:
        path.write_bytes(b"model")
    assert face_swap_service.face_swap_available() == (True, [])


def _face_swap_controls(html: str) -> tuple[str, str]:
    checkbox = re.search(r'<input type="checkbox" id="applyFaceSwap"[^>]*>', html).group(0)
    container = re.search(r'<div class="video-detail" id="videoDetailApp"[^>]*>', html).group(0)
    return checkbox, container


async def test_video_page_turns_off_face_swap_without_its_models(client, monkeypatch):
    import api.main
    monkeypatch.setattr(api.main, 'face_swap_available', lambda: (False, ["models/inswapper/inswapper_128.onnx"]))

    html = (await client.get("/video/1")).text

    checkbox, container = _face_swap_controls(html)
    assert 'disabled' in checkbox and 'checked' not in checkbox
    # The reason is written under the option, not hidden in a tooltip
    reason = re.search(r'<span class="form-hint unavailable-reason">(.*?)</span>\s*</span>', html, re.S).group(1)
    assert 'models/inswapper/inswapper_128.onnx' in reason
    # The thumbnails' "fix face" buttons read why from the page
    assert 'data-face-swap-missing="' in container and 'models/inswapper/inswapper_128.onnx' in container


async def test_video_page_offers_face_swap_with_its_models(client, monkeypatch):
    import api.main
    monkeypatch.setattr(api.main, 'face_swap_available', lambda: (True, []))

    html = (await client.get("/video/1")).text

    checkbox, container = _face_swap_controls(html)
    # Whether it starts checked comes from the settings, in the page's script
    assert 'disabled' not in checkbox
    assert 'class="unavailable-warning"' not in html
    assert 'data-face-swap-missing=""' in container


async def test_missing_restoration_does_not_disable_the_swap(client, monkeypatch):
    import api.main

    monkeypatch.setattr(api.main, 'face_swap_available', lambda: (True, []))
    monkeypatch.setattr(api.main, 'face_restoration_available', lambda: (False, ["models/gfpgan/GFPGANv1.4.pth"]))
    html = (await client.get("/video/1")).text
    checkbox, _ = _face_swap_controls(html)
    restoration = re.search(r'<input type="checkbox" id="applyFaceRestoration"[^>]*>', html).group(0)
    assert 'disabled' not in checkbox
    assert 'checked' not in restoration
    assert 'disabled' in restoration
    assert 'models/gfpgan/GFPGANv1.4.pth' in html

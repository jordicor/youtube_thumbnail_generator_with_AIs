"""GPU coordination: bounded admission, cancellation, and verified release."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest
import requests

from services import local_gpu_service as gpu

ACQUIRE_ID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=0)

    def sleep(seconds):
        state.now += seconds

    monkeypatch.setattr(gpu, "time", SimpleNamespace(monotonic=lambda: state.now, sleep=sleep))
    return state


def reservation(client, tmp_path, cancellation_check=None, timeout=5):
    return client.reserve_images("gpu-4090", ACQUIRE_ID.replace("-", ""), tmp_path / "reservation.json", timeout, cancellation_check)


def response(request_id=ACQUIRE_ID, state="reserved", complete=True):
    return (200 if complete else 202), {
        "resource_id": "gpu-4090", "request_id": request_id, "state": state, "complete": complete,
        "reservation": ({"reservation_id": ACQUIRE_ID, "request_id": ACQUIRE_ID, "ready": state == "reserved"}
                        if state in ("reserved", "draining", "unloading") else None),
    }


def test_busy_then_pending_uses_one_identity_and_waits_for_release(tmp_path, monkeypatch, clock):
    calls = []
    acquisitions = iter([gpu.LocalGPUError("busy", 409, "resource_busy"), response(state="draining", complete=False), response()])
    release_count = 0

    def request(method, path, *, body, **_kwargs):
        nonlocal release_count
        calls.append((path, body))
        if path.endswith("/release"):
            release_count += 1
            return response(body["request_id"], "releasing" if release_count == 1 else "idle", release_count > 1)
        result = next(acquisitions)
        if isinstance(result, Exception):
            raise result
        return result

    with gpu.Client() as client:
        monkeypatch.setattr(client, "request", request)
        with reservation(client, tmp_path):
            assert [p for p, _ in calls] == ["/local-resources/gpu-4090/reserve"] * 3
            assert json.loads((tmp_path / "reservation.json").read_text())["state"] == "reserved"
    assert len(calls) == 5
    assert {body["request_id"] for path, body in calls if path.endswith("/reserve")} == {ACQUIRE_ID}
    releases = [body for path, body in calls if path.endswith("/release")]
    assert releases[0] == releases[1]
    assert releases[0]["reservation_id"] == ACQUIRE_ID
    assert str(UUID(releases[0]["request_id"])) != ACQUIRE_ID
    receipt = json.loads((tmp_path / "reservation.json").read_text())
    assert receipt["state"] == "released"
    assert receipt["release_request_id"] == releases[0]["request_id"]


def test_cancellation_cleans_up_pending_acquire_and_waits_through_busy_release(tmp_path, monkeypatch, clock):
    calls = []

    def request(method, path, *, body, **_kwargs):
        calls.append(path)
        if path.endswith("/reserve"):
            return response(state="draining", complete=False)
        if len(calls) == 2:
            raise gpu.LocalGPUError("Original acquire is still running", 409, "resource_busy")
        return response(body["request_id"], "unverified")

    with gpu.Client() as client:
        monkeypatch.setattr(client, "request", request)
        with pytest.raises(gpu.LocalGPUCancelled), reservation(client, tmp_path, lambda: clock.now >= 0.4):
            pytest.fail("Image generation started while pending")
    assert len(calls) == 3
    assert calls[-1].endswith("/release")
    assert json.loads((tmp_path / "reservation.json").read_text())["state"] == "released"


def test_busy_timeout_does_not_release_another_owners_reservation(tmp_path, monkeypatch, clock):
    calls = []

    def request(method, path, **_kwargs):
        calls.append(path)
        raise gpu.LocalGPUError("busy", 409, "resource_busy")

    with gpu.Client() as client:
        monkeypatch.setattr(client, "request", request)
        with pytest.raises(gpu.LocalGPUError, match="Timed out"), reservation(client, tmp_path, timeout=1):
            pytest.fail("Image generation started without a reservation")
    assert calls == ["/local-resources/gpu-4090/reserve"]
    assert clock.now == 1


def test_uncertain_acquire_is_not_retried_or_reported_as_released(tmp_path, monkeypatch):
    request = MagicMock(side_effect=gpu.LocalGPUError("Connection lost after the request was sent"))
    with gpu.Client() as client:
        monkeypatch.setattr(client, "request", request)
        with pytest.raises(gpu.LocalGPUError, match="Connection lost"), reservation(client, tmp_path):
            pytest.fail("Image generation started after an uncertain acquire")
    assert request.call_count == 1
    receipt = json.loads((tmp_path / "reservation.json").read_text())
    assert receipt["state"] == "acquire_failed"
    assert receipt["reservation_id"] == ACQUIRE_ID
    assert str(UUID(receipt["release_request_id"])) != ACQUIRE_ID


@pytest.mark.parametrize("failure", [gpu.LocalGPUError("Still running", 503), (202, {"state": "releasing"})])
def test_unverified_release_fails_closed(tmp_path, monkeypatch, failure):
    def request(method, path, **_kwargs):
        if path.endswith("/reserve"):
            return response()
        if isinstance(failure, Exception):
            raise failure
        return failure

    with gpu.Client() as client:
        monkeypatch.setattr(client, "request", request)
        with pytest.raises(gpu.LocalGPUError, match="could not be released"), reservation(client, tmp_path):
            pass
    assert json.loads((tmp_path / "reservation.json").read_text())["state"] == "release_failed"


@pytest.mark.parametrize("state", ["released", "canceled"])
def test_historical_acquire_never_allows_image_submission(tmp_path, monkeypatch, state):
    request = MagicMock(return_value=response(state=state))
    with gpu.Client() as client:
        monkeypatch.setattr(client, "request", request)
        with pytest.raises(gpu.LocalGPUError, match="invalid GPU reservation"), reservation(client, tmp_path):
            pytest.fail("A historical acquire permitted image submission")
    assert request.call_count == 1


def test_manual_unload_polls_the_same_uuid_and_keeps_transport_errors_visible(monkeypatch, clock):
    request_id = "33333333-3333-4333-8333-333333333333"
    with gpu.Client() as client:
        request = MagicMock(side_effect=[response(request_id, "unloading", False), response(request_id, "idle")])
        monkeypatch.setattr(client, "request", request)
        assert client.unload("gpu-4090", {"request_id": request_id, "target": "external"})["complete"]
        assert request.call_args_list[0].kwargs["body"] == request.call_args_list[1].kwargs["body"]
    with gpu.Client() as client:
        monkeypatch.setattr(client.session, "request", MagicMock(side_effect=requests.Timeout("offline")))
        with pytest.raises(gpu.LocalGPUError, match="unavailable"):
            client.resources()


@pytest.mark.parametrize("action", ["reserve", "unload"])
def test_conflicting_id_and_busy_manual_unload_are_not_retried(monkeypatch, action):
    code = "idempotency_conflict" if action == "reserve" else "resource_busy"
    with gpu.Client() as client:
        request = MagicMock(side_effect=gpu.LocalGPUError(code, 409, code))
        monkeypatch.setattr(client, "request", request)
        with pytest.raises(gpu.LocalGPUError, match=code):
            client.operation("gpu-4090", action, {"request_id": ACQUIRE_ID}, 5, wait_busy=action == "reserve")
        assert request.call_count == 1


def test_resource_error_envelope_preserves_busy_code(monkeypatch):
    http_response = MagicMock(status_code=409)
    http_response.__enter__.return_value = http_response
    http_response.json.return_value = {"error": {"code": "resource_busy", "detail": "Local inference is active"}}
    with gpu.Client() as client:
        monkeypatch.setattr(client.session, "request", MagicMock(return_value=http_response))
        with pytest.raises(gpu.LocalGPUError) as caught:
            client.request("POST", "/local-resources/gpu-4090/reserve")
        assert caught.value.code == "resource_busy"
        assert caught.value.status_code == 409
        assert "Local inference is active" in str(caught.value)


def test_comfyui_requires_one_exact_registered_target(monkeypatch):
    with gpu.Client() as client:
        monkeypatch.setattr(client, "resources", lambda: [{"id": "gpu", "external_service_url": "http://render:8188/"}])
        assert client.resource_for_comfyui("http://RENDER:8188") == "gpu"
        with pytest.raises(gpu.LocalGPUError, match="exactly one"):
            client.resource_for_comfyui("http://another-render:8188")


async def test_manual_gpu_api_reports_busy_and_validates_targets(client, monkeypatch):
    from api.routes import settings

    backend = MagicMock()
    backend.resources.return_value = [{"id": "gpu", "state": "idle", "active_dispatches": 0,
                                      "external_service_url": "http://render:8188", "requires_reconciliation": False,
                                      "last_error": None, "backend_errors": [], "external_service": {"idle": True},
                                      "loaded_models": [{"endpoint_id": "ollama", "name": "qwen", "size_vram": 1024}]}]
    factory = MagicMock()
    factory.return_value.__enter__.return_value = backend
    monkeypatch.setattr(settings, "GPUClient", factory)
    result = await client.get('/api/settings/local-resources')
    assert result.status_code == 200
    assert result.json()['resources'][0]['loaded_models'][0]['name'] == 'qwen'
    assert backend.unload.call_count == 0
    target = '/api/settings/local-resources/gpu/unload'
    assert (await client.post(target, json={'target': 'ollama'})).status_code == 422
    assert (await client.post(target, json={'target': 'external', 'model': 'qwen'})).status_code == 422
    backend.unload.side_effect = gpu.LocalGPUError('GPU busy', 409, 'resource_busy')
    result = await client.post(target, json={'target': 'ollama', 'endpoint_id': 'ollama', 'model': 'qwen'})
    assert result.status_code == 409
    assert 'GPU busy' in result.json()['detail']
    resource_id, body = backend.unload.call_args.args
    assert resource_id == 'gpu' and str(UUID(body.pop('request_id')))
    assert body == {'target': 'ollama', 'endpoint_id': 'ollama', 'model': 'qwen'}


@pytest.mark.parametrize("invalid", ["not_ready", "pending_http", "wrong_reservation"])
def test_incomplete_or_mismatched_acquire_never_submits_images(tmp_path, monkeypatch, invalid):
    status, result = response()
    if invalid == "not_ready":
        result["reservation"]["ready"] = False
    elif invalid == "pending_http":
        status = 202
    else:
        result["reservation"]["reservation_id"] = "99999999-9999-4999-8999-999999999999"

    def request(method, path, *, body, **_kwargs):
        if path.endswith("/release"):
            return response(body["request_id"], "idle")
        return status, result

    with gpu.Client() as client:
        monkeypatch.setattr(client, "request", request)
        with pytest.raises(gpu.LocalGPUError), reservation(client, tmp_path):
            pytest.fail("An incomplete or mismatched reservation permitted image submission")

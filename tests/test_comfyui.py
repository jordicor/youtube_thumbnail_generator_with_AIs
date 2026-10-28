"""Native ComfyUI protocol, local-only dispatch, and application integration."""
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

import pytest
from PIL import Image
from pydantic import ValidationError

from services import comfyui_service as comfy
from services.settings_service import default_settings
from prompt_generation import ThumbnailImage


@pytest.fixture
def profile():
    return comfy.Profile.model_validate({
        "id": "fixture", "name": "Reference workflow",
        "workflow": {
            "1": {"class_type": "ReferenceGenerator", "inputs": {"prompt": "", "seed": 1, "model": "local.safetensors"}},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "fixture"}},
        },
        "prompt": [{"node": "1", "input": "prompt"}], "seed": [{"node": "1", "input": "seed"}],
        "references": [{"node": "1", "input": "image1"}, {"node": "1", "input": "image2"}],
        "output_node": "2", "prompt_format": "qwen",
        "parameters": {"model": {"node": "1", "input": "model", "label": "Model"}},
    })


@pytest.fixture
def server(monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (160, 90), "navy").save(buffer, "PNG")
    state = SimpleNamespace(mode="success", uploads=[], prompts=[], downloads=[], canceled=[], active=[],
                            image=buffer.getvalue(), api_node=False, gpu_events=[], gpu_request=None,
                            gpu_failure=None, resource_url=None)
    schemas = {
        "ReferenceGenerator": {"input": {"required": {"prompt": ["STRING"], "seed": ["INT", {"min": 0}],
                                                      "model": [["local.safetensors", "other.safetensors"]]},
                                         "optional": {"image1": ["IMAGE"], "image2": ["IMAGE"]}}},
        "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING"]}}, "output_node": True},
        "LoadImage": {"input": {"required": {"image": [["reference.png"]]}}},
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, value, status=200, content_type="application/json"):
            body = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/local-resources":
                return self.send({"resources": [{"id": "fixture-gpu", "external_service_url": state.resource_url or state.url}]})
            if path == "/system_stats":
                return self.send({"system": {"comfyui_version": "0.37.0"}})
            if path.startswith("/object_info/"):
                name = path.rsplit("/", 1)[1]
                return self.send({name: {**schemas[name], "api_node": state.api_node}})
            if path == "/queue":
                return self.send({"queue_running": [[0, job] for job in state.active], "queue_pending": []})
            if path.startswith("/history/"):
                prompt_id = path.rsplit("/", 1)[1]
                if state.mode == "pending":
                    return self.send({})
                status = {"completed": True, "status_str": "success"}
                images = [{"filename": "result.png", "subfolder": "fixture", "type": "output"}]
                if state.mode == "error":
                    status = {"completed": False, "status_str": "error", "messages": [["execution_error", {"exception_message": "GPU out of memory"}]]}
                if state.mode == "missing_output":
                    images = []
                return self.send({prompt_id: {"status": status, "outputs": {"2": {"images": images}}}})
            if path == "/view":
                state.downloads.append(self.path)
                return self.send(state.image, content_type="image/png")
            self.send({"error": "unexpected route"}, 404)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/local-resources/fixture-gpu/reserve":
                request = json.loads(body)
                state.gpu_events.append("reserve")
                state.gpu_request = request["request_id"]
                if state.gpu_failure == "reserve":
                    return self.send({"detail": "Resource unavailable"}, 503)
                return self.send({"resource_id": "fixture-gpu", "request_id": state.gpu_request, "state": "reserved", "complete": True,
                                  "reservation": {"reservation_id": state.gpu_request, "request_id": state.gpu_request, "ready": True}})
            if self.path == "/local-resources/fixture-gpu/release":
                request = json.loads(body)
                assert request["reservation_id"] == state.gpu_request
                assert request["request_id"] != state.gpu_request
                state.gpu_events.append("release")
                if state.gpu_failure == "release":
                    return self.send({"detail": "Memory release was not confirmed"}, 503)
                return self.send({"resource_id": "fixture-gpu", "request_id": request["request_id"], "state": "idle", "complete": True, "reservation": None})
            if self.path == "/upload/image":
                state.uploads.append(body)
                return self.send({"name": f"reference_{len(state.uploads)}.png", "subfolder": "uploaded", "type": "input"})
            if self.path == "/prompt":
                prompt = json.loads(body)
                state.prompts.append(prompt)
                state.gpu_events.append("prompt")
                state.active.append(prompt["prompt_id"])
                if state.mode == "disconnect":
                    self.close_connection = True
                    return
                if state.mode == "reject":
                    return self.send({"error": "invalid workflow"}, 400)
                return self.send({"prompt_id": prompt["prompt_id"]})
            if self.path.startswith("/api/jobs/") and self.path.endswith("/cancel"):
                job = self.path.split("/")[3]
                state.canceled.append(job)
                state.active.remove(job)
                return self.send({"status": "canceled"})
            self.send({"error": "unexpected route"}, 404)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{httpd.server_port}"
    monkeypatch.setattr(comfy.config, "GRANSABIO_LLM_URL", state.url)
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def idea(index=1):
    return ThumbnailImage(image_index=index, concept_name="Fixture", thumbnail_concept="Two people",
                          suggested_title="Local AI", image_prompt="person_01 and person_02 at a desk",
                          text_overlay="LOCAL AI", mood="calm", colors=["#000000"], key_topics=[],
                          scene="person_01 and person_02 at a desk",
                          reference_mapping=[{"image_index": 1, "person_id": "person_01", "reference_role": "face"},
                                             {"image_index": 2, "person_id": "person_02", "reference_role": "body"}])


def test_batch_uses_all_references_once_and_preserves_native_output(server, profile, tmp_path, monkeypatch):
    monkeypatch.setattr(comfy.config, "KEEP_RAW_THUMBNAILS", True)
    references = []
    for index, color in enumerate(("red", "blue")):
        path = tmp_path / f"reference_{index}.png"
        Image.new("RGB", (91 + index, 67), color).save(path)
        references.append(path)
    progress = []
    paths = comfy.generate_batch([idea(1), idea(2)], references, SimpleNamespace(output_dir=tmp_path),
                                 connection={"url": server.url}, snapshot=profile.model_dump(),
                                 progress_callback=lambda *args: progress.append(args))
    assert len(paths) == len(server.prompts) == len(server.uploads) == 2
    assert server.gpu_events == ["reserve", "prompt", "prompt", "release"]
    assert json.loads(next((tmp_path / "thumbnails").glob("gpu-*.json")).read_text())["state"] == "released"
    assert [item[0] for item in progress] == [1, 2]
    for path in paths:
        assert Image.open(path).size == (1280, 720)
        receipt = json.loads(path.with_suffix(".comfyui.json").read_text())
        assert receipt["state"] == "completed"
        graph = receipt["workflow"]
        assert graph["1"]["inputs"]["image1"] == ["app_reference_1", 0]
        assert graph["1"]["inputs"]["image2"] == ["app_reference_2", 0]
        assert graph["app_reference_2"]["inputs"]["image"] == "uploaded/reference_2.png"
        assert "<image2> = image_index 2: body for person_02" in graph["1"]["inputs"]["prompt"]
    assert [p.read_bytes() for p in (tmp_path / "thumbnails" / "raw").glob("*.png")] == [server.image, server.image]
    assert references[0].read_bytes() in server.uploads[0]
    assert references[1].read_bytes() in server.uploads[1]
    assert "image1" not in profile.workflow["1"]["inputs"]


@pytest.mark.parametrize("mode, message", [("reject", "HTTP 400"), ("error", "GPU out of memory"),
                                          ("missing_output", "Expected one image"), ("disconnect", "failed")])
def test_failed_or_uncertain_submission_is_never_retried(server, profile, tmp_path, mode, message):
    server.mode = mode
    receipt = tmp_path / "job.json"
    with comfy.Client(comfy.Connection(url=server.url)) as client:
        with pytest.raises(comfy.ComfyUIError, match=message):
            client.generate(profile, "test", 123, ["reference.png"], receipt)
    assert len(server.prompts) == 1
    assert server.downloads == []
    assert json.loads(receipt.read_text())["prompt_id"] == server.prompts[0]["prompt_id"]


def test_cancel_only_stops_our_job(server, profile, tmp_path):
    server.mode = "pending"
    server.active.append("another-users-job")
    checks = iter([False, True])
    with comfy.Client(comfy.Connection(url=server.url)) as client:
        with pytest.raises(comfy.ComfyUICancelledError):
            client.generate(profile, "test", 1, ["reference.png"], tmp_path / "job.json", lambda: next(checks))
    assert server.canceled == [server.prompts[0]["prompt_id"]]
    assert server.active == ["another-users-job"]


def test_timeout_cancels_without_resubmitting(server, profile, tmp_path):
    server.mode = "pending"
    with comfy.Client(comfy.Connection(url=server.url, timeout=5)) as client:
        with pytest.raises(comfy.ComfyUIError, match="timed out"):
            client.generate(profile, "test", 1, ["reference.png"], tmp_path / "job.json")
    assert len(server.prompts) == len(server.canceled) == 1
    assert json.loads((tmp_path / "job.json").read_text())["state"] == "timeout_cancel_requested"


def test_invalid_model_and_paid_nodes_fail_before_submission(server, profile):
    with comfy.Client(comfy.Connection(url=server.url, parameters={"model": "missing.safetensors"})) as client:
        with pytest.raises(comfy.ComfyUIError, match="Unavailable value"):
            client.validate(profile)
    server.api_node = True
    with comfy.Client(comfy.Connection(url=server.url)) as client:
        with pytest.raises(comfy.ComfyUIError, match="Paid/API node"):
            client.validate(profile)
    assert server.prompts == server.uploads == []


def test_local_provider_never_calls_paid_generators(tmp_path, monkeypatch, server, profile):
    import image_generation
    from api.routes.generation import GenerationRequest
    from services.settings_service import AppSettings

    def forbidden(*_args, **_kwargs):
        pytest.fail("A paid provider was called")

    for name in ("gemini", "openai", "poe", "replicate"):
        monkeypatch.setattr(image_generation, f"generate_with_{name}", forbidden)
    with pytest.raises(ValueError, match="fallback"):
        image_generation.generate_thumbnails_from_images([], [], SimpleNamespace(output_dir=tmp_path),
                                                         image_provider="comfyui", allow_provider_fallback=True)
    with pytest.raises(ValidationError, match="fallback"):
        GenerationRequest(cluster_ids=[1], image_provider="comfyui", allow_provider_fallback=True)
    settings = default_settings().model_dump()
    settings["image"] = {"provider": "comfyui", "model": "qwen-image-2.1"}
    settings["generation"]["provider_fallback"] = True
    with pytest.raises(ValidationError, match="fallback"):
        AppSettings.model_validate(settings)
    with pytest.raises(ValueError, match="reference"):
        image_generation.generate_thumbnails_from_images([idea()], [], SimpleNamespace(output_dir=tmp_path), image_provider="comfyui")

    server.mode = "error"
    path = tmp_path / "reference.png"
    Image.new("RGB", (64, 64), "red").save(path)
    with pytest.raises(comfy.ComfyUIError, match="GPU out of memory"):
        image_generation.generate_thumbnails_from_images(
            [idea()], [path, path], SimpleNamespace(output_dir=tmp_path), image_provider="comfyui",
            comfyui_config={"url": server.url}, comfyui_profile=profile.model_dump(),
        )
    assert len(server.prompts) == 1


def test_native_qwen_profile_and_reference_contract():
    profile = comfy.load_profile("qwen-image-2.1")
    graph = profile.graph("prompt", 123, [f"reference_{i}.png" for i in range(16)], {})
    assert len(profile.references) == 16
    assert graph["4"]["inputs"]["images.image_16"] == ["app_reference_16", 0]
    assert graph["5"]["inputs"] == {"width": 1536, "height": 864, "batch_size": 1}
    with pytest.raises(ValueError, match="1-16"):
        profile.graph("prompt", 1, ["image.png"] * 17, {})
    with pytest.raises(ValueError, match="mapping"):
        comfy.build_prompt(profile, idea(), 1)


async def test_settings_import_check_and_save(client, server, profile):
    assert (await client.post("/api/settings/comfyui/workflows", json=profile.model_dump())).status_code == 201
    assert (await client.post("/api/settings/comfyui/workflows", json=profile.model_dump())).status_code == 400
    connection = {"url": server.url, "timeout": 120, "parameters": {"model": "other.safetensors"}}
    checked = await client.post("/api/settings/comfyui/check", json={"workflow": profile.id, "connection": connection})
    assert checked.status_code == 200
    assert checked.json()["validation_error"] is None
    assert checked.json()["parameters"]["model"]["choices"] == ["local.safetensors", "other.safetensors"]
    settings = default_settings().model_dump()
    settings["image"] = {"provider": "comfyui", "model": profile.id}
    settings["generation"]["face_photos"] = 2
    settings["comfyui"] = connection
    saved = await client.put("/api/settings", json=settings)
    assert saved.status_code == 200
    assert saved.json()["settings"] == settings
    assert (await client.get("/api/settings")).json()["image_models"]["comfyui"][profile.id] == 2
    settings["comfyui"]["parameters"]["model"] = "missing.safetensors"
    assert (await client.put("/api/settings", json=settings)).status_code == 400
    assert server.uploads == server.prompts == []


async def test_enqueue_captures_workflow_and_connection(client, test_db, sample_cluster, server, profile, monkeypatch):
    from api.routes import generation
    from tests.conftest import provision_uploaded_face

    comfy.import_profile(profile)
    await provision_uploaded_face(test_db, sample_cluster["id"])
    enqueue = AsyncMock(return_value="fixture-job")
    monkeypatch.setattr(generation, "enqueue_generation", enqueue)
    response = await client.post(f'/api/generation/{sample_cluster["video_id"]}/start', json={
        "cluster_ids": [sample_cluster["id"]], "num_images": 1, "image_provider": "comfyui",
        "comfyui_workflow": profile.id, "comfyui": {"url": server.url}, "include_scene_reference": False,
    })
    assert response.status_code == 200, response.text
    values = enqueue.call_args.kwargs
    assert values["comfyui_profile"] == profile.model_dump()
    assert values["comfyui_config"]["url"] == server.url
    assert values["allow_provider_fallback"] is False
    assert server.prompts == []


async def test_connection_migration_preserves_saved_preferences(test_db):
    from database.db import _run_migrations
    from services.settings_service import load_settings, save_settings

    previous = default_settings().model_dump()
    del previous["comfyui"]
    previous["generation"]["num_images"] = 9
    await test_db.execute("INSERT INTO app_settings (id, data) VALUES (1, ?)", (json.dumps(previous),))
    await _run_migrations(test_db)
    saved = await load_settings(test_db)
    expected = {**previous, "comfyui": comfy.Connection().model_dump()}
    assert saved.model_dump() == expected
    saved.comfyui.url = "http://localhost:9000"
    await save_settings(test_db, saved)
    await _run_migrations(test_db)
    assert (await load_settings(test_db)).comfyui.url == "http://localhost:9000"


def test_cli_passes_connection_and_worker_uses_generation_timeout():
    import main
    from workers.main import WorkerSettings
    from job_queue.settings import WorkerSettings as QueueSettings

    args = main._build_arg_parser().parse_args(['generate', '--video', '1', '--clusters', '0',
                                               '--image-provider', 'comfyui', '--comfyui-url', 'http://localhost:9000'])
    assert main._comfyui_cli_config(args)["url"] == "http://localhost:9000"
    generation = next(fn for fn in WorkerSettings.functions if getattr(fn, 'name', '') == 'run_generation')
    assert generation.timeout_s == QueueSettings.JOB_TIMEOUT_GENERATION


def test_callback_failure_stops_the_accepted_remote_job(server, profile, tmp_path):
    calls = 0
    server.mode = "pending"

    def cancellation_check():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("Cancellation database is unavailable")
        return False

    with comfy.Client(comfy.Connection(url=server.url)) as client:
        with pytest.raises(comfy.ComfyUIError, match="Cancellation database"):
            client.generate(profile, "test", 1, ["reference.png"], tmp_path / "job.json", cancellation_check)
    assert server.canceled == [server.prompts[0]["prompt_id"]]
    receipt = json.loads((tmp_path / "job.json").read_text(encoding="utf-8"))
    assert receipt["state"] == "failed_canceled"


def test_config_file_and_fallback_guard(tmp_path):
    import main
    path = tmp_path / "connection.json"
    path.write_text(json.dumps({"url": "http://localhost:9000", "timeout": 90, "parameters": {"model": "local.safetensors"}}), encoding="utf-8")
    args = main._build_arg_parser().parse_args(['generate', '--video', '1', '--clusters', '0',
                                               '--image-provider', 'comfyui', '--comfyui-config', str(path)])
    assert main._comfyui_cli_config(args)["parameters"] == {"model": "local.safetensors"}
    args.provider_fallback = True
    with pytest.raises(ValueError, match="fallback"):
        main._comfyui_cli_config(args)


@pytest.mark.parametrize("failure", ["reserve", "release", "unregistered"])
def test_batch_never_ignores_a_gpu_coordination_failure(server, profile, tmp_path, failure):
    from services.local_gpu_service import LocalGPUError

    references = []
    for index in range(2):
        path = tmp_path / f"face-{index}.png"
        Image.new("RGB", (32, 32), "navy").save(path)
        references.append(path)
    if failure == "unregistered":
        server.resource_url = "http://different-device:8188"
    else:
        server.gpu_failure = failure
    with pytest.raises((comfy.ComfyUIError, LocalGPUError)):
        comfy.generate_batch([idea()], references, SimpleNamespace(output_dir=tmp_path),
                             connection={"url": server.url}, snapshot=profile.model_dump())
    if failure in ("reserve", "unregistered"):
        assert not server.prompts
    else:
        assert len(server.prompts) == 1
        receipt = json.loads(next((tmp_path / "thumbnails").glob("gpu-*.json")).read_text())
        assert receipt['state'] == 'release_failed'

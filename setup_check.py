"""Check installation without processing videos or calling AI providers."""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent


def inspect_installation(args):
    checks = []

    def record(name, operation):
        try:
            checks.append({"name": name, "status": "ok", "detail": operation() or "Available."})
        except Exception as exc:
            checks.append({"name": name, "status": "error", "detail": str(exc)})

    def require(condition, explanation):
        if not condition:
            raise ValueError(explanation)

    record("Python", lambda: require(sys.version_info[:2] == (3, 11) and sys.maxsize > 2**32,
                                      "Use 64-bit Python 3.11 and rerun installation."))

    def packages():
        for module in ("cv2", "numpy", "PIL", "scenedetect", "insightface", "mediapipe", "torchvision",
                       "whisper", "fastapi", "uvicorn", "arq", "redis", "aiosqlite", "orjson", "psutil",
                       "anthropic", "openai", "google.genai", "replicate", "ai_json_cleanroom", "gfpgan", "client"):
            importlib.import_module(module)
        importlib.import_module("api.main")
        importlib.import_module("workers.main")
        return "Required packages and actual web/worker entry points import successfully."

    record("Application packages", packages)

    def ffmpeg():
        for name in ("ffmpeg", "ffprobe"):
            require(shutil.which(name), f"{name} is missing. Run install.ps1 -InstallPrerequisites.")
            subprocess.run([name, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=True, timeout=10)
        return "FFmpeg and FFprobe both run."

    record("Video tools", ffmpeg)

    def hardware():
        import numpy as np
        import torch
        import onnxruntime as ort
        from onnx import TensorProto, helper
        if args.hardware == "cuda":
            require(torch.cuda.is_available(), "NVIDIA acceleration is unavailable in PyTorch. Check the driver and CUDA build; setup does not silently switch to CPU.")
            torch.ones(1, device="cuda").add_(1).cpu()
            provider = "CUDAExecutionProvider"
        else:
            provider = "CPUExecutionProvider"
        graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "setup_probe",
                                  [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        session = ort.InferenceSession(model.SerializeToString(), providers=[provider])
        require(provider in session.get_providers(), "ONNX could not load the selected hardware provider. Check CUDA/cuDNN compatibility.")
        result = session.run(None, {"x": np.array([2], dtype=np.float32)})
        require(result[0][0] == 2, "The ONNX execution check returned an unexpected result.")
        return f"A small real ONNX operation passed on {provider}; PyTorch is usable."

    record("Analysis hardware", hardware)

    def models():
        from services.portrait_quality import MODEL_PATH, MODEL_SHA256, SEGMENTER_PATH, SEGMENTER_SHA256
        from services.body_references import MODEL_PATH as BODY, MODEL_SHA256 as BODY_SHA
        from services.body_references import SEGMENTATION_PATH, SEGMENTATION_SHA256
        for path, expected in ((MODEL_PATH, MODEL_SHA256), (SEGMENTER_PATH, SEGMENTER_SHA256),
                               (BODY, BODY_SHA), (SEGMENTATION_PATH, SEGMENTATION_SHA256)):
            require(path.is_file(), f"Missing {path.name}. Run install.py install --non-interactive --models core.")
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"{path.name} has the wrong checksum; inspect it before downloading again.")
        return "Required face/body selection models match their published checksums."

    if args.models == "none":
        checks.append({"name": "Analysis models", "status": "action", "detail": "Downloads were deferred. Install core models before analyzing a video."})
    else:
        record("Analysis models", models)
    if args.models == "all":
        def face_swap_models():
            from scripts.download_face_swap_models import MODELS, _sha256_of
            for spec in MODELS.values():
                require(spec.target.is_file(), f"Missing optional model: {spec.name}.")
                require(_sha256_of(spec.target) == spec.expected_sha256, f"Wrong checksum for {spec.name}.")
            return "All optional face replacement/restoration models match their checksums."
        record("Face replacement models", face_swap_models)

    from dotenv import dotenv_values
    env = {**dotenv_values(ROOT / ".env"), **os.environ}
    gran_env = dotenv_values(args.gransabio_dir / ".env") if args.gransabio_dir else {}

    def configured(value):
        return bool(value) and "your-" not in value and "your_" not in value

    missing = []
    if not configured(gran_env.get("ANTHROPIC_API_KEY")):
        missing.append("Default Claude ideas: add ANTHROPIC_API_KEY to GranSabio's .env, or select your configured text engine in Settings.")
    image_key = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "poe": "POE_API_KEY",
                 "replicate": "REPLICATE_API_TOKEN"}.get(env.get("IMAGE_PROVIDER", "gemini"))
    if image_key and not configured(env.get(image_key)):
        missing.append(f"Images: add {image_key} to this app's .env, or select your configured image engine in Settings.")
    if env.get("USE_LOCAL_WHISPER", "false").lower() != "true" and not configured(env.get("ELEVENLABS_API_KEY")):
        missing.append("Transcription: set USE_LOCAL_WHISPER=true for local transcription, or configure ElevenLabs.")
    checks.append({"name": "AI configuration", "status": "action" if missing else "ok",
                   "detail": " ".join(missing) if missing else "Default cloud credentials are present; validity and billing were not tested."})

    if args.services:
        def http_health(url, expected):
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    require(json.load(response).get("status") == expected, "Unhealthy service.")
            except Exception:
                raise ValueError("Service did not pass its health check. Start it and check its local log.") from None

        record("GranSabio service", lambda: http_health(env.get("GRANSABIO_LLM_URL", "http://127.0.0.1:8000").rstrip("/") + "/health", "healthy"))
        def web():
            url = f"http://127.0.0.1:{args.web_port}"
            http_health(url + "/health", "ok")
            with urllib.request.urlopen(url, timeout=5) as response:
                require(response.status == 200 and "text/html" in response.headers.get("Content-Type", ""),
                        "The web home page did not render successfully.")
            return "The health endpoint responds and the home page renders."

        record("Web service", web)

        def queue():
            import redis
            from services.queue_health import WORKER_HEALTH_KEY
            try:
                with redis.Redis.from_url(env.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
                                          socket_connect_timeout=3, socket_timeout=3) as client:
                    client.ping()
                    require(client.exists(WORKER_HEALTH_KEY), "The worker has no current heartbeat. Check .install/worker.log.")
            except redis.RedisError:
                raise ValueError("Redis is unreachable. Start it and check REDIS_URL locally.") from None
            return "Redis responds and a worker has a current heartbeat."

        record("Job queue and worker", queue)
    failed = any(item["status"] == "error" for item in checks)
    pending = any(item["status"] == "action" for item in checks)
    return {"checks": checks, "installation_ok": not failed, "configuration_pending": pending,
            "paid_generation_tested": False, "exit_code": 1 if failed else 3 if pending else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--hardware", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--models", choices=("core", "all", "none"), default="core")
    parser.add_argument("--gransabio-dir", type=Path)
    parser.add_argument("--web-port", type=int, default=8057)
    parser.add_argument("--services", action="store_true")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    with redirect_stdout(sys.stderr):
        result = inspect_installation(args)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for item in result["checks"]:
            print(f"[{item['status'].upper()}] {item['name']}: {item['detail']}")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

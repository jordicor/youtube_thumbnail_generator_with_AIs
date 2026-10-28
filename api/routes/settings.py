"""
Settings API Routes

The app settings (AI engines and the values every video's Generate form
starts with), read and saved by the header's Settings dialog.
"""

from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import Field, model_validator
from services.local_gpu_service import Client as GPUClient, LocalGPUError
from starlette.concurrency import run_in_threadpool
from services.comfyui_service import (
    Client, Connection, Profile, StrictModel, ComfyUIError,
    profiles, load_profile, import_profile,
)

from database.db import get_db
from gransabio_prompt_generator import GranSabioError, CATALOG_TEXT_PROVIDERS, get_available_models
from services.face_swap_service import face_restoration_available, face_swap_available
from services.settings_service import (
    DEFAULT_IMAGE_MODELS,
    image_models,
    AppSettings,
    load_settings,
    save_settings,
)

router = APIRouter()


def _response(settings: AppSettings) -> dict:
    # Face swap is shown turned off, saying why, while its models are missing
    _, face_swap_missing = face_swap_available()
    _, face_restoration_missing = face_restoration_available()
    return {
        "settings": settings.model_dump(),
        "image_models": image_models(),
        "comfyui_profiles": {
            key: {
                "name": profile.name,
                "parameters": {
                    name: {"label": binding.label, "default": profile.workflow[binding.node]["inputs"][binding.input]}
                    for name, binding in profile.parameters.items()
                },
            }
            for key, profile in profiles().items()
        },
        "default_image_models": DEFAULT_IMAGE_MODELS,
        "face_swap_missing": face_swap_missing,
        "face_restoration_missing": face_restoration_missing,
    }


@router.get("")
async def get_settings():
    """The saved settings (or the defaults), with the image models they can use."""
    async with get_db() as db:
        return _response(await load_settings(db))


@router.put("")
async def put_settings(settings: AppSettings):
    """Replace the settings. Videos already open keep the values they changed."""
    catalog_engines = [(name, getattr(settings, name)) for name in ("thumbnail_ideas", "texts")
                     if getattr(settings, name).provider in CATALOG_TEXT_PROVIDERS]
    if catalog_engines:
        try:
            catalog = await get_available_models()
        except GranSabioError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        for name, engine in catalog_engines:
            model = next((item for item in catalog.get(engine.provider, []) if item["key"] == engine.model), None)
            if model is None:
                raise HTTPException(status_code=400, detail=f"Text model is not available: {engine.model}")
            if name == "thumbnail_ideas" and not model["vision"]:
                raise HTTPException(status_code=400, detail=f"Thumbnail ideas require a vision model: {engine.model}")
            if engine.thinking_enabled and not model["reasoning"]:
                raise HTTPException(status_code=400, detail=f"This model does not support thinking: {engine.model}")
    if settings.image.provider == "comfyui":
        try:
            await run_in_threadpool(_check, settings.comfyui, load_profile(settings.image.model), True)
        except (ComfyUIError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    async with get_db() as db:
        await save_settings(db, settings)
    return _response(settings)


class ComfyUICheck(StrictModel):
    connection: Connection
    workflow: str


def _check(connection, profile, require_valid=False):
    with Client(connection) as client:
        info = client.inspect(profile)
        error = None
        try:
            client.validate(profile, info)
        except (ComfyUIError, ValueError) as exception:
            if require_valid:
                raise
            error = str(exception)
        return {"version": info["version"], "parameters": info["parameters"], "validation_error": error}


@router.post("/comfyui/check")
async def check_comfyui(request: ComfyUICheck):
    """Inspect native nodes and model choices without submitting a prompt."""
    try:
        return await run_in_threadpool(_check, request.connection, load_profile(request.workflow))
    except (ComfyUIError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/comfyui/workflows", status_code=201)
async def add_comfyui_workflow(profile: Profile):
    """Store an explicitly imported profile in private runtime data."""
    try:
        import_profile(profile)
    except (ValueError, FileExistsError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"id": profile.id}


class GPUUnload(StrictModel):
    target: Literal["ollama", "external"]
    endpoint_id: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_target(self):
        if self.target == "ollama" and (not self.endpoint_id or not self.model):
            raise ValueError("Choose an Ollama endpoint and model to unload")
        if self.target == "external" and (self.endpoint_id is not None or self.model is not None):
            raise ValueError("ComfyUI unload applies to its image models, not an Ollama model")
        return self


def _gpu_resources():
    with GPUClient() as client:
        resources = client.resources()
    try:
        return {"resources": [{
            "id": item["id"], "state": item["state"],
            "active_dispatches": item["active_dispatches"], "loaded_models": item["loaded_models"],
            "has_comfyui": bool(item["external_service_url"]), "external_service": item.get("external_service"),
            "requires_reconciliation": item["requires_reconciliation"], "last_error": item["last_error"],
            "backend_errors": item["backend_errors"],
        } for item in resources]}
    except (KeyError, TypeError) as error:
        raise LocalGPUError("Gran Sabio returned incomplete GPU resource details") from error


def _gpu_unload(resource_id, request):
    with GPUClient() as client:
        return client.unload(resource_id, {"request_id": str(uuid4()), **request.model_dump(exclude_none=True)})


def _gpu_http_error(error):
    status = error.status_code if error.status_code in (400, 408, 409, 422, 503) else 502
    return HTTPException(status_code=status, detail=str(error))


@router.get("/local-resources")
async def get_gpu_resources():
    """Read configured GPU resources without loading or unloading any models."""
    try:
        return await run_in_threadpool(_gpu_resources)
    except LocalGPUError as error:
        raise _gpu_http_error(error) from error


@router.post("/local-resources/{resource_id}/unload")
async def unload_gpu_model(resource_id: str, request: GPUUnload):
    """Ask Gran Sabio to release idle model memory under its admission lock."""
    try:
        return await run_in_threadpool(_gpu_unload, resource_id, request)
    except LocalGPUError as error:
        raise _gpu_http_error(error) from error

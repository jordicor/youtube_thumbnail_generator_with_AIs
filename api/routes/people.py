"""
People API Routes

Endpoints of the person editor: the frames of a video with their faces, and
creating and editing people (clusters of one view) with the photos the image
model gets of them. The setting photos belong to the video. Frames are named by
their filename in the video's frames/ folder.
"""

from typing import Dict, List, Optional, Literal
import asyncio
import base64
import io

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, ORJSONResponse, Response
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from database.db import get_db
from i18n.i18n import translate as t
from services.person_service import PersonEditError, PersonService
from services.reference_service import ReferenceService, VisualReference, reference_limit
from utils import ReferenceImageError


router = APIRouter()

VIEW_MODE_PATTERN = '^(person|person_scene)$'


# ============================================================================
# MODELS
# ============================================================================

class FramesRequest(BaseModel):
    filenames: List[str] = Field(..., min_length=1)


class MatchFramesRequest(BaseModel):
    view_mode: str = Field(..., pattern=VIEW_MODE_PATTERN)  # View of the person being built
    filenames: List[str]  # Frames of the person being built (may be empty)


class AutomaticReferencesRequest(BaseModel):
    view_mode: str = Field(..., pattern=VIEW_MODE_PATTERN)
    person_id: Optional[int] = None  # None while creating the person
    members: List[str] = Field(..., min_length=1)  # The person's frames in the editor


class UpdatePersonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Optional[str] = Field(None, max_length=128)
    description: Optional[str] = Field(None, max_length=2000)
    references: List[VisualReference] = Field(..., max_length=100)  # Face and body photos
    reference_format: Literal["photos", "collage"] = "photos"
    revision: int = Field(..., ge=0)


class CreatePersonRequest(UpdatePersonRequest):
    view_mode: str = Field(..., pattern=VIEW_MODE_PATTERN)
    members: List[str] = Field(default_factory=list)


class EnvironmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    references: List[VisualReference] = Field(..., max_length=20)
    environment_mode: Literal["same", "alternatives", "split"]
    revision: int = Field(..., ge=0)


class EnvironmentSourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["photos", "ai"]


class UploadReferenceRequest(BaseModel):
    filename: str = Field(..., max_length=255)
    image_base64: str = Field(..., max_length=14_000_000)


class PreviewReferencesRequest(BaseModel):
    cluster_ids: List[int] = Field(..., min_length=1, max_length=20)
    references: Optional[List[VisualReference]] = Field(None, max_length=100)
    formats: Dict[str, Literal["photos", "collage"]] = Field(default_factory=dict)
    environment_mode: Optional[Literal["same", "alternatives", "split"]] = None
    include_environment: bool = True
    image_provider: Literal["gemini", "openai", "poe", "replicate", "comfyui"] = "gemini"
    image_model: Optional[str] = None
    limit: Optional[int] = Field(None, ge=1)
    style_base64: Optional[str] = Field(None, max_length=14_000_000)
    draft_person: bool = False
    # Longest side of small JPEG copies to show instead of the exact PNGs
    thumbnail: Optional[int] = Field(None, ge=64, le=1024)


class MoveFramesRequest(BaseModel):
    filenames: List[str] = Field(..., min_length=1)
    target_id: Optional[int] = None  # None moves them to a new person


def _edit_error(error: PersonEditError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(error))


@router.get("/{video_id}/references")
async def reference_document(video_id: int):
    async with get_db() as db:
        try:
            return await ReferenceService(db).document(video_id)
        except ReferenceImageError as error:
            raise _edit_error(error) from error


@router.put("/{video_id}/environment")
async def save_environment(video_id: int, request: EnvironmentRequest):
    """Set the photos of the setting the thumbnails of the video may use."""
    async with get_db() as db:
        try:
            return await ReferenceService(db).save_environment(
                video_id, [r.model_dump() for r in request.references], request.environment_mode, request.revision
            )
        except ReferenceImageError as error:
            raise _edit_error(error) from error


@router.put("/{video_id}/environment/source")
async def save_environment_source(video_id: int, request: EnvironmentSourceRequest):
    """Remember whether the video's thumbnails use its setting photos or let the AI choose."""
    async with get_db() as db:
        try:
            await ReferenceService(db).save_environment_source(video_id, request.source)
        except ReferenceImageError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
    return {"environment_source": request.source}


@router.post("/{video_id}/references/upload")
async def upload_reference(video_id: int, request: UploadReferenceRequest):
    async with get_db() as db:
        try:
            return await ReferenceService(db).upload(video_id, request.filename, request.image_base64)
        except ReferenceImageError as error:
            raise _edit_error(error) from error


@router.get("/{video_id}/references/uploads/{asset_id}")
async def reference_original(video_id: int, asset_id: str):
    async with get_db() as db:
        try:
            path = await ReferenceService(db).source_path(video_id, {"source_type": "upload", "source": asset_id})
        except ReferenceImageError as error:
            raise _edit_error(error) from error
    return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})


@router.post("/{video_id}/references/crop")
async def preview_reference_crop(video_id: int, request: VisualReference):
    async with get_db() as db:
        service = ReferenceService(db)
        try:
            people = await service.people(video_id)
            if request.person_id == -1:
                people.append({"id": -1, "label": "Draft"})
            refs = await service.validate(video_id, [request.model_dump()], people, "same", require_primary=False)
            png, box = await service.crop_with_box(video_id, refs[0])
        except (ReferenceImageError, ValueError) as error:
            raise _edit_error(error) from error
    # The crop used, so the editor can start adjusting from the automatic one
    headers = {"Cache-Control": "no-store", "X-Reference-Crop": ",".join(map(str, box)) if box else ""}
    return Response(png, media_type="image/png", headers=headers)


@router.post("/{video_id}/references/preview")
async def preview_references(video_id: int, request: PreviewReferencesRequest):
    async with get_db() as db:
        try:
            maximum = reference_limit(request.image_provider, request.image_model, request.limit)
            prepared = await ReferenceService(db).prepare(
                video_id, request.cluster_ids,
                references=None if request.references is None else [r.model_dump() for r in request.references],
                formats=request.formats, environment_mode=request.environment_mode,
                include_environment=request.include_environment, style_base64=request.style_base64,
                draft_person=request.draft_person,
            )
        except (ReferenceImageError, ValueError) as error:
            raise _edit_error(error) from error
    encode = _thumbnail_url if request.thumbnail else _png_url
    images = [await asyncio.to_thread(encode, entry["png"], request.thumbnail) for entry in prepared]
    return {"count": len(prepared), "limit": maximum, "over_budget": len(prepared) > maximum,
            "images": [{**{k: v for k, v in entry.items() if k != "png"}, "image": image}
                       for entry, image in zip(prepared, images)]}


def _png_url(png: bytes, _size: Optional[int]) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _thumbnail_url(png: bytes, size: int) -> str:
    """A small JPEG copy of a reference, to show it."""
    with Image.open(io.BytesIO(png)) as image:
        image.thumbnail((size, size))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ============================================================================
# FRAMES OF A VIDEO
# ============================================================================

@router.get("/{video_id}/frames")
async def list_frames(
    video_id: int,
    view_mode: str = Query(default="person", pattern=VIEW_MODE_PATTERN),
    person_id: Optional[int] = None
):
    """Every frame of the video with the people of the view that hold it.

    With person_id, each frame carries the face closest to that person and its
    similarity, and the view is the person's own.
    """
    async with get_db() as db:
        result = await PersonService(db).list_frames(video_id, view_mode, person_id)
    if result is None:
        raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))
    # Serialized by orjson directly: FastAPI's jsonable_encoder walks every value
    # of a plain dict, about 50 ms for the frames of a long video
    return ORJSONResponse(result)


@router.post("/{video_id}/frames/match")
async def match_frames(video_id: int, request: MatchFramesRequest):
    """Face of every frame closest to the new person of a view these frames would form."""
    async with get_db() as db:
        try:
            matches = await PersonService(db).match_frames(video_id, request.view_mode, request.filenames)
        except (ReferenceImageError, ValueError) as error:
            raise _edit_error(error) from error
    if matches is None:
        raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))
    return ORJSONResponse({"matches": matches})


@router.post("/{video_id}/references/automatic")
async def automatic_references(video_id: int, request: AutomaticReferencesRequest):
    """Face photos picked for a person: their best faces, skipping black and white frames."""
    async with get_db() as db:
        try:
            references = await PersonService(db).automatic_references(
                video_id, request.view_mode, request.person_id, request.members
            )
        except (ReferenceImageError, ValueError) as error:
            raise _edit_error(error) from error
    if references is None:
        raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))
    return {"references": references}


@router.post("/{video_id}/frames/delete")
async def delete_frames(video_id: int, request: FramesRequest):
    """Delete frames from disk and from every person; people left empty are deleted."""
    async with get_db() as db:
        try:
            result = await PersonService(db).delete_frames(video_id, request.filenames)
        except (ReferenceImageError, ValueError) as error:
            raise _edit_error(error) from error
    if result is None:
        raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))
    return result


@router.get("/{video_id}/frames/{filename}/image")
async def get_frame_image(video_id: int, filename: str):
    """A frame image of the video."""
    async with get_db() as db:
        frame_path = await PersonService(db).get_frame_path(video_id, filename)
    if frame_path is None:
        raise HTTPException(status_code=404, detail=t('api.errors.frame_not_found'))
    return FileResponse(str(frame_path), media_type="image/jpeg")


# ============================================================================
# PEOPLE
# ============================================================================

@router.post("/{video_id}")
async def create_person(video_id: int, request: CreatePersonRequest):
    """Create a person of a view from frames of the video."""
    async with get_db() as db:
        try:
            person = await PersonService(db).create_person(
                video_id, request.view_mode, request.label, request.description,
                request.members, [r.model_dump() for r in request.references],
                request.reference_format, request.revision
            )
        except (ReferenceImageError, ValueError) as error:
            raise _edit_error(error) from error
    if person is None:
        raise HTTPException(status_code=404, detail=t('api.errors.video_not_found'))
    return person


@router.get("/{video_id}/{person_id}")
async def get_person(video_id: int, person_id: int):
    """A person with its frames and its ordered references."""
    async with get_db() as db:
        person = await PersonService(db).get_person(video_id, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))
    return ORJSONResponse(person)


@router.get("/{video_id}/{person_id}/body")
async def get_body_candidates(video_id: int, person_id: int):
    """Body crops of a person found by an earlier body analysis, best first."""
    async with get_db() as db:
        result = await PersonService(db).body_candidates(video_id, person_id)
    if result is None:
        raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))
    return ORJSONResponse(result)


@router.post("/{video_id}/{person_id}/references/automatic-body")
async def automatic_body(video_id: int, person_id: int):
    """Complementary body photos, analyzing locally when current measurements are missing."""
    async with get_db() as db:
        try:
            references = await PersonService(db).automatic_body(video_id, person_id)
        except (ReferenceImageError, RuntimeError, ValueError) as error:
            raise _edit_error(error) from error
    if references is None:
        raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))
    return {"references": references}


@router.put("/{video_id}/{person_id}")
async def update_person(video_id: int, person_id: int, request: UpdatePersonRequest):
    """Set the name, the notes and the face and body photos of a person."""
    async with get_db() as db:
        try:
            person = await PersonService(db).update_person(
                video_id, person_id, request.label, request.description, [r.model_dump() for r in request.references],
                request.reference_format, request.revision,
            )
        except (ReferenceImageError, ValueError) as error:
            raise _edit_error(error) from error
    if person is None:
        raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))
    return person


@router.post("/{video_id}/{person_id}/frames/add")
async def add_frames(video_id: int, person_id: int, request: FramesRequest):
    """Add frames of the video to a person."""
    async with get_db() as db:
        try:
            person = await PersonService(db).add_frames(video_id, person_id, request.filenames)
        except (ReferenceImageError, ValueError) as error:
            raise _edit_error(error) from error
    if person is None:
        raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))
    return person


@router.post("/{video_id}/{person_id}/frames/remove")
async def remove_frames(video_id: int, person_id: int, request: FramesRequest):
    """Take frames out of a person; they stay in the video."""
    async with get_db() as db:
        try:
            person = await PersonService(db).remove_frames(video_id, person_id, request.filenames)
        except (ReferenceImageError, ValueError) as error:
            raise _edit_error(error) from error
    if person is None:
        raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))
    return person


@router.post("/{video_id}/{person_id}/frames/move")
async def move_frames(video_id: int, person_id: int, request: MoveFramesRequest):
    """Move frames of a person to another person of its view, or to a new one."""
    async with get_db() as db:
        try:
            result = await PersonService(db).move_frames(
                video_id, person_id, request.filenames, request.target_id
            )
        except (ReferenceImageError, ValueError) as error:
            raise _edit_error(error) from error
    if result is None:
        raise HTTPException(status_code=404, detail=t('api.errors.cluster_not_found'))
    return result


@router.get("/{video_id}/{person_id}/face")
async def get_person_face(video_id: int, person_id: int):
    """Square crop of the person's face in its representative frame."""
    async with get_db() as db:
        image = await PersonService(db).render_avatar(video_id, person_id)
    if image is None:
        raise HTTPException(status_code=404, detail=t('api.errors.image_not_found'))
    # Callers add the representative frame ID to the URL, so a new face gets a new URL
    return Response(image, media_type="image/png", headers={"Cache-Control": "no-store"})

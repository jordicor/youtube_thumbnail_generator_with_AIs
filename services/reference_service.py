"""One collection of visual references and one preparation path for every consumer."""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import io
import uuid
from pathlib import Path
from typing import Literal

import orjson
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field

from i18n import t
from services.face_index import identity_face_row, load_face_index, portrait_crop_box, stored_centroid
from utils import ReferenceImageError, reference_png


class VisualReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    role: Literal["face", "body", "environment"]
    person_id: int | None = None
    source_type: Literal["frame", "upload"]
    source: str = Field(min_length=1, max_length=255)
    crop: tuple[int, int, int, int] | None = None
    primary: bool = False
    view: Literal["front", "three_quarter", "profile", "back"] | None = None
    expression: str | None = Field(default=None, max_length=64)
    coverage: Literal["torso", "three_quarter", "full_body", "unknown"] = "unknown"
    posture: Literal["standing", "seated", "reclined", "unknown"] = "unknown"
    side: Literal["left", "right"] | None = None
    identity_confirmed: bool = False
    # Chosen by the analysis; a new analysis may replace it. A choice the user
    # makes or changes is never automatic.
    automatic: bool = False


REFERENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS reference_assets (
    id TEXT PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    display_name TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS visual_references (
    id TEXT PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    person_id INTEGER REFERENCES clusters(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('face', 'body', 'environment')),
    position INTEGER NOT NULL,
    data TEXT NOT NULL,
    CHECK((role = 'environment' AND person_id IS NULL)
       OR (role != 'environment' AND person_id IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_visual_references_video ON visual_references(video_id, position);
CREATE INDEX IF NOT EXISTS idx_visual_references_person ON visual_references(person_id, role);
"""


def reference_limit(provider, model=None, requested=None, comfyui_profile=None):
    from config import (GEMINI_IMAGE_MODEL, GEMINI_MODEL_MAX_REFS, OPENAI_IMAGE_MODEL,
                        OPENAI_MODEL_MAX_REFS, POE_IMAGE_MODEL, POE_MODEL_MAX_REFS,
                        REPLICATE_MODEL_MAX_REFS)
    catalog = {
        "gemini": (GEMINI_IMAGE_MODEL, GEMINI_MODEL_MAX_REFS),
        "openai": (OPENAI_IMAGE_MODEL, OPENAI_MODEL_MAX_REFS),
        "poe": (POE_IMAGE_MODEL, POE_MODEL_MAX_REFS),
        "replicate": ("flux-1.1-pro", REPLICATE_MODEL_MAX_REFS),
    }
    if provider == "comfyui":
        from services.comfyui_service import Profile, load_profile
        profile = Profile.model_validate(comfyui_profile) if comfyui_profile else load_profile(model)
        catalog["comfyui"] = (profile.id, {profile.id: len(profile.references)})
    if provider not in catalog:
        raise ReferenceImageError(f"Unknown image provider: {provider}")
    default, limits = catalog[provider]
    model = model or default
    if model not in limits:
        raise ReferenceImageError(f"Unknown {provider} image model: {model}")
    if requested is not None and requested < 1:
        raise ReferenceImageError("The reference budget must be positive.")
    return min(requested, limits[model]) if requested else limits[model]


def environment_instruction(mode, side=None):
    if mode == "same":
        return "These are views of ONE shared setting. Reconstruct a coherent place."
    if mode == "alternatives":
        return "These are ALTERNATIVE settings. Choose ONE background; do not combine them."
    if mode == "split":
        return f"Use this setting in the {side.upper()} panel of a split-screen composition."
    raise ReferenceImageError("Choose a valid relationship between environment images.")


def decode_upload(encoded):
    if len(encoded) > 14_000_000:
        raise ReferenceImageError("The uploaded image exceeds 10 MB.")
    try:
        raw = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ReferenceImageError("The uploaded image is not valid base64.") from error
    if len(raw) > 10_000_000:
        raise ReferenceImageError("The uploaded image exceeds 10 MB.")
    try:
        with Image.open(io.BytesIO(raw)) as header:
            if header.width * header.height > 40_000_000:
                raise ReferenceImageError("The uploaded image exceeds 40 megapixels.")
    except (OSError, Image.DecompressionBombError) as error:
        raise ReferenceImageError("Cannot decode the uploaded image.") from error
    # Use the application's decoder, then retain the original bytes separately.
    png = reference_png(raw, "uploaded image")
    with Image.open(io.BytesIO(png)) as image:
        if image.width * image.height > 40_000_000:
            raise ReferenceImageError("The uploaded image exceeds 40 megapixels.")
        size = image.size
    with Image.open(io.BytesIO(raw)) as image:
        extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format)
    if extension is None:
        raise ReferenceImageError("Upload a JPEG, PNG, or WebP image.")
    return raw, extension, size


class ReferenceService:
    def __init__(self, db):
        self.db = db
        from services.analysis_service import AnalysisService
        self.analysis = AnalysisService(db)

    async def output_dir(self, video_id):
        video = await self.analysis.get_video(video_id)
        if video is None:
            raise ReferenceImageError("Video not found.")
        return self.analysis._get_video_output_dir(video)

    async def list(self, video_id, person_id=None):
        async with self.db.execute(
            "SELECT id, person_id, role, data FROM visual_references WHERE video_id = ?"
            + (" AND person_id = ?" if person_id is not None else "") + " ORDER BY position, id",
            [video_id, person_id] if person_id is not None else [video_id],
        ) as cursor:
            return [dict(orjson.loads(data), id=key, person_id=person, role=role)
                    for key, person, role, data in await cursor.fetchall()]

    async def people(self, video_id):
        async with self.db.execute(
            "SELECT id, label, reference_format, view_mode FROM clusters WHERE video_id = ? ORDER BY cluster_index, id",
            [video_id],
        ) as cursor:
            return [dict(zip(("id", "label", "reference_format", "view_mode"), row))
                    for row in await cursor.fetchall()]

    async def document(self, video_id):
        async with self.db.execute(
            "SELECT reference_revision, environment_mode, environment_source FROM videos WHERE id = ?", [video_id]
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise ReferenceImageError("Video not found.")
        return {"revision": row[0], "environment_mode": row[1], "environment_source": row[2],
                "references": await self.list(video_id), "people": await self.people(video_id)}

    async def save_environment_source(self, video_id, source):
        """Remember whether the video's thumbnails use its setting photos ('photos') or not ('ai')."""
        cursor = await self.db.execute("UPDATE videos SET environment_source=? WHERE id=?", [source, video_id])
        if cursor.rowcount != 1:
            raise ReferenceImageError("Video not found.")
        await self.db.commit()

    async def source_path(self, video_id, ref):
        output = await self.output_dir(video_id)
        source = ref["source"]
        if Path(source).name != source or "/" in source or "\\" in source or source in (".", ".."):
            raise ReferenceImageError("Invalid reference source.")
        if ref["source_type"] == "frame":
            path = output / "frames" / source
        else:
            async with self.db.execute(
                "SELECT filename FROM reference_assets WHERE id = ? AND video_id = ?", [source, video_id]
            ) as cursor:
                asset = await cursor.fetchone()
            if asset is None:
                raise ReferenceImageError("The uploaded original does not belong to this video.")
            path = output / "references" / "originals" / asset[0]
        if not path.is_file():
            raise ReferenceImageError(f"Reference original not found: {source}")
        return path

    async def upload(self, video_id, filename, encoded):
        output = await self.output_dir(video_id)
        raw, extension, (width, height) = await asyncio.to_thread(decode_upload, encoded)
        key = uuid.uuid4().hex
        stored_name = key + extension
        folder = output / "references" / "originals"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / stored_name
        path.write_bytes(raw)
        try:
            await self.db.execute(
                "INSERT INTO reference_assets VALUES (?, ?, ?, ?, ?, ?)",
                [key, video_id, stored_name, Path(filename).name[:255], width, height],
            )
            await self.db.commit()
        except BaseException:
            path.unlink()
            raise
        return {"source_type": "upload", "source": key, "width": width, "height": height,
                "display_name": Path(filename).name[:255]}

    async def insert(self, video_id, references):
        for position, value in enumerate(references):
            ref = VisualReference.model_validate(value).model_dump(mode="json")
            key, person, role = (ref.pop(name) for name in ("id", "person_id", "role"))
            await self.db.execute(
                "INSERT INTO visual_references (id, video_id, person_id, role, position, data) VALUES (?, ?, ?, ?, ?, ?)",
                [key, video_id, person, role, position, orjson.dumps(ref).decode()],
            )

    async def keep_automatic(self, video_id, references):
        """The references with automatic kept only where they are unchanged automatic choices."""
        stored = {ref["id"]: ref for ref in await self.list(video_id)}
        fields = ("role", "person_id", "source_type", "source", "crop", "primary", "side", "view")
        result = []
        for ref in references:
            before = stored.get(ref["id"])
            same = before is not None and before.get("automatic", False) and all(
                (list(before[f]) if f == "crop" and before[f] is not None else before[f])
                == (list(ref[f]) if f == "crop" and ref[f] is not None else ref[f]) for f in fields)
            result.append({**ref, "automatic": same})
        return result

    async def initialize_context(self, video_id, index, body_index, environment_frame):
        """Freeze available body/context evidence after analysis, preserving shared choices."""
        from services.body_references import person_body_candidates, select_body_references
        from services.face_index import match_faces
        from services.face_selection import same_look_frames

        refs = await self.list(video_id)
        additions = []
        # A setting the user chose stays; an automatic one follows the new analysis
        if all(ref.get("automatic") for ref in refs if ref["role"] == "environment"):
            await self.db.execute("DELETE FROM visual_references WHERE video_id=? AND role='environment'", [video_id])
            additions.append(VisualReference(role="environment", source_type="frame",
                                              source=environment_frame, automatic=True).model_dump())
        if body_index is not None:
            output = await self.output_dir(video_id)
            async with self.db.execute("""
                SELECT c.id, c.embedding_centroid, vf.frame_path
                FROM clusters c JOIN cluster_frame_assignments a ON a.cluster_id=c.id
                JOIN video_frames vf ON vf.id=a.frame_id WHERE c.video_id=? ORDER BY c.id, vf.id
            """, [video_id]) as cursor:
                rows = await cursor.fetchall()
            groups = {}
            for person_id, blob, path in rows:
                groups.setdefault(person_id, (blob, []))[1].append(Path(path).name)
            selected = {ref["person_id"] for ref in refs if ref["role"] == "body"}
            for person_id, (blob, members) in groups.items():
                if person_id in selected:
                    continue
                centroid = stored_centroid(blob, index, members, person_id)
                matches = match_faces(index, centroid, members)
                primary = next((ref["source"] for ref in refs if ref["person_id"] == person_id
                                and ref["role"] == "face" and ref["primary"] and ref["source_type"] == "frame"), None)
                allowed = same_look_frames(matches, index, primary)
                candidates = person_body_candidates(body_index, {name: match for name, match in matches.items()
                                                                  if name in allowed})["candidates"]
                for candidate in select_body_references(output / "frames", candidates):
                    additions.append(VisualReference(
                        role="body", person_id=person_id, source_type="frame", source=candidate["filename"],
                        crop=candidate["box"], coverage=candidate["coverage"], posture=candidate["posture"],
                        view=candidate["view"], identity_confirmed=True, automatic=True,
                    ).model_dump())
        await self.insert(video_id, additions)

    async def replace_person_faces(self, video_id, person_id, names):
        """Used when analysis creates a person; membership is a separate relation."""
        await self.db.execute("DELETE FROM visual_references WHERE person_id = ? AND role = 'face'", [person_id])
        await self.insert(video_id, [
            VisualReference(role="face", person_id=person_id, source_type="frame",
                            source=Path(name).name, primary=position == 0).model_dump()
            for position, name in enumerate(dict.fromkeys(names))
        ])

    async def validate(self, video_id, references, people, environment_mode, require_primary=True):
        known = {p["id"] for p in people}
        ids, sources = set(), set()
        cleaned = []
        for item in references:
            ref = VisualReference.model_validate(item).model_dump(mode="json")
            if ref["id"] in ids:
                raise ReferenceImageError("A reference ID occurs more than once.")
            ids.add(ref["id"])
            if ref["role"] == "environment":
                if ref["person_id"] is not None or ref["primary"]:
                    raise ReferenceImageError("Environment references belong to the video, not a person.")
            elif ref["person_id"] not in known:
                raise ReferenceImageError("A reference is assigned to a person outside this video.")
            if ref["view"] == "back" and ref["role"] != "body":
                raise ReferenceImageError("Back views are only available for body references.")
            if ref["primary"] and ref["role"] != "face":
                raise ReferenceImageError("Only a face reference can be primary.")
            key = (ref["person_id"], ref["role"], ref["source_type"], ref["source"])
            if key in sources:
                raise ReferenceImageError(t('api.errors.reference_duplicate'))
            sources.add(key)
            path = await self.source_path(video_id, ref)
            try:
                with Image.open(path) as original:
                    width, height = original.size
                    if original.getexif().get(274, 1) in (5, 6, 7, 8):
                        width, height = height, width
            except (OSError, Image.DecompressionBombError) as error:
                raise ReferenceImageError(f"Cannot read reference {ref['source']}.") from error
            crop = ref["crop"]
            if crop is not None and not (0 <= crop[0] < crop[2] <= width and 0 <= crop[1] < crop[3] <= height):
                raise ReferenceImageError(t('api.errors.reference_crop_outside'))
            if ref["role"] != "environment" and ref["source_type"] == "upload":
                if not ref["identity_confirmed"] or crop is None:
                    raise ReferenceImageError(t('api.errors.reference_upload_confirm'))
            if ref["role"] == "body" and crop is not None and not ref["identity_confirmed"]:
                raise ReferenceImageError(t('api.errors.reference_upload_confirm'))
            cleaned.append(ref)
        settings = [r for r in cleaned if r["role"] == "environment"]
        if settings:
            environment_instruction(environment_mode, "left")
            if environment_mode == "split" and {r["side"] for r in settings} != {"left", "right"}:
                raise ReferenceImageError(t('api.errors.split_needs_sides'))
        if require_primary:
            for person in people:
                faces = [r for r in cleaned if r["person_id"] == person["id"] and r["role"] == "face"]
                if sum(r["primary"] for r in faces) != 1:
                    raise ReferenceImageError(t('api.errors.primary_face_required',
                                                name=person.get("label") or person["id"]))
        return cleaned

    async def save_environment(self, video_id, references, environment_mode, revision):
        """Replace the video's setting references; people's references stay as they are."""
        if any(ref["role"] != "environment" for ref in references):
            raise ReferenceImageError("Only setting references belong to the video.")
        cleaned = await self.keep_automatic(
            video_id, await self.validate(video_id, references, [], environment_mode, require_primary=False))
        for ref in cleaned:
            await self.crop_bytes(video_id, ref)
        cursor = await self.db.execute(
            "UPDATE videos SET reference_revision=reference_revision+1, environment_mode=? WHERE id=? AND reference_revision=?",
            [environment_mode, video_id, revision],
        )
        if cursor.rowcount != 1:
            raise ReferenceImageError(t('api.errors.references_changed'))
        await self.db.execute("DELETE FROM visual_references WHERE video_id=? AND role='environment'", [video_id])
        await self.insert(video_id, cleaned)
        await self.db.commit()
        return await self.document(video_id)

    async def crop_bytes(self, video_id, ref, index=None, centroids=None):
        """PNG of the reference exactly as generation sends it."""
        return (await self.crop_with_box(video_id, ref, index, centroids))[0]

    async def crop_with_box(self, video_id, ref, index=None, centroids=None):
        """PNG of the reference and the crop it used (None for the whole image).

        Explicit crops are never redetected, enlarged, or replaced by another source;
        a face or body of a frame without one gets the automatic crop of that person.
        """
        path = await self.source_path(video_id, ref)
        try:
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
        except (OSError, Image.DecompressionBombError) as error:
            raise ReferenceImageError(f"Cannot decode reference {ref['source']}.") from error
        crop = ref["crop"]
        if ref["role"] != "environment" and ref["source_type"] == "frame":
            if index is None:
                index = await asyncio.to_thread(load_face_index, (await self.output_dir(video_id)) / "faces.json")
            # A reviewed back view can have no face. Other frame references retain
            # the normal identity/crop checks, including protection against other people.
            no_face_in_crop = crop is not None and not any(
                min(crop[2], box[2]) > max(crop[0], box[0])
                and min(crop[3], box[3]) > max(crop[1], box[1])
                for box in index.boxes[slice(*index.spans.get(path.name, (0, 0)))])
            manual_back = ref["role"] == "body" and ref["view"] == "back" and ref["identity_confirmed"] and no_face_in_crop
            if not manual_back:
                if centroids is None:
                    centroids = {}
                person = ref["person_id"]
                if person == -1:
                    if crop is None or not ref["identity_confirmed"]:
                        raise ReferenceImageError(t('api.errors.reference_new_person_crop'))
                    rows = [i for i in range(*index.spans.get(path.name, (0, 0)))
                            if crop[0] <= (index.boxes[i][0] + index.boxes[i][2]) / 2 < crop[2]
                            and crop[1] <= (index.boxes[i][1] + index.boxes[i][3]) / 2 < crop[3]]
                    if len(rows) != 1:
                        raise ReferenceImageError(t('api.errors.reference_new_person_crop'))
                    row = rows[0]
                else:
                    if person not in centroids:
                        async with self.db.execute("SELECT embedding_centroid FROM clusters WHERE id = ?", [person]) as cursor:
                            record = await cursor.fetchone()
                        centroids[person] = stored_centroid(record[0] if record else None, index, [path.name], person)
                    try:
                        row = identity_face_row(index, centroids[person], path.name)
                    except ValueError as error:
                        raise ReferenceImageError(t('api.errors.reference_wrong_person')) from error
                face = index.boxes[row].tolist()
                others = [index.boxes[i].tolist() for i in range(*index.spans[path.name]) if i != row]
                if crop is None:
                    if ref["role"] == "face":
                        crop = portrait_crop_box(face, others, *image.size)
                    else:
                        from services.body_references import load_body_index
                        data = await asyncio.to_thread(load_body_index, await self.output_dir(video_id))
                        records = data["frames"].get(path.name, []) if data else []
                        body = next((r for r in records if r["status"] == "matched" and r["face_box"] == face), None)
                        if body is None:
                            raise ReferenceImageError(t('api.errors.reference_body_crop'))
                        crop = body["box"]
                # An explicit crop must keep this person's face and exclude other indexed faces.
                if not (crop[0] <= (face[0] + face[2]) / 2 < crop[2]
                        and crop[1] <= (face[1] + face[3]) / 2 < crop[3]):
                    raise ReferenceImageError(t('api.errors.reference_crop_misses_face'))
                if any(min(crop[2], b[2]) > max(crop[0], b[0]) and
                       min(crop[3], b[3]) > max(crop[1], b[1]) for b in others):
                    raise ReferenceImageError(t('api.errors.reference_crop_other_face'))
        if crop:
            crop = tuple(int(value) for value in crop)
            if not (0 <= crop[0] < crop[2] <= image.width and 0 <= crop[1] < crop[3] <= image.height):
                raise ReferenceImageError(t('api.errors.reference_crop_outside'))
            image = image.crop(crop)
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return buffer.getvalue(), crop or None

    async def prepare(self, video_id, cluster_ids, *, references=None, formats=None,
                      environment_mode=None, include_environment=True, limit=None,
                      style_base64=None, draft_person=False):
        """Ordered, lossless bytes and mapping shared by preview, prompts, and generation."""
        if not cluster_ids or len(set(cluster_ids)) != len(cluster_ids):
            raise ReferenceImageError("Choose distinct people for the thumbnail.")
        document = await self.document(video_id)
        by_id = {p["id"]: p for p in document["people"]}
        if draft_person:
            by_id[-1] = {"id": -1, "label": "Draft", "reference_format": "photos"}
        if any(key not in by_id for key in cluster_ids):
            raise ReferenceImageError("A selected person does not belong to this video.")
        people = [by_id[key] for key in cluster_ids]
        refs = document["references"] if references is None else references
        mode = environment_mode or document["environment_mode"]
        refs = [r for r in refs if r["person_id"] in cluster_ids or (r["role"] == "environment" and include_environment)]
        refs = await self.validate(video_id, refs, people, mode)
        formats = formats or {}
        result = []
        if style_base64:
            raw, _, _ = await asyncio.to_thread(decode_upload, style_base64)
            result.append({"png": reference_png(raw, "style"), "reference_role": "style", "is_style_reference": True,
                           "cluster_id": None, "person_id": None, "source_type": "upload", "source_frame": None})
        output = await self.output_dir(video_id)
        needs_index = any(r["role"] != "environment" and r["source_type"] == "frame" for r in refs)
        index = await asyncio.to_thread(load_face_index, output / "faces.json") if needs_index else None
        centroids = {}
        for number, person in enumerate(people, 1):
            person_refs = [r for r in refs if r["person_id"] == person["id"]]
            faces = [r for r in person_refs if r["role"] == "face"]
            bodies = [r for r in person_refs if r["role"] == "body"]
            faces.sort(key=lambda r: not r["primary"])
            presentation = formats.get(str(person["id"]), person["reference_format"])
            if presentation not in ("photos", "collage"):
                raise ReferenceImageError("Unknown face presentation format.")
            face_entries = []
            for ref in faces + bodies:
                png = await self.crop_bytes(video_id, ref, index, centroids)
                entry = {
                    "png": png, "reference_id": ref["id"], "reference_role": ref["role"],
                    "cluster_id": person["id"], "person_id": f"person_{number:02d}",
                    "source_type": ref["source_type"],
                    "source_frame": ref["source"] if ref["source_type"] == "frame" else None,
                    "source_asset": ref["source"] if ref["source_type"] == "upload" else None,
                    "crop": ref["crop"], "view": ref["view"], "expression": ref["expression"],
                    "is_primary": ref["primary"], "is_style_reference": False,
                    "body_coverage": ref["coverage"], "body_posture": ref["posture"],
                    "body_reference_status": "available" if bodies else "unavailable",
                }
                if ref["role"] == "face":
                    face_entries.append(entry)
                else:
                    result.append(entry)
            if presentation == "collage":
                from services.face_sheet import render_selected_faces
                png = await asyncio.to_thread(render_selected_faces, [entry["png"] for entry in face_entries])
                face_entries = [{
                    **face_entries[0], "png": png, "reference_role": "face_sheet",
                    "reference_ids": [r["id"] for r in faces],
                    "face_sheet_views": ", ".join(
                        " / ".join(filter(None, (r["view"], r["expression"]))) or "selected face" for r in faces
                    ),
                }]
            # Faces precede bodies for each identity, without changing user order within a role.
            offset = len(result) - len(bodies)
            result[offset:offset] = face_entries
        for ref in refs:
            if ref["role"] == "environment":
                result.append({
                    "png": await self.crop_bytes(video_id, ref), "reference_id": ref["id"],
                    "reference_role": "environment", "cluster_id": None, "person_id": None,
                    "source_type": ref["source_type"], "source_frame": ref["source"] if ref["source_type"] == "frame" else None,
                    "source_asset": ref["source"] if ref["source_type"] == "upload" else None,
                    "crop": ref["crop"], "is_style_reference": False, "environment_mode": mode,
                    "environment_side": ref["side"], "environment_instruction": environment_instruction(mode, ref["side"]),
                })
        if limit is not None and len(result) > limit:
            raise ReferenceImageError(t('api.errors.references_over_limit', count=len(result), limit=limit))
        for number, entry in enumerate(result, 1):
            entry["image_index"] = number
            entry["sha256"] = hashlib.sha256(entry["png"]).hexdigest()
        return result

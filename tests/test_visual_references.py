"""Reference photos: one collection, exact pixels, explicit budgets and migration."""
import base64
import hashlib
import io
import re
from pathlib import Path
from unittest.mock import AsyncMock

import orjson
import pytest
from PIL import Image

from services.person_service import PersonService
from services.reference_service import ReferenceService, VisualReference, reference_limit
from i18n.i18n import translate as t
from tests.test_person_editor import LEFT_FACE, RIGHT_FACE
from tests.test_person_editor import people_video as people_video  # noqa: PLC0414 -- pytest fixture re-export
from utils import ReferenceImageError


def face(person, source, **kwargs):
    return VisualReference(role="face", person_id=person, source_type="frame", source=source, **kwargs).model_dump()


def picture(color=(30, 120, 180), size=(100, 80), fmt="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, fmt)
    return stream.getvalue()


async def test_merge_keeps_manual_crops_uploads_and_primary(test_db, people_video):
    from services.analysis_service import AnalysisService

    video = people_video
    service = ReferenceService(test_db)
    asset = await service.upload(video['video_id'], 'portrait.png', base64.b64encode(picture()).decode())
    refs = [
        face(video['ana'], video['names'][0], crop=(0, 0, 140, 180), identity_confirmed=True),
        VisualReference(role='face', person_id=video['ana'], source_type='upload',
                        source=asset['source'], crop=(0, 0, 100, 80), primary=True,
                        identity_confirmed=True).model_dump(),
        face(video['bea'], video['names'][1], crop=(0, 0, 150, 180), identity_confirmed=True),
    ]
    await test_db.execute("DELETE FROM visual_references WHERE person_id IN (?, ?)",
                          [video['ana'], video['bea']])
    await service.insert(video['video_id'], refs)
    await test_db.commit()

    result = await AnalysisService(test_db).merge_clusters(video['video_id'], [0, 1], 0)

    assert result['success']
    merged = await service.list(video['video_id'], video['ana'])
    assert {ref['id'] for ref in merged} == {ref['id'] for ref in refs}
    assert [ref['id'] for ref in merged if ref['primary']] == [refs[1]['id']]
    assert {tuple(ref['crop']) for ref in merged} == {(0, 0, 140, 180), (0, 0, 150, 180), (0, 0, 100, 80)}
    assert all(ref['person_id'] == video['ana'] for ref in merged)
    assert (await service.source_path(video['video_id'], merged[0])).exists()
    prepared = await service.prepare(video['video_id'], [video['ana']], formats={str(video['ana']): 'photos'})
    assert {entry['reference_id'] for entry in prepared} == {ref['id'] for ref in refs}


async def save_payload(db, video, refs=None):
    document = await ReferenceService(db).document(video["video_id"])
    return {
        "label": "Ana", "description": "Video notes",
        "references": refs if refs is not None else [r for r in document["references"] if r["person_id"] == video["ana"]],
        "reference_format": "photos", "revision": document["revision"],
    }


async def environment_payload(db, video, refs, mode="same"):
    document = await ReferenceService(db).document(video["video_id"])
    return {"references": refs, "environment_mode": mode, "revision": document["revision"]}


async def test_upload_keeps_original_and_never_creates_a_video_frame(client, test_db, people_video):
    raw = picture(fmt="JPEG")
    service = ReferenceService(test_db)
    async with test_db.execute("SELECT COUNT(*) FROM video_frames") as cursor:
        before = (await cursor.fetchone())[0]
    response = await client.post(f"/api/people/{people_video['video_id']}/references/upload",
                                 json={"filename": "portrait.jpg", "image_base64": base64.b64encode(raw).decode()})
    assert response.status_code == 200, response.text
    asset = response.json()
    path = await service.source_path(people_video["video_id"], asset)
    assert path.read_bytes() == raw
    async with test_db.execute("SELECT COUNT(*) FROM video_frames") as cursor:
        assert (await cursor.fetchone())[0] == before
    assert asset["source_type"] == "upload"
    assert (await client.get(f"/api/people/{people_video['video_id']}/references/uploads/{asset['source']}")).content == raw
    from tests.conftest import insert_video
    await insert_video(test_db, "other.mp4", "analyzed")
    with pytest.raises(ReferenceImageError, match="does not belong"):
        await service.source_path(people_video["video_id"] + 1, asset)


async def test_uploaded_person_can_be_created_without_inventing_frame_membership(client, test_db, people_video):
    service = ReferenceService(test_db)
    raw = picture(size=(180, 240))
    asset = await service.upload(people_video["video_id"], "person.png", base64.b64encode(raw).decode())
    ref = VisualReference(role="face", person_id=-1, source_type="upload", source=asset["source"],
                          primary=True, crop=(5, 10, 170, 230), identity_confirmed=True).model_dump()
    payload = await save_payload(test_db, people_video, [ref])
    payload.update(view_mode="person", members=[], label="Uploaded person")
    response = await client.post(f"/api/people/{people_video['video_id']}", json=payload)
    assert response.status_code == 200, response.text
    person = response.json()
    assert person["members"] == [] and person["num_frames"] == 0
    assert person["references"][0]["source_type"] == "upload"
    assert person["references"][0]["person_id"] == person["id"]
    prepared = await service.prepare(people_video["video_id"], [person["id"]])
    assert prepared[0]["source_frame"] is None
    assert Image.open(io.BytesIO(prepared[0]["png"])).size == (165, 220)


async def test_upload_requires_explicit_crop_and_identity_confirmation(client, test_db, people_video):
    service = ReferenceService(test_db)
    asset = await service.upload(people_video["video_id"], "group.png", base64.b64encode(picture()).decode())
    ref = VisualReference(role="body", person_id=people_video["ana"], source_type="upload", source=asset["source"]).model_dump()
    url = f"/api/people/{people_video['video_id']}/references/crop"
    assert (await client.post(url, json=ref)).status_code == 400
    ref.update(crop=[10, 10, 70, 70], identity_confirmed=True)
    cropped = await client.post(url, json=ref)
    assert cropped.status_code == 200
    assert Image.open(io.BytesIO(cropped.content)).size == (60, 60)
    ref["crop"] = [-1, 0, 80, 80]
    assert (await client.post(url, json=ref)).status_code == 400
    assert (await service.source_path(people_video["video_id"], ref)).read_bytes() == picture()


async def test_shared_frame_crops_keep_the_selected_person(client, test_db, people_video):
    service = ReferenceService(test_db)
    ref = face(people_video["bea"], people_video["names"][4], primary=True)
    data = await service.crop_bytes(people_video["video_id"], ref)
    assert Image.open(io.BytesIO(data)).size == (150, 180)
    ref.update(crop=[0, 0, 320, 180], identity_confirmed=True)
    response = await client.post(f"/api/people/{people_video['video_id']}/references/crop", json=ref)
    assert response.status_code == 400 and t('api.errors.reference_crop_other_face') in response.json()["detail"]
    ref["crop"] = [0, 0, 140, 180]
    assert (await client.post(f"/api/people/{people_video['video_id']}/references/crop", json=ref)).status_code == 400


async def test_review_uses_identical_bytes_order_and_roles_without_saving(client, test_db, people_video):
    video_id = people_video["video_id"]
    service = ReferenceService(test_db)
    initial = await service.document(video_id)
    refs = [
        face(people_video["ana"], people_video["names"][0], primary=True, view="front", expression="smiling"),
        face(people_video["bea"], people_video["names"][4], primary=True),
        VisualReference(role="body", person_id=people_video["ana"], source_type="frame",
                        source=people_video["names"][0], crop=(0, 0, 160, 180),
                        coverage="torso", posture="seated", identity_confirmed=True).model_dump(),
        VisualReference(role="environment", source_type="frame", source=people_video["names"][2]).model_dump(),
    ]
    style = base64.b64encode(picture()).decode()
    response = await client.post(f"/api/people/{video_id}/references/preview", json={
        "cluster_ids": [people_video["bea"], people_video["ana"]], "references": refs,
        "image_provider": "openai", "image_model": "gpt-image-2", "style_base64": style,
        "environment_mode": "same", "limit": 10,
    })
    assert response.status_code == 200, response.text
    review = response.json()
    prepared = await service.prepare(video_id, [people_video["bea"], people_video["ana"]],
                                     references=refs, environment_mode="same", style_base64=style, limit=10)
    assert [r["reference_role"] for r in prepared] == ["style", "face", "face", "body", "environment"]
    assert [r["person_id"] for r in prepared] == [None, "person_01", "person_02", "person_02", None]
    for shown, sent in zip(review["images"], prepared):
        assert base64.b64decode(shown["image"].split(",", 1)[1]) == sent["png"]
        assert shown["sha256"] == hashlib.sha256(sent["png"]).hexdigest()
    assert await service.document(video_id) == initial


async def test_budget_counts_every_person_body_environment_and_style_and_never_truncates(client, test_db, people_video):
    service = ReferenceService(test_db)
    refs = [face(people_video["ana"], people_video["names"][0], primary=True),
            face(people_video["ana"], people_video["names"][1])]
    style = base64.b64encode(picture()).decode()
    response = await client.post(f"/api/people/{people_video['video_id']}/references/preview", json={
        "cluster_ids": [people_video["ana"]], "references": refs, "limit": 2,
        "style_base64": style,
    })
    assert response.status_code == 200
    assert response.json()["over_budget"] and len(response.json()["images"]) == 3
    with pytest.raises(ReferenceImageError, match=re.escape(t('api.errors.references_over_limit', count=3, limit=2))):
        await service.prepare(people_video["video_id"], [people_video["ana"]], references=refs, style_base64=style, limit=2)


async def test_collage_is_an_alternate_format_of_exactly_the_same_faces(test_db, people_video):
    service = ReferenceService(test_db)
    refs = [face(people_video["ana"], people_video["names"][0], primary=True),
            face(people_video["ana"], people_video["names"][1], view="profile")]
    entries = await service.prepare(people_video["video_id"], [people_video["ana"]], references=refs,
                                    formats={str(people_video["ana"]): "collage"}, limit=1)
    assert len(entries) == 1 and entries[0]["reference_role"] == "face_sheet"
    assert entries[0]["reference_ids"] == [r["id"] for r in refs]
    assert (await service.document(people_video["video_id"]))["people"][0]["reference_format"] == "photos"


async def test_one_photo_can_cover_angle_and_expression_without_duplication(client, test_db, people_video):
    refs = [face(people_video["ana"], people_video["names"][0], primary=True, view="three_quarter", expression="smiling")]
    payload = await save_payload(test_db, people_video, refs)
    url = f"/api/people/{people_video['video_id']}/{people_video['ana']}"
    result = await client.put(url, json=payload)
    assert result.status_code == 200, result.text
    assert len(result.json()["references"]) == 1
    payload = await save_payload(test_db, people_video, [*refs, dict(refs[0], id="duplicate", primary=False)])
    result = await client.put(url, json=payload)
    assert result.status_code == 400 and result.json()["detail"] == t('api.errors.reference_duplicate')


async def test_primary_is_required_and_stale_editor_cannot_overwrite_choices(client, test_db, people_video):
    url = f"/api/people/{people_video['video_id']}/{people_video['ana']}"
    payload = await save_payload(test_db, people_video)
    stale = dict(payload, label="Stale name")
    assert (await client.put(url, json=payload)).status_code == 200
    result = await client.put(url, json=stale)
    assert result.status_code == 400 and result.json()["detail"] == t('api.errors.references_changed')
    current = await save_payload(test_db, people_video)
    current["references"] = [dict(ref, primary=False) for ref in current["references"]]
    assert (await client.put(url, json=current)).status_code == 400
    assert (await client.get(url)).json()["label"] == "Ana"


async def test_environment_is_shared_and_split_assignments_reach_both_prompts(client, test_db, people_video):
    from gransabio_prompt_generator import _build_reference_mapping_section
    from image_generation import _build_face_source_header
    settings = [VisualReference(role="environment", source_type="frame", source=source, side=side).model_dump()
                for source, side in zip(people_video["names"][2:4], ("left", "right"))]
    url = f"/api/people/{people_video['video_id']}/environment"
    response = await client.put(url, json=await environment_payload(test_db, people_video, settings, "split"))
    assert response.status_code == 200, response.text
    assert response.json()["environment_mode"] == "split"
    prepared = await ReferenceService(test_db).prepare(people_video["video_id"], [people_video["ana"]])
    assert [r["environment_side"] for r in prepared if r["reference_role"] == "environment"] == ["left", "right"]
    mapping = [{k: v for k, v in r.items() if k != "png"} for r in prepared]
    for text in (_build_reference_mapping_section(mapping), _build_face_source_header(mapping)):
        assert "LEFT" in text and "RIGHT" in text
    assert all(ref["person_id"] is None for ref in settings)
    one_side = await client.put(url, json=await environment_payload(test_db, people_video, [settings[0]], "split"))
    assert one_side.status_code == 400 and one_side.json()["detail"] == t('api.errors.split_needs_sides')


async def test_invalid_budget_fails_before_enqueue_and_keeps_video_unclaimed(client, test_db, people_video, monkeypatch):
    from api.routes import generation
    queue = AsyncMock()
    monkeypatch.setattr(generation, "enqueue_generation", queue)
    response = await client.post(f"/api/generation/{people_video['video_id']}/start",
                                 json={"cluster_ids": [people_video["ana"]], "num_reference_images": 1})
    assert response.status_code == 400
    assert response.json()["detail"] == t('api.errors.references_over_limit', count=2, limit=1)
    queue.assert_not_called()
    async with test_db.execute("SELECT status FROM videos WHERE id=?", [people_video["video_id"]]) as cursor:
        assert (await cursor.fetchone())[0] == "analyzed"


async def test_suggestions_do_not_fill_a_quota_with_redundant_angles(client, people_video):
    response = await client.post(f"/api/people/{people_video['video_id']}/references/automatic", json={
        "view_mode": "person", "person_id": people_video["ana"], "members": people_video["names"][:5],
    })
    assert response.status_code == 200, response.text
    refs = response.json()["references"]
    assert len(refs) == 1 and refs[0]["primary"]
    assert refs[0]["view"] is None  # These fixtures contain no yaw; do not invent an angle.


async def test_migration_preserves_manual_choices_deduplicates_and_is_idempotent(test_db, people_video):
    from database.reference_migration import migrate_visual_references
    video_id, ana = people_video["video_id"], people_video["ana"]
    names = people_video["names"]
    await test_db.execute("DROP TABLE visual_references")
    await test_db.execute("DROP TABLE reference_assets")
    for name, definition in (("face_sheet", "TEXT"), ("face_sheet_enabled", "INTEGER DEFAULT 0"),
                             ("body_reference", "TEXT"), ("environment_reference", "TEXT")):
        await test_db.execute(f"ALTER TABLE clusters ADD COLUMN {name} {definition}")
    await test_db.execute("ALTER TABLE cluster_frame_assignments ADD COLUMN is_reference INTEGER DEFAULT 0")
    await test_db.execute("ALTER TABLE cluster_frame_assignments ADD COLUMN reference_order INTEGER")
    await test_db.execute("""
        UPDATE cluster_frame_assignments SET is_reference=1, reference_order=1
        WHERE cluster_id=? AND frame_id=(SELECT id FROM video_frames WHERE frame_path LIKE ?)
    """, [ana, "%/" + names[0]])
    sheet = {"front": names[0], "smiling": names[0], "profile": names[1], "mouth_open": None}
    await test_db.execute("UPDATE clusters SET face_sheet=?, body_reference=?, environment_reference=? WHERE id=?",
                         [orjson.dumps(sheet).decode(), names[0], names[2], ana])
    await test_db.execute("UPDATE clusters SET environment_reference=? WHERE id=?", [names[3], people_video["bea"]])
    originals = {p.name: p.read_bytes() for p in (people_video["dir"] / "frames").glob("*.jpg")}
    await migrate_visual_references(test_db)
    service = ReferenceService(test_db)
    migrated = await service.document(video_id)
    faces = [r for r in migrated["references"] if r["person_id"] == ana and r["role"] == "face"]
    assert [r["source"] for r in faces] == names[:2]
    assert faces[0]["view"] == "front" and faces[0]["expression"] == "smiling" and faces[0]["primary"]
    assert migrated["environment_mode"] == "alternatives"
    assert all(r["person_id"] is None for r in migrated["references"] if r["role"] == "environment")
    await migrate_visual_references(test_db)
    assert await service.document(video_id) == migrated
    async with test_db.execute("PRAGMA table_info(clusters)") as cursor:
        assert not {"face_sheet", "body_reference", "environment_reference"} & {r[1] for r in await cursor.fetchall()}
    async with test_db.execute("PRAGMA table_info(cluster_frame_assignments)") as cursor:
        assert "is_reference" not in {r[1] for r in await cursor.fetchall()}
    assert originals == {p.name: p.read_bytes() for p in (people_video["dir"] / "frames").glob("*.jpg")}


async def test_editor_pages_show_frames_photos_and_crop_in_both_languages(client, people_video):
    import json
    root = Path(__file__).resolve().parents[1]
    texts = {language: json.loads((root / "static/locales" / f"{language}.json").read_text(encoding="utf-8"))
             for language in ("en", "es")}
    for section in ("person_editor", "crop"):
        assert texts["en"][section].keys() == texts["es"][section].keys()
    assert "reference_editor" not in texts["es"]
    for url in (f"/video/{people_video['video_id']}/person/{people_video['ana']}",
                f"/video/{people_video['video_id']}/person/new?view_mode=person"):
        response = await client.get(url)
        assert response.status_code == 200
        for element in ("peGrid", "peFaceSlots", "peBodySlots", "peSelectionBar", "cropDialog", "pePreviewDialog"):
            assert f'id="{element}"' in response.text
    page = await client.get(f"/video/{people_video['video_id']}")
    assert 'id="backgroundStep"' in page.text and 'id="cropDialog"' in page.text
    # A nested form would close the Generate form early and leave its button outside
    form = page.text.split('<form id="generateForm"', 1)[1].split("</form>", 1)[0]
    assert "<form" not in form and 'id="generateBtn"' in form



async def test_migration_failure_rolls_back_schema_and_data():
    import aiosqlite
    from database.reference_migration import migrate_visual_references
    async with aiosqlite.connect(":memory:") as db:
        await db.executescript("""
            CREATE TABLE videos(id INTEGER PRIMARY KEY, filename TEXT, filepath TEXT);
            CREATE TABLE clusters(id INTEGER PRIMARY KEY, video_id INTEGER, cluster_index INTEGER,
                view_mode TEXT, face_sheet TEXT, face_sheet_enabled INTEGER, body_reference TEXT,
                environment_reference TEXT);
            CREATE TABLE video_frames(id INTEGER PRIMARY KEY, frame_path TEXT, quality_score REAL);
            CREATE TABLE cluster_frame_assignments(cluster_id INTEGER, frame_id INTEGER,
                is_reference INTEGER, reference_order INTEGER);
            INSERT INTO videos VALUES(1, 'fixture.mp4', '/fake/fixture.mp4');
            INSERT INTO clusters VALUES(1,1,0,'person',NULL,0,'','');
            INSERT INTO clusters VALUES(2,1,1,'person','invalid json',0,'','');
        """)
        with pytest.raises(orjson.JSONDecodeError):
            await migrate_visual_references(db)
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
            assert "visual_references" not in {row[0] for row in await cursor.fetchall()}
        async with db.execute("PRAGMA table_info(clusters)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
            assert "face_sheet" in columns and "reference_format" not in columns
        async with db.execute("SELECT COUNT(*) FROM clusters") as cursor:
            assert (await cursor.fetchone())[0] == 2


async def test_uploaded_only_person_has_explicit_no_embedding_score_reference(test_db, people_video):
    from services.face_match import load_job_references
    from services.generation_service import GenerationService
    service = ReferenceService(test_db)
    asset = await service.upload(people_video["video_id"], "person.png", base64.b64encode(picture()).decode())
    ref = VisualReference(role="face", person_id=-1, source_type="upload", source=asset["source"],
                          primary=True, crop=(0, 0, 100, 80), identity_confirmed=True).model_dump()
    person = await PersonService(test_db).create_person(
        people_video["video_id"], "person", "Uploaded", None, [], [ref], "photos", 0,
    )
    job = await GenerationService(test_db).create_generation_job(people_video["video_id"], [person["id"]])
    (people_video["dir"] / "faces.json").unlink()
    refs = await load_job_references(test_db, job["id"])
    assert len(refs) == 1 and refs[0].embedding is None



async def test_manual_face_references_can_exceed_automatic_default(client, test_db, people_video):
    refs = [face(people_video["ana"], name, primary=i == 0)
            for i, name in enumerate(people_video["names"][:4])]
    payload = await save_payload(test_db, people_video, refs)
    response = await client.put(f"/api/people/{people_video['video_id']}/{people_video['ana']}", json=payload)
    assert response.status_code == 200, response.text
    prepared = await ReferenceService(test_db).prepare(people_video["video_id"], [people_video["ana"]], limit=6)
    assert len(prepared) == 4 and all(r["reference_role"] == "face" for r in prepared)
    assert [r["source_frame"] for r in prepared] == people_video["names"][:4]


async def test_new_person_automatic_faces_have_reviewable_crops(client, people_video):
    response = await client.post(f"/api/people/{people_video['video_id']}/references/automatic", json={
        "view_mode": "person_scene", "person_id": None, "members": [people_video["names"][1]],
    })
    assert response.status_code == 200, response.text
    refs = response.json()["references"]
    assert refs and refs[0]["crop"] and refs[0]["identity_confirmed"]
    preview = await client.post(f"/api/people/{people_video['video_id']}/references/preview", json={
        "cluster_ids": [-1], "references": refs, "draft_person": True, "include_environment": False,
    })
    assert preview.status_code == 200, preview.text
    assert preview.json()["count"] == len(refs)


async def test_optional_uploaded_back_view_survives_save_and_generation(client, test_db, people_video):
    video_id, person_id = people_video["video_id"], people_video["ana"]
    service = ReferenceService(test_db)
    asset = await service.upload(video_id, "back.png", base64.b64encode(picture(size=(180, 240))).decode())
    payload = await save_payload(test_db, people_video)
    back = VisualReference(role="body", person_id=person_id, source_type="upload", source=asset["source"],
                           crop=(10, 10, 170, 230), view="back", identity_confirmed=True).model_dump()
    payload["references"].append(back)
    response = await client.put(f"/api/people/{video_id}/{person_id}", json=payload)
    assert response.status_code == 200, response.text
    prepared = await service.prepare(video_id, [person_id])
    bodies = [entry for entry in prepared if entry["reference_role"] == "body"]
    assert len(bodies) == 1 and bodies[0]["view"] == "back"
    assert Image.open(io.BytesIO(bodies[0]["png"])).size == (160, 220)


async def test_faceless_back_frame_requires_manual_identity_and_keeps_other_person_checks(client, people_video):
    video_id, person_id = people_video["video_id"], people_video["ana"]
    url = f"/api/people/{video_id}/references/crop"
    ref = VisualReference(role="body", person_id=person_id, source_type="frame", source=people_video["names"][7],
                          crop=(0, 0, 160, 180), view="back").model_dump()
    assert (await client.post(url, json=ref)).status_code == 400
    ref["identity_confirmed"] = True
    assert (await client.post(url, json=ref)).status_code == 200
    ref["view"] = "front"
    assert (await client.post(url, json=ref)).status_code == 400
    ref.update(view="back", source=people_video["names"][5], crop=(0, 0, 320, 180))
    assert (await client.post(url, json=ref)).status_code == 400
    ref.update(role="face")
    assert (await client.post(url, json=ref)).status_code == 400

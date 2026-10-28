"""Photos of people and of the setting as the editors save them, crop them and show them."""
import base64
import io

from PIL import Image

from i18n.i18n import translate as t
from services.body_references import select_body_references
from services.face_index import portrait_crop_box
from services.reference_service import ReferenceService, VisualReference
from tests.test_person_editor import FRAME_SIZE, LEFT_FACE, RIGHT_FACE, people_video  # noqa: F401


def setting(source, **fields):
    return VisualReference(role="environment", source_type="frame", source=source, **fields).model_dump()


async def revision(db, video):
    return (await ReferenceService(db).document(video["video_id"]))["revision"]


async def person_payload(db, video):
    document = await ReferenceService(db).document(video["video_id"])
    return {"label": "Ana", "description": None, "reference_format": "photos", "revision": document["revision"],
            "references": [r for r in document["references"] if r["person_id"] == video["ana"]]}


async def test_saving_a_person_keeps_the_setting_of_the_video(client, test_db, people_video):
    video_id = people_video["video_id"]
    saved = await client.put(f"/api/people/{video_id}/environment", json={
        "references": [setting(people_video["names"][2])], "environment_mode": "same",
        "revision": await revision(test_db, people_video),
    })
    assert saved.status_code == 200, saved.text

    url = f"/api/people/{video_id}/{people_video['ana']}"
    payload = await person_payload(test_db, people_video)
    assert (await client.put(url, json=payload)).status_code == 200
    document = await ReferenceService(test_db).document(video_id)
    assert [r["source"] for r in document["references"] if r["role"] == "environment"] == [people_video["names"][2]]

    payload = await person_payload(test_db, people_video)
    payload["references"].append(setting(people_video["names"][3]))
    assert (await client.put(url, json=payload)).status_code == 400


async def test_the_setting_takes_only_its_photos_and_refuses_stale_editors(client, test_db, people_video):
    url = f"/api/people/{people_video['video_id']}/environment"
    current = await revision(test_db, people_video)
    face = VisualReference(role="face", person_id=people_video["ana"], source_type="frame",
                           source=people_video["names"][0], primary=True).model_dump()
    assert (await client.put(url, json={"references": [face], "environment_mode": "same",
                                        "revision": current})).status_code == 400
    assert (await client.put(url, json={"references": [], "environment_mode": "same",
                                        "revision": current})).status_code == 200
    stale = await client.put(url, json={"references": [], "environment_mode": "same", "revision": current})
    assert stale.status_code == 400 and stale.json()["detail"] == t('api.errors.references_changed')


async def test_crop_preview_says_which_crop_it_used(client, people_video):
    url = f"/api/people/{people_video['video_id']}/references/crop"
    names = people_video["names"]

    alone = VisualReference(role="face", person_id=people_video["ana"], source_type="frame", source=names[0]).model_dump()
    response = await client.post(url, json=alone)
    assert response.status_code == 200, response.text
    expected = portrait_crop_box(LEFT_FACE, [], *FRAME_SIZE)
    assert response.headers["X-Reference-Crop"] == ",".join(map(str, expected))
    assert Image.open(io.BytesIO(response.content)).size == (expected[2] - expected[0], expected[3] - expected[1])

    # Beside another face the automatic crop stops halfway
    together = dict(alone, source=names[4])
    response = await client.post(url, json=together)
    assert response.headers["X-Reference-Crop"] == ",".join(map(str, portrait_crop_box(LEFT_FACE, [RIGHT_FACE], *FRAME_SIZE)))

    explicit = dict(alone, crop=[10, 10, 130, 170], identity_confirmed=True)
    assert (await client.post(url, json=explicit)).headers["X-Reference-Crop"] == "10,10,130,170"
    assert (await client.post(url, json=setting(names[2]))).headers["X-Reference-Crop"] == ""


async def test_preview_thumbnails_are_small_copies(client, people_video):
    response = await client.post(f"/api/people/{people_video['video_id']}/references/preview", json={
        "cluster_ids": [people_video["ana"]], "include_environment": False, "thumbnail": 64,
    })
    assert response.status_code == 200, response.text
    images = response.json()["images"]
    assert len(images) == 2
    for entry in images:
        assert entry["image"].startswith("data:image/jpeg;base64,")
        image = Image.open(io.BytesIO(base64.b64decode(entry["image"].split(",", 1)[1])))
        assert max(image.size) <= 64


async def test_frames_leaving_a_person_take_its_photos_and_pass_on_the_primary(client, test_db, people_video):
    video_id, ana = people_video["video_id"], people_video["ana"]
    f1, f2 = people_video["names"][:2]
    before = await revision(test_db, people_video)
    response = await client.post(f"/api/people/{video_id}/{ana}/frames/remove", json={"filenames": [f2]})
    assert response.status_code == 200, response.text
    faces = [r for r in response.json()["references"] if r["role"] == "face"]
    assert [(r["source"], r["primary"]) for r in faces] == [(f1, True)]
    assert await revision(test_db, people_video) == before + 1


async def test_automatic_body_skips_black_and_white_frames(client, people_video, monkeypatch, tmp_path):
    color = tmp_path / "scene_000_frame_000002.jpg"
    gray = tmp_path / "scene_000_frame_000001.jpg"
    Image.new("RGB", FRAME_SIZE, (200, 60, 40)).save(color)
    Image.new("RGB", FRAME_SIZE, (90, 90, 90)).save(gray)
    import orjson
    (tmp_path.parent / "scenes.json").write_bytes(orjson.dumps({"scenes": [
        {"index": 0, "start_frame": 0, "end_frame": 3, "duration": 3, "frame_indices": [1, 2]}
    ]}))
    candidates = [{"filename": path.name, "face_box": LEFT_FACE, "quality": 80, "view": "front"} for path in [gray, color]]
    assert select_body_references(tmp_path, candidates)[0]["filename"] == color.name
    assert select_body_references(tmp_path, candidates[:1]) == []

    # The fixture's frames are all gray: there is no body to pick
    from services.person_service import PersonService

    async def gray_bodies(self, video_id, person_id, analyze=False):
        return {"status": "ready", "excluded": {}, "candidates": [{
            "filename": people_video["names"][0], "face_box": LEFT_FACE, "box": [0, 0, 160, 180],
            "coverage": "torso", "view": "front", "posture": "seated", "quality": 1.0, "status": "matched",
        }]}

    monkeypatch.setattr(PersonService, "body_candidates", gray_bodies)
    response = await client.post(f"/api/people/{people_video['video_id']}/{people_video['ana']}/references/automatic-body")
    assert response.status_code == 400 and response.json()["detail"] == t('api.errors.no_color_body')

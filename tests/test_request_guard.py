"""Other websites cannot use the app through the user's browser, and odd
requests get a clean answer instead of touching the disk or crashing."""

import os

import pytest

from api.middleware import request_guard
from api.middleware.request_guard import RequestGuardMiddleware
from tests.conftest import insert_video
from tests.test_thumbnail_downloads import _insert_job, _insert_thumbnail


async def test_cross_site_post_is_refused(client, test_db):
    video_id = await insert_video(test_db, "kept.mp4", "pending")

    response = await client.delete(f"/api/videos/{video_id}", headers={"Origin": "https://evil.example"})

    assert response.status_code == 403
    async with test_db.execute("SELECT COUNT(*) FROM videos WHERE id = ?", (video_id,)) as cursor:
        assert (await cursor.fetchone())[0] == 1


async def test_plain_text_json_from_another_site_is_refused(client):
    # No CORS preflight for text/plain, and FastAPI would still parse the JSON
    response = await client.post(
        "/api/lang", content='{"lang": "en"}',
        headers={"Content-Type": "text/plain", "Origin": "https://evil.example"},
    )

    assert response.status_code == 403
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize("origin", ["http://localhost", "http://192.168.1.20:3000", "http://127.0.0.1:8057"])
async def test_own_pages_and_lan_origins_can_post(client, origin):
    response = await client.post("/api/lang", json={"lang": "en"}, headers={"Origin": origin})

    assert response.status_code == 200


async def test_clients_without_origin_can_post(client):
    # The CLI and scripts send no Origin
    response = await client.post("/api/lang", json={"lang": "en"})

    assert response.status_code == 200


async def test_browser_request_without_origin_from_another_site_is_refused(client):
    response = await client.post("/api/lang", json={"lang": "en"}, headers={"Sec-Fetch-Site": "cross-site"})

    assert response.status_code == 403


async def test_unknown_host_is_refused(client):
    # A website pointing its domain at a LAN address (DNS rebinding)
    response = await client.get("/api/lang", headers={"Host": "evil.example"})

    assert response.status_code == 400
    assert "ALLOWED_HOSTS" in response.json()["detail"]


@pytest.mark.parametrize("host", ["192.168.1.5:8057", "[::1]:8057", "localhost", "app.localhost:8057"])
async def test_addresses_and_local_names_are_allowed(client, host):
    response = await client.get("/api/lang", headers={"Host": host})

    assert response.status_code == 200


def test_extra_host_names_come_from_the_configuration(monkeypatch):
    monkeypatch.setattr(request_guard, "ALLOWED_HOSTS", ["thumbs.home"])
    guard = RequestGuardMiddleware(app=None)

    assert guard._host_allowed("thumbs.home:8057")
    assert not guard._host_allowed("other.home:8057")


@pytest.mark.parametrize("path", [
    "/static/%5C%5Cattacker%5Cshare%5Cfile.css",
    "/output/%5C%5Cattacker%5Cshare%5Cfile.jpg",
    "/output/C:%5Cwindows%5Cwin.ini",
])
async def test_static_paths_with_backslash_or_colon_are_not_looked_up(client, monkeypatch, path):
    # StaticFiles resolves the joined path before checking it stays in its folder;
    # resolving a UNC path already connects to that host
    resolved = []
    real_realpath = os.path.realpath
    monkeypatch.setattr(os.path, "realpath", lambda target, *args, **kwargs: resolved.append(str(target)) or real_realpath(target, *args, **kwargs))
    monkeypatch.setattr(os.path, "abspath", lambda target, real=os.path.abspath: resolved.append(str(target)) or real(target))

    response = await client.get(path)

    assert response.status_code == 404
    assert not [target for target in resolved if "attacker" in target or "win.ini" in target]


async def test_id_beyond_sqlite_integers_is_a_client_error(client):
    response = await client.get(f"/api/videos/{2 ** 64}")

    assert response.status_code == 422


async def test_download_names_keep_any_character(client, test_db):
    video_id = await insert_video(test_db, "Ñandú 🎬.mp4", "completed")
    job_id = await _insert_job(test_db, video_id)
    await _insert_thumbnail(test_db, job_id, 0, "Título", "a.png", b"png-a")
    await test_db.commit()

    response = await client.get(f"/api/thumbnails/video/{video_id}/download-all")

    assert response.status_code == 200
    disposition = response.headers["content-disposition"]
    # Old clients get every character outside ASCII as "_"
    assert disposition.startswith('attachment; filename="all_thumbnails__and_ __')
    assert "filename*=UTF-8''all_thumbnails_%C3%91and%C3%BA%20%F0%9F%8E%AC_" in disposition


def test_removed_endpoints_are_gone(app):
    paths = app.openapi()["paths"]

    # A GET that registered videos and an upload that overwrote files
    assert "/api/videos/scan" not in paths
    assert "/api/videos/upload" not in paths


@pytest.mark.parametrize("name", ["..", ".", "...", "   ", ""])
def test_names_that_point_at_a_parent_folder_are_refused(name):
    # Imported here: importing utils rewraps stdout, which breaks pytest's capture during collection
    from utils import sanitize_filename

    with pytest.raises(ValueError):
        sanitize_filename(name)


def test_ordinary_names_are_kept():
    from utils import sanitize_filename

    assert sanitize_filename("my.video") == "my.video"
    assert sanitize_filename("a:b") == "a_b"

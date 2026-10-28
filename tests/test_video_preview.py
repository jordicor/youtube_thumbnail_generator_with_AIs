"""The library's video previews: unreachable video files and non-ASCII names."""

import os

OFFLINE_VIDEO = r"\\offline-share\videos\clip.mp4"


async def _insert_offline_video(db) -> int:
    cursor = await db.execute(
        "INSERT INTO videos (filename, filepath, status) VALUES (?, ?, 'pending')",
        ("clip.mp4", OFFLINE_VIDEO),
    )
    await db.commit()
    return cursor.lastrowid


async def test_video_on_a_share_that_does_not_answer_has_no_preview(client, test_db, monkeypatch):
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if str(path) == OFFLINE_VIDEO:
            # What Windows raises for a mapped share that went away (WinError 64)
            raise OSError(None, "The specified network name is no longer available", str(path), 64)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    video_id = await _insert_offline_video(test_db)

    response = await client.get(f"/api/videos/{video_id}/thumbnail")

    # The page shows its placeholder instead of the request failing with a 500
    assert response.status_code == 404


def test_preview_of_a_video_with_an_accent_in_its_name_is_saved(tmp_path):
    import cv2
    import numpy as np

    from utils import extract_first_frame

    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 64))
    writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    preview = tmp_path / "Transformación personal" / "preview.jpg"

    assert extract_first_frame(video, preview) == preview
    # cv2.imwrite wrote it under a mangled name and reported success, so the
    # library extracted it again from the video every time it was shown
    assert preview.is_file()

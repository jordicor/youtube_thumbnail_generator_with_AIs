"""A transcribed video works while its file is unreachable (e.g. a disconnected network drive)."""

from pathlib import Path

import orjson
import pytest
from fastapi import HTTPException

import config
from api.routes.titles import get_video_info
from utils import VideoOutput


async def _insert_video(db, filepath: Path) -> int:
    cursor = await db.execute(
        "INSERT INTO videos (filename, filepath, status) VALUES (?, ?, 'analyzed')",
        (filepath.name, str(filepath)),
    )
    await db.commit()
    return cursor.lastrowid


@pytest.fixture
def unreachable_video(tmp_path) -> Path:
    """A video whose file is not there (its drive is not connected)."""
    return tmp_path / "offline_drive" / "talk.mp4"


def _transcribe(video_path: Path) -> VideoOutput:
    output = VideoOutput(video_path, Path(config.OUTPUT_DIR))
    output.output_dir.mkdir(parents=True)
    output.transcription_file.write_text("# Transcription\n\nhola a todos\n", encoding='utf-8')
    (output.output_dir / "transcription_segments.json").write_bytes(orjson.dumps(
        {'segments': [{'timestamp': '00:01', 'speaker': 'speaker_0', 'text': 'hola a todos'}]}
    ))
    return output


async def test_transcription_is_read_without_the_video_file(client, test_db, unreachable_video):
    _transcribe(unreachable_video)
    video_id = await _insert_video(test_db, unreachable_video)

    response = await client.get(f"/api/titles/transcription/{video_id}")
    assert response.status_code == 200
    assert response.json() == {"exists": True, "text": "[00:01] speaker_0: hola a todos"}

    response = await client.get(f"/api/titles/check-transcription/{video_id}")
    assert response.status_code == 200
    assert response.json() == {"exists": True}


async def test_untranscribed_video_without_file_has_no_transcription(client, test_db, unreachable_video):
    video_id = await _insert_video(test_db, unreachable_video)

    response = await client.get(f"/api/titles/transcription/{video_id}")
    assert response.json() == {"exists": False, "text": None}
    response = await client.get(f"/api/titles/check-transcription/{video_id}")
    assert response.json() == {"exists": False}


async def test_generating_text_needs_the_file_only_to_transcribe(app, test_db, unreachable_video):
    video_id = await _insert_video(test_db, unreachable_video)

    # Not transcribed yet: the video has to be read
    with pytest.raises(HTTPException) as error:
        await get_video_info(video_id)
    assert error.value.status_code == 404

    _transcribe(unreachable_video)
    assert await get_video_info(video_id) == ("talk.mp4", unreachable_video)

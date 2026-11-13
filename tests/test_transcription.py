"""A video where nobody speaks has an empty transcription, saved like any other
so it is not sent again; a transcription that cannot be made fails with its
reason. No ffmpeg, no ElevenLabs."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

# transcription imports utils, which rewraps stdout on Windows: imported inside the fixtures

@pytest.fixture
def video(tmp_path, monkeypatch):
    """A video file with its output folder, ffprobe and ffmpeg replaced, ElevenLabs configured."""
    import transcription
    from utils import VideoOutput

    path = tmp_path / "videos" / "clip.mp4"
    path.parent.mkdir()
    path.write_bytes(b"synthetic video")
    output = VideoOutput(path, tmp_path / "output")
    output.setup()

    tools = SimpleNamespace(audio_track=True, ffprobe_error=None, ffmpeg_error=None, calls=[], audio_files=[])

    def run(command, **kwargs):
        tool = command[0]
        tools.calls.append(tool)
        if tool == "ffprobe":
            if tools.ffprobe_error:
                return subprocess.CompletedProcess(command, 1, b"", tools.ffprobe_error)
            return subprocess.CompletedProcess(command, 0, b"0\n" if tools.audio_track else b"", b"")
        audio = Path(command[-1])
        tools.audio_files.append(audio)
        if tools.ffmpeg_error:
            return subprocess.CompletedProcess(command, 1, b"", tools.ffmpeg_error)
        audio.write_bytes(b"synthetic audio")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(transcription.subprocess, "run", run)
    monkeypatch.setattr(transcription, "USE_LOCAL_WHISPER", False)
    monkeypatch.setattr(transcription, "ELEVENLABS_API_KEY", "test-key")

    def heard(text, words=()):
        """What ElevenLabs answers from now on; returns the list of its calls."""
        calls = []

        def transcribe(audio_path):
            calls.append(audio_path)
            return text, {"text": text, "words": list(words)}

        monkeypatch.setattr(transcription, "transcribe_elevenlabs", transcribe)
        return calls

    return SimpleNamespace(module=transcription, path=path, output=output, tools=tools, heard=heard)


# =============================================================================
# NOTHING TO TRANSCRIBE
# =============================================================================

def test_a_video_without_an_audio_track_has_an_empty_transcription(video):
    video.tools.audio_track = False
    sent = video.heard("never asked")

    assert video.module.transcribe_video(video.path, video.output) == ""

    assert video.tools.calls == ["ffprobe"] and sent == []
    assert video.module.read_transcription(video.output) == ""


def test_a_video_where_nobody_speaks_has_an_empty_transcription(video):
    sent = video.heard("  ")

    assert video.module.transcribe_video(video.path, video.output) == ""

    assert len(sent) == 1
    assert video.module.read_transcription(video.output) == ""
    assert not video.tools.audio_files[0].exists()


def test_an_empty_transcription_is_not_asked_for_again(video):
    sent = video.heard("")
    video.module.transcribe_video(video.path, video.output)
    video.tools.calls.clear()

    assert video.module.transcribe_video(video.path, video.output) == ""

    assert video.tools.calls == [] and len(sent) == 1


def test_a_transcription_is_read_back_without_the_header_of_its_file(video):
    video.heard("Hola a todos.\nBienvenidos al canal.")

    assert video.module.transcribe_video(video.path, video.output) == "Hola a todos.\nBienvenidos al canal."

    assert video.output.transcription_file.read_text(encoding="utf-8").startswith("# Transcription")
    assert video.module.read_transcription(video.output) == "Hola a todos.\nBienvenidos al canal."


def test_a_video_never_transcribed_has_no_transcription(video):
    assert video.module.read_transcription(video.output) is None


# =============================================================================
# IT COULD NOT BE MADE
# =============================================================================

def test_a_video_ffprobe_cannot_read_fails_with_what_ffprobe_said(video):
    video.tools.ffprobe_error = b"clip.mp4: Invalid data found when processing input\n"

    with pytest.raises(video.module.TranscriptionError, match="Invalid data found"):
        video.module.transcribe_video(video.path, video.output)

    assert video.module.read_transcription(video.output) is None


def test_audio_ffmpeg_cannot_extract_fails_with_the_end_of_its_output(video):
    video.tools.ffmpeg_error = b"ffmpeg version 7.1\n  built with gcc\n\nError opening output file.\nConversion failed!\n"

    with pytest.raises(video.module.TranscriptionError) as failure:
        video.module.transcribe_video(video.path, video.output)

    assert "Error opening output file. | Conversion failed!" in str(failure.value)
    assert video.module.read_transcription(video.output) is None
    assert not video.tools.audio_files[0].exists()


def test_ffmpeg_that_is_not_installed_fails_saying_so(video, monkeypatch):
    def missing(command, **kwargs):
        raise FileNotFoundError(2, "The system cannot find the file specified")

    monkeypatch.setattr(video.module.subprocess, "run", missing)

    with pytest.raises(video.module.TranscriptionError, match="ffprobe did not run"):
        video.module.transcribe_video(video.path, video.output)


def test_an_answer_of_elevenlabs_that_is_not_a_transcription_fails_with_it(video, monkeypatch, tmp_path):
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"synthetic audio")
    answer = SimpleNamespace(status_code=401, text='{"detail": {"status": "invalid_api_key"}}')
    monkeypatch.setattr(video.module.requests, "post", lambda *args, **kwargs: answer)

    with pytest.raises(video.module.TranscriptionError, match="HTTP 401.*invalid_api_key"):
        video.module.transcribe_elevenlabs(audio)


def test_elevenlabs_without_its_key_fails_before_sending_anything(video, monkeypatch, tmp_path):
    monkeypatch.setattr(video.module, "ELEVENLABS_API_KEY", "")
    monkeypatch.setattr(video.module.requests, "post", lambda *args, **kwargs: pytest.fail("sent"))

    with pytest.raises(video.module.TranscriptionError, match="ELEVENLABS_API_KEY"):
        video.module.transcribe_elevenlabs(tmp_path / "audio.mp3")


def test_a_failed_transcription_saves_nothing_so_it_is_tried_again(video, monkeypatch):
    def refused(audio_path):
        raise video.module.TranscriptionError("ElevenLabs could not transcribe the audio: HTTP 500: busy")

    monkeypatch.setattr(video.module, "transcribe_elevenlabs", refused)
    with pytest.raises(video.module.TranscriptionError):
        video.module.transcribe_video(video.path, video.output)
    assert not video.tools.audio_files[0].exists()

    video.heard("Ahora sí.")
    assert video.module.transcribe_video(video.path, video.output) == "Ahora sí."


# =============================================================================
# WHAT THE ANALYSIS, TITLES AND DESCRIPTIONS MAKE OF IT
# =============================================================================

@pytest.fixture
def analysis(test_db, video, monkeypatch):
    """Analyze the video with scenes and faces already made: only the transcription is real."""
    from services import analysis_service

    monkeypatch.setattr(analysis_service, "OUTPUT_DIR", video.output.output_dir.parent)
    monkeypatch.setattr(analysis_service, "process_video_scenes", lambda path, output: (SimpleNamespace(), []))
    monkeypatch.setattr(analysis_service, "process_faces", lambda frames, output: SimpleNamespace(all_faces=[]))

    async def run():
        cursor = await test_db.execute(
            "INSERT INTO videos (filename, filepath, status) VALUES (?, ?, 'pending')",
            (video.path.name, str(video.path)),
        )
        await test_db.commit()
        service = analysis_service.AnalysisService(test_db)
        await service.run_full_analysis(cursor.lastrowid)
        return await service.get_video(cursor.lastrowid)

    return run


async def test_a_video_without_speech_ends_its_analysis_analyzed(analysis, video):
    """It used to end in error, with its people already found, and could not generate."""
    video.tools.audio_track = False

    analyzed = await analysis()

    assert (analyzed["status"], analyzed["error_message"]) == ("analyzed", None)


async def test_an_analysis_whose_transcription_fails_says_why(analysis, video):
    video.tools.ffmpeg_error = b"clip.mp4: Permission denied\n"

    analyzed = await analysis()

    assert analyzed["status"] == "error"
    assert analyzed["error_message"] == (
        "transcription_error: The audio could not be extracted: ffmpeg: clip.mp4: Permission denied"
    )


def test_titles_get_the_saved_transcription_without_its_header(video, monkeypatch):
    from api.routes import titles

    monkeypatch.setattr(titles, "OUTPUT_DIR", video.output.output_dir.parent)
    video.tools.audio_track = False

    # Transcribed now (nothing to hear), then read from its file
    assert titles.get_or_create_transcription(video.path, "clip.mp4") == ("", True)
    assert titles.get_or_create_transcription(video.path, "clip.mp4") == ("", False)


def test_titles_of_a_video_that_cannot_be_transcribed_fail_with_the_reason(video, monkeypatch):
    from api.routes import titles

    monkeypatch.setattr(titles, "OUTPUT_DIR", video.output.output_dir.parent)
    video.tools.ffmpeg_error = b"clip.mp4: Permission denied\n"

    with pytest.raises(HTTPException) as failure:
        titles.get_or_create_transcription(video.path, "clip.mp4")

    assert failure.value.status_code == 502
    assert "Permission denied" in failure.value.detail

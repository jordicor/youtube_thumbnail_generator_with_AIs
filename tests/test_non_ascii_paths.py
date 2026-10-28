"""Videos with an accent in their name ("Psicológicos.mp4") on Windows.

The accent reaches every path under the video's output folder, and cv2.imread
and cv2.imwrite cannot open such a path: imwrite wrote nowhere and still
returned True, so those videos were analyzed without a single frame.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from PIL import Image

from services.analysis_service import AnalysisService
from services.face_swap_service import FaceSwapAction

ACCENTED = "Psicológicos"


def _video(path, frames=20):
    """A small MJPG video, dark frames then bright ones, saved at this path."""
    import cv2

    # Written under a plain name: the test is about reading it, not cv2.VideoWriter
    plain = path.parent / "clip.avi"
    plain.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(plain), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    for i in range(frames):
        writer.write(np.full((64, 64, 3), 20 if i < frames // 2 else 230, np.uint8))
    writer.release()
    return plain.rename(path)


def _image(path, value=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (value, value, value)).save(path)
    return path


def _output(video, tmp_path):
    from utils import VideoOutput

    output = VideoOutput(video, tmp_path / "output")
    output.setup()
    return output


# =============================================================================
# SCENE FRAMES
# =============================================================================

def test_the_frames_of_a_video_with_an_accent_in_its_name_are_written(tmp_path):
    from scene_detection import process_video_scenes

    video = _video(tmp_path / "videos" / f"{ACCENTED}.avi")
    output = _output(video, tmp_path)

    result, frames = process_video_scenes(video, output)

    assert result and frames
    assert ACCENTED in str(output.frames_dir)
    assert sorted(output.frames_dir.iterdir()) == sorted(frames)
    with Image.open(frames[0]) as frame:
        assert frame.size == (64, 64)


def test_a_frame_that_cannot_be_written_fails_the_extraction(tmp_path):
    from scene_detection import process_video_scenes

    video = _video(tmp_path / "videos" / f"{ACCENTED}.avi")
    output = _output(video, tmp_path)
    output.frames_dir.rmdir()

    with pytest.raises(FileNotFoundError):
        process_video_scenes(video, output)


def test_an_extraction_that_reads_no_frame_fails(tmp_path):
    from scene_detection import Scene, SceneDetectionResult, extract_scene_frames

    video = _video(tmp_path / "videos" / f"{ACCENTED}.avi", frames=10)
    output = _output(video, tmp_path)
    beyond_the_end = Scene(index=0, start_frame=100, end_frame=110, start_time=10.0,
                           end_time=11.0, duration=1.0, frame_indices=[100, 105])
    result = SceneDetectionResult(video_path=str(video), total_scenes=1,
                                  total_frames_to_extract=2, scenes=[beyond_the_end])

    with pytest.raises(RuntimeError, match="None of the frames"):
        extract_scene_frames(video, result, output)
    assert list(output.frames_dir.iterdir()) == []


# =============================================================================
# FULL ANALYSIS
# =============================================================================

@pytest.fixture
def analysis(test_db, tmp_path, monkeypatch):
    """A video with an accent in its name, analyzed without faces or a real transcription."""
    from services import analysis_service

    seen = {}

    def process_faces(frames, output):
        seen['frames on disk'] = [frame.is_file() for frame in frames]
        return SimpleNamespace(all_faces=[])

    monkeypatch.setattr(analysis_service, 'process_faces', process_faces)
    monkeypatch.setattr(analysis_service, 'transcribe_video', lambda video_path, output: "Hola")
    video = _video(tmp_path / "videos" / f"{ACCENTED}.avi")

    async def run():
        cursor = await test_db.execute(
            "INSERT INTO videos (filename, filepath, status) VALUES (?, ?, 'pending')",
            (video.name, str(video)),
        )
        await test_db.commit()
        service = AnalysisService(test_db)
        await service.run_full_analysis(cursor.lastrowid, force_scenes=True, force_faces=True)
        return await service.get_video(cursor.lastrowid)

    return SimpleNamespace(run=run, seen=seen)


async def test_a_video_with_an_accent_in_its_name_is_analyzed_with_its_frames(analysis):
    video = await analysis.run()

    assert video['status'] == 'analyzed'
    frames_on_disk = analysis.seen['frames on disk']
    assert frames_on_disk and all(frames_on_disk)


async def test_an_analysis_whose_frames_cannot_be_written_ends_in_error(analysis, monkeypatch):
    import cv2

    monkeypatch.setattr(cv2, 'imencode', lambda extension, image, params: (False, None))

    video = await analysis.run()

    assert video['status'] == 'error'
    assert video['error_message'].startswith('scene_detection_error: Could not encode')
    assert 'frames on disk' not in analysis.seen


# =============================================================================
# FACES IN THE FRAMES
# =============================================================================

def test_faces_are_searched_in_a_frame_under_a_folder_with_an_accent(tmp_path, monkeypatch):
    import face_extraction

    frame = _image(tmp_path / ACCENTED / "frames" / "scene_000_frame_000000.jpg")
    seen = []
    monkeypatch.setattr(face_extraction, 'get_face_app',
                        lambda: SimpleNamespace(get=lambda image: seen.append(image) or []))

    assert face_extraction.analyze_frame(frame) == []
    assert [image.shape for image in seen] == [(8, 8, 3)]


def test_a_frame_that_cannot_be_read_fails_the_face_extraction(tmp_path, monkeypatch):
    import face_extraction

    monkeypatch.setattr(face_extraction, 'get_face_app', lambda: pytest.fail("no frame to search"))

    # It used to count as a frame without faces, and the analysis ended as analyzed
    with pytest.raises(FileNotFoundError):
        face_extraction.analyze_frame(tmp_path / ACCENTED / "frames" / "missing.jpg")


# =============================================================================
# FACE SWAP
# =============================================================================

async def test_the_thumbnail_to_fix_is_read_from_a_folder_with_an_accent(tmp_path, monkeypatch):
    from services import face_matcher

    thumbnail = _image(tmp_path / ACCENTED / "thumbnails" / "thumbnail_p1_v1.png")
    seen = []
    app = SimpleNamespace(get=lambda image: seen.append(image) or [])
    monkeypatch.setitem(sys.modules, 'face_extraction', SimpleNamespace(get_face_app=lambda: app))

    image, pairs, actions = await face_matcher.match_thumbnail_faces(None, thumbnail, [])

    assert image.shape == (8, 8, 3) and seen[0] is image
    assert pairs == [] and actions == []


async def test_the_swapped_thumbnail_is_written_to_a_folder_with_an_accent(tmp_path, monkeypatch):
    from services import face_swap_pipeline

    thumbnail = _image(tmp_path / ACCENTED / "thumbnails" / "thumbnail_p1_v1.png")
    image = np.full((8, 8, 3), 100, np.uint8)
    swap = FaceSwapAction(target_face_index=0, target_bbox=(0, 0, 4, 4), person_id='person_01',
                          cluster_id=1, similarity=0.7, decision='swap', source_frame=None)
    monkeypatch.setattr(face_swap_pipeline, 'face_swap_available', lambda: (True, []))
    monkeypatch.setattr(face_swap_pipeline, 'match_thumbnail_faces',
                        AsyncMock(return_value=(image, [('target', 'source')], [swap])))
    swapper = SimpleNamespace(apply=lambda target, pairs, restore_faces=False: (target + 1, [0.0]))
    monkeypatch.setattr(face_swap_pipeline, 'FaceSwapService', SimpleNamespace(get=lambda: swapper))

    result = await face_swap_pipeline.apply_face_swap_to_thumbnail(None, thumbnail, [{'cluster_id': 1}])

    assert result.success
    with Image.open(thumbnail) as swapped:
        assert (np.asarray(swapped) == 101).all()
    with Image.open(result.raw_path) as raw:
        assert (np.asarray(raw) == 100).all()

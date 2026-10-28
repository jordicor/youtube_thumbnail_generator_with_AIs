"""Face extraction: the InsightFace models it runs and how faces are written to faces.json."""

from dataclasses import asdict

import orjson

import face_extraction
from face_extraction import FaceInfo


def _face() -> FaceInfo:
    return FaceInfo(
        frame_path="video/frames/scene_001_frame_000010.jpg",
        bbox=[10.5, 20.25, 110.0, 140.75],
        confidence=0.91,
        quality_score=73.4,
        pose=[1.5, -12.25, 0.5],
        expression="smiling",
        expression_scores={"mar": 0.12, "smile_ratio": 0.4},
        embedding=[0.10000000149011612, -0.25, 0.5] * 170 + [0.75, 1.0],
    )


def test_faces_are_written_as_before():
    face = _face()

    # Same faces.json as dataclasses.asdict, which deep-copies every float
    assert face.to_dict() == asdict(face)
    assert orjson.dumps(face.to_dict(), option=orjson.OPT_INDENT_2) == orjson.dumps(asdict(face), option=orjson.OPT_INDENT_2)

    without_embedding = face.to_dict(include_embedding=False)
    assert 'embedding' not in without_embedding
    assert face.embedding is not None


def test_only_the_models_whose_output_is_read_run(monkeypatch):
    import insightface.app

    created = {}

    class FakeFaceAnalysis:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def prepare(self, **kwargs):
            created['prepared'] = kwargs

    monkeypatch.setattr(insightface.app, 'FaceAnalysis', FakeFaceAnalysis)
    monkeypatch.setattr(face_extraction, '_face_app', None)

    face_extraction.get_face_app()

    # Boxes, embeddings, and the 3D landmarks that give pose and expression
    assert sorted(created['allowed_modules']) == ['detection', 'landmark_3d_68', 'recognition']
    assert created['prepared'] == {'ctx_id': 0, 'det_size': (640, 640)}


def test_old_face_cache_is_recomputed_in_incremental_mode(monkeypatch):
    from pathlib import Path

    import config
    from utils import VideoOutput

    output = VideoOutput(Path("stale.mp4"), config.OUTPUT_DIR)
    output.frames_dir.mkdir(parents=True)
    frame = output.frames_dir / "scene_000_frame_000001.jpg"
    frame.write_bytes(b"fixture")
    old = _face()
    old.frame_path = "stale/frames/" + frame.name
    output.faces_file.write_bytes(orjson.dumps({
        "analysis_version": 1, "all_faces": [old.to_dict()],
        "processed_frames": [old.frame_path], "total_frames_analyzed": 1, "frames_with_faces": 1,
    }))
    calls = []
    def analyze(path):
        calls.append(path)
        fresh = _face()
        fresh.frame_path = str(path)
        fresh.portrait_quality = {"eye_openness": 0.3, "score": 70, "issues": []}
        return [fresh]
    monkeypatch.setattr(face_extraction, "analyze_frame", analyze)
    result = face_extraction.extract_faces([frame], output, incremental=True)
    assert calls == [frame]
    assert len(result.all_faces) == 1 and result.all_faces[0].portrait_quality["score"] == 70

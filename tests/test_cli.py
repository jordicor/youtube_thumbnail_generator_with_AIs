"""File-based CLI processing with synthetic media and no AI or GPU calls."""

import base64
import importlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from PIL import Image

import utils


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "LOG_FILE", tmp_path / "configured" / "pipeline.log")
    cli = importlib.import_module("main")
    configured = tmp_path / "configured"
    monkeypatch.setattr(cli, "OUTPUT_DIR", configured)
    monkeypatch.setattr(config, "OUTPUT_DIR", configured)
    calls = []
    references = []

    def scenes(video, output):
        calls.append(("scenes", video.name))
        frame = output.frames_dir / "frame.png"
        Image.new("RGB", (1200, 800), "blue").save(frame)
        output.scenes_file.write_bytes(b"{}")
        return SimpleNamespace(total_scenes=1), [frame]

    def faces(frames, output):
        calls.append(("faces", output.video_path.name))
        # Exercise the same configured-root conversion used by the real face cache.
        output.faces_file.write_bytes(orjson.dumps({"frame_path": utils.to_output_relative(frames[0])}))
        face = SimpleNamespace(frame_path=str(frames[0]), quality_score=90)
        return SimpleNamespace(all_faces=[face], frames_with_faces=1)

    def transcribe(video, output):
        output.transcription_file.write_text("Synthetic speech.", encoding="utf-8")
        return "Synthetic speech."

    def ideas(**kwargs):
        calls.append(("ideas", kwargs["video_title"]))
        references.extend(kwargs["reference_images_base64"])
        folder = kwargs["output"].output_dir / "prompts"
        folder.mkdir(exist_ok=True)
        (folder / "images.json").write_bytes(b"{}")
        return [SimpleNamespace(suggested_title="Synthetic idea") for _ in range(kwargs["num_images"])]

    def images(**kwargs):
        calls.append(("images", kwargs["output"].video_path.name))
        folder = kwargs["output"].output_dir / "thumbnails"
        folder.mkdir(exist_ok=True)
        result = []
        for index, _ in enumerate(kwargs["images"], 1):
            path = folder / f"gpt-image_2026_11_12_{index}.png"
            Image.new("RGB", (16, 9), "red").save(path)
            result.append(path)
        return result

    modules = {
        "scene_detection": SimpleNamespace(process_video_scenes=scenes),
        "face_extraction": SimpleNamespace(process_faces=faces),
        "transcription": SimpleNamespace(transcribe_video=transcribe, read_transcription=lambda output: None),
        "prompt_generation": SimpleNamespace(generate_thumbnail_images=ideas),
        "image_generation": SimpleNamespace(generate_thumbnails_from_images=images),
        "gransabio_prompt_generator": SimpleNamespace(PromptGenerationConfig=SimpleNamespace),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(cli, "get_video_info", lambda path: {"duration": 1, "width": 16, "height": 9, "size_mb": 0})

    source = tmp_path / "videos"
    source.mkdir()
    for name in ("a.mp4", "b.mp4"):
        (source / name).write_bytes(b"synthetic video")

    def run(*arguments):
        monkeypatch.setattr(sys, "argv", ["main.py", "process", *map(str, arguments)])
        return cli.main()

    return SimpleNamespace(cli=cli, source=source, output=tmp_path / "chosen", configured=configured,
                           calls=calls, references=references, run=run, modules=modules)


@pytest.mark.parametrize("single", [False, True])
def test_output_applies_to_every_stage_and_completed_videos_are_skipped(pipeline, single):
    target = ["--single", pipeline.source / "a.mp4"] if single else ["--source", pipeline.source]
    arguments = [*target, "--output", pipeline.output, "--num-images", "2"]

    assert pipeline.run(*arguments) == 0

    for name in (["a"] if single else ["a", "b"]):
        folder = pipeline.output / name
        assert (folder / "scenes.json").is_file()
        face = orjson.loads((folder / "faces.json").read_bytes())
        assert face["frame_path"] == f"{name}/frames/frame.png"
        assert (folder / "transcription.txt").is_file()
        assert (folder / "prompts" / "images.json").is_file()
        metadata = orjson.loads((folder / "metadata.json").read_bytes())
        assert metadata["total_thumbnails"] == 2
        assert all(Path(path).is_file() and Path(path).is_relative_to(pipeline.output)
                   for path in metadata["thumbnail_paths"])
    assert not (pipeline.configured / "a").exists()
    assert (pipeline.output / "pipeline.log").is_file()
    with Image.open(io.BytesIO(base64.b64decode(pipeline.references[0]))) as reference:
        assert reference.format == "PNG"
        assert reference.size == (1200, 800)

    pipeline.calls.clear()
    assert pipeline.run(*arguments) == 0
    assert pipeline.calls == []


@pytest.mark.parametrize("single", [False, True])
def test_dry_run_does_not_process_or_create_output(pipeline, single):
    target = ["--single", pipeline.source / "a.mp4"] if single else ["--source", pipeline.source]

    assert pipeline.run(*target, "--output", pipeline.output, "--dry-run") == 0

    assert pipeline.calls == []
    assert not pipeline.output.exists()


def test_force_prompts_includes_completed_videos_in_a_batch(pipeline):
    arguments = ["--source", pipeline.source, "--output", pipeline.output]
    assert pipeline.run(*arguments) == 0
    pipeline.calls.clear()

    assert pipeline.run(*arguments, "--force-prompts") == 0

    assert [name for stage, name in pipeline.calls if stage == "ideas"] == ["a", "b"]


def test_batch_stops_on_first_failure_and_reports_failure(pipeline):
    def failed(video, output):
        pipeline.calls.append(("scenes", video.name))
        raise RuntimeError("Synthetic scene failure")

    pipeline.modules["scene_detection"].process_video_scenes = failed

    assert pipeline.run("--source", pipeline.source, "--output", pipeline.output) == 1
    assert pipeline.calls == [("scenes", "a.mp4")]


def test_missing_thumbnail_causes_the_video_to_be_processed_again(pipeline):
    arguments = ["--single", pipeline.source / "a.mp4", "--output", pipeline.output]
    assert pipeline.run(*arguments) == 0
    next((pipeline.output / "a" / "thumbnails").glob("*.png")).unlink()
    pipeline.calls.clear()

    assert pipeline.run(*arguments) == 0
    assert ("images", "a.mp4") in pipeline.calls


def test_incomplete_generation_invalidates_previous_completion(pipeline):
    arguments = ["--single", pipeline.source / "a.mp4", "--output", pipeline.output]
    assert pipeline.run(*arguments) == 0
    pipeline.modules["image_generation"].generate_thumbnails_from_images = lambda **kwargs: []

    assert pipeline.run(*arguments, "--force-thumbnails") == 1
    assert not (pipeline.output / "a" / "metadata.json").exists()


def test_replacing_scenes_invalidates_the_faces_cached_for_old_frames(pipeline):
    arguments = ["--single", pipeline.source / "a.mp4", "--output", pipeline.output]
    assert pipeline.run(*arguments) == 0
    process_faces = pipeline.modules["face_extraction"].process_faces

    def faces(frames, output):
        assert not output.faces_file.exists()
        return process_faces(frames, output)

    pipeline.modules["face_extraction"].process_faces = faces
    assert pipeline.run(*arguments, "--force-scenes") == 0


@pytest.mark.parametrize("arguments", [[], ["process", "--force"], ["process", "--image-pro"],
                                        ["process", "--num-images", "0"]])
def test_cli_rejects_missing_command_obsolete_flags_and_invalid_count(pipeline, arguments):
    with pytest.raises(SystemExit) as error:
        pipeline.cli._build_arg_parser().parse_args(arguments)
    assert error.value.code == 2


def test_missing_source_is_an_error_even_in_dry_run(pipeline):
    assert pipeline.run("--source", pipeline.source / "missing", "--output", pipeline.output,
                        "--dry-run") == 1
    assert not pipeline.output.exists()


def test_same_output_folder_for_two_inputs_is_rejected_before_processing(pipeline):
    (pipeline.source / "nested").mkdir()
    (pipeline.source / "nested" / "a.mp4").write_bytes(b"another synthetic video")

    assert pipeline.run("--source", pipeline.source, "--output", pipeline.output) == 1
    assert pipeline.calls == []

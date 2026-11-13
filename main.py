#!/usr/bin/env python3
"""
YouTube Thumbnail Generator - Main Pipeline & CLI
=================================================
Single entry point for batch processing AND AI-agent-friendly inspection.

Usage (file processing):
    python main.py process --source /path/to/videos
    python main.py process --single video.mp4 --output /path/to/output
    python main.py process --dry-run

Usage (subcommands for AI agents and scripting — all support --json):
    python main.py list-videos --json
    python main.py list-clusters --video VIDEO_ID --json
    python main.py list-frames --cluster CLUSTER_ID [--reference-only] --json
    python main.py list-thumbnails --video VIDEO_ID --json
    python main.py inspect --video VIDEO_ID --json
    python main.py generate --video VIDEO_ID --clusters 0,1 --json
"""

import argparse
import base64
import logging
import shutil
import json
import asyncio
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

import orjson

import config

from config import (
    VIDEOS_DIR,
    OUTPUT_DIR,
    VIDEO_EXTENSIONS,
    EXCLUDE_FOLDERS,
    LOG_LEVEL,
    NUM_PROMPT_VARIATIONS,
    GEMINI_MODELS,
    IMAGE_PROVIDER,
    OPENAI_MODELS,
    POE_MODELS,
)
from utils import (
    setup_logger,
    find_videos,
    VideoOutput,
    get_video_info,
    resolve_output_path,
    load_reference_images,
)
# NOTE: heavy imports (scene_detection, face_extraction, transcription,
# prompt_generation, image_generation, gransabio_prompt_generator) are
# deferred to where they're used. AI-agent subcommands like list-videos
# should not pay the import cost of insightface/onnx/cv2.

# Setup main logger
logger = setup_logger("main", level=LOG_LEVEL)


# =============================================================================
# FILE PROCESSING
# =============================================================================


def _process_video(video_path: Path, output: VideoOutput, args,
                   comfyui_config=None, comfyui_profile=None) -> None:
    """Process one file using the shared analysis, prompt, and image stages."""
    # Imports stay here so inspection and dry runs do not load GPU libraries.
    from scene_detection import process_video_scenes
    from face_extraction import process_faces
    from transcription import read_transcription, transcribe_video
    from prompt_generation import generate_thumbnail_images
    from image_generation import generate_thumbnails_from_images
    from gransabio_prompt_generator import PromptGenerationConfig

    output.setup()
    # A failed rerun must not retain the previous run's completion record.
    output.metadata_file.unlink(missing_ok=True)
    info = get_video_info(video_path)
    logger.info(f"Video: {video_path.name}, {info['duration']:.1f}s, {info['width']}x{info['height']}")

    if args.force_scenes:
        output.scenes_file.unlink(missing_ok=True)
        if not output.frames_dir.resolve().is_relative_to(output.output_dir.resolve()):
            raise ValueError("The frames directory is outside this video's output directory")
        shutil.rmtree(output.frames_dir)
        output.frames_dir.mkdir()
    # Replacing the frames also invalidates the faces extracted from them.
    if args.force_scenes or args.force_faces:
        output.faces_file.unlink(missing_ok=True)

    scene_result, frames = process_video_scenes(video_path, output)
    if not scene_result or not frames:
        raise RuntimeError(f"No frames could be extracted from {video_path}")
    face_result = process_faces(frames, output)
    if face_result and face_result.all_faces:
        face_by_path = {}
        for face in face_result.all_faces:
            if face.frame_path not in face_by_path or face.quality_score > face_by_path[face.frame_path].quality_score:
                face_by_path[face.frame_path] = face
        best = sorted(face_by_path.values(), key=lambda face: face.quality_score, reverse=True)
        reference_frames = [Path(face.frame_path) for face in best[:8]]
    else:
        # A video without people can still supply setting references.
        reference_frames = frames[:8]

    if args.skip_transcription:
        transcription = read_transcription(output)
    else:
        if args.force_transcription:
            output.transcription_file.unlink(missing_ok=True)
        transcription = transcribe_video(video_path, output)

    if args.force_prompts:
        (output.output_dir / "prompts" / "images.json").unlink(missing_ok=True)
    # Use the same strict, native-resolution loader as the image providers.
    references = [base64.b64encode(data).decode("ascii") for data in load_reference_images(reference_frames)]
    images = generate_thumbnail_images(
        transcription=transcription or video_path.stem,
        video_title=video_path.stem,
        output=output,
        num_images=args.num_images,
        prompt_config=PromptGenerationConfig(provider="anthropic"),
        reference_images_base64=references,
    )
    paths = generate_thumbnails_from_images(
        images=images,
        best_frames=reference_frames,
        output=output,
        use_composite=args.composite,
        gemini_model=args.gemini_model,
        openai_model=args.openai_model,
        poe_model=args.poe_model,
        comfyui_workflow=args.comfyui_workflow,
        comfyui_config=comfyui_config,
        comfyui_profile=comfyui_profile,
        image_provider=args.image_provider,
        allow_provider_fallback=args.provider_fallback,
    )
    if len(paths) != args.num_images or not all(path.is_file() for path in paths):
        raise RuntimeError(f"Expected {args.num_images} finished thumbnails for {video_path.name}, got {len(paths)}")
    output.save_metadata({
        "scenes_detected": scene_result.total_scenes,
        "frames_extracted": len(frames),
        "faces_detected": face_result.frames_with_faces if face_result else 0,
        "reference_frames": len(reference_frames),
        "num_images": len(images),
        "total_thumbnails": len(paths),
        "thumbnail_paths": [str(path) for path in paths],
        "use_composite": args.composite,
    })
    logger.success(f"Saved {len(paths)} thumbnails to {output.output_dir / 'thumbnails'}")


# =============================================================================
# AI-AGENT SUBCOMMANDS (JSON-friendly inspection and generation)
# =============================================================================
#
# These subcommands are designed for programmatic use by AI agents (and by
# the user from scripts/notebooks). All produce stable, structured output
# when --json is passed. File paths returned are ABSOLUTE so a vision-capable
# agent can open frames or thumbnails directly.


def _emit_json(data) -> None:
    """Write a Python value as pretty JSON to the process stdout."""
    stdout = sys.__stdout__.buffer
    stdout.write(
        orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS)
    )
    stdout.write(b'\n')
    stdout.flush()


def _resolve_path(p) -> str:
    """Return an absolute filesystem path as a string, or None."""
    if not p:
        return None
    try:
        return str(Path(p).resolve())
    except Exception:
        return str(p)


def _output_path(stored) -> Optional[str]:
    """Return the absolute path of a value stored relative to OUTPUT_DIR, or None."""
    return str(resolve_output_path(stored)) if stored else None


async def _open_db():
    """Initialise the DB schema used by all subcommands."""
    from database.db import init_db
    await init_db()


async def _run_subcommand(handler, args) -> int:
    """Run a subcommand and close the init connection so the process can exit."""
    from database.db import close_db
    try:
        if getattr(args, 'json', False):
            # Keep stdout pure JSON: messages printed by imports, schema
            # initialization or the pipeline go to stderr instead.
            with redirect_stdout(sys.stderr):
                return await handler(args)
        return await handler(args)
    finally:
        await close_db()


async def cmd_list_videos(args) -> int:
    from database.db import get_db
    await _open_db()
    async with get_db() as db:
        query = "SELECT * FROM videos"
        params: list = []
        if args.status:
            query += " WHERE status = ?"
            params.append(args.status)
        query += " ORDER BY created_at DESC"

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            columns = [d[0] for d in cursor.description]
            videos = [dict(zip(columns, row)) for row in rows]

        for v in videos:
            v['filepath'] = _resolve_path(v.get('filepath'))

    if args.json:
        _emit_json({"videos": videos})
    else:
        print(f"{len(videos)} video(s):")
        for v in videos:
            print(f"  [{v['id']}] {v['filename']} - status={v['status']}")
    return 0


async def cmd_list_clusters(args) -> int:
    from database.db import get_db
    await _open_db()
    async with get_db() as db:
        query = """
            SELECT * FROM clusters
            WHERE video_id = ? AND view_mode = ?
            ORDER BY cluster_index
        """
        async with db.execute(query, [args.video, args.view_mode]) as cursor:
            rows = await cursor.fetchall()
            columns = [d[0] for d in cursor.description]
            clusters = [dict(zip(columns, row)) for row in rows]

        for c in clusters:
            c['representative_frame'] = _output_path(c.get('representative_frame'))
            # Drop the binary blob from JSON; expose its presence instead.
            c['has_embedding_centroid'] = c.pop('embedding_centroid', None) is not None

    if args.json:
        _emit_json({
            "video_id": args.video,
            "view_mode": args.view_mode,
            "clusters": clusters,
        })
    else:
        print(f"Video {args.video} - {len(clusters)} cluster(s) ({args.view_mode}):")
        for c in clusters:
            label = c.get('label') or f"cluster_{c['cluster_index']}"
            print(f"  [{c['id']}] index={c['cluster_index']} {label} - {c['num_frames']} frames")
    return 0


async def cmd_list_frames(args) -> int:
    from database.db import get_db
    await _open_db()
    async with get_db() as db:
        async with db.execute(
            "SELECT data, position FROM visual_references WHERE person_id = ? AND role = 'face' "
            "ORDER BY position, id", [args.cluster]
        ) as cursor:
            reference_order = {}
            for data, position in await cursor.fetchall():
                ref = orjson.loads(data)
                if ref['source_type'] == 'frame':
                    reference_order.setdefault(ref['source'], position)
        query = """
            SELECT vf.id, vf.frame_path, vf.quality_score, vf.expression, vf.scene_index
            FROM video_frames vf
            JOIN cluster_frame_assignments cfa ON cfa.frame_id = vf.id
            WHERE cfa.cluster_id = ?
            ORDER BY vf.quality_score DESC, vf.id
        """
        async with db.execute(query, [args.cluster]) as cursor:
            rows = await cursor.fetchall()
            columns = [d[0] for d in cursor.description]
            frames = [dict(zip(columns, row)) for row in rows]

        for f in frames:
            order = reference_order.get(Path(f['frame_path']).name)
            f['is_reference'] = int(order is not None)
            f['reference_order'] = order
            f['frame_path'] = _output_path(f.get('frame_path'))
        frames.sort(key=lambda f: (not f['is_reference'], f['reference_order'] or 0))
        if args.reference_only:
            frames = [f for f in frames if f['is_reference']]

    if args.json:
        _emit_json({
            "cluster_id": args.cluster,
            "reference_only": args.reference_only,
            "frames": frames,
        })
    else:
        print(f"Cluster {args.cluster} - {len(frames)} frame(s):")
        for f in frames:
            mark = '*' if f.get('is_reference') else ' '
            print(f"  {mark}[{f['id']}] q={f['quality_score']:.2f} {f['expression']} -> {f['frame_path']}")
    return 0


async def cmd_list_thumbnails(args) -> int:
    from database.db import get_db
    from services.face_match import face_match_fields
    await _open_db()
    async with get_db() as db:
        query = """
            SELECT t.id, t.job_id, t.image_index, t.filepath, t.prompt_text,
                   t.suggested_title, t.text_overlay, t.created_at,
                   t.face_match_status, t.face_match_score, t.face_match_people
            FROM thumbnails t
            JOIN generation_jobs j ON t.job_id = j.id
            WHERE j.video_id = ?
            ORDER BY t.created_at DESC, t.image_index
        """
        async with db.execute(query, [args.video]) as cursor:
            rows = await cursor.fetchall()
            columns = [d[0] for d in cursor.description]
            thumbs = [face_match_fields(dict(zip(columns, row))) for row in rows]

        for t in thumbs:
            t['filepath'] = _output_path(t.get('filepath'))

    if args.json:
        _emit_json({"video_id": args.video, "thumbnails": thumbs})
    else:
        print(f"Video {args.video} - {len(thumbs)} thumbnail(s):")
        for t in thumbs:
            match = t['face_match']
            if match is None:
                face = 'unscored'
            elif match['score'] is None:
                face = match['band']
            else:
                face = f"{match['band']} {match['score']:.2f}"
            print(f"  [{t['id']}] job={t['job_id']} #{t['image_index']} face={face} -> {t['filepath']}")
    return 0


async def cmd_score_thumbnails(args) -> int:
    """Score the face match of thumbnails generated before it existed."""
    from database.db import get_db
    from services.face_match import score_existing_thumbnails
    await _open_db()

    def report(done: int, total: int) -> None:
        if done == total or done % 25 == 0:
            print(f"  checked {done}/{total}", file=sys.stderr)

    async with get_db() as db:
        bands, missing_files = await score_existing_thumbnails(
            db, video_id=args.video, rescore=args.rescore, on_scored=report
        )

    counts = {str(band): count for band, count in bands.items()}
    if args.json:
        _emit_json({
            "video_id": args.video,
            "rescore": args.rescore,
            "scored": sum(bands.values()),
            "bands": counts,
            "missing_files": missing_files,
        })
    else:
        print(f"Scored {sum(bands.values())} thumbnail(s): {counts}")
        if missing_files:
            print(f"Not scored, image no longer on disk ({len(missing_files)}):")
            for missing in missing_files:
                print(f"  [{missing['thumbnail_id']}] video={missing['video_id']} -> {missing['path']}")
    return 0


async def cmd_inspect(args) -> int:
    """Single-call snapshot of a video: video row + clusters + thumbnails + jobs."""
    from database.db import get_db
    from services.face_match import face_match_fields
    from services.generation_service import GenerationService
    await _open_db()

    async with get_db() as db:
        # Video
        async with db.execute("SELECT * FROM videos WHERE id = ?", [args.video]) as cursor:
            row = await cursor.fetchone()
            if not row:
                print(f"Error: video {args.video} not found", file=sys.stderr)
                return 1
            video = dict(zip([d[0] for d in cursor.description], row))
        video['filepath'] = _resolve_path(video.get('filepath'))

        # Clusters (both view modes)
        async def fetch_clusters(view_mode: str):
            async with db.execute(
                "SELECT * FROM clusters WHERE video_id = ? AND view_mode = ? ORDER BY cluster_index",
                [args.video, view_mode]
            ) as cursor:
                rows = await cursor.fetchall()
                cols = [d[0] for d in cursor.description]
                out = [dict(zip(cols, r)) for r in rows]
            for c in out:
                c['representative_frame'] = _output_path(c.get('representative_frame'))
                c['has_embedding_centroid'] = c.pop('embedding_centroid', None) is not None
            return out

        clusters_person = await fetch_clusters('person')
        clusters_scene = await fetch_clusters('person_scene')

        # Generation jobs (enriched with cluster mapping)
        service = GenerationService(db)
        jobs = await service.get_video_jobs(args.video)

        # Thumbnails
        async with db.execute("""
            SELECT t.id, t.job_id, t.image_index, t.filepath, t.suggested_title,
                   t.text_overlay, t.created_at,
                   t.face_match_status, t.face_match_score, t.face_match_people
            FROM thumbnails t
            JOIN generation_jobs j ON t.job_id = j.id
            WHERE j.video_id = ?
            ORDER BY t.created_at DESC, t.image_index
        """, [args.video]) as cursor:
            rows = await cursor.fetchall()
            columns = [d[0] for d in cursor.description]
            thumbs = [face_match_fields(dict(zip(columns, row))) for row in rows]
        for t in thumbs:
            t['filepath'] = _output_path(t.get('filepath'))

    snapshot = {
        "video": video,
        "clusters": {"person": clusters_person, "person_scene": clusters_scene},
        "jobs": jobs,
        "thumbnails": thumbs,
    }

    if args.json:
        _emit_json(snapshot)
    else:
        print(f"=== Video {video['id']}: {video['filename']} ===")
        print(f"  status={video['status']}  duration={video['duration_seconds']}s")
        print(f"  clusters: {len(clusters_person)} person / {len(clusters_scene)} person+scene")
        print(f"  jobs: {len(jobs)} ({sum(1 for j in jobs if j['status']=='completed')} completed)")
        print(f"  thumbnails: {len(thumbs)}")
    return 0


async def cmd_generate(args) -> int:
    """Run a full generation pipeline synchronously (no Redis worker required).

    The same orchestrator the web/queue uses is called directly here.
    """
    comfyui_config = _comfyui_cli_config(args)
    from database.db import get_db
    from services.generation_service import GenerationService, GenerationNotReadyError, create_sync_cancellation_check
    await _open_db()

    cluster_indices = [int(x.strip()) for x in args.clusters.split(',') if x.strip()]
    if not cluster_indices:
        print("Error: --clusters must contain at least one cluster_index", file=sys.stderr)
        return 1

    async with get_db() as db:
        service = GenerationService(db)

        # Resolve cluster_index → cluster_id
        cluster_ids: list = []
        for idx in cluster_indices:
            cluster = await service.get_cluster(args.video, idx, view_mode=args.view_mode)
            if not cluster:
                print(f"Error: cluster_index {idx} not found for video {args.video} (view_mode={args.view_mode})", file=sys.stderr)
                return 1
            cluster_ids.append(cluster['id'])

        # The same cluster cannot map to two person_XX roles.
        if len(set(cluster_ids)) != len(cluster_ids):
            print("Error: --clusters contains duplicates", file=sys.stderr)
            return 1

        # Claim the video and create the job in one transaction
        try:
            job = await service.create_generation_job(
                video_id=args.video,
                cluster_ids=cluster_ids,
                num_images=args.num_images,
            )
        except GenerationNotReadyError:
            print(f"Error: video {args.video} must be in status 'analyzed' or 'completed' to generate", file=sys.stderr)
            return 1

        # Run the orchestrator directly (no Redis)
        try:
            await service.run_generation_pipeline(
                job_id=job['id'],
                cancellation_check=create_sync_cancellation_check(job['id']),
                image_provider=args.image_provider,
                gemini_model=args.gemini_model,
                openai_model=args.openai_model,
                poe_model=args.poe_model,
                comfyui_workflow=args.comfyui_workflow,
                comfyui_config=comfyui_config,
                force_prompts=args.force_prompts,
                allow_provider_fallback=args.provider_fallback,
                include_scene_reference=not args.no_scene_reference,
            )
        except asyncio.CancelledError:
            # Ctrl+C cancels the coroutine. Release this job without
            # overwriting a newer generation, as the worker does.
            await service.update_job_status(
                job['id'], 'error', 0, 'Generation CLI was interrupted'
            )
            raise
        finally:
            # A cancelled pipeline keeps the video claimed until its steps have
            # stopped; then the video can generate again. Its finished
            # thumbnails stay.
            current = await service.get_job(job['id'])
            if current and current['status'] == 'cancelled':
                await service.update_job_status(job['id'], 'cancelled')

        # Fetch generated thumbnails for output
        thumbs = await service.get_job_thumbnails(job['id'])
        for t in thumbs:
            t['filepath'] = _resolve_path(t.get('filepath'))

        # Refresh job status
        final_status = await service.get_job_status(job['id'])

    result = {
        "job_id": job['id'],
        "video_id": args.video,
        "cluster_indices": cluster_indices,
        "cluster_ids": cluster_ids,
        "status": final_status.get('status') if final_status else 'unknown',
        "progress": final_status.get('progress') if final_status else None,
        "error_message": final_status.get('error_message') if final_status else None,
        "thumbnails": thumbs,
    }
    if args.json:
        _emit_json(result)
    else:
        print(f"Job {job['id']} finished with status={result['status']}, {len(thumbs)} thumbnail(s)")
        for t in thumbs:
            print(f"  #{t['image_index']} -> {t['filepath']}")
    return 0 if result['status'] == 'completed' else 1


def _add_common_json_flag(p):
    p.add_argument('--json', action='store_true', help='Emit machine-readable JSON to stdout')


def _comfyui_cli_config(args):
    if (args.image_provider or IMAGE_PROVIDER) != 'comfyui':
        return None
    if args.provider_fallback or getattr(args, 'composite', False):
        raise ValueError("ComfyUI does not allow provider fallback or composite generation")
    from services.comfyui_service import Connection
    values = json.loads(args.comfyui_config.read_text(encoding='utf-8-sig')) if args.comfyui_config else {}
    if args.comfyui_url:
        values['url'] = args.comfyui_url
    return Connection.model_validate(values).model_dump()


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("The number of images must be at least 1")
    return number


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='YouTube Thumbnail Generator — batch processing and AI-agent CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # ----- process (batch / single video) -----
    p_proc = subparsers.add_parser('process', help='Generate thumbnails directly from video files', allow_abbrev=False)
    source = p_proc.add_mutually_exclusive_group()
    source.add_argument('--source', '-s', type=Path, default=VIDEOS_DIR, help='Directory to scan for videos')
    source.add_argument('--single', type=Path, help='Process a single video file')
    p_proc.add_argument('--output', '-o', type=Path, default=OUTPUT_DIR)
    p_proc.add_argument('--dry-run', '-n', action='store_true')
    p_proc.add_argument('--skip-transcription', action='store_true')
    p_proc.add_argument('--composite', action='store_true')
    p_proc.add_argument('--force-all', action='store_true')
    p_proc.add_argument('--force-scenes', action='store_true')
    p_proc.add_argument('--force-faces', action='store_true')
    p_proc.add_argument('--force-transcription', action='store_true')
    p_proc.add_argument('--force-prompts', action='store_true')
    p_proc.add_argument('--force-thumbnails', action='store_true')
    p_proc.add_argument('--verbose', '-v', action='store_true')
    p_proc.add_argument('--num-images', type=_positive_int, default=NUM_PROMPT_VARIATIONS)
    p_proc.add_argument('--image-provider', choices=['gemini', 'openai', 'replicate', 'poe', 'comfyui'], default=None,
                        help=f'Image generation provider (default from config: {IMAGE_PROVIDER}). Reference image support depends on the selected model.')
    p_proc.add_argument('--gemini-model', choices=list(GEMINI_MODELS), default=None)
    p_proc.add_argument('--openai-model', choices=list(OPENAI_MODELS), default=None)
    p_proc.add_argument('--poe-model', choices=list(POE_MODELS), default=None)
    p_proc.add_argument('--provider-fallback', action='store_true',
                        help='If the chosen image provider fails, try the others (default: that image fails)')

    # ----- list-videos -----
    p_lv = subparsers.add_parser('list-videos', help='List videos registered in the database')
    p_lv.add_argument('--status', help='Filter by status (pending, analyzing, analyzed, generating, completed, error)')
    _add_common_json_flag(p_lv)

    # ----- list-clusters -----
    p_lc = subparsers.add_parser('list-clusters', help='List clusters of a video')
    p_lc.add_argument('--video', type=int, required=True, help='Video ID')
    p_lc.add_argument('--view-mode', choices=['person', 'person_scene'], default='person')
    _add_common_json_flag(p_lc)

    # ----- list-frames -----
    p_lf = subparsers.add_parser('list-frames', help='List frames of a cluster (with absolute paths)')
    p_lf.add_argument('--cluster', type=int, required=True, help='Cluster ID (clusters.id, not cluster_index)')
    p_lf.add_argument('--reference-only', action='store_true', help='Only frames marked as references')
    _add_common_json_flag(p_lf)

    # ----- list-thumbnails -----
    p_lt = subparsers.add_parser('list-thumbnails', help='List thumbnails generated for a video')
    p_lt.add_argument('--video', type=int, required=True, help='Video ID')
    _add_common_json_flag(p_lt)

    # ----- score-thumbnails -----
    p_st = subparsers.add_parser(
        'score-thumbnails',
        help='Score how much the people look like themselves in thumbnails not scored yet'
    )
    p_st.add_argument('--video', type=int, default=None, help='Only this video (default: all)')
    p_st.add_argument('--rescore', action='store_true',
                      help='Score again the thumbnails already scored (e.g. after editing people)')
    _add_common_json_flag(p_st)

    # ----- inspect -----
    p_in = subparsers.add_parser('inspect', help='Full snapshot of a video (clusters + jobs + thumbnails)')
    p_in.add_argument('--video', type=int, required=True, help='Video ID')
    _add_common_json_flag(p_in)

    # ----- generate -----
    p_g = subparsers.add_parser('generate', help='Run a generation pipeline synchronously (no Redis required)')
    p_g.add_argument('--video', type=int, required=True, help='Video ID')
    p_g.add_argument('--clusters', type=str, required=True,
                     help='Comma-separated cluster_index values, in person_XX order. E.g. "0,2" → person_01=cluster 0, person_02=cluster 2')
    p_g.add_argument('--view-mode', choices=['person', 'person_scene'], default='person')
    p_g.add_argument('--num-images', type=_positive_int, default=NUM_PROMPT_VARIATIONS)
    p_g.add_argument('--image-provider', choices=['gemini', 'openai', 'replicate', 'poe', 'comfyui'], default=IMAGE_PROVIDER)
    p_g.add_argument('--gemini-model', choices=list(GEMINI_MODELS), default=None)
    p_g.add_argument('--openai-model', choices=list(OPENAI_MODELS), default=None)
    p_g.add_argument('--poe-model', choices=list(POE_MODELS), default=None)
    p_g.add_argument('--provider-fallback', action='store_true',
                     help='If the chosen image provider fails, try the others (default: that image fails)')
    p_g.add_argument('--no-scene-reference', action='store_true',
                     help='With several people, do not send a frame of them together (default: sent, in one reference slot)')
    p_g.add_argument('--force-prompts', action='store_true', help='Regenerate prompts even if cached')
    _add_common_json_flag(p_g)

    for command in (p_proc, p_g):
        command.add_argument('--comfyui-url', help='Native ComfyUI server URL (default: COMFYUI_URL)')
        command.add_argument('--comfyui-workflow', help='Imported or bundled workflow ID (default: COMFYUI_WORKFLOW)')
        command.add_argument('--comfyui-config', type=Path, help='Connection JSON: url, timeout, parameters')

    return parser


# =============================================================================
# CLI
# =============================================================================

def _run_process(args) -> int:
    """Select files once for both single-video and batch processing."""
    if args.single is not None:
        if not args.single.is_file():
            raise ValueError(f"Video file not found: {args.single}")
        videos = [args.single.resolve()]
    else:
        if not args.source.is_dir():
            raise ValueError(f"Source directory not found: {args.source}")
        videos = find_videos(args.source.resolve(), VIDEO_EXTENSIONS, EXCLUDE_FOLDERS)
        if not videos:
            raise ValueError(f"No videos found in {args.source}")

    if args.skip_transcription and (args.force_transcription or args.force_all):
        raise ValueError("Cannot skip transcription and force it in the same command")
    if args.force_all:
        args.force_scenes = args.force_faces = args.force_transcription = args.force_prompts = args.force_thumbnails = True
    force = any((args.force_scenes, args.force_faces, args.force_transcription, args.force_prompts, args.force_thumbnails))
    args.output = args.output.resolve()
    selected = [(video, VideoOutput(video, args.output)) for video in videos]
    if len({output.output_dir for _, output in selected}) != len(selected):
        raise ValueError("Multiple input videos have the same output folder name; rename them before processing")
    pending = [(video, output) for video, output in selected if force or not output.is_complete()]
    logger.setLevel("DEBUG" if args.verbose else LOG_LEVEL)
    logger.info(f"Output: {args.output}; pending: {len(pending)}; completed: {len(selected) - len(pending)}")
    if args.dry_run:
        for video, output in pending:
            print(f"{video} -> {output.output_dir}")
        return 0
    if not pending:
        return 0

    comfyui_config = _comfyui_cli_config(args)
    comfyui_profile = None
    if (args.image_provider or IMAGE_PROVIDER) == "comfyui":
        from services.comfyui_service import prepare
        comfyui_profile, comfyui_config = prepare(args.comfyui_workflow, comfyui_config)

    args.output.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(args.output / "pipeline.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    # Face-cache serialization uses this root too. Override it only for this
    # file command, without changing the web's environment or database paths.
    previous_output = config.OUTPUT_DIR
    config.OUTPUT_DIR = args.output
    try:
        for index, (video, output) in enumerate(pending, 1):
            logger.info(f"Processing {index}/{len(pending)}: {video}")
            _process_video(video, output, args, comfyui_config, comfyui_profile)
    finally:
        config.OUTPUT_DIR = previous_output
        logger.removeHandler(handler)
        handler.close()
    return 0


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # All AI-agent subcommands are async; the rest run via asyncio.run.
    handlers = {
        'list-videos': cmd_list_videos,
        'list-clusters': cmd_list_clusters,
        'list-frames': cmd_list_frames,
        'list-thumbnails': cmd_list_thumbnails,
        'score-thumbnails': cmd_score_thumbnails,
        'inspect': cmd_inspect,
        'generate': cmd_generate,
    }
    try:
        if args.command == 'process':
            return _run_process(args)
        return asyncio.run(_run_subcommand(handlers[args.command], args))
    except KeyboardInterrupt:
        print('Interrupted by user', file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

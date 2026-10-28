#!/usr/bin/env python3
"""
YouTube Thumbnail Generator - Main Pipeline & CLI
=================================================
Single entry point for batch processing AND AI-agent-friendly inspection.

Usage (batch processing — default when no subcommand is given):
    python main.py                      # Process all videos in configured folder
    python main.py --source /path/to/videos
    python main.py --single video.mp4   # Process single video
    python main.py --dry-run            # Show what would be processed

Usage (subcommands for AI agents and scripting — all support --json):
    python main.py list-videos --json
    python main.py list-clusters --video VIDEO_ID --json
    python main.py list-frames --cluster CLUSTER_ID [--reference-only] --json
    python main.py list-thumbnails --video VIDEO_ID --json
    python main.py inspect --video VIDEO_ID --json
    python main.py generate --video VIDEO_ID --clusters 0,1 --json
"""

import argparse
import json
import asyncio
import sys
from contextlib import redirect_stdout
from pathlib import Path
from datetime import datetime
from typing import Optional

import orjson

# Add project directory to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from config import (
    VIDEOS_DIR,
    OUTPUT_DIR,
    VIDEO_EXTENSIONS,
    EXCLUDE_FOLDERS,
    LOG_FILE,
    LOG_LEVEL,
    NUM_PROMPT_VARIATIONS,
    NUM_IMAGE_VARIATIONS,
    GEMINI_IMAGE_MODEL,
    GEMINI_MODELS,
    IMAGE_PROVIDER,
    OPENAI_IMAGE_MODEL,
    OPENAI_MODELS,
    POE_IMAGE_MODEL,
    POE_MODELS,
)
from utils import (
    setup_logger,
    find_videos,
    VideoOutput,
    ProgressTracker,
    get_video_info,
    ensure_dir,
    resolve_output_path,
)
# NOTE: heavy imports (scene_detection, face_extraction, transcription,
# prompt_generation, image_generation, gransabio_prompt_generator) are
# deferred to where they're used. AI-agent subcommands like list-videos
# should not pay the import cost of insightface/onnx/cv2.

# Setup main logger
logger = setup_logger("main", LOG_FILE, LOG_LEVEL)


# =============================================================================
# HELPERS
# =============================================================================

def _frames_to_base64(frames: list[Path]) -> list[str]:
    """Convert a list of frame paths to base64-encoded JPEGs (max 1024px).

    Used by the CLI to send reference frames to Gran Sabio LLM for vision
    analysis. Mirrors the conversion done by the web service.
    """
    import base64
    import io
    from PIL import Image as _PILImage

    encoded = []
    for path in frames:
        if not path or not path.exists():
            continue
        try:
            with _PILImage.open(path) as img:
                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        except Exception as e:
            logger.warning(f"Could not encode frame {path.name}: {e}")
    return encoded


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def process_single_video(
    video_path: Path,
    skip_transcription: bool = False,
    use_composite: bool = False,
    force: bool = False,  # Deprecated, kept for compatibility
    num_images: int = None,
    use_pro_model: bool = False,
    force_scenes: bool = False,
    force_faces: bool = False,
    force_transcription: bool = False,
    force_prompts: bool = False,
    force_thumbnails: bool = False,
    image_provider: str = None,
    gemini_model: str = None,
    openai_model: str = None,
    poe_model: str = None,
    comfyui_workflow: str = None,
    comfyui_config: dict = None,
    comfyui_profile: dict = None,
    allow_provider_fallback: bool = False
) -> bool:
    """
    Process a single video through the entire pipeline.

    Steps:
    1. Scene detection & frame extraction
    2. Face detection & best frame selection
    3. Audio transcription
    4. Thumbnail prompt generation
    5. Thumbnail image generation

    Args:
        video_path: Path to the video file
        skip_transcription: Skip the transcription step
        use_composite: Use composite thumbnail (real face + AI background)
        force: Force reprocessing even if already done

    Returns:
        True if successful, False otherwise
    """

    print()  # Empty line before header
    logger.info("=" * 60)
    logger.info(f"Processing: {video_path.name}")
    logger.info("=" * 60)

    # Setup output structure
    output = VideoOutput(video_path, OUTPUT_DIR)

    # Check if already complete (skip only if NO force flags are set)
    any_force = force or force_scenes or force_faces or force_transcription or force_prompts or force_thumbnails
    if not any_force and output.is_complete():
        logger.info("Video already processed, skipping (use --force-* flags to reprocess)")
        return True

    output.setup()

    # Get video info
    info = get_video_info(video_path)
    logger.info(f"Video: {info['duration']:.1f}s, {info['width']}x{info['height']}, {info['size_mb']:.1f}MB")

    try:
        if (image_provider or IMAGE_PROVIDER) == "comfyui":
            if allow_provider_fallback or use_composite:
                raise ValueError("ComfyUI does not allow provider fallback or composite generation")
            from services.comfyui_service import prepare
            comfyui_profile, comfyui_config = prepare(comfyui_workflow, comfyui_config, comfyui_profile)
            comfyui_workflow = comfyui_profile["id"]

        # Deferred imports: heavy pipeline modules (insightface, cv2, onnx).
        from scene_detection import process_video_scenes
        from face_extraction import process_faces
        from transcription import transcribe_video
        from prompt_generation import generate_thumbnail_images
        from image_generation import generate_thumbnails_from_images
        from gransabio_prompt_generator import PromptGenerationConfig

        # Step 1: Scene Detection
        logger.info("\n[Step 1/5] Scene Detection...")
        if force_scenes:
            logger.info("Force mode: Deleting cached scene data...")
            output.scenes_file.unlink(missing_ok=True)
            import shutil
            shutil.rmtree(output.frames_dir, ignore_errors=True)
            output.frames_dir.mkdir(exist_ok=True)

        scene_result, extracted_frames = process_video_scenes(video_path, output)

        if not scene_result:
            logger.error("Scene detection failed")
            return False

        logger.success(f"Extracted {len(extracted_frames)} frames from {scene_result.total_scenes} scenes")

        # Step 2: Face Extraction
        logger.info("\n[Step 2/5] Face Detection & Selection...")
        if force_faces:
            logger.info("Force mode: Deleting cached face data...")
            output.faces_file.unlink(missing_ok=True)

        face_result = process_faces(extracted_frames, output)

        # For CLI mode (without cluster selection), use frames with detected faces
        # sorted by quality as reference images for the AI
        if face_result and face_result.all_faces:
            # Get unique frame paths sorted by quality score
            face_by_path = {}
            for face in face_result.all_faces:
                path = face.frame_path
                if path not in face_by_path or face.quality_score > face_by_path[path].quality_score:
                    face_by_path[path] = face

            sorted_faces = sorted(face_by_path.values(), key=lambda f: f.quality_score, reverse=True)
            reference_frames = [Path(f.frame_path) for f in sorted_faces[:8]]
        else:
            logger.warning("Face detection returned no results")
            reference_frames = extracted_frames[:8]  # Fallback to first few frames

        logger.success(f"Selected {len(reference_frames)} reference frames for image generation")

        # Step 3: Transcription
        transcription = ""
        if not skip_transcription:
            logger.info("\n[Step 3/5] Transcription...")
            if force_transcription:
                logger.info("Force mode: Deleting cached transcription...")
                output.transcription_file.unlink(missing_ok=True)

            transcription = transcribe_video(video_path, output)

            if not transcription:
                logger.warning("Transcription failed, will use video title for prompt")
                transcription = video_path.stem  # Use filename as fallback
        else:
            logger.info("\n[Step 3/5] Transcription... SKIPPED")
            # Try to load existing transcription
            if output.transcription_file.exists():
                with open(output.transcription_file, 'r', encoding='utf-8') as f:
                    transcription = f.read()
            else:
                transcription = video_path.stem

        # Use config defaults or override
        n_images = num_images if num_images else NUM_PROMPT_VARIATIONS

        # Select Gemini model
        # Priority: explicit --gemini-model > --image-pro > default
        selected_gemini_model = gemini_model
        if not selected_gemini_model and use_pro_model:
            selected_gemini_model = "gemini-3-pro-image"
        if selected_gemini_model == "gemini-3-pro-image":
            logger.info("Using Nano Banana Pro (gemini-3-pro-image)")

        # Log image provider and model
        effective_provider = image_provider or IMAGE_PROVIDER
        if effective_provider == "openai":
            effective_openai_model = openai_model or OPENAI_IMAGE_MODEL
            logger.info(f"Using OpenAI ({effective_openai_model})")
        elif effective_provider == "gemini":
            logger.info(f"Using Gemini ({selected_gemini_model or GEMINI_IMAGE_MODEL})")

        # Step 4: Prompt Generation (via Gran Sabio LLM)
        logger.info(f"\n[Step 4/5] Generating {n_images} Thumbnail Concepts via Gran Sabio LLM...")
        if force_prompts:
            logger.info("Force mode: Deleting cached prompts...")
            import shutil
            prompts_dir = output.output_dir / "prompts"
            shutil.rmtree(prompts_dir, ignore_errors=True)

        # Convert reference frames to base64 so the LLM can see who's in the video
        ref_images_b64 = _frames_to_base64(reference_frames) if reference_frames else None

        prompt_config = PromptGenerationConfig(provider="anthropic")

        thumbnail_images = generate_thumbnail_images(
            transcription=transcription,
            video_title=video_path.stem,
            output=output,
            num_images=n_images,
            prompt_config=prompt_config,
            reference_images_base64=ref_images_b64,
        )

        logger.success(f"Generated {len(thumbnail_images)} thumbnail concepts")
        for i, img in enumerate(thumbnail_images, 1):
            logger.info(f"  Concept {i}: '{img.suggested_title}'")

        # Step 5: Image Generation
        logger.info(f"\n[Step 5/5] Generating {n_images} Thumbnail Images...")

        thumbnail_paths = generate_thumbnails_from_images(
            images=thumbnail_images,
            best_frames=reference_frames,
            output=output,
            use_composite=use_composite,
            gemini_model=selected_gemini_model,
            openai_model=openai_model,
            poe_model=poe_model,
            comfyui_workflow=comfyui_workflow,
            comfyui_config=comfyui_config,
            comfyui_profile=comfyui_profile,
            image_provider=image_provider,
            allow_provider_fallback=allow_provider_fallback,
        )

        if not thumbnail_paths:
            logger.error("Thumbnail generation failed")
            return False

        # Save final metadata
        output.save_metadata({
            "scenes_detected": scene_result.total_scenes,
            "frames_extracted": len(extracted_frames),
            "faces_detected": face_result.frames_with_faces if face_result else 0,
            "reference_frames": len(reference_frames),
            "num_images": len(thumbnail_images),
            "total_thumbnails": len(thumbnail_paths),
            "thumbnail_paths": [str(p) for p in thumbnail_paths],
            "use_composite": use_composite,
            "use_pro_model": use_pro_model,
        })

        logger.success(f"\n{len(thumbnail_paths)} thumbnails generated successfully!")
        logger.success(f"Output directory: {output.output_dir / 'thumbnails'}")

        return True

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        raise
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def process_batch(
    source_dir: Path,
    skip_transcription: bool = False,
    use_composite: bool = False,
    force: bool = False,  # Deprecated
    dry_run: bool = False,
    num_images: int = None,
    use_pro_model: bool = False,
    force_scenes: bool = False,
    force_faces: bool = False,
    force_transcription: bool = False,
    force_prompts: bool = False,
    force_thumbnails: bool = False,
    image_provider: str = None,
    gemini_model: str = None,
    openai_model: str = None,
    poe_model: str = None,
    comfyui_workflow: str = None,
    comfyui_config: dict = None,
    comfyui_profile: dict = None,
    allow_provider_fallback: bool = False
) -> dict:
    """
    Process all videos in a directory.

    Args:
        source_dir: Directory containing videos
        skip_transcription: Skip transcription step
        use_composite: Use composite thumbnails
        force: Force reprocessing
        dry_run: Only show what would be processed

    Returns:
        Summary dict with statistics
    """

    logger.info(f"\n{'='*60}")
    logger.info("YouTube Thumbnail Generator - Batch Processing")
    logger.info(f"{'='*60}")
    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # Find videos
    videos = find_videos(source_dir, VIDEO_EXTENSIONS, EXCLUDE_FOLDERS)

    if not videos:
        logger.error("No videos found!")
        return {"total": 0, "processed": 0, "success": 0, "failed": 0}

    logger.info(f"Found {len(videos)} video(s)")

    # Check which need processing
    pending = []
    skipped = []

    for video in videos:
        output = VideoOutput(video, OUTPUT_DIR)
        if force or not output.is_complete():
            pending.append(video)
        else:
            skipped.append(video)

    if skipped:
        logger.info(f"Skipping {len(skipped)} already processed video(s)")

    if not pending:
        logger.success("All videos already processed!")
        return {"total": len(videos), "processed": 0, "success": len(skipped), "failed": 0}

    logger.info(f"Videos to process: {len(pending)}")

    # Dry run mode
    if dry_run:
        logger.info("\n[DRY RUN] Videos that would be processed:")
        for i, video in enumerate(pending, 1):
            rel_path = video.relative_to(source_dir) if source_dir in video.parents else video.name
            print(f"  {i}. {rel_path}")
        return {"total": len(videos), "processed": 0, "success": 0, "failed": 0, "dry_run": True}

    # Process videos
    progress = ProgressTracker(len(pending), "Processing videos")
    failed_videos = []

    for video in pending:
        try:
            success = process_single_video(
                video,
                skip_transcription=skip_transcription,
                use_composite=use_composite,
                force=force,
                num_images=num_images,
                use_pro_model=use_pro_model,
                force_scenes=force_scenes,
                force_faces=force_faces,
                force_transcription=force_transcription,
                force_prompts=force_prompts,
                force_thumbnails=force_thumbnails,
                image_provider=image_provider,
                gemini_model=gemini_model,
                openai_model=openai_model,
                poe_model=poe_model,
                comfyui_workflow=comfyui_workflow,
                comfyui_config=comfyui_config,
                comfyui_profile=comfyui_profile,
                allow_provider_fallback=allow_provider_fallback
            )
            progress.update(success, video.name)

            if not success:
                failed_videos.append(video)

        except KeyboardInterrupt:
            logger.warning("\nBatch processing interrupted")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            progress.update(False, video.name)
            failed_videos.append(video)

    # Summary
    summary = progress.summary()
    summary["skipped"] = len(skipped)

    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.success(f"Successful: {summary['successes']}")
    if summary['failures'] > 0:
        logger.error(f"Failed: {summary['failures']}")
    if skipped:
        logger.info(f"Skipped: {len(skipped)}")
    logger.info(f"Total time: {summary['elapsed']}")

    if failed_videos:
        logger.info("\nFailed videos:")
        for video in failed_videos:
            print(f"  - {video}")

    return summary


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
            # A cancelled pipeline keeps the video claimed until its partial
            # output is removed, then the video becomes retryable.
            current = await service.get_job(job['id'])
            if current and current['status'] == 'cancelled':
                await service.cleanup_partial_generation(job['id'])
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


SUBCOMMANDS = {
    'list-videos', 'list-clusters', 'list-frames', 'list-thumbnails',
    'score-thumbnails', 'inspect', 'generate', 'process',
}


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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='YouTube Thumbnail Generator — batch processing and AI-agent CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command')

    # ----- process (batch / single video) -----
    p_proc = subparsers.add_parser('process', help='Process videos to generate thumbnails (default if no subcommand)')
    p_proc.add_argument('--source', '-s', type=Path, default=VIDEOS_DIR)
    p_proc.add_argument('--single', type=Path, help='Process a single video file')
    p_proc.add_argument('--output', '-o', type=Path, default=OUTPUT_DIR)
    p_proc.add_argument('--dry-run', '-n', action='store_true')
    p_proc.add_argument('--skip-transcription', action='store_true')
    p_proc.add_argument('--composite', action='store_true')
    p_proc.add_argument('--force', '-f', action='store_true', help='[DEPRECATED] Use --force-all')
    p_proc.add_argument('--force-all', action='store_true')
    p_proc.add_argument('--force-scenes', action='store_true')
    p_proc.add_argument('--force-faces', action='store_true')
    p_proc.add_argument('--force-transcription', action='store_true')
    p_proc.add_argument('--force-prompts', action='store_true')
    p_proc.add_argument('--force-thumbnails', action='store_true')
    p_proc.add_argument('--verbose', '-v', action='store_true')
    p_proc.add_argument('--num-images', type=int, default=NUM_PROMPT_VARIATIONS)
    p_proc.add_argument('--image-pro', action='store_true')
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
    p_g.add_argument('--num-images', type=int, default=NUM_PROMPT_VARIATIONS)
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
    """Execute the legacy batch/single video pipeline."""
    comfyui_config = _comfyui_cli_config(args)
    # Deprecated --force flag → treat as --force-all
    if args.force:
        logger.warning("--force is deprecated, use --force-all instead")
        args.force_all = True

    if args.force_all:
        args.force_scenes = True
        args.force_faces = True
        args.force_transcription = True
        args.force_prompts = True
        args.force_thumbnails = True

    ensure_dir(args.output)

    if args.single:
        if not args.single.exists():
            logger.error(f"Video not found: {args.single}")
            return 1
        success = process_single_video(
            args.single,
            skip_transcription=args.skip_transcription,
            use_composite=args.composite,
            force=args.force,
            num_images=args.num_images,
            use_pro_model=args.image_pro,
            force_scenes=args.force_scenes,
            force_faces=args.force_faces,
            force_transcription=args.force_transcription,
            force_prompts=args.force_prompts,
            force_thumbnails=args.force_thumbnails,
            image_provider=args.image_provider,
            gemini_model=args.gemini_model,
            openai_model=args.openai_model,
            poe_model=args.poe_model,
            comfyui_workflow=args.comfyui_workflow,
            comfyui_config=comfyui_config,
            allow_provider_fallback=args.provider_fallback
        )
        return 0 if success else 1

    if not args.source.exists():
        logger.error(f"Source directory not found: {args.source}")
        return 1

    summary = process_batch(
        args.source,
        skip_transcription=args.skip_transcription,
        use_composite=args.composite,
        force=args.force,
        dry_run=args.dry_run,
        num_images=args.num_images,
        use_pro_model=args.image_pro,
        force_scenes=args.force_scenes,
        force_faces=args.force_faces,
        force_transcription=args.force_transcription,
        force_prompts=args.force_prompts,
        force_thumbnails=args.force_thumbnails,
        image_provider=args.image_provider,
        gemini_model=args.gemini_model,
        openai_model=args.openai_model,
        poe_model=args.poe_model,
        comfyui_workflow=args.comfyui_workflow,
        comfyui_config=comfyui_config,
        allow_provider_fallback=args.provider_fallback
    )
    return 0 if summary.get('failures', 0) == 0 else 1


def main():
    # Compatibility shim: if the user invoked `python main.py` or
    # `python main.py --flag ...` (old-style), inject the implicit
    # `process` subcommand so argparse routes correctly.
    if len(sys.argv) == 1:
        sys.argv.append('process')
    elif sys.argv[1] not in SUBCOMMANDS and sys.argv[1].startswith('-'):
        sys.argv.insert(1, 'process')

    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == 'process':
        return _run_process(args)

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
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return asyncio.run(_run_subcommand(handler, args))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
YouTube Thumbnail Generator - Main Pipeline
============================================
Automated batch processing of videos to generate thumbnails.

Usage:
    python main.py                      # Process all videos in configured folder
    python main.py --source /path/to/videos
    python main.py --single video.mp4   # Process single video
    python main.py --dry-run            # Show what would be processed
    python main.py --skip-transcription # Skip transcription step
    python main.py --composite          # Use composite thumbnails (keeps real face)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

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
    IMAGE_PROVIDER,
    OPENAI_IMAGE_MODEL,
    POE_IMAGE_MODEL,
)
from utils import (
    setup_logger,
    find_videos,
    VideoOutput,
    ProgressTracker,
    get_video_info,
    ensure_dir,
)
from scene_detection import process_video_scenes
from face_extraction import process_faces
from transcription import transcribe_video
from prompt_generation import generate_thumbnail_prompts
from image_generation import generate_thumbnails

# Setup main logger
logger = setup_logger("main", LOG_FILE, LOG_LEVEL)


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
    poe_model: str = None
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
        # Step 1: Scene Detection
        logger.info("\n[Step 1/5] Scene Detection...")
        if force_scenes:
            logger.info("Force mode: Deleting cached scene data...")
            output.scenes_file.unlink(missing_ok=True)
            import shutil
            shutil.rmtree(output.frames_dir, ignore_errors=True)
            output.frames_dir.mkdir(exist_ok=True)

        scene_result, extracted_frames = process_video_scenes(video_path, output)

        if not scene_result or not extracted_frames:
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
            selected_gemini_model = "gemini-3-pro-image-preview"
        if selected_gemini_model == "gemini-3-pro-image-preview":
            logger.info("Using Nano Banana Pro (gemini-3-pro-image-preview)")

        # Log image provider and model
        effective_provider = image_provider or IMAGE_PROVIDER
        if effective_provider == "openai":
            effective_openai_model = openai_model or OPENAI_IMAGE_MODEL
            logger.info(f"Using OpenAI ({effective_openai_model})")
        elif effective_provider == "gemini":
            logger.info(f"Using Gemini ({selected_gemini_model or GEMINI_IMAGE_MODEL})")

        # Step 4: Prompt Generation
        logger.info(f"\n[Step 4/5] Generating {n_images} Thumbnail Prompts...")
        if force_prompts:
            logger.info("Force mode: Deleting cached prompts...")
            import shutil
            prompts_dir = output.output_dir / "prompts"
            shutil.rmtree(prompts_dir, ignore_errors=True)

        thumbnail_prompts = generate_thumbnail_prompts(
            transcription=transcription,
            video_title=video_path.stem,
            output=output,
            num_variations=n_images
        )

        if not thumbnail_prompts:
            logger.error("Prompt generation failed")
            return False

        logger.success(f"Generated {len(thumbnail_prompts)} prompts")
        for i, p in enumerate(thumbnail_prompts, 1):
            logger.info(f"  Prompt {i}: '{p.suggested_title}'")

        # Step 5: Image Generation
        logger.info(f"\n[Step 5/5] Generating {n_images} Thumbnail Images...")

        thumbnail_paths = generate_thumbnails(
            prompts=thumbnail_prompts,
            best_frames=reference_frames,
            output=output,
            num_variations_per_prompt=1,  # One image per prompt (no variations)
            use_composite=use_composite,
            gemini_model=selected_gemini_model,
            openai_model=openai_model,
            poe_model=poe_model,
            image_provider=image_provider
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
            "num_images": len(thumbnail_prompts),
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
    poe_model: str = None
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
                poe_model=poe_model
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
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Automated YouTube Thumbnail Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Process all videos in default folder
  python main.py --source "D:/MyVideos"   # Process videos from specific folder
  python main.py --single video.mp4       # Process single video
  python main.py --dry-run                # Preview what would be processed
  python main.py --composite              # Create composite thumbnails (keep real face)
  python main.py --skip-transcription     # Skip transcription (faster but less accurate)
  python main.py --force                  # Reprocess even if already done
        """
    )

    parser.add_argument(
        '--source', '-s',
        type=Path,
        default=VIDEOS_DIR,
        help=f'Source directory with videos (default: {VIDEOS_DIR})'
    )

    parser.add_argument(
        '--single',
        type=Path,
        help='Process a single video file'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=OUTPUT_DIR,
        help=f'Output directory (default: {OUTPUT_DIR})'
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be processed without actually doing it'
    )

    parser.add_argument(
        '--skip-transcription',
        action='store_true',
        help='Skip transcription step (faster but prompts less accurate)'
    )

    parser.add_argument(
        '--composite',
        action='store_true',
        help='Create composite thumbnails (real face + AI background/text)'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='[DEPRECATED] Use --force-all instead'
    )

    parser.add_argument(
        '--force-all',
        action='store_true',
        help='Force ALL steps (scenes, faces, transcription, prompts, thumbnails)'
    )

    parser.add_argument(
        '--force-scenes',
        action='store_true',
        help='Force scene detection (ignore cached scenes)'
    )

    parser.add_argument(
        '--force-faces',
        action='store_true',
        help='Force face analysis (ignore cached face data)'
    )

    parser.add_argument(
        '--force-transcription',
        action='store_true',
        help='Force transcription (ignore cached transcription)'
    )

    parser.add_argument(
        '--force-prompts',
        action='store_true',
        help='Force prompt generation (ignore cached prompts)'
    )

    parser.add_argument(
        '--force-thumbnails',
        action='store_true',
        help='Force thumbnail generation (regenerate all images)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--num-images',
        type=int,
        default=NUM_PROMPT_VARIATIONS,
        help=f'Number of thumbnail images to generate (default: {NUM_PROMPT_VARIATIONS})'
    )

    parser.add_argument(
        '--image-pro',
        action='store_true',
        help='Use Nano Banana Pro (gemini-3-pro-image-preview) for higher quality images'
    )

    parser.add_argument(
        '--image-provider',
        choices=['gemini', 'openai', 'replicate', 'poe'],
        default=None,
        help=f'Image generation provider (default from config: {IMAGE_PROVIDER}). Reference image support: gemini=YES, poe=YES, replicate=YES, openai=NO'
    )

    parser.add_argument(
        '--gemini-model',
        choices=['gemini-2.5-flash-image', 'gemini-3-pro-image-preview'],
        default=None,
        help=f'Gemini model to use (default from config: {GEMINI_IMAGE_MODEL}). gemini-3-pro-image-preview is higher quality 4K'
    )

    parser.add_argument(
        '--openai-model',
        choices=['gpt-image-1', 'gpt-image-1.5', 'gpt-image-1-mini', 'dall-e-3'],
        default=None,
        help=f'OpenAI model to use (default from config: {OPENAI_IMAGE_MODEL}). WARNING: OpenAI does NOT support reference images - generated person will not match the YouTuber'
    )

    parser.add_argument(
        '--poe-model',
        choices=['flux2pro', 'flux2flex', 'fluxkontextpro', 'seedream40', 'nanobananapro', 'Ideogram-v3'],
        default=None,
        help=f'Poe model to use (default from config: {POE_IMAGE_MODEL}). All models support reference images for face consistency'
    )

    args = parser.parse_args()

    # OpenAI reference images warning
    effective_provider = args.image_provider or IMAGE_PROVIDER
    if effective_provider == "openai":
        logger.warning("=" * 70)
        logger.warning("WARNING: OpenAI does NOT support reference images!")
        logger.warning("The generated thumbnails will NOT preserve the YouTuber's face.")
        logger.warning("A new person will be created based on text description only.")
        logger.warning("Consider using --image-provider gemini or poe for face consistency.")
        logger.warning("=" * 70)

    # Handle force flags
    # If --force (deprecated) is used, treat as --force-all
    if args.force:
        logger.warning("--force is deprecated, use --force-all instead")
        args.force_all = True

    # If --force-all, activate all force flags
    if args.force_all:
        args.force_scenes = True
        args.force_faces = True
        args.force_transcription = True
        args.force_prompts = True
        args.force_thumbnails = True

    # Use specified output dir or default
    output_dir = args.output
    ensure_dir(output_dir)

    # Single video mode
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
            poe_model=args.poe_model
        )

        return 0 if success else 1

    # Batch mode
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
        poe_model=args.poe_model
    )

    return 0 if summary.get('failures', 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

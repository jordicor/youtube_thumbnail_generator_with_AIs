"""
Generation Service

Business logic for thumbnail generation operations.
"""

import asyncio
from pathlib import Path
from typing import Optional, List, Callable
import aiosqlite
import sqlite3
import shutil
import logging
from contextlib import closing

from utils import ReferenceImageError, resolve_output_path, resolve_row_paths, to_output_relative

logger = logging.getLogger(__name__)


class GenerationCancelledError(Exception):
    """Raised when generation is cancelled by user."""
    pass


class ImageGenerationError(Exception):
    """Raised when image generation fails."""
    pass


class GenerationNotReadyError(Exception):
    """Raised when a video cannot be claimed for a new generation job."""


def create_sync_cancellation_check(job_id: int):
    """Share cancellation status with synchronous image transports and the CLI."""
    from config import DATABASE_PATH

    def check() -> bool:
        with closing(sqlite3.connect(str(DATABASE_PATH), timeout=5)) as connection:
            row = connection.execute("SELECT status FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
            return row is None or row[0] in {"cancelled", "error"}

    return check


class GenerationService:
    """Service for thumbnail generation operations."""

    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    async def get_video(self, video_id: int) -> Optional[dict]:
        """Get video by ID."""
        query = "SELECT * FROM videos WHERE id = ?"
        async with self.db.execute(query, [video_id]) as cursor:
            row = await cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None

    async def get_video_cluster(self, video_id: int, cluster_id: int) -> Optional[dict]:
        """Get a cluster by its database ID, only if it belongs to the video."""
        query = "SELECT * FROM clusters WHERE id = ? AND video_id = ?"
        async with self.db.execute(query, [cluster_id, video_id]) as cursor:
            row = await cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None

    async def get_cluster(self, video_id: int, cluster_index: int, view_mode: str = 'person') -> Optional[dict]:
        """Get a specific cluster."""
        query = """
            SELECT * FROM clusters
            WHERE video_id = ? AND cluster_index = ? AND view_mode = ?
        """
        async with self.db.execute(query, [video_id, cluster_index, view_mode]) as cursor:
            row = await cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None

    async def get_cluster_frames(
        self,
        video_id: int,
        cluster_index: int,
        limit: int = 20,
        view_mode: str = 'person'
    ) -> List[dict]:
        """
        Get frames for a specific cluster.

        V2 Architecture: Uses cluster_frame_assignments JOIN video_frames.
        """
        # First get cluster ID
        query = """
            SELECT id FROM clusters
            WHERE video_id = ? AND cluster_index = ? AND view_mode = ?
        """
        async with self.db.execute(query, [video_id, cluster_index, view_mode]) as cursor:
            row = await cursor.fetchone()
            if not row:
                return []
            cluster_id = row[0]

        # Get frames ordered by quality (best first)
        query = """
            SELECT vf.frame_path, vf.quality_score, vf.expression
            FROM video_frames vf
            JOIN cluster_frame_assignments cfa ON cfa.frame_id = vf.id
            WHERE cfa.cluster_id = ?
            ORDER BY vf.quality_score DESC
            LIMIT ?
        """

        async with self.db.execute(query, [cluster_id, limit]) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [resolve_row_paths(dict(zip(columns, row)), 'frame_path') for row in rows]

    async def create_generation_job(
        self,
        video_id: int,
        cluster_ids: List[int],
        num_images: int = 5,
        preferred_expression: Optional[str] = None
    ) -> dict:
        """Atomically claim the video and create its owning generation job.

        The job references one or more clusters. Each cluster in `cluster_ids`
        is mapped to a sequential person_id (person_01, person_02, ...) by its
        position in the list, inside the same transaction as the claim.
        """
        if not cluster_ids:
            raise ValueError("cluster_ids must contain at least one cluster")

        job_id = None
        try:
            async with self.db.execute(
                "UPDATE videos SET status = 'generating' "
                "WHERE id = ? AND status IN ('analyzed', 'completed')",
                [video_id],
            ) as claim:
                if claim.rowcount != 1:
                    raise GenerationNotReadyError("Video is not ready for generation")
            cursor = await self.db.execute(
                """
                INSERT INTO generation_jobs
                (video_id, num_images, preferred_expression, status)
                VALUES (?, ?, ?, 'pending')
                """,
                [video_id, num_images, preferred_expression]
            )
            job_id = cursor.lastrowid
            await cursor.close()

            for order, cluster_id in enumerate(cluster_ids):
                person_id = f"person_{order + 1:02d}"
                await self.db.execute(
                    """
                    INSERT INTO generation_job_clusters
                    (job_id, cluster_id, person_id, cluster_order)
                    VALUES (?, ?, ?, ?)
                    """,
                    [job_id, cluster_id, person_id, order]
                )

            await self.db.commit()
        except BaseException:
            await self.db.rollback()
            # The SQLite thread may finish commit just as this coroutine is
            # cancelled. If the row survived rollback, compensate that job.
            if job_id is not None:
                await self.fail_enqueue(job_id, 'Generation request stopped before enqueue')
            raise

        return {
            "id": job_id,
            "video_id": video_id,
            "cluster_ids": list(cluster_ids),
            "status": "pending"
        }

    async def get_job_clusters(self, job_id: int) -> List[dict]:
        """Return clusters linked to a job, ordered by cluster_order.

        Each row contains: cluster_id, person_id, cluster_order.
        """
        query = """
            SELECT cluster_id, person_id, cluster_order
            FROM generation_job_clusters
            WHERE job_id = ?
            ORDER BY cluster_order ASC
        """
        async with self.db.execute(query, [job_id]) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    async def update_job_status(
        self,
        job_id: int,
        status: str,
        progress: int = 0,
        error_message: Optional[str] = None,
        *,
        release_video: bool = True,
    ):
        """Transition an active job; terminal states cannot be overwritten."""
        terminal = status in ('completed', 'cancelled', 'error')
        try:
            async with self.db.execute("""
                UPDATE generation_jobs
                SET status = ?, progress = ?, error_message = ?,
                    completed_at = CASE WHEN ? = 'completed'
                        THEN CURRENT_TIMESTAMP ELSE completed_at END
                WHERE id = ? AND status IN ('pending', 'transcribing', 'prompting', 'generating')
            """, [status, progress, error_message, status, job_id]) as cursor:
                changed = cursor.rowcount == 1
            if terminal and release_video:
                # The most recently created job owns the video's generating
                # state. A delayed worker must never release a newer job.
                await self.db.execute("""
                    UPDATE videos
                    SET status = CASE WHEN EXISTS (
                        SELECT 1 FROM generation_jobs done
                        WHERE done.video_id = videos.id AND done.status = 'completed'
                    ) THEN 'completed' ELSE 'analyzed' END
                    WHERE status = 'generating'
                      AND id = (SELECT video_id FROM generation_jobs WHERE id = ?)
                      AND ? = (SELECT MAX(id) FROM generation_jobs WHERE video_id = videos.id)
                      AND EXISTS (SELECT 1 FROM generation_jobs
                          WHERE id = ? AND status = ?)
                """, [job_id, job_id, job_id, status])
            await self.db.commit()
        except BaseException:
            await self.db.rollback()
            raise
        if not changed and not terminal:
            raise GenerationCancelledError(f"Job {job_id} is no longer active")
        return changed

    async def _run_generation_step(self, job_id, function, *args):
        """Do not release shared video output while an executor is still writing."""
        future = asyncio.get_running_loop().run_in_executor(None, function, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Cancelling an asyncio future cannot stop its underlying thread.
            # Keep ownership until the synchronous stage has actually exited.
            try:
                await self.update_job_status(job_id, 'cancelled', release_video=False)
            finally:
                while not future.done():
                    try:
                        await asyncio.shield(future)
                    except asyncio.CancelledError:
                        continue
                    except GenerationCancelledError:
                        break
                try:
                    future.result()
                except GenerationCancelledError:
                    pass
            raise

    @staticmethod
    def save_generation_progress(database_path, job_id, progress, thumbnail_info=None):
        """Persist progress atomically and always close the callback connection."""
        with closing(sqlite3.connect(str(database_path), timeout=30)) as conn:
            with conn:
                cursor = conn.execute(
                    "UPDATE generation_jobs SET progress = ? "
                    "WHERE id = ? AND status = 'generating'",
                    [progress, job_id],
                )
                if cursor.rowcount != 1:
                    raise GenerationCancelledError(f"Job {job_id} is no longer generating")
                if thumbnail_info and thumbnail_info.get('path'):
                    img = thumbnail_info.get('image')
                    conn.execute("""
                        INSERT INTO thumbnails
                        (job_id, image_index, filepath, prompt_text, suggested_title, text_overlay)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, [
                        job_id, thumbnail_info['image_index'], to_output_relative(thumbnail_info['path']),
                        img.image_prompt if img else None,
                        img.suggested_title if img else None,
                        img.text_overlay if img else None,
                    ])

    async def run_generation_pipeline(
        self,
        job_id: int,
        force_transcription: bool = False,
        force_prompts: bool = False,
        image_provider: str = "gemini",
        gemini_model: Optional[str] = None,
        openai_model: Optional[str] = None,
        poe_model: Optional[str] = None,
        comfyui_workflow: Optional[str] = None,
        comfyui_config: Optional[dict] = None,
        comfyui_profile: Optional[dict] = None,
        num_reference_images: Optional[int] = None,
        # Prompt generation AI settings
        prompt_provider: Optional[str] = None,
        prompt_model: Optional[str] = None,
        prompt_thinking_enabled: bool = False,
        prompt_thinking_level: str = "medium",
        prompt_custom_instructions: Optional[str] = None,
        prompt_include_history: bool = False,
        # Selected titles to guide image generation
        selected_titles: Optional[List[str]] = None,
        # External reference image(s)
        reference_image_base64: Optional[str] = None,
        reference_images_base64: Optional[List[str]] = None,
        reference_image_use_for_prompts: bool = False,
        reference_image_include_in_refs: bool = False,
        # Face swap is an explicit opt-in post-processing step.
        apply_face_swap: bool = False,
        restore_faces: bool = False,
        # Try other image providers when the chosen one fails
        allow_provider_fallback: bool = False,
        # Include the full setting frame chosen for the first selected person
        include_scene_reference: bool = True,
        # Cancellation check callback
        cancellation_check: Optional[Callable[[], bool]] = None
    ):
        """
        Run the full generation pipeline.

        Steps:
        1. Load video and cluster info
        2. Transcribe audio (if needed)
        3. Generate prompts
        4. Generate thumbnail images

        Args:
            num_reference_images: Number of reference images to use (None = use model's max)
            prompt_provider: AI provider for prompt generation (anthropic, openai, google, xai)
            prompt_model: Specific model for prompt generation
            prompt_thinking_enabled: Enable thinking/reasoning mode
            prompt_thinking_level: Thinking level (low, medium, high)
            prompt_custom_instructions: Additional user instructions
            prompt_include_history: Include previous prompts to avoid repetition
            selected_titles: User-selected titles to guide image generation
            reference_image_base64: External reference image as base64
            reference_images_base64: List of external reference images as base64 (max 20)
            reference_image_use_for_prompts: Use reference image(s) for prompt analysis
            reference_image_include_in_refs: Include reference image in generation refs
            include_scene_reference: Include the video's shared environment selection
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from config import OUTPUT_DIR
        from utils import VideoOutput
        from transcription import transcribe_video
        from prompt_generation import generate_thumbnail_images
        from image_generation import generate_thumbnails_from_images

        try:
            # Get job info
            job = await self.get_job(job_id)
            if not job:
                return

            if image_provider == "comfyui":
                if allow_provider_fallback:
                    raise ValueError("ComfyUI does not allow provider fallback")
                from services.comfyui_service import prepare
                comfyui_profile, comfyui_config = await self._run_generation_step(
                    job_id, prepare, comfyui_workflow, comfyui_config, comfyui_profile
                )
                comfyui_workflow = comfyui_profile["id"]

            video = await self.get_video(job['video_id'])
            if not video:
                await self.update_job_status(job_id, 'error', 0, 'Video not found')
                return

            video_path = Path(video['filepath'])
            output = VideoOutput(video_path, Path(OUTPUT_DIR))

            # Get all clusters linked to this job (multi-cluster: person_01, person_02, ...)
            job_clusters = await self.get_job_clusters(job_id)
            if not job_clusters:
                await self.update_job_status(job_id, 'error', 0, 'Job has no clusters')
                return

            # Load full info for each linked cluster, in person_id order
            clusters_info: List[dict] = []
            for jc in job_clusters:
                async with self.db.execute(
                    "SELECT * FROM clusters WHERE id = ?", [jc['cluster_id']]
                ) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        await self.update_job_status(
                            job_id, 'error', 0,
                            f"Cluster {jc['cluster_id']} (person {jc['person_id']}) not found"
                        )
                        return
                    columns = [description[0] for description in cursor.description]
                    cluster_dict = dict(zip(columns, row))
                    cluster_dict['person_id'] = jc['person_id']
                    cluster_dict['cluster_order'] = jc['cluster_order']
                    clusters_info.append(cluster_dict)

            # Concatenate cluster descriptions labelled by person_id for the LLM
            description_parts = []
            for ci in clusters_info:
                desc = ci.get('description')
                if desc and desc.strip():
                    description_parts.append(f"{ci['person_id']}: {desc.strip()}")
            cluster_description_combined = "\n".join(description_parts) if description_parts else None

            # Build prompt config - always use Gran Sabio LLM (no fallback)
            from gransabio_prompt_generator import PromptGenerationConfig, get_prompt_history_for_video, require_vision_model

            # Get prompt history if enabled
            history_prompts = None
            if prompt_include_history:
                history_prompts = get_prompt_history_for_video(job['video_id'])

            # Check the photo requirement before transcription or generation.
            model_info = await require_vision_model(prompt_provider or "anthropic", prompt_model)

            # Use specified provider or default to anthropic
            prompt_config = PromptGenerationConfig(
                provider=prompt_provider or "anthropic",
                model=prompt_model,
                thinking_enabled=prompt_thinking_enabled,
                thinking_level=prompt_thinking_level,
                custom_instructions=prompt_custom_instructions,
                include_history=prompt_include_history,
                history_prompts=history_prompts,
                facial_expression=job['preferred_expression'],
                model_info=model_info,
            )

            from services.reference_service import ReferenceService, reference_limit
            import base64
            model = {"gemini": gemini_model, "openai": openai_model, "poe": poe_model, "replicate": None, "comfyui": comfyui_workflow}[image_provider]
            limit = reference_limit(image_provider, model, num_reference_images, comfyui_profile)
            if reference_images_base64:
                raise ReferenceImageError("Additional prompt-only reference images are unsupported. Add them in the reference editor.")
            if reference_image_base64 and not reference_image_include_in_refs:
                raise ReferenceImageError("The style reference must be included in both prompts and generation.")
            entries = await ReferenceService(self.db).prepare(
                job["video_id"], [ci["id"] for ci in clusters_info], limit=limit,
                include_environment=include_scene_reference,
                style_base64=reference_image_base64 if reference_image_include_in_refs else None,
            )
            directory = output.output_dir / "temp_refs" / f"job_{job_id}"
            directory.mkdir(parents=True, exist_ok=True)
            best_frames = []
            reference_mapping = []
            ref_images_for_prompts = []
            for entry in entries:
                png = entry["png"]
                if image_provider == "openai" and len(png) >= 50_000_000:
                    raise ReferenceImageError("A reference exceeds the provider's 50 MB limit.")
                path = directory / f"reference_{entry['image_index']:02d}.png"
                path.write_bytes(png)
                best_frames.append(path)
                reference_mapping.append({key: value for key, value in entry.items() if key != "png"})
                ref_images_for_prompts.append(base64.b64encode(png).decode("ascii"))
            has_external_style_ref_for_prompts = any(entry["is_style_reference"] for entry in entries)

            # Force regeneration only when explicitly requested. The
            # avoid-repeat (prompt_include_history) flag does NOT invalidate
            # the cache anymore: with the skeleton+N pipeline, the history
            # is part of the concepts_hash, so changing it just regenerates
            # the concept stage while preserving the identity skeleton.
            if force_prompts:
                import shutil
                prompts_dir = output.output_dir / "prompts"
                shutil.rmtree(prompts_dir, ignore_errors=True)

            # Step 1: Transcription
            await self.update_job_status(job_id, 'transcribing', 10)

            if force_transcription:
                output.transcription_file.unlink(missing_ok=True)

            transcription = await self._run_generation_step(
                job_id, transcribe_video, video_path, output
            )

            if not transcription:
                transcription = video_path.stem

            # Check for cancellation after transcription
            if cancellation_check and cancellation_check():
                raise GenerationCancelledError(f"Job {job_id} cancelled after transcription")

            await self.update_job_status(job_id, 'prompting', 30)

            from functools import partial
            _gen_prompts_fn = partial(
                generate_thumbnail_images,
                transcription=transcription,
                video_title=video_path.stem,
                output=output,
                num_images=job['num_images'],
                cluster_description=cluster_description_combined,
                prompt_config=prompt_config,
                selected_titles=selected_titles,
                reference_images_base64=ref_images_for_prompts if ref_images_for_prompts else None,
                has_style_reference=has_external_style_ref_for_prompts,
                reference_mapping=reference_mapping if reference_mapping else None
            )
            thumbnail_images = await self._run_generation_step(job_id, _gen_prompts_fn)

            # Check for cancellation after prompt generation
            if cancellation_check and cancellation_check():
                raise GenerationCancelledError(f"Job {job_id} cancelled after prompt generation")

            # Step 3: Generate thumbnail images
            await self.update_job_status(job_id, 'generating', 50)

            has_external_style_ref = has_external_style_ref_for_prompts
            for image in thumbnail_images:
                image.reference_mapping = reference_mapping

            # Step 4: Generate thumbnails with progress callback
            # Create a sync callback that updates DB directly (since generate_thumbnails is sync)
            from config import DATABASE_PATH

            def progress_callback(current: int, total: int, thumbnail_info: dict = None):
                """Update progress and insert thumbnails in DB synchronously during image generation."""
                # Check for cancellation before continuing
                # This allows aborting between image generations
                if cancellation_check and cancellation_check():
                    raise GenerationCancelledError(f"Job {job_id} cancelled during image generation")

                # Calculate progress: 50% to 95% during image generation
                if total > 0:
                    image_progress = int((current / total) * 45)  # 0-45%
                    overall_progress = 50 + image_progress  # 50-95%
                else:
                    overall_progress = 50

                self.save_generation_progress(
                    DATABASE_PATH, job_id, overall_progress, thumbnail_info
                )

            _gen_thumbs_fn = partial(
                generate_thumbnails_from_images,
                images=thumbnail_images,
                best_frames=best_frames,
                output=output,
                image_provider=image_provider,
                gemini_model=gemini_model,
                openai_model=openai_model,
                poe_model=poe_model,
                comfyui_workflow=comfyui_workflow,
                comfyui_config=comfyui_config,
                comfyui_profile=comfyui_profile,
                progress_callback=progress_callback,
                has_external_style_ref=has_external_style_ref,
                cancellation_check=cancellation_check,
                allow_provider_fallback=allow_provider_fallback
            )
            thumbnail_paths = await self._run_generation_step(job_id, _gen_thumbs_fn)

            if not thumbnail_paths:
                error_msg = 'Thumbnail generation failed'
                logger.error(f"Job {job_id}: {error_msg} - marking as error and aborting")
                await self.update_job_status(job_id, 'error', 0, error_msg)
                raise ImageGenerationError(f"Job {job_id}: {error_msg}")

            # Step 5: Face match of each thumbnail, on the images as generated:
            # the face swap would paste the people's identity back and hide a miss.
            from services.face_match import score_job_thumbnails
            bands = await score_job_thumbnails(self.db, job_id, cancellation_check=cancellation_check)
            logger.info(f"Job {job_id}: face match — {dict(bands)}")

            # Step 6: Optional face-swap post-processing.
            # Runs after every thumbnail has been generated and inserted into
            # the DB by the progress_callback. Each thumbnail is processed
            # individually so a swap failure on one doesn't take down the job.
            if apply_face_swap:
                # Check for cancellation before starting the swap pass
                if cancellation_check and cancellation_check():
                    raise GenerationCancelledError(f"Job {job_id} cancelled before face swap")

                await self.update_job_status(job_id, 'generating', 95)
                try:
                    from services.face_swap_pipeline import apply_face_swap_to_job
                    counters = await apply_face_swap_to_job(
                        self.db, job_id, restore_faces=restore_faces, cancellation_check=cancellation_check
                    )
                    logger.info(
                        f"Job {job_id}: face swap pass — "
                        f"swapped={counters['swapped']} "
                        f"skipped={counters['skipped']} "
                        f"errors={counters['errors']}"
                    )
                except GenerationCancelledError:
                    # Cancellation is not a swap failure: it must stay terminal.
                    raise
                except Exception as swap_err:
                    # Swap is a best-effort enhancement: never fail the whole
                    # job because of it. Log and continue.
                    logger.error(f"Job {job_id}: face swap pass failed: {swap_err}")

            # A cancellation that won the race must remain terminal.
            if not await self.update_job_status(job_id, 'completed', 100):
                raise GenerationCancelledError(f"Job {job_id} stopped before completion")

        except GenerationCancelledError as e:
            # Handle cancellation gracefully
            logger.info(f"Generation job {job_id} cancelled: {e}")
            await self.update_job_status(job_id, 'cancelled', release_video=False)
            # Don't raise - just return so worker can handle cleanup
            return

        except ImageGenerationError:
            # Already logged and marked as error before raising - just re-raise
            # so worker can handle appropriately (e.g., arq counts as j_failed)
            raise

        except Exception as e:
            logger.error(f"Job {job_id}: Unexpected error - {e}")
            await self.update_job_status(job_id, 'error', 0, str(e))
            raise

    async def get_job(self, job_id: int) -> Optional[dict]:
        """Get a generation job by ID."""
        query = "SELECT * FROM generation_jobs WHERE id = ?"
        async with self.db.execute(query, [job_id]) as cursor:
            row = await cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None

    async def get_job_status(self, job_id: int) -> Optional[dict]:
        """Get detailed job status."""
        job = await self.get_job(job_id)
        if not job:
            return None

        # Count generated thumbnails
        async with self.db.execute(
            "SELECT COUNT(*) FROM thumbnails WHERE job_id = ?",
            [job_id]
        ) as cursor:
            row = await cursor.fetchone()
            thumbnails_count = row[0] if row else 0

        total_thumbnails = job['num_images']

        return {
            'job_id': job_id,
            'video_id': job['video_id'],
            'status': job['status'],
            'progress': job['progress'],
            'current_step': job['status'],
            'thumbnails_generated': thumbnails_count,
            'total_thumbnails': total_thumbnails,
            'error_message': job.get('error_message')
        }

    async def get_job_thumbnails(self, job_id: int) -> List[dict]:
        """Get all thumbnails for a job."""
        from services.face_match import face_match_fields

        query = """
            SELECT id, filepath, image_index, suggested_title, text_overlay,
                   face_match_status, face_match_score, face_match_people
            FROM thumbnails
            WHERE job_id = ?
            ORDER BY image_index
        """

        async with self.db.execute(query, [job_id]) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [
                face_match_fields(resolve_row_paths(dict(zip(columns, row)), 'filepath'))
                for row in rows
            ]

    async def cancel_job(self, job_id: int) -> bool:
        """Cancel a generation job."""
        return await self._stop_job(job_id, 'cancelled')

    async def fail_enqueue(self, job_id: int, error_message: str) -> bool:
        """Stop a failed enqueue, including an ambiguous accepted-then-timeout."""
        return await self._stop_job(job_id, 'error', error_message)

    async def _stop_job(self, job_id, pending_status, error_message=None):
        # Serialize the read with the worker's pending -> transcribing claim.
        # Pending jobs can release immediately; running stages must drain first.
        try:
            await self.db.execute('BEGIN IMMEDIATE')
            job = await self.get_job(job_id)
            if not job or job['status'] in ('completed', 'cancelled', 'error'):
                await self.db.rollback()
                return False
            pending = job['status'] == 'pending'
            return await self.update_job_status(
                job_id, pending_status if pending else 'cancelled',
                error_message=error_message, release_video=pending,
            )
        except BaseException:
            await self.db.rollback()
            raise

    async def get_video_jobs(self, video_id: int) -> List[dict]:
        """Get all generation jobs for a video, enriched with their cluster list."""
        query = """
            SELECT id, num_images, status, progress, created_at, completed_at
            FROM generation_jobs
            WHERE video_id = ?
            ORDER BY created_at DESC
        """

        async with self.db.execute(query, [video_id]) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            jobs = [dict(zip(columns, row)) for row in rows]

        # Attach cluster info (cluster_ids + person_ids) per job
        for job in jobs:
            job_clusters = await self.get_job_clusters(job['id'])
            job['cluster_ids'] = [jc['cluster_id'] for jc in job_clusters]
            job['person_mapping'] = {jc['person_id']: jc['cluster_id'] for jc in job_clusters}

        return jobs

    async def cleanup_partial_generation(self, job_id: int) -> dict:
        """
        Clean up partial resources after generation cancellation.

        This method removes incomplete files and database records that were
        created during an interrupted generation job. It always cleans:
        - This job's temp_refs/job_ID/ directory (temporary reference images)
        - Thumbnail files that were generated for this job, including
          pre-face-swap copies
        - Thumbnail database records for this job

        Prompts are NOT deleted as they may be valid and reusable.

        Args:
            job_id: The generation job ID whose partial resources should be cleaned

        Returns:
            dict with cleanup details: {
                'cleaned_files': list of file paths that were deleted,
                'cleaned_db_records': number of DB records deleted,
                'job_status_was': the status the job was in when cancelled
            }
        """
        from utils import VideoOutput
        from config import OUTPUT_DIR

        cleaned_files = []
        cleaned_db = 0

        # Get job info
        job = await self.get_job(job_id)
        if not job:
            return {"cleaned_files": [], "cleaned_db_records": 0, "job_status_was": None}

        job_status = job.get("status", "")
        video_id = job.get("video_id")

        # Get video info for output directory
        video = await self.get_video(video_id)
        if not video:
            return {"cleaned_files": [], "cleaned_db_records": 0, "job_status_was": job_status}

        # A stopped worker retains its claim until cleanup finishes. A late
        # duplicate delivery must never remove artifacts of a newer generation.
        async with self.db.execute(
            'SELECT MAX(id) FROM generation_jobs WHERE video_id = ?', [video_id]
        ) as cursor:
            latest_job_id = (await cursor.fetchone())[0]
        if video['status'] != 'generating' or latest_job_id != job_id:
            return {"cleaned_files": [], "cleaned_db_records": 0, "job_status_was": job_status}

        video_path = Path(video.get("filepath", ""))
        output = VideoOutput(video_path, Path(OUTPUT_DIR))

        logger.info(f"Cleaning up partial generation for job {job_id} (status was: {job_status})")

        # 1. Never delete temporary references belonging to a newer job.
        temp_refs_dir = output.output_dir / "temp_refs" / f"job_{job_id}"
        if temp_refs_dir.exists():
            try:
                shutil.rmtree(temp_refs_dir, ignore_errors=True)
                cleaned_files.append(str(temp_refs_dir))
                logger.debug(f"Deleted {temp_refs_dir}")
            except Exception as e:
                logger.warning(f"Could not delete {temp_refs_dir}: {e}")

        # 2. Get and delete partial thumbnails (files + DB)
        try:
            async with self.db.execute(
                "SELECT filepath, raw_path FROM thumbnails WHERE job_id = ?",
                [job_id]
            ) as cursor:
                thumbnails = await cursor.fetchall()

            for filepath, swap_raw_path in thumbnails:
                thumb_path = resolve_output_path(filepath)

                # Delete the pre-face-swap copy (thumbnails/raw_swap/)
                if swap_raw_path:
                    swap_raw = resolve_output_path(swap_raw_path)
                    if swap_raw.exists():
                        try:
                            swap_raw.unlink()
                            cleaned_files.append(str(swap_raw))
                            logger.debug(f"Deleted pre-swap thumbnail {swap_raw}")
                        except Exception as e:
                            logger.warning(f"Could not delete {swap_raw}: {e}")

                # Delete the thumbnail file
                if thumb_path.exists():
                    try:
                        thumb_path.unlink()
                        cleaned_files.append(str(thumb_path))
                        logger.debug(f"Deleted thumbnail {thumb_path}")
                    except Exception as e:
                        logger.warning(f"Could not delete {thumb_path}: {e}")

                # Also check for raw version in thumbnails/raw/
                if thumb_path.parent.name == "thumbnails":
                    raw_dir = thumb_path.parent / "raw"
                    # Match the filename used by image_generation and ComfyUI.
                    parts = thumb_path.stem.split("_", 1)
                    raw_stem = f"{parts[0]}_raw_{parts[1]}" if len(parts) == 2 else f"raw_{parts[0]}"
                    raw_filename = raw_stem + thumb_path.suffix
                    raw_path = raw_dir / raw_filename
                    if raw_path.exists():
                        try:
                            raw_path.unlink()
                            cleaned_files.append(str(raw_path))
                            logger.debug(f"Deleted raw thumbnail {raw_path}")
                        except Exception as e:
                            logger.warning(f"Could not delete {raw_path}: {e}")

        except Exception as e:
            logger.warning(f"Error fetching thumbnails for cleanup: {e}")

        # 3. Delete thumbnail DB records
        try:
            cursor = await self.db.execute(
                "DELETE FROM thumbnails WHERE job_id = ?",
                [job_id]
            )
            cleaned_db = cursor.rowcount
            await self.db.commit()
            if cleaned_db > 0:
                logger.debug(f"Deleted {cleaned_db} thumbnail records from DB")
        except Exception as e:
            logger.warning(f"Could not delete thumbnail DB records: {e}")

        if cleaned_files or cleaned_db:
            logger.info(
                f"Cleanup complete for job {job_id}: "
                f"{len(cleaned_files)} files, {cleaned_db} DB records"
            )

        return {
            "cleaned_files": cleaned_files,
            "cleaned_db_records": cleaned_db,
            "job_status_was": job_status
        }

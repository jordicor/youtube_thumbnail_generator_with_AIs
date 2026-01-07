"""
Generation Service

Business logic for thumbnail generation operations.
"""

from pathlib import Path
from typing import Optional, List
import aiosqlite
import sqlite3


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

    async def get_cluster(self, video_id: int, cluster_index: int) -> Optional[dict]:
        """Get a specific cluster."""
        query = """
            SELECT * FROM clusters
            WHERE video_id = ? AND cluster_index = ?
        """
        async with self.db.execute(query, [video_id, cluster_index]) as cursor:
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
            return [dict(zip(columns, row)) for row in rows]

    async def create_generation_job(
        self,
        video_id: int,
        cluster_id: int,
        num_prompts: int = 5,
        num_variations: int = 1,
        preferred_expression: Optional[str] = None
    ) -> dict:
        """Create a new generation job."""
        query = """
            INSERT INTO generation_jobs
            (video_id, cluster_id, num_prompts, num_variations, preferred_expression, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
        """

        cursor = await self.db.execute(query, [
            video_id,
            cluster_id,
            num_prompts,
            num_variations,
            preferred_expression
        ])
        await self.db.commit()

        return {
            "id": cursor.lastrowid,
            "video_id": video_id,
            "cluster_id": cluster_id,
            "status": "pending"
        }

    async def update_job_status(
        self,
        job_id: int,
        status: str,
        progress: int = 0,
        error_message: Optional[str] = None
    ):
        """Update job status and progress."""
        if error_message:
            query = """
                UPDATE generation_jobs
                SET status = ?, progress = ?, error_message = ?
                WHERE id = ?
            """
            await self.db.execute(query, [status, progress, error_message, job_id])
        else:
            query = """
                UPDATE generation_jobs
                SET status = ?, progress = ?
                WHERE id = ?
            """
            await self.db.execute(query, [status, progress, job_id])
        await self.db.commit()

    async def run_generation_pipeline(
        self,
        job_id: int,
        force_transcription: bool = False,
        force_prompts: bool = False,
        image_provider: str = "gemini",
        gemini_model: Optional[str] = None,
        openai_model: Optional[str] = None,
        poe_model: Optional[str] = None,
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
        reference_image_include_in_refs: bool = False
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
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from config import (
            OUTPUT_DIR,
            GEMINI_MODEL_MAX_REFS,
            OPENAI_MODEL_MAX_REFS,
            POE_MODEL_MAX_REFS,
            REPLICATE_MODEL_MAX_REFS,
            GEMINI_IMAGE_MODEL,
            OPENAI_IMAGE_MODEL,
            POE_IMAGE_MODEL,
        )
        from utils import VideoOutput
        from transcription import transcribe_video
        from prompt_generation import generate_thumbnail_concepts
        from image_generation import generate_thumbnails

        try:
            # Get job info
            job = await self.get_job(job_id)
            if not job:
                return

            video = await self.get_video(job['video_id'])
            if not video:
                await self.update_job_status(job_id, 'error', 0, 'Video not found')
                return

            video_path = Path(video['filepath'])
            output = VideoOutput(video_path, Path(OUTPUT_DIR))

            # Step 1: Transcription
            await self.update_job_status(job_id, 'transcribing', 10)

            if force_transcription:
                output.transcription_file.unlink(missing_ok=True)

            transcription = transcribe_video(video_path, output)

            if not transcription:
                transcription = video_path.stem

            # Get cluster information by its database ID (needed for description in prompts)
            query = "SELECT * FROM clusters WHERE id = ?"
            async with self.db.execute(query, [job['cluster_id']]) as cursor:
                row = await cursor.fetchone()
                if not row:
                    await self.update_job_status(job_id, 'error', 0, 'Cluster not found')
                    return
                columns = [description[0] for description in cursor.description]
                cluster = dict(zip(columns, row))

            # Step 2: Generate prompts
            await self.update_job_status(job_id, 'prompting', 30)

            if force_prompts:
                import shutil
                prompts_dir = output.output_dir / "prompts"
                shutil.rmtree(prompts_dir, ignore_errors=True)

            # Build prompt config - always use Gran Sabio LLM (no fallback)
            from gransabio_prompt_generator import PromptGenerationConfig, get_prompt_history_for_video

            # Get prompt history if enabled
            history_prompts = None
            if prompt_include_history:
                history_prompts = get_prompt_history_for_video(job['video_id'])

            # Use specified provider or default to anthropic
            prompt_config = PromptGenerationConfig(
                provider=prompt_provider or "anthropic",
                model=prompt_model,
                thinking_enabled=prompt_thinking_enabled,
                thinking_level=prompt_thinking_level,
                custom_instructions=prompt_custom_instructions,
                include_history=prompt_include_history,
                history_prompts=history_prompts
            )

            # Prepare reference image(s) for prompt generation if requested
            ref_image_for_prompts = None
            ref_images_for_prompts = None
            if reference_image_use_for_prompts:
                if reference_image_base64:
                    ref_image_for_prompts = reference_image_base64
                if reference_images_base64:
                    ref_images_for_prompts = reference_images_base64

            thumbnail_concepts = generate_thumbnail_concepts(
                transcription=transcription,
                video_title=video_path.stem,
                output=output,
                num_concepts=job['num_prompts'],
                num_variations_per_concept=job['num_variations'],
                cluster_description=cluster.get('description'),
                prompt_config=prompt_config,
                selected_titles=selected_titles,
                reference_image_base64=ref_image_for_prompts,
                reference_images_base64=ref_images_for_prompts
            )

            if not thumbnail_concepts:
                await self.update_job_status(job_id, 'error', 0, 'Prompt generation failed')
                return

            # Step 3: Get best frames from the SELECTED cluster
            await self.update_job_status(job_id, 'generating', 50)

            # Determine max reference images based on provider and model
            if image_provider == "gemini":
                model = gemini_model or GEMINI_IMAGE_MODEL
                model_max = GEMINI_MODEL_MAX_REFS.get(model, 3)
            elif image_provider == "openai":
                model = openai_model or OPENAI_IMAGE_MODEL
                model_max = OPENAI_MODEL_MAX_REFS.get(model, 0)
            elif image_provider == "poe":
                model = poe_model or POE_IMAGE_MODEL
                model_max = POE_MODEL_MAX_REFS.get(model, 8)
            elif image_provider == "replicate":
                model_max = REPLICATE_MODEL_MAX_REFS.get("flux-1.1-pro", 1)
            else:
                model_max = 14  # Default fallback

            # Use user-specified limit if provided, but don't exceed model's max
            ref_limit = min(num_reference_images, model_max) if num_reference_images else model_max

            # Load reference frames from the selected cluster
            # Uses frames marked as references by user, or falls back to top quality
            cluster_frames_data = await self.get_reference_frames_for_generation(
                job['cluster_id'],
                limit=ref_limit
            )

            # Convert to Path objects and filter existing files
            best_frames = [
                Path(frame['frame_path'])
                for frame in cluster_frames_data
                if Path(frame['frame_path']).exists()
            ]

            # Fallback: if no frames found for cluster, use extracted frames
            if not best_frames:
                best_frames = list(output.frames_dir.glob("*.jpg"))[:14]

            # Handle external reference image for image generation
            has_external_style_ref = False
            if reference_image_base64 and reference_image_include_in_refs:
                external_ref_path = self._save_temp_reference_image(
                    reference_image_base64,
                    output.output_dir
                )
                if external_ref_path:
                    # If we're at the limit, remove the last cluster frame to make room
                    if len(best_frames) >= ref_limit:
                        best_frames = best_frames[:-1]

                    # Add external reference at the beginning (highest priority)
                    best_frames.insert(0, external_ref_path)
                    has_external_style_ref = True

            # Step 4: Generate thumbnails with progress callback
            # Create a sync callback that updates DB directly (since generate_thumbnails is sync)
            from config import DATABASE_PATH

            def progress_callback(current: int, total: int, thumbnail_info: dict = None):
                """Update progress and insert thumbnails in DB synchronously during image generation."""
                # Calculate progress: 50% to 95% during image generation
                if total > 0:
                    image_progress = int((current / total) * 45)  # 0-45%
                    overall_progress = 50 + image_progress  # 50-95%
                else:
                    overall_progress = 50

                # Use sync sqlite3 to update (generate_thumbnails is blocking)
                # timeout=30 prevents indefinite blocking if DB is locked
                # isolation_level=None enables autocommit for faster writes
                conn = None
                try:
                    conn = sqlite3.connect(
                        str(DATABASE_PATH),
                        timeout=30,
                        isolation_level=None
                    )
                    # Enable WAL mode for better concurrency
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA busy_timeout=30000")

                    # Update progress
                    conn.execute(
                        "UPDATE generation_jobs SET progress = ? WHERE id = ?",
                        [overall_progress, job_id]
                    )

                    # Insert thumbnail if generated successfully
                    if thumbnail_info and thumbnail_info.get('path'):
                        concept = thumbnail_info.get('concept')
                        variation = thumbnail_info.get('variation')
                        conn.execute("""
                            INSERT INTO thumbnails
                            (job_id, prompt_index, variation_index, filepath, prompt_text, suggested_title, text_overlay)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, [
                            job_id,
                            thumbnail_info['concept_index'],
                            thumbnail_info['variation_index'],
                            str(thumbnail_info['path']),
                            variation.image_prompt if variation else None,
                            concept.suggested_title if concept else None,
                            variation.text_overlay if variation else None
                        ])

                except Exception:
                    pass  # Don't fail generation if progress/thumbnail update fails
                finally:
                    if conn:
                        conn.close()

            thumbnail_paths = generate_thumbnails(
                concepts=thumbnail_concepts,
                best_frames=best_frames,
                output=output,
                image_provider=image_provider,
                gemini_model=gemini_model,
                openai_model=openai_model,
                poe_model=poe_model,
                progress_callback=progress_callback,
                has_external_style_ref=has_external_style_ref
            )

            if not thumbnail_paths:
                await self.update_job_status(job_id, 'error', 0, 'Thumbnail generation failed')
                return

            # Thumbnails are already saved to database in progress_callback during generation
            # Just update video status
            await self.db.execute(
                "UPDATE videos SET status = 'completed' WHERE id = ?",
                [job['video_id']]
            )

            # Mark job as completed
            await self.db.execute("""
                UPDATE generation_jobs
                SET status = 'completed', progress = 100, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, [job_id])

            await self.db.commit()

        except Exception as e:
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

        total_thumbnails = job['num_prompts'] * job['num_variations']

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
        query = """
            SELECT id, filepath, prompt_index, variation_index, suggested_title, text_overlay
            FROM thumbnails
            WHERE job_id = ?
            ORDER BY prompt_index, variation_index
        """

        async with self.db.execute(query, [job_id]) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    async def cancel_job(self, job_id: int) -> bool:
        """Cancel a generation job."""
        job = await self.get_job(job_id)
        if not job:
            return False

        if job['status'] in ('completed', 'cancelled', 'error'):
            return False

        await self.update_job_status(job_id, 'cancelled')
        return True

    async def get_video_jobs(self, video_id: int) -> List[dict]:
        """Get all generation jobs for a video."""
        query = """
            SELECT id, cluster_id, num_prompts, num_variations, status, progress, created_at, completed_at
            FROM generation_jobs
            WHERE video_id = ?
            ORDER BY created_at DESC
        """

        async with self.db.execute(query, [video_id]) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    async def get_reference_frames_for_generation(
        self,
        cluster_id: int,
        limit: int = 10
    ) -> List[dict]:
        """
        Get reference frames for AI generation.

        V2 Architecture: Uses cluster_frame_assignments JOIN video_frames.

        Returns frames marked as references, ordered by reference_order.
        If no references are marked, falls back to top frames by quality_score.
        """
        # Try to get explicitly marked references first
        query = """
            SELECT vf.frame_path, vf.quality_score, vf.expression
            FROM video_frames vf
            JOIN cluster_frame_assignments cfa ON cfa.frame_id = vf.id
            WHERE cfa.cluster_id = ? AND cfa.is_reference = 1
            ORDER BY cfa.reference_order ASC
            LIMIT ?
        """
        async with self.db.execute(query, [cluster_id, limit]) as cursor:
            rows = await cursor.fetchall()
            if rows:
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in rows]

        # Fallback: top frames by quality score
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
            return [dict(zip(columns, row)) for row in rows]

    def _save_temp_reference_image(
        self,
        base64_data: str,
        output_dir: Path
    ) -> Optional[Path]:
        """
        Save base64 image to temporary file for reference.

        Args:
            base64_data: Base64 encoded image data
            output_dir: Directory to save the image

        Returns:
            Path to saved image or None if failed
        """
        try:
            import base64
            from PIL import Image
            import io

            # Decode base64
            img_bytes = base64.b64decode(base64_data)
            img = Image.open(io.BytesIO(img_bytes))

            # Resize if too large
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024))

            # Convert to RGB if necessary (e.g., for PNG with transparency)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # Save to temp location
            ref_dir = output_dir / "temp_refs"
            ref_dir.mkdir(exist_ok=True)
            ref_path = ref_dir / "external_reference.jpg"

            img.save(ref_path, 'JPEG', quality=90)
            return ref_path

        except Exception as e:
            # Log but don't fail - just skip the external reference
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not save external reference image: {e}")
            return None

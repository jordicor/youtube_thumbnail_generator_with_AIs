"""
YouTube Thumbnail Generator - Image Generation Module
======================================================
Generates thumbnail images using Gemini (Nano Banana Pro),
OpenAI DALL-E, or Replicate APIs.
"""

import base64
import io
from pathlib import Path
from datetime import datetime
from typing import Optional
from PIL import Image

from config import (
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    POE_API_KEY,
    GEMINI_IMAGE_MODEL,
    GEMINI_PRO_OUTPUT_RESOLUTION,
    OPENAI_IMAGE_MODEL,
    OPENAI_IMAGE_SIZE,
    OPENAI_IMAGE_FORMAT,
    OPENAI_IMAGE_QUALITY,
    POE_IMAGE_MODEL,
    POE_IMAGE_QUALITY,
    POE_IMAGE_ASPECT,
    POE_MODEL_MAX_REFS,
    GEMINI_MODEL_MAX_REFS,
    OPENAI_MODEL_MAX_REFS,
    REPLICATE_MODEL_MAX_REFS,
    THUMBNAIL_WIDTH,
    THUMBNAIL_HEIGHT,
    IMAGE_PROVIDER,
    KEEP_RAW_THUMBNAILS,
)
from utils import setup_logger, VideoOutput
from prompt_generation import (
    ThumbnailPrompt,
    ThumbnailConcept,
    ThumbnailVariation,
    enhance_prompt_with_face_description,
    enhance_prompt_with_face_description_from_text,
    get_json_prompt_for_generation
)

logger = setup_logger(__name__)


# =============================================================================
# FILENAME HELPERS
# =============================================================================

def normalize_model_name_for_filename(model_name: str) -> str:
    """Convert model name to filename-safe format (dots to dashes)."""
    return model_name.replace(".", "-")


def save_raw_thumbnail(image_data: bytes, output_path: Path, logger) -> Optional[Path]:
    """
    Save the original (raw) image before resizing if KEEP_RAW_THUMBNAILS is enabled.

    Saves to: thumbnails/raw/{filename_with_raw_prefix}

    Args:
        image_data: Raw image bytes from API
        output_path: The final output path (used to derive raw path)
        logger: Logger instance

    Returns:
        Path to saved raw image or None if not saved
    """
    if not KEEP_RAW_THUMBNAILS:
        return None

    try:
        # Create raw directory
        raw_dir = output_path.parent / "raw"
        raw_dir.mkdir(exist_ok=True)

        # Build raw filename: add _raw_ before timestamp
        # Original: model_timestamp_cX_vY.png
        # Raw: model_raw_timestamp_cX_vY.png
        stem = output_path.stem
        parts = stem.split("_", 1)  # Split at first underscore
        if len(parts) >= 2:
            raw_filename = f"{parts[0]}_raw_{parts[1]}{output_path.suffix}"
        else:
            raw_filename = f"raw_{stem}{output_path.suffix}"

        raw_path = raw_dir / raw_filename

        # Save raw image
        with open(raw_path, 'wb') as f:
            f.write(image_data)

        # Get dimensions for logging
        img = Image.open(raw_path)
        logger.info(f"Raw image saved: {raw_path.name} ({img.width}x{img.height})")

        return raw_path

    except Exception as e:
        logger.warning(f"Could not save raw thumbnail: {e}")
        return None


def get_effective_model(provider: str, gemini_model: str = None,
                        openai_model: str = None, poe_model: str = None) -> str:
    """Get the model name that will be used for a given provider."""
    if provider == "gemini":
        return gemini_model or GEMINI_IMAGE_MODEL
    elif provider == "openai":
        return openai_model or OPENAI_IMAGE_MODEL
    elif provider == "poe":
        return poe_model or POE_IMAGE_MODEL
    elif provider == "replicate":
        return "flux-1-1-pro"
    return "unknown"


# =============================================================================
# IMAGE GENERATION PROVIDERS
# =============================================================================

def generate_with_gemini(
    prompt: str,
    reference_images: list[Path] = None,
    output_path: Path = None,
    model: str = None
) -> Optional[Path]:
    """
    Generate image using Google Gemini API (Nano Banana / Nano Banana Pro).

    Args:
        prompt: Text prompt for image generation
        reference_images: Optional list of reference image paths
        output_path: Where to save the generated image

    Returns:
        Path to generated image or None if failed
    """

    if not GEMINI_API_KEY:
        logger.error("Gemini API key not set")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Build the prompt with STRICT identity preservation
        if reference_images and len(reference_images) > 0:
            full_prompt = f"""STRICT IDENTITY CLONING - THIS IS THE SAME PERSON, NOT A SIMILAR ONE:
The generated image MUST show an EXACT CLONE of the person in the reference photos.
This is NOT "inspired by" or "similar to" - it must be THE EXACT SAME PERSON.

═══════════════════════════════════════════════════════════════
ABSOLUTELY FORBIDDEN - DO NOT DO ANY OF THESE:
═══════════════════════════════════════════════════════════════
❌ Add facial hair (beard, mustache, stubble) if NOT present in reference
❌ Remove facial hair if present in reference
❌ Change body weight or build (no thinner, no heavier, no more muscular)
❌ Modify face shape, jaw line, chin, or cheekbones
❌ Alter nose size, shape, or bridge
❌ Change eye size, shape, spacing, or color
❌ Modify lip size, shape, or fullness
❌ Add or remove glasses/sunglasses
❌ Change hairstyle, hair length, or hair color
❌ Alter skin tone, complexion, or texture
❌ Add wrinkles, scars, moles, or marks not in reference
❌ "Beautify", "idealize", or "enhance" any features
❌ Age the person (older or younger)
❌ Change clothing style or colors from reference

═══════════════════════════════════════════════════════════════
MANDATORY - COPY EXACTLY FROM REFERENCE:
═══════════════════════════════════════════════════════════════
✅ Exact facial proportions (interpupillary distance, nose-to-lip ratio)
✅ Same body type and physical build
✅ Identical skin texture, tone, and any visible features
✅ Same hair (exact style, color, length, texture)
✅ Same facial hair status (if clean-shaven in reference, MUST be clean-shaven)
✅ Same clothing from reference images
✅ Same accessories (glasses, earrings, etc.)

SCENE TO CREATE (THIS IS THE PRIMARY INSTRUCTION):
{prompt}

IMPORTANT: The scene description above is the MAIN guide for what to create.
Reference images provide identity to preserve, but the scene, composition,
background, and action should follow the prompt above, not the reference images.

TECHNICAL:
- 16:9 aspect ratio (1280x720)
- Eye-catching and professional
- Bold, contrasting colors
- DO NOT include any text in the image

VERIFICATION: A friend of this person should recognize them INSTANTLY.
"""
        else:
            full_prompt = f"""Create a professional YouTube thumbnail.

{prompt}

Requirements:
- 16:9 aspect ratio (1280x720 pixels)
- Eye-catching and bold design
- Professional YouTube thumbnail style
- DO NOT include any text in the image
"""

        # Use provided model or default
        gemini_model = model or GEMINI_IMAGE_MODEL

        logger.info(f"Generating image with Gemini ({gemini_model})...")

        # Prepare content parts
        contents = []

        # Add reference images if provided
        # Limit depends on model: Gemini 3 Pro = 14, Gemini 2.5 Flash = 3
        if reference_images:
            max_refs = GEMINI_MODEL_MAX_REFS.get(gemini_model, 3)  # Default 3 for Flash
            logger.info(f"Using up to {max_refs} reference images for {gemini_model}")
            for img_path in reference_images[:max_refs]:
                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        if max(img.size) > 1024:
                            img.thumbnail((1024, 1024))

                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format='JPEG', quality=85)
                        img_bytes = img_buffer.getvalue()

                        contents.append(types.Part.from_bytes(
                            data=img_bytes,
                            mime_type="image/jpeg"
                        ))
                        logger.info(f"Added reference image: {img_path.name}")
                    except Exception as e:
                        logger.warning(f"Could not load reference image {img_path}: {e}")

        # Add text prompt
        contents.append(full_prompt)

        # Build generation config
        # Gemini 3 Pro supports image_size and aspect_ratio for higher resolution output
        is_gemini_3_pro = "gemini-3-pro" in gemini_model

        if is_gemini_3_pro:
            gen_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(
                    aspect_ratio="16:9",
                    image_size=GEMINI_PRO_OUTPUT_RESOLUTION,  # "1K", "2K", or "4K"
                ),
            )
            logger.info(f"Using Gemini 3 Pro with {GEMINI_PRO_OUTPUT_RESOLUTION} resolution (16:9)")
        else:
            # Standard config for other models (gemini-2.5-flash-image, etc.)
            gen_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            )

        # Generate with image output modality
        response = client.models.generate_content(
            model=gemini_model,
            contents=contents,
            config=gen_config,
        )

        # Extract image from response
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            if part.inline_data.mime_type.startswith('image/'):
                                image_data = part.inline_data.data

                                # Save image
                                if output_path:
                                    # Save raw image first (if enabled)
                                    save_raw_thumbnail(image_data, output_path, logger)

                                    # Load image for resizing
                                    img = Image.open(io.BytesIO(image_data))

                                    # Resize to exact thumbnail dimensions
                                    img = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)
                                    img.save(output_path, 'PNG', quality=95)

                                    logger.success(f"Image saved to: {output_path} ({THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT})")
                                    return output_path

        logger.warning("No image in Gemini response")
        return None

    except ImportError:
        logger.error("google-genai package not installed. Install with: pip install google-genai")
        return None
    except Exception as e:
        logger.error(f"Gemini image generation failed: {e}")
        return None


def generate_with_openai(
    prompt: str,
    reference_images: list[Path] = None,
    output_path: Path = None,
    model: str = None
) -> Optional[Path]:
    """
    Generate image using OpenAI GPT Image or DALL-E.

    Supports:
    - gpt-image-1.5 (best quality, recommended) - up to 16 reference images
    - gpt-image-1 (standard) - up to 16 reference images
    - gpt-image-1-mini (faster, lower cost) - up to 16 reference images
    - dall-e-3 (legacy, deprecated) - NO reference support

    Reference images are passed as base64 encoded images in the prompt.
    """

    if not OPENAI_API_KEY:
        logger.error("OpenAI API key not set")
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)

        # Use provided model or default from config
        selected_model = model or OPENAI_IMAGE_MODEL
        is_gpt_image = selected_model.startswith("gpt-image")

        # Check reference image support for this model
        max_refs = OPENAI_MODEL_MAX_REFS.get(selected_model, 0)
        has_refs = reference_images and len(reference_images) > 0 and max_refs > 0

        if reference_images and max_refs == 0:
            logger.warning(
                f"{selected_model} does NOT support reference images. "
                f"The {len(reference_images)} provided reference image(s) will be IGNORED."
            )

        logger.info(f"Generating image with OpenAI ({selected_model})...")
        if has_refs:
            logger.info(f"Using up to {max_refs} reference images")

        # Build prompt with YouTube specific instructions and reference context
        if has_refs:
            # Build prompt with reference image instructions
            ref_count = min(len(reference_images), max_refs)
            full_prompt = f"""STRICT IDENTITY CLONING from the {ref_count} reference image(s) provided.
The generated image MUST show the EXACT SAME PERSON from the reference photos.

Professional YouTube video thumbnail.

{prompt}

MANDATORY - COPY EXACTLY FROM REFERENCE:
- Exact facial features and proportions
- Same hair style, color, and length
- Same skin tone and texture
- Same body type

Style: Bold, eye-catching YouTube thumbnail with high contrast colors.
Do NOT include any text or letters in the image.
Photorealistic style, professional quality.

VERIFICATION: A friend of this person should recognize them INSTANTLY.
"""
        else:
            full_prompt = f"""Professional YouTube video thumbnail.

{prompt}

Style: Bold, eye-catching YouTube thumbnail with high contrast colors.
Do NOT include any text or letters in the image.
Photorealistic style, professional quality.
"""

        if is_gpt_image:
            # GPT Image models (gpt-image-1, gpt-image-1.5, gpt-image-1-mini)
            response = None

            if has_refs:
                # Use images.edit endpoint with reference images
                # images.edit accepts an array of images (up to 16 for gpt-image models)
                ref_image_files = []
                for img_path in reference_images[:max_refs]:
                    if img_path.exists():
                        try:
                            # Load and resize image for efficiency
                            img = Image.open(img_path)
                            if max(img.size) > 1024:
                                img.thumbnail((1024, 1024))

                            # Convert to PNG bytes (required format)
                            img_buffer = io.BytesIO()
                            img.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            ref_image_files.append(img_buffer)
                            logger.info(f"Added reference image: {img_path.name}")
                        except Exception as e:
                            logger.warning(f"Could not load reference image {img_path}: {e}")

                if ref_image_files:
                    # Use images.edit with reference images
                    # input_fidelity="high" preserves facial features better (not supported on mini)
                    # Pass as tuples with filename and mime type for proper detection
                    image_tuples = [
                        (f"ref_{i}.png", img_buffer, "image/png")
                        for i, img_buffer in enumerate(ref_image_files)
                    ]

                    # Build edit params - input_fidelity only supported on gpt-image-1 and gpt-image-1.5
                    edit_params = {
                        "model": selected_model,
                        "image": image_tuples if len(image_tuples) > 1 else image_tuples[0],
                        "prompt": full_prompt,
                        "size": OPENAI_IMAGE_SIZE,
                    }
                    if selected_model in {"gpt-image-1", "gpt-image-1.5"}:
                        edit_params["input_fidelity"] = "high"

                    response = client.images.edit(**edit_params)

            # Fallback to standard generation if no refs or refs failed to load
            if response is None:
                response = client.images.generate(
                    model=selected_model,
                    prompt=full_prompt,
                    size=OPENAI_IMAGE_SIZE,
                    quality=OPENAI_IMAGE_QUALITY,
                    n=1,
                )

            # Extract image from response
            if response.data and len(response.data) > 0:
                image_data = response.data[0]

                # GPT Image returns b64_json by default
                if hasattr(image_data, 'b64_json') and image_data.b64_json:
                    img_bytes = base64.b64decode(image_data.b64_json)
                elif hasattr(image_data, 'url') and image_data.url:
                    # Fallback to URL if available
                    import requests
                    img_response = requests.get(image_data.url)
                    if img_response.status_code != 200:
                        logger.error("Failed to download image from URL")
                        return None
                    img_bytes = img_response.content
                else:
                    logger.error("No image data in response")
                    return None

                # Save raw image first (if enabled)
                save_raw_thumbnail(img_bytes, output_path, logger)

                # Load and resize to exact thumbnail dimensions
                img = Image.open(io.BytesIO(img_bytes))
                img = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)

                # Save with configured format
                save_format = OPENAI_IMAGE_FORMAT.upper()
                if save_format == "JPG":
                    save_format = "JPEG"

                img.save(output_path, save_format, quality=95)
                logger.success(f"Image saved to: {output_path} ({THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT})")
                return output_path

        else:
            # Legacy DALL-E 3 (deprecated but still supported until 2026)
            response = client.images.generate(
                model=selected_model,
                prompt=full_prompt[:4000],  # DALL-E 3 has a 4000 char limit
                size="1792x1024",  # Closest to 16:9
                quality="hd",
                n=1,
            )

            if response.data and len(response.data) > 0:
                image_url = response.data[0].url

                # Download image
                import requests
                img_response = requests.get(image_url)

                if img_response.status_code == 200:
                    img_bytes = img_response.content

                    # Save raw image first (if enabled)
                    save_raw_thumbnail(img_bytes, output_path, logger)

                    # Load and resize to exact thumbnail dimensions
                    img = Image.open(io.BytesIO(img_bytes))
                    img = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)

                    # Save
                    img.save(output_path, 'PNG', quality=95)
                    logger.success(f"Image saved to: {output_path} ({THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT})")
                    return output_path

        return None

    except Exception as e:
        logger.error(f"OpenAI image generation failed: {e}")
        return None


def generate_with_replicate(
    prompt: str,
    reference_images: list[Path] = None,
    output_path: Path = None,
    model: str = None
) -> Optional[Path]:
    """
    Generate image using Replicate API (FLUX, SDXL, etc.).

    Supports face reference via image_prompt (FLUX) or face-to-many.
    Reference limits per model:
    - flux-1.1-pro: 1 image (via image_prompt)
    - face-to-many: 1 face image
    """

    try:
        import replicate
        import requests

        # Use provided model or default
        selected_model = model or "black-forest-labs/flux-1.1-pro"
        model_short_name = selected_model.split("/")[-1] if "/" in selected_model else selected_model

        # Get max refs for this model
        max_refs = REPLICATE_MODEL_MAX_REFS.get(model_short_name, 1)

        logger.info(f"Generating image with Replicate ({selected_model})...")

        # Prepare input
        input_data = {
            "prompt": f"Professional YouTube thumbnail. {prompt}. Bold colors, eye-catching, 16:9 aspect ratio.",
            "aspect_ratio": "16:9",
            "output_format": "png",
            "output_quality": 95,
        }

        # If we have reference images, use them according to model limits
        if reference_images and len(reference_images) > 0:
            ref_img = reference_images[0]  # Most Replicate models only support 1 reference
            if ref_img.exists():
                try:
                    logger.info(f"Using {min(len(reference_images), max_refs)} reference image(s) for {model_short_name}")

                    # Resize image for efficiency
                    img = Image.open(ref_img)
                    if max(img.size) > 1024:
                        img.thumbnail((1024, 1024))
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='JPEG', quality=85)
                    img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

                    if "face-to-many" in selected_model:
                        # Use face-to-many for stylized face transformations
                        input_data = {
                            "image": f"data:image/jpeg;base64,{img_data}",
                            "prompt": prompt,
                            "style": "Video game",  # or "3D", "Emoji", etc.
                            "negative_prompt": "blurry, low quality",
                        }
                    else:
                        # FLUX 1.1 Pro uses image_prompt for reference
                        input_data["image_prompt"] = f"data:image/jpeg;base64,{img_data}"
                        logger.info(f"Added reference image: {ref_img.name}")

                except Exception as e:
                    logger.warning(f"Could not use face reference: {e}")

        output = replicate.run(selected_model, input=input_data)

        # Handle output (can be URL or list)
        if isinstance(output, list):
            image_url = output[0] if output else None
        else:
            image_url = output

        if image_url:
            # Download and save
            response = requests.get(image_url)
            if response.status_code == 200:
                img_bytes = response.content

                # Save raw image first (if enabled)
                save_raw_thumbnail(img_bytes, output_path, logger)

                # Load and resize
                img = Image.open(io.BytesIO(img_bytes))
                img = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)
                img.save(output_path, 'PNG', quality=95)
                logger.success(f"Image saved to: {output_path} ({THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT})")
                return output_path

        return None

    except ImportError:
        logger.error("replicate package not installed")
        return None
    except Exception as e:
        logger.error(f"Replicate image generation failed: {e}")
        return None


def generate_with_poe(
    prompt: str,
    reference_images: list[Path] = None,
    output_path: Path = None,
    model: str = None
) -> Optional[Path]:
    """
    Generate image using Poe API with FLUX/Ideogram models.

    Supports reference images for character/face consistency.
    Available models (bot names verified 2025-12-28):
        - flux2pro: Best quality, up to 8 references
        - flux2flex: High resolution (14MP)
        - fluxkontextpro: Best prompt following
        - seedream40: Good for combining references
        - nanobananapro: Gemini 3 Pro via Poe
        - Ideogram-v3: Best for text/logos

    Args:
        prompt: Text prompt for image generation
        reference_images: Optional list of reference image paths (up to 3)
        output_path: Where to save the generated image
        model: Model to use (default from config: flux2pro)

    Returns:
        Path to generated image or None if failed
    """

    if not POE_API_KEY:
        logger.error("Poe API key not set")
        return None

    try:
        from openai import OpenAI
        import requests

        # Initialize Poe client (OpenAI-compatible)
        client = OpenAI(
            api_key=POE_API_KEY,
            base_url="https://api.poe.com/v1",
        )

        selected_model = model or POE_IMAGE_MODEL
        logger.info(f"Generating image with Poe ({selected_model})...")

        # Build message content
        content = []

        # Add reference images
        # Each model has different limits (nanobananapro: 14, fluxkontextpro: 1, etc.)
        if reference_images:
            max_images = POE_MODEL_MAX_REFS.get(selected_model, 8)
            for img_path in reference_images[:max_images]:
                if img_path.exists():
                    try:
                        # Load and resize image for efficiency
                        img = Image.open(img_path)
                        if max(img.size) > 1024:
                            img.thumbnail((1024, 1024))

                        # Convert to base64
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format='JPEG', quality=85)
                        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        })
                        logger.info(f"Added reference image: {img_path.name}")
                    except Exception as e:
                        logger.warning(f"Could not load reference image {img_path}: {e}")

        # Build prompt with STRICT identity preservation instructions
        if reference_images and len(reference_images) > 0:
            full_prompt = f"""STRICT IDENTITY CLONING - THIS IS THE SAME PERSON, NOT A SIMILAR ONE:
The generated image MUST show an EXACT CLONE of the person in the reference photos.
This is NOT "inspired by" or "similar to" - it must be THE EXACT SAME PERSON.

═══════════════════════════════════════════════════════════════
ABSOLUTELY FORBIDDEN - DO NOT DO ANY OF THESE:
═══════════════════════════════════════════════════════════════
❌ Add facial hair (beard, mustache, stubble) if NOT present in reference
❌ Remove facial hair if present in reference
❌ Change body weight or build (no thinner, no heavier, no more muscular)
❌ Modify face shape, jaw line, chin, or cheekbones
❌ Alter nose size, shape, or bridge
❌ Change eye size, shape, spacing, or color
❌ Modify lip size, shape, or fullness
❌ Add or remove glasses/sunglasses
❌ Change hairstyle, hair length, or hair color
❌ Alter skin tone, complexion, or texture
❌ Add wrinkles, scars, moles, or marks not in reference
❌ "Beautify", "idealize", or "enhance" any features
❌ Age the person (older or younger)
❌ Change clothing style or colors from reference

═══════════════════════════════════════════════════════════════
MANDATORY - COPY EXACTLY FROM REFERENCE:
═══════════════════════════════════════════════════════════════
✅ Exact facial proportions (interpupillary distance, nose-to-lip ratio)
✅ Same body type and physical build
✅ Identical skin texture, tone, and any visible features
✅ Same hair (exact style, color, length, texture)
✅ Same facial hair status (if clean-shaven in reference, MUST be clean-shaven)
✅ Same clothing from reference images
✅ Same accessories (glasses, earrings, etc.)

SCENE TO CREATE:
{prompt}

TECHNICAL:
- YouTube thumbnail style: bold, eye-catching, professional
- 16:9 aspect ratio (1280x720)
- High contrast colors
- DO NOT include any text in the image

VERIFICATION: A friend of this person should recognize them INSTANTLY."""
        else:
            full_prompt = f"""Create a professional YouTube thumbnail.

{prompt}

Requirements:
- 16:9 aspect ratio (1280x720)
- Bold, eye-catching design
- Professional YouTube thumbnail style
- DO NOT include any text in the image"""

        content.append({
            "type": "text",
            "text": full_prompt
        })

        # Make API call (stream=False is required for image models)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": content}],
            stream=False,
            extra_body={
                "aspect": POE_IMAGE_ASPECT,
                "quality": POE_IMAGE_QUALITY
            }
        )

        # Extract image from response
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message

            # Check for image in content
            if hasattr(message, 'content') and message.content:
                response_content = message.content

                # Handle different response formats
                image_url = None

                # If response is a list of content blocks
                if isinstance(response_content, list):
                    for block in response_content:
                        if isinstance(block, dict):
                            if block.get('type') == 'image_url':
                                image_url = block.get('image_url', {}).get('url')
                            elif block.get('type') == 'image':
                                image_url = block.get('url') or block.get('image_url')
                        elif hasattr(block, 'image_url'):
                            image_url = block.image_url.url if hasattr(block.image_url, 'url') else block.image_url

                # If response contains markdown image
                elif isinstance(response_content, str):
                    import re
                    # Look for markdown image: ![...](URL)
                    match = re.search(r'!\[.*?\]\((https?://[^\)]+)\)', response_content)
                    if match:
                        image_url = match.group(1)
                    # Look for direct URL
                    elif response_content.startswith('http'):
                        image_url = response_content.strip()

                # Download and save image
                if image_url:
                    if image_url.startswith('data:image'):
                        # Base64 encoded image
                        header, data = image_url.split(',', 1)
                        img_bytes = base64.b64decode(data)
                    else:
                        # URL - download image
                        img_response = requests.get(image_url, timeout=60)
                        if img_response.status_code != 200:
                            logger.error(f"Failed to download image: HTTP {img_response.status_code}")
                            return None
                        img_bytes = img_response.content

                    # Save raw image first (if enabled)
                    save_raw_thumbnail(img_bytes, output_path, logger)

                    # Load and resize to exact thumbnail dimensions
                    img = Image.open(io.BytesIO(img_bytes))
                    img = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)

                    # Save
                    img.save(output_path, 'PNG', quality=95)
                    logger.success(f"Image saved to: {output_path} ({THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT})")
                    return output_path

                else:
                    logger.warning(f"No image URL found in Poe response. Content: {response_content[:500] if response_content else 'None'}")

        logger.warning("No image in Poe response")
        return None

    except ImportError:
        logger.error("openai package not installed. Install with: pip install openai")
        return None
    except Exception as e:
        logger.error(f"Poe image generation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


# =============================================================================
# COMPOSITE THUMBNAIL CREATION
# =============================================================================

def create_composite_thumbnail(
    background_prompt: str,
    face_frame: Path,
    text_overlay: str,
    output_path: Path,
    colors: list[str] = None
) -> Optional[Path]:
    """
    Create a composite thumbnail with:
    - AI-generated or solid background
    - Real face/frame overlaid
    - Text overlay

    This is an alternative approach that keeps your real face.
    """

    try:
        from PIL import Image, ImageDraw, ImageFont

        logger.info("Creating composite thumbnail...")

        # Create base canvas
        thumbnail = Image.new('RGB', (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), '#1a1a2e')

        # Load face frame
        if face_frame and face_frame.exists():
            face_img = Image.open(face_frame)

            # Resize to fit nicely (about 60% of thumbnail height)
            face_height = int(THUMBNAIL_HEIGHT * 0.8)
            face_ratio = face_img.width / face_img.height
            face_width = int(face_height * face_ratio)

            face_img = face_img.resize((face_width, face_height), Image.Resampling.LANCZOS)

            # Position on right side
            x_pos = THUMBNAIL_WIDTH - face_width - 20
            y_pos = (THUMBNAIL_HEIGHT - face_height) // 2

            thumbnail.paste(face_img, (x_pos, y_pos))

        # Add gradient overlay on left for text area
        draw = ImageDraw.Draw(thumbnail)

        # Simple gradient effect
        primary_color = colors[0] if colors else '#FF0000'
        for i in range(THUMBNAIL_WIDTH // 2):
            alpha = int(255 * (1 - i / (THUMBNAIL_WIDTH // 2)))
            # This is simplified - real gradient would need alpha compositing

        # Add text overlay
        if text_overlay:
            try:
                # Try to use a bold font
                font_size = 80
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()

                # Text position (left side, vertically centered)
                text_x = 40
                text_y = THUMBNAIL_HEIGHT // 2 - font_size

                # Draw text with shadow for readability
                shadow_offset = 3
                draw.text((text_x + shadow_offset, text_y + shadow_offset),
                         text_overlay.upper(), fill='#000000', font=font)
                draw.text((text_x, text_y),
                         text_overlay.upper(), fill='#FFFFFF', font=font)

            except Exception as e:
                logger.warning(f"Could not add text overlay: {e}")

        # Save
        thumbnail.save(output_path, 'PNG', quality=95)
        logger.success(f"Composite thumbnail saved to: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Composite thumbnail creation failed: {e}")
        return None


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_thumbnails(
    concepts: list[ThumbnailConcept],
    best_frames: list[Path],
    output: VideoOutput,
    use_composite: bool = False,
    gemini_model: str = None,
    openai_model: str = None,
    poe_model: str = None,
    image_provider: str = None,
    progress_callback: callable = None,
    has_external_style_ref: bool = False
) -> list[Path]:
    """
    Generate thumbnails from concepts with variations.

    Args:
        concepts: List of ThumbnailConcept objects, each containing variations
        best_frames: List of best frame paths for reference
        output: VideoOutput instance
        use_composite: If True, create composite instead of full AI generation
        gemini_model: Override Gemini model (e.g., "gemini-3-pro-image-preview")
        openai_model: Override OpenAI model (e.g., "gpt-image-1.5")
        poe_model: Override Poe model (e.g., "flux2pro", "fluxkontextpro", "Ideogram-v3")
        image_provider: Override image provider ("gemini", "openai", "replicate", "poe")
        progress_callback: Optional callback function(current, total, thumbnail_info) to report progress
                          thumbnail_info is a dict with: path, concept_index, variation_index, concept, variation
        has_external_style_ref: If True, first reference image is an external style reference (not a person)

    Returns:
        List of paths to generated thumbnails
    """

    thumbnails_dir = output.output_dir / "thumbnails"
    thumbnails_dir.mkdir(exist_ok=True)

    # Calculate total images from concepts and their variations
    total_images = sum(len(concept.variations) for concept in concepts)
    num_concepts = len(concepts)
    logger.info(f"Generating {total_images} thumbnails ({num_concepts} concepts with variations)")

    generated_thumbnails = []
    image_counter = 0

    for concept in concepts:
        concept_idx = concept.concept_index
        logger.info(f"\n[Concept {concept_idx}/{num_concepts}] {concept.concept_name}: {concept.suggested_title}")

        for variation in concept.variations:
            variation_idx = variation.variation_index
            image_counter += 1
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

            # Determine model prefix for filename
            selected_provider = image_provider or IMAGE_PROVIDER
            effective_model = get_effective_model(
                selected_provider, gemini_model, openai_model, poe_model
            )
            model_prefix = normalize_model_name_for_filename(effective_model)

            output_path = thumbnails_dir / f"{model_prefix}_{timestamp}_c{concept_idx}_v{variation_idx}.png"

            logger.info(f"  Variation {variation_idx} [{variation.variation_focus}] ({image_counter}/{total_images})...")

            # Option 1: Composite thumbnail
            if use_composite and best_frames:
                composite_path = thumbnails_dir / f"composite_{timestamp}_c{concept_idx}_v{variation_idx}.png"
                result = create_composite_thumbnail(
                    background_prompt=variation.image_prompt,
                    face_frame=best_frames[0],
                    text_overlay=variation.text_overlay,
                    output_path=composite_path,
                    colors=concept.colors
                )
                if result:
                    generated_thumbnails.append(result)
                    if progress_callback:
                        progress_callback(image_counter, total_images, {
                            'path': result,
                            'concept_index': concept_idx,
                            'variation_index': variation_idx,
                            'concept': concept,
                            'variation': variation
                        })
                else:
                    if progress_callback:
                        progress_callback(image_counter, total_images, None)
                continue

            # Option 2: Full AI generation
            # Each variation has its own unique image_prompt from the LLM
            enhanced_prompt = enhance_prompt_with_face_description_from_text(
                variation.image_prompt,
                best_frames[0] if best_frames else None
            )

            # Add note about external style reference if present
            if has_external_style_ref and best_frames:
                style_ref_note = """
═══════════════════════════════════════════════════════════════════════════════
STYLE REFERENCE (INSPIRATION ONLY - DO NOT COPY EXACTLY):
═══════════════════════════════════════════════════════════════════════════════
The FIRST reference image is for STYLE INSPIRATION only:
- Use it as a loose guide for colors, mood, or aesthetic if relevant
- The SCENE DESCRIPTION BELOW takes PRIORITY over the style reference
- Do NOT replicate the style reference exactly - use it as inspiration only
- Do NOT try to clone any person from the style reference

PRIORITY ORDER:
1. HIGHEST: The scene description below (image prompt)
2. SECOND: Identity of the person from reference images 2 onwards
3. LOWEST: Style inspiration from first reference (use loosely)

The remaining reference images show the actual person who must appear in the
thumbnail. Clone their identity ONLY from reference images 2 onwards.
═══════════════════════════════════════════════════════════════════════════════

"""
                enhanced_prompt = style_ref_note + enhanced_prompt

            # Try configured provider
            selected_provider = image_provider or IMAGE_PROVIDER

            generators = {
                "gemini": generate_with_gemini,
                "openai": generate_with_openai,
                "replicate": generate_with_replicate,
                "poe": generate_with_poe,
            }

            success = False

            # Try primary provider
            primary_generator = generators.get(selected_provider)
            if primary_generator:
                kwargs = {
                    "prompt": enhanced_prompt,
                    "reference_images": best_frames[:14] if best_frames else None,
                    "output_path": output_path
                }
                if selected_provider == "gemini" and gemini_model:
                    kwargs["model"] = gemini_model
                elif selected_provider == "openai" and openai_model:
                    kwargs["model"] = openai_model
                elif selected_provider == "poe" and poe_model:
                    kwargs["model"] = poe_model

                result = primary_generator(**kwargs)
                if result:
                    generated_thumbnails.append(result)
                    success = True

            # Try fallbacks if primary failed
            if not success:
                for provider, generator in generators.items():
                    if provider != selected_provider:
                        logger.info(f"    Trying fallback provider: {provider}")

                        fallback_model = get_effective_model(
                            provider, gemini_model, openai_model, poe_model
                        )
                        fallback_prefix = normalize_model_name_for_filename(fallback_model)
                        fallback_path = thumbnails_dir / f"{fallback_prefix}_{timestamp}_c{concept_idx}_v{variation_idx}.png"

                        kwargs = {
                            "prompt": enhanced_prompt,
                            "reference_images": best_frames[:14] if best_frames else None,
                            "output_path": fallback_path
                        }
                        if provider == "gemini" and gemini_model:
                            kwargs["model"] = gemini_model
                        elif provider == "openai" and openai_model:
                            kwargs["model"] = openai_model
                        elif provider == "poe" and poe_model:
                            kwargs["model"] = poe_model

                        result = generator(**kwargs)
                        if result:
                            generated_thumbnails.append(result)
                            success = True
                            break

            # Last resort: composite
            if not success and best_frames:
                logger.warning("    All AI providers failed, creating composite")
                composite_path = thumbnails_dir / f"composite_{timestamp}_c{concept_idx}_v{variation_idx}.png"
                result = create_composite_thumbnail(
                    background_prompt=variation.image_prompt,
                    face_frame=best_frames[0],
                    text_overlay=variation.text_overlay,
                    output_path=composite_path,
                    colors=concept.colors
                )
                if result:
                    generated_thumbnails.append(result)
                    success = True

            # Report progress after each image
            if progress_callback:
                last_thumbnail = generated_thumbnails[-1] if success and generated_thumbnails else None
                thumbnail_info = {
                    'path': last_thumbnail,
                    'concept_index': concept_idx,
                    'variation_index': variation_idx,
                    'concept': concept,
                    'variation': variation
                } if last_thumbnail else None
                progress_callback(image_counter, total_images, thumbnail_info)

    logger.success(f"Generated {len(generated_thumbnails)}/{total_images} thumbnails")
    logger.info(f"Thumbnails saved to: {thumbnails_dir}")

    return generated_thumbnails


if __name__ == "__main__":
    # Test thumbnail generation
    from prompt_generation import ThumbnailPrompt

    test_prompt = ThumbnailPrompt(
        video_title="Test Video",
        suggested_title="Amazing Gaming Tips",
        thumbnail_concept="A gamer looking excited at the camera",
        image_prompt="Professional YouTube thumbnail, excited gamer with headphones, gaming setup background, neon lights, bold colors",
        text_overlay="PRO TIPS",
        mood="excited",
        colors=["#FF0000", "#00FF00", "#0000FF"],
        key_topics=["gaming", "tips", "youtube"]
    )

    from config import OUTPUT_DIR

    output = VideoOutput(Path("test_video.mp4"), OUTPUT_DIR)
    output.setup()

    result = generate_thumbnail(test_prompt, [], output)

    if result:
        print(f"Thumbnail generated: {result}")
    else:
        print("Thumbnail generation failed")

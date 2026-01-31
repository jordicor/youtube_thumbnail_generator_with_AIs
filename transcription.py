"""
YouTube Thumbnail Generator - Transcription Module
===================================================
Transcribes video audio using ElevenLabs API or local Whisper.
Based on your existing batch_transcribe.py script.

When using ElevenLabs, diarization is enabled to get word-level timestamps
and speaker identification for generating accurate timestamps in descriptions.
"""

import subprocess
import tempfile
import requests
import orjson
from pathlib import Path
from typing import Optional, Tuple

from config import (
    USE_LOCAL_WHISPER,
    WHISPER_MODEL,
    TRANSCRIPTION_LANGUAGE,
    ELEVENLABS_API_KEY,
    ELEVENLABS_API_URL,
    ELEVENLABS_MODEL,
    ELEVENLABS_KEYTERMS,
    ELEVENLABS_ENTITY_DETECTION,
)
from utils import setup_logger, VideoOutput

logger = setup_logger(__name__)


# =============================================================================
# AUDIO EXTRACTION
# =============================================================================

def extract_audio(video_path: Path) -> Optional[Path]:
    """Extract audio from video using ffmpeg"""

    logger.info("Extracting audio with ffmpeg...")

    # Create temporary file for audio
    temp_audio = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    temp_audio_path = Path(temp_audio.name)
    temp_audio.close()

    try:
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',                  # No video
            '-acodec', 'libmp3lame',
            '-ab', '128k',          # Bitrate
            '-ar', '44100',         # Sample rate
            '-ac', '1',             # Mono
            '-y',                   # Overwrite
            str(temp_audio_path)
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600  # 10 minutes
        )

        if result.returncode == 0:
            audio_size_mb = temp_audio_path.stat().st_size / (1024 * 1024)
            logger.success(f"Audio extracted: {audio_size_mb:.2f} MB")
            return temp_audio_path
        else:
            logger.error(f"ffmpeg failed: {result.stderr.decode('utf-8', errors='ignore')[:500]}")
            temp_audio_path.unlink(missing_ok=True)
            return None

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timeout (video too long)")
        temp_audio_path.unlink(missing_ok=True)
        return None
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        temp_audio_path.unlink(missing_ok=True)
        return None


# =============================================================================
# ELEVENLABS TRANSCRIPTION
# =============================================================================

def transcribe_elevenlabs(audio_path: Path) -> Tuple[Optional[str], Optional[dict]]:
    """
    Transcribe audio using ElevenLabs API with diarization enabled.

    Uses Scribe v2 by default, which supports:
    - Keyterm prompting (up to 100 terms) for better recognition of specific words
    - Entity detection (56 types) with exact timestamps
    - Up to 48 speakers for diarization

    Returns:
        Tuple of (text, full_response_dict)
        - text: Plain text transcription
        - full_response: Complete API response with words, timestamps, speakers
    """

    audio_size_mb = audio_path.stat().st_size / (1024 * 1024)

    if audio_size_mb > 3000:  # Scribe v2 supports up to 3GB
        logger.error(f"Audio file too large for ElevenLabs: {audio_size_mb:.2f} MB (max 3GB)")
        return None, None

    headers = {
        'xi-api-key': ELEVENLABS_API_KEY
    }

    data = {
        'model_id': ELEVENLABS_MODEL,
        'language_code': TRANSCRIPTION_LANGUAGE,
        'diarize': 'true',  # Enable diarization for word-level timestamps & speaker ID
        'timestamps_granularity': 'word',  # Get word-level timestamps
    }

    # Parse keyterms if configured (Scribe v2 only)
    keyterms = []
    if ELEVENLABS_KEYTERMS:
        keyterms = [k.strip() for k in ELEVENLABS_KEYTERMS.split(',') if k.strip()]
        if keyterms:
            logger.info(f"Using {len(keyterms)} keyterms for improved recognition")

    # Add entity detection if configured (Scribe v2 only, additional cost)
    if ELEVENLABS_ENTITY_DETECTION:
        data['entity_detection'] = ELEVENLABS_ENTITY_DETECTION
        logger.info(f"Entity detection enabled: {ELEVENLABS_ENTITY_DETECTION}")

    logger.info(f"Uploading audio ({audio_size_mb:.2f} MB) to ElevenLabs API ({ELEVENLABS_MODEL}, diarization enabled)...")

    try:
        with open(audio_path, 'rb') as audio_file:
            files = {
                'file': (audio_path.name, audio_file, 'audio/mpeg')
            }

            # Convert data to list of tuples to support multiple keyterms fields
            # ElevenLabs API expects each keyterm as a separate form field
            data_tuples = list(data.items())
            for term in keyterms:
                data_tuples.append(('keyterms', term))

            response = requests.post(
                ELEVENLABS_API_URL,
                headers=headers,
                data=data_tuples,
                files=files,
                timeout=900  # 15 minutes
            )

        if response.status_code == 200:
            result = response.json()
            text = result.get('text', '')
            word_count = len(result.get('words', []))
            logger.success(f"Transcription complete: {len(text)} chars, {word_count} words with timestamps")
            return text, result
        else:
            logger.error(f"API Error {response.status_code}: {response.text[:500]}")
            return None, None

    except requests.exceptions.Timeout:
        logger.error("ElevenLabs API timeout")
        return None, None
    except Exception as e:
        logger.error(f"ElevenLabs API failed: {e}")
        return None, None


# =============================================================================
# LOCAL WHISPER TRANSCRIPTION
# =============================================================================

def transcribe_whisper_local(audio_path: Path) -> Optional[str]:
    """Transcribe audio using local Whisper model (GPU)"""

    try:
        import whisper

        logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
        model = whisper.load_model(WHISPER_MODEL)

        logger.info("Transcribing with Whisper (GPU)...")

        result = model.transcribe(
            str(audio_path),
            language=TRANSCRIPTION_LANGUAGE,
            verbose=False
        )

        text = result.get('text', '')
        logger.success(f"Transcription complete: {len(text)} characters")

        return text

    except ImportError:
        logger.error("Whisper not installed. Install with: pip install openai-whisper")
        return None
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return None


# =============================================================================
# MAIN TRANSCRIPTION FUNCTION
# =============================================================================

def transcribe_video(video_path: Path, output: VideoOutput) -> Optional[str]:
    """
    Transcribe a video file.

    Args:
        video_path: Path to the video file
        output: VideoOutput instance for saving results

    Returns:
        Transcription text or None if failed
    """

    logger.info(f"Transcribing: {video_path.name}")

    # Check if already transcribed
    if output.transcription_file.exists():
        logger.info("Loading cached transcription...")
        with open(output.transcription_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Skip header lines
            lines = content.split('\n')
            text_start = 0
            for i, line in enumerate(lines):
                if line.startswith('='):
                    text_start = i + 2
                    break
            return '\n'.join(lines[text_start:]).strip()

    # Extract audio
    audio_path = extract_audio(video_path)
    if not audio_path:
        return None

    try:
        # Transcribe
        full_response = None
        if USE_LOCAL_WHISPER:
            text = transcribe_whisper_local(audio_path)
        else:
            text, full_response = transcribe_elevenlabs(audio_path)

        if not text:
            return None

        # Save transcription (text file and JSON with timestamps if available)
        save_transcription(text, output, video_path, full_response)

        return text

    finally:
        # Clean up temp audio
        if audio_path and audio_path.exists():
            audio_path.unlink()


def save_transcription(
    text: str,
    output: VideoOutput,
    video_path: Path,
    full_response: Optional[dict] = None
):
    """
    Save transcription to file.

    Args:
        text: Plain text transcription
        output: VideoOutput instance
        video_path: Original video path
        full_response: Complete API response with words/timestamps (ElevenLabs only)
    """

    from datetime import datetime

    output.transcription_file.parent.mkdir(parents=True, exist_ok=True)

    # Save plain text transcription
    with open(output.transcription_file, 'w', encoding='utf-8') as f:
        f.write(f"# Transcription\n")
        f.write(f"# Video: {video_path.name}\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Language: {TRANSCRIPTION_LANGUAGE}\n")
        f.write(f"# Method: {'Whisper (local)' if USE_LOCAL_WHISPER else 'ElevenLabs API'}\n")
        f.write(f"\n{'='*80}\n\n")
        f.write(text)
        f.write(f"\n")

    logger.success(f"Transcription saved: {output.transcription_file}")

    # Save JSON with word-level timestamps if available (ElevenLabs)
    if full_response and full_response.get('words'):
        json_path = output.transcription_file.with_suffix('.json')
        with open(json_path, 'wb') as f:
            f.write(orjson.dumps(full_response, option=orjson.OPT_INDENT_2))
        word_count = len(full_response.get('words', []))
        logger.success(f"Transcription JSON saved: {json_path} ({word_count} words with timestamps)")

        # Generate and save segments cache for faster timestamp lookups
        from transcript_processing import save_segments_cache
        cache_path = save_segments_cache(json_path)
        if cache_path:
            logger.success(f"Segments cache saved: {cache_path}")


# =============================================================================
# TRANSCRIPTION SUMMARY FOR PROMPTS
# =============================================================================

def get_transcription_summary(text: str, max_length: int = 2000) -> str:
    """
    Get a summary of the transcription suitable for prompt generation.

    Returns the first portion of the transcription, trying to end at a sentence.
    """

    if not text:
        return ""

    if len(text) <= max_length:
        return text

    # Try to cut at a sentence boundary
    truncated = text[:max_length]

    # Find last sentence ending
    for sep in ['. ', '! ', '? ', '\n']:
        last_sep = truncated.rfind(sep)
        if last_sep > max_length * 0.5:  # Only if we keep at least half
            return truncated[:last_sep + 1].strip()

    return truncated.strip() + "..."

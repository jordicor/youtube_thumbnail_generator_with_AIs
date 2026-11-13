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


class TranscriptionError(Exception):
    """The audio could not be extracted or transcribed. The message says why."""


def _last_lines(output: bytes, count: int = 3) -> str:
    """The end of a tool's output, where ffmpeg and ffprobe say what went wrong."""
    lines = [line.strip() for line in output.decode('utf-8', errors='ignore').splitlines() if line.strip()]
    return ' | '.join(lines[-count:])[:500]


# =============================================================================
# AUDIO EXTRACTION
# =============================================================================

def has_audio_track(video_path: Path) -> bool:
    """Whether the video has an audio track at all (ffprobe)."""

    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream=index',
        '-of', 'csv=p=0',
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    except subprocess.TimeoutExpired:
        raise TranscriptionError("The video could not be read: ffprobe gave no answer in 2 minutes") from None
    except OSError as error:
        raise TranscriptionError(f"The video could not be read: ffprobe did not run ({error})") from error

    if result.returncode != 0:
        raise TranscriptionError(f"The video could not be read: ffprobe: {_last_lines(result.stderr)}")
    return bool(result.stdout.strip())


def extract_audio(video_path: Path) -> Path:
    """Extract audio from video using ffmpeg. Raises TranscriptionError if it cannot."""

    logger.info("Extracting audio with ffmpeg...")

    # Create temporary file for audio
    temp_audio = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    temp_audio_path = Path(temp_audio.name)
    temp_audio.close()

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

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600  # 10 minutes
        )
    except subprocess.TimeoutExpired:
        temp_audio_path.unlink(missing_ok=True)
        raise TranscriptionError("The audio could not be extracted: ffmpeg did not finish in 10 minutes") from None
    except OSError as error:
        temp_audio_path.unlink(missing_ok=True)
        raise TranscriptionError(f"The audio could not be extracted: ffmpeg did not run ({error})") from error

    if result.returncode != 0:
        temp_audio_path.unlink(missing_ok=True)
        raise TranscriptionError(f"The audio could not be extracted: ffmpeg: {_last_lines(result.stderr)}")

    audio_size_mb = temp_audio_path.stat().st_size / (1024 * 1024)
    logger.success(f"Audio extracted: {audio_size_mb:.2f} MB")
    return temp_audio_path


# =============================================================================
# ELEVENLABS TRANSCRIPTION
# =============================================================================

def transcribe_elevenlabs(audio_path: Path) -> Tuple[str, dict]:
    """
    Transcribe audio using ElevenLabs API with diarization enabled.

    Uses Scribe v2 by default, which supports:
    - Keyterm prompting (up to 100 terms) for better recognition of specific words
    - Entity detection (56 types) with exact timestamps
    - Up to 48 speakers for diarization

    Returns:
        Tuple of (text, full_response_dict)
        - text: Plain text transcription, empty when nobody speaks
        - full_response: Complete API response with words, timestamps, speakers

    Raises:
        TranscriptionError: ElevenLabs did not transcribe the audio
    """

    if not ELEVENLABS_API_KEY:
        raise TranscriptionError("ElevenLabs could not transcribe the audio: ELEVENLABS_API_KEY is not set in .env")

    audio_size_mb = audio_path.stat().st_size / (1024 * 1024)

    if audio_size_mb > 3000:  # Scribe v2 supports up to 3GB
        raise TranscriptionError(
            f"ElevenLabs could not transcribe the audio: it is {audio_size_mb:.0f} MB and the limit is 3 GB"
        )

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

    except requests.exceptions.Timeout:
        raise TranscriptionError("ElevenLabs could not transcribe the audio: no answer in 15 minutes") from None
    except requests.exceptions.RequestException as error:
        raise TranscriptionError(f"ElevenLabs could not transcribe the audio: {error}") from error

    if response.status_code != 200:
        raise TranscriptionError(
            f"ElevenLabs could not transcribe the audio: HTTP {response.status_code}: {response.text[:300]}"
        )

    result = response.json()
    text = result.get('text') or ''
    word_count = len(result.get('words', []))
    logger.success(f"Transcription complete: {len(text)} chars, {word_count} words with timestamps")
    return text, result


# =============================================================================
# LOCAL WHISPER TRANSCRIPTION
# =============================================================================

def transcribe_whisper_local(audio_path: Path) -> str:
    """Transcribe audio using local Whisper model (GPU). Raises TranscriptionError if it cannot."""

    try:
        import whisper
    except ImportError as error:
        raise TranscriptionError(
            "Whisper could not transcribe the audio: it is not installed (pip install openai-whisper)"
        ) from error

    try:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
        model = whisper.load_model(WHISPER_MODEL)

        logger.info("Transcribing with Whisper (GPU)...")

        result = model.transcribe(
            str(audio_path),
            language=TRANSCRIPTION_LANGUAGE,
            verbose=False
        )
    except Exception as error:
        raise TranscriptionError(f"Whisper could not transcribe the audio: {error}") from error

    text = result.get('text') or ''
    logger.success(f"Transcription complete: {len(text)} characters")

    return text


# =============================================================================
# MAIN TRANSCRIPTION FUNCTION
# =============================================================================

def read_transcription(output: VideoOutput) -> Optional[str]:
    """
    The saved transcription of a video, without the header of its file.

    Returns:
        The text (empty for a video without speech), or None when the video
        has no transcription saved
    """

    if not output.transcription_file.exists():
        return None

    lines = output.transcription_file.read_text(encoding='utf-8').split('\n')
    for i, line in enumerate(lines):
        # The header ends with a line of '='
        if line.startswith('='):
            return '\n'.join(lines[i + 1:]).strip()
    return '\n'.join(lines).strip()


def transcribe_video(video_path: Path, output: VideoOutput) -> str:
    """
    Transcribe a video file, or read the transcription it already has.

    A video without an audio track, or where nobody speaks, has an empty
    transcription. That is a result like any other and is saved, so the video
    is not sent (and paid for) again.

    Args:
        video_path: Path to the video file
        output: VideoOutput instance for saving results

    Returns:
        Transcription text, empty when the video has no speech

    Raises:
        TranscriptionError: the audio could not be extracted or transcribed
    """

    logger.info(f"Transcribing: {video_path.name}")

    # Check if already transcribed
    saved = read_transcription(output)
    if saved is not None:
        logger.info("Loading cached transcription...")
        return saved

    if not has_audio_track(video_path):
        logger.info("The video has no audio track: its transcription is empty")
        save_transcription('', output, video_path)
        return ''

    audio_path = extract_audio(video_path)

    try:
        # Transcribe
        full_response = None
        if USE_LOCAL_WHISPER:
            text = transcribe_whisper_local(audio_path)
        else:
            text, full_response = transcribe_elevenlabs(audio_path)

        text = text.strip()
        if not text:
            logger.info("Nobody speaks in the video: its transcription is empty")

        # Save transcription (text file and JSON with timestamps if available)
        save_transcription(text, output, video_path, full_response)

        return text

    finally:
        # Clean up temp audio
        if audio_path.exists():
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

# How much of a transcription the AI reads to write ideas, titles and
# descriptions: about half an hour of speech. Ideas used to read 3,000, the
# first three minutes of the video.
TRANSCRIPT_CHARS_FOR_AI = 30000


def get_transcription_summary(text: str, max_length: int = TRANSCRIPT_CHARS_FOR_AI) -> str:
    """
    The part of the transcription sent to the AI that writes the thumbnail ideas.

    Returns the first max_length characters, trying to end at a sentence.
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

"""
YouTube Thumbnail Generator - Transcript Processing Module
===========================================================
Processes ElevenLabs transcription responses to group words into segments
based on speaker changes and pauses. Provides timestamps for video descriptions.
"""

from datetime import timedelta
from typing import Optional
from pathlib import Path
import json
import re
import logging

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def group_words_into_segments(
    words: list[dict],
    pause_threshold: float = 1.5,
) -> list[dict]:
    """
    Group words into segments based on speaker changes and pauses.

    Args:
        words: List of word objects from ElevenLabs response
        pause_threshold: Seconds of pause to consider a new segment

    Returns:
        List of segments with start, end, speaker, and text
    """
    if not words:
        return []

    segments = []
    current_segment = {
        "start": words[0].get("start", 0),
        "end": words[0].get("end", 0),
        "speaker": words[0].get("speaker_id", "unknown"),
        "words": [words[0].get("text", "")]
    }
    last_end = words[0].get("end", 0)

    for word in words[1:]:
        word_start = word.get("start", 0)
        word_end = word.get("end", 0)
        word_text = word.get("text", "")
        word_speaker = word.get("speaker_id", "unknown")

        # Skip spacing and audio events for segment detection
        if word.get("type") == "spacing":
            continue

        # Detect segment break by pause or speaker change
        pause_detected = (word_start - last_end) > pause_threshold
        speaker_changed = word_speaker != current_segment["speaker"]

        if pause_detected or speaker_changed:
            # Save current segment
            current_segment["end"] = last_end
            current_segment["text"] = " ".join(current_segment["words"])
            segments.append(current_segment)

            # Start new segment
            current_segment = {
                "start": word_start,
                "end": word_end,
                "speaker": word_speaker,
                "words": [word_text]
            }
        else:
            current_segment["words"].append(word_text)

        last_end = word_end

    # Add final segment
    current_segment["end"] = last_end
    current_segment["text"] = " ".join(current_segment["words"])
    segments.append(current_segment)

    return segments


def group_words_into_segments_improved(
    words: list[dict],
    pause_threshold: float = 0.8,
    max_segment_duration: float = 30.0,
    min_segment_duration: float = 15.0,
    split_on_punctuation: bool = True,
) -> list[dict]:
    """
    Improved algorithm: groups words into segments based on:
    1. Speaker changes (always split)
    2. Pauses > pause_threshold
    3. Sentence-ending punctuation (. ? !) after min_segment_duration
    4. Maximum segment duration

    Args:
        words: List of word objects from ElevenLabs response
        pause_threshold: Seconds of pause to consider a new segment
        max_segment_duration: Force split if segment exceeds this duration
        min_segment_duration: Only split on punctuation after this duration
        split_on_punctuation: Enable splitting on sentence-ending punctuation

    Returns:
        List of segments with start, end, speaker, and text
    """
    if not words:
        return []

    # Filter out spacing tokens
    filtered_words = [w for w in words if w.get("type") != "spacing"]
    if not filtered_words:
        return []

    segments = []
    current_segment = {
        "start": filtered_words[0].get("start", 0),
        "end": filtered_words[0].get("end", 0),
        "speaker": filtered_words[0].get("speaker_id", "unknown"),
        "words": [filtered_words[0].get("text", "")]
    }
    last_end = filtered_words[0].get("end", 0)

    # Regex for sentence-ending punctuation
    sentence_end_pattern = re.compile(r'[.!?]$')

    for word in filtered_words[1:]:
        word_start = word.get("start", 0)
        word_end = word.get("end", 0)
        word_text = word.get("text", "")
        word_speaker = word.get("speaker_id", "unknown")

        # Check split conditions
        pause_detected = (word_start - last_end) > pause_threshold
        speaker_changed = word_speaker != current_segment["speaker"]
        segment_duration = word_start - current_segment["start"]
        segment_too_long = segment_duration > max_segment_duration

        # Check if previous word ended a sentence
        prev_word = current_segment["words"][-1] if current_segment["words"] else ""
        sentence_ended = split_on_punctuation and sentence_end_pattern.search(prev_word)

        # Split conditions
        should_split = speaker_changed or pause_detected or (sentence_ended and segment_too_long)

        # Also split at natural sentence breaks after min duration
        if sentence_ended and segment_duration > min_segment_duration:
            should_split = True

        if should_split:
            # Save current segment
            current_segment["end"] = last_end
            current_segment["text"] = " ".join(current_segment["words"])
            segments.append(current_segment)

            # Start new segment
            current_segment = {
                "start": word_start,
                "end": word_end,
                "speaker": word_speaker,
                "words": [word_text]
            }
        else:
            current_segment["words"].append(word_text)

        last_end = word_end

    # Add final segment
    current_segment["end"] = last_end
    current_segment["text"] = " ".join(current_segment["words"])
    segments.append(current_segment)

    return segments


def format_segments_for_display(
    segments: list[dict],
    show_speaker: bool = True,
    show_end_time: bool = False
) -> str:
    """
    Format segments into readable text with timestamps.

    Args:
        segments: List of grouped segments
        show_speaker: Include speaker identifier
        show_end_time: Include end time in addition to start

    Returns:
        Formatted transcript string
    """
    lines = []

    for seg in segments:
        timestamp = format_timestamp(seg["start"])
        if show_end_time:
            timestamp += f" - {format_timestamp(seg['end'])}"

        if show_speaker and seg.get("speaker") and seg["speaker"] != "unknown":
            speaker = seg["speaker"].replace("speaker_", "Speaker ")
            line = f"[{timestamp}] {speaker}: {seg['text']}"
        else:
            line = f"[{timestamp}] {seg['text']}"

        lines.append(line)

    return "\n".join(lines)


def format_segments_for_ai(
    segments: list[dict],
    include_speaker: bool = False
) -> str:
    """
    Format segments optimized for AI description generation.
    Focuses on timestamps and content for identifying key moments.

    Args:
        segments: List of grouped segments
        include_speaker: Include speaker info (useful for interviews/podcasts)

    Returns:
        Formatted string for AI prompt
    """
    lines = []

    for i, seg in enumerate(segments):
        timestamp = format_timestamp(seg["start"])
        text = seg["text"].strip()

        # Truncate very long segments for AI context efficiency
        if len(text) > 300:
            text = text[:300] + "..."

        if include_speaker and seg.get("speaker") and seg["speaker"] != "unknown":
            speaker = seg["speaker"].replace("speaker_", "Speaker ")
            line = f"[{timestamp}] ({speaker}) {text}"
        else:
            line = f"[{timestamp}] {text}"

        lines.append(line)

    return "\n".join(lines)


def process_elevenlabs_response(
    response: dict,
    pause_threshold: float = 1.5,
    show_speaker: bool = True,
    show_end_time: bool = False
) -> str:
    """
    Process complete ElevenLabs API response into formatted transcript.

    Args:
        response: JSON response from ElevenLabs API
        pause_threshold: Seconds of pause to separate segments
        show_speaker: Show speaker in output
        show_end_time: Show end time

    Returns:
        Formatted transcript string
    """
    words = response.get("words", [])
    segments = group_words_into_segments(words, pause_threshold)
    return format_segments_for_display(segments, show_speaker, show_end_time)


def get_segments_from_json_file(
    json_path: Path,
    pause_threshold: float = 1.5
) -> list[dict]:
    """
    Load and process segments from saved JSON file.

    Args:
        json_path: Path to the transcription JSON file
        pause_threshold: Seconds of pause to separate segments

    Returns:
        List of processed segments
    """
    if not json_path.exists():
        return []

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        words = data.get("words", [])
        return group_words_into_segments(words, pause_threshold)
    except Exception:
        return []


def get_formatted_transcript_with_timestamps(
    json_path: Path,
    pause_threshold: float = 0.8,
    for_ai: bool = False,
    include_speaker: bool = False
) -> Optional[str]:
    """
    Main function to get formatted transcript from saved JSON.
    Uses cached segments if available, otherwise generates and caches them.

    Args:
        json_path: Path to transcription JSON file
        pause_threshold: Seconds of pause for segment breaks
        for_ai: Use AI-optimized format (shorter, focused on timestamps)
        include_speaker: Include speaker identification

    Returns:
        Formatted transcript string or None if file doesn't exist
    """
    # Try to use cached segments first (much faster)
    cache = get_or_create_segments_cache(json_path, pause_threshold)
    if cache and cache.get("segments"):
        segments = cache["segments"]
    else:
        # Fallback to old method if cache fails
        segments = get_segments_from_json_file(json_path, pause_threshold)

    if not segments:
        return None

    if for_ai:
        return format_segments_for_ai(segments, include_speaker)
    else:
        return format_segments_for_display(segments, include_speaker)


def has_multiple_speakers(json_path: Path) -> bool:
    """
    Check if the transcription has multiple speakers detected.

    Args:
        json_path: Path to transcription JSON file

    Returns:
        True if multiple speakers were detected
    """
    if not json_path.exists():
        return False

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        words = data.get("words", [])
        speakers = set()

        for word in words:
            speaker = word.get("speaker_id")
            if speaker:
                speakers.add(speaker)

        return len(speakers) > 1
    except Exception:
        return False


def get_video_duration_from_transcript(json_path: Path) -> Optional[float]:
    """
    Get video duration from the last word's end time.

    Args:
        json_path: Path to transcription JSON file

    Returns:
        Duration in seconds or None
    """
    if not json_path.exists():
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        words = data.get("words", [])
        if words:
            return words[-1].get("end", 0)
        return None
    except Exception:
        return None


# =============================================================================
# SEGMENT CACHE FUNCTIONS
# =============================================================================

def get_segments_cache_path(json_path: Path) -> Path:
    """Get the path for the segments cache file."""
    return json_path.with_name("transcription_segments.json")


def generate_segments_cache(json_path: Path, pause_threshold: float = 0.8) -> Optional[dict]:
    """
    Generate segments cache from transcription JSON using improved algorithm.

    Args:
        json_path: Path to transcription JSON file
        pause_threshold: Seconds of pause to consider a new segment

    Returns:
        Cache dict with segments and metadata, or None on error
    """
    if not json_path.exists():
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        words = data.get("words", [])
        if not words:
            return None

        # Process segments with improved algorithm
        raw_segments = group_words_into_segments_improved(
            words,
            pause_threshold=pause_threshold,
            max_segment_duration=30.0,
            min_segment_duration=15.0,
            split_on_punctuation=True
        )

        # Clean up segments (remove internal 'words' list, add timestamp string)
        segments = []
        for seg in raw_segments:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "text": seg["text"],
                "timestamp": format_timestamp(seg["start"])
            })

        # Detect unique speakers
        speakers = list(set(
            seg["speaker"] for seg in segments
            if seg["speaker"] != "unknown"
        ))

        return {
            "segments": segments,
            "metadata": {
                "pause_threshold": pause_threshold,
                "total_segments": len(segments),
                "speakers": sorted(speakers),
                "duration_seconds": segments[-1]["end"] if segments else 0,
                "language": data.get("language_code", "unknown")
            }
        }
    except Exception as e:
        logger.error(f"Error generating segments cache: {e}")
        return None


def save_segments_cache(json_path: Path, pause_threshold: float = 0.8) -> Optional[Path]:
    """
    Generate and save segments cache to file.

    Args:
        json_path: Path to transcription JSON file
        pause_threshold: Seconds of pause to consider a new segment

    Returns:
        Path to saved cache file, or None on error
    """
    cache = generate_segments_cache(json_path, pause_threshold)
    if not cache:
        return None

    cache_path = get_segments_cache_path(json_path)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        logger.info(f"Segments cache saved: {cache_path} ({cache['metadata']['total_segments']} segments)")
        return cache_path
    except Exception as e:
        logger.error(f"Error saving segments cache: {e}")
        return None


def load_segments_cache(json_path: Path) -> Optional[dict]:
    """
    Load segments from cache file if it exists.

    Args:
        json_path: Path to transcription JSON file

    Returns:
        Cache dict with segments and metadata, or None if not found
    """
    cache_path = get_segments_cache_path(json_path)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading segments cache: {e}")
        return None


def get_or_create_segments_cache(json_path: Path, pause_threshold: float = 0.8) -> Optional[dict]:
    """
    Get segments from cache, or create cache if it doesn't exist.

    Args:
        json_path: Path to transcription JSON file
        pause_threshold: Seconds of pause to consider a new segment

    Returns:
        Cache dict with segments and metadata
    """
    # Try to load from cache first
    cache = load_segments_cache(json_path)
    if cache:
        return cache

    # Generate and save cache
    cache = generate_segments_cache(json_path, pause_threshold)
    if cache:
        save_segments_cache(json_path, pause_threshold)
    return cache


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    # Test with example data
    example_response = {
        "language_code": "es",
        "language_probability": 0.98,
        "text": "Hola que tal como estas Muy bien gracias y tu Pues aqui andamos trabajando en el proyecto",
        "words": [
            {"text": "Hola", "start": 0.0, "end": 0.3, "type": "word", "speaker_id": "speaker_0"},
            {"text": "que", "start": 0.32, "end": 0.45, "type": "word", "speaker_id": "speaker_0"},
            {"text": "tal", "start": 0.47, "end": 0.6, "type": "word", "speaker_id": "speaker_0"},
            {"text": "como", "start": 0.62, "end": 0.8, "type": "word", "speaker_id": "speaker_0"},
            {"text": "estas", "start": 0.82, "end": 1.1, "type": "word", "speaker_id": "speaker_0"},
            # 2 second pause + speaker change
            {"text": "Muy", "start": 3.2, "end": 3.4, "type": "word", "speaker_id": "speaker_1"},
            {"text": "bien", "start": 3.42, "end": 3.7, "type": "word", "speaker_id": "speaker_1"},
            {"text": "gracias", "start": 3.72, "end": 4.1, "type": "word", "speaker_id": "speaker_1"},
            {"text": "y", "start": 4.12, "end": 4.2, "type": "word", "speaker_id": "speaker_1"},
            {"text": "tu", "start": 4.22, "end": 4.5, "type": "word", "speaker_id": "speaker_1"},
            # Pause + speaker change
            {"text": "Pues", "start": 6.0, "end": 6.3, "type": "word", "speaker_id": "speaker_0"},
            {"text": "aqui", "start": 6.32, "end": 6.5, "type": "word", "speaker_id": "speaker_0"},
            {"text": "andamos", "start": 6.52, "end": 6.9, "type": "word", "speaker_id": "speaker_0"},
            {"text": "trabajando", "start": 6.92, "end": 7.4, "type": "word", "speaker_id": "speaker_0"},
            {"text": "en", "start": 7.42, "end": 7.5, "type": "word", "speaker_id": "speaker_0"},
            {"text": "el", "start": 7.52, "end": 7.6, "type": "word", "speaker_id": "speaker_0"},
            {"text": "proyecto", "start": 7.62, "end": 8.1, "type": "word", "speaker_id": "speaker_0"},
        ]
    }

    print("=" * 60)
    print("GROUPED TRANSCRIPT (with speakers)")
    print("=" * 60)
    result = process_elevenlabs_response(
        example_response,
        pause_threshold=1.5,
        show_speaker=True
    )
    print(result)

    print("\n" + "=" * 60)
    print("AI-OPTIMIZED FORMAT")
    print("=" * 60)
    segments = group_words_into_segments(example_response["words"], 1.5)
    ai_format = format_segments_for_ai(segments, include_speaker=True)
    print(ai_format)

"""
Workers module for arq job queue.

This module provides background workers for:
- Video analysis (scene detection, face extraction, clustering, transcription)
- Thumbnail generation (prompts, image generation)
"""

from workers.tasks import analyze_video, run_generation

__all__ = [
    "analyze_video",
    "run_generation",
]

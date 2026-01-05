"""
YouTube Thumbnail Generator
===========================
Automated pipeline for generating YouTube thumbnails from videos.

Modules:
    - scene_detection: Detect scenes and extract representative frames
    - face_extraction: Detect and select best frames with target face
    - transcription: Transcribe video audio
    - prompt_generation: Generate thumbnail prompts using LLM
    - image_generation: Generate thumbnail images using AI

Usage:
    python main.py                  # Process all videos
    python main.py --single video.mp4  # Process single video
"""

__version__ = "1.0.0"
__author__ = "Jordi"

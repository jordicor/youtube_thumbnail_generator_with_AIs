#!/usr/bin/env python3
"""
YouTube Thumbnail Generator - Setup Checker
============================================
Verifies that all dependencies are correctly installed.
"""

import sys
from pathlib import Path

def check_import(module_name, package_name=None):
    """Try to import a module and report status"""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"  [OK] {package_name}")
        return True
    except ImportError as e:
        print(f"  [X]  {package_name} - Not installed")
        return False

def check_ffmpeg():
    """Check if FFmpeg is available"""
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            print("  [OK] FFmpeg")
            return True
    except Exception:
        pass
    print("  [X]  FFmpeg - Not found in PATH")
    return False

def check_cuda():
    """Check CUDA availability for GPU acceleration"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  [OK] CUDA - {gpu_name}")
            return True
        else:
            print("  [!]  CUDA - Not available (will use CPU)")
            return False
    except ImportError:
        # Check via onnxruntime
        try:
            import onnxruntime
            providers = onnxruntime.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                print("  [OK] CUDA (via ONNX Runtime)")
                return True
            else:
                print("  [!]  CUDA - Not available (will use CPU)")
                return False
        except ImportError:
            print("  [?]  CUDA - Cannot verify (torch/onnxruntime not installed)")
            return False

def check_api_keys():
    """Check if API keys are configured"""
    import os

    keys = {
        'GEMINI_API_KEY': 'Gemini (Nano Banana Pro)',
        'ANTHROPIC_API_KEY': 'Claude (Anthropic)',
        'OPENAI_API_KEY': 'OpenAI (GPT/DALL-E)',
        'ELEVENLABS_API_KEY': 'ElevenLabs (Transcription)',
    }

    print("\nAPI Keys:")

    # Also check config.py
    try:
        from config import GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY
        config_keys = {
            'GEMINI_API_KEY': GEMINI_API_KEY,
            'ANTHROPIC_API_KEY': ANTHROPIC_API_KEY,
            'OPENAI_API_KEY': OPENAI_API_KEY,
            'ELEVENLABS_API_KEY': ELEVENLABS_API_KEY,
        }
    except ImportError:
        config_keys = {}

    configured = 0
    for key, name in keys.items():
        env_val = os.getenv(key, '')
        config_val = config_keys.get(key, '')

        if env_val or config_val:
            print(f"  [OK] {name}")
            configured += 1
        else:
            print(f"  [!]  {name} - Not configured")

    return configured > 0

def check_reference_face():
    """Info about the clustering-based workflow (reference_face no longer needed)"""
    print("  [OK] Face detection: Automatic clustering (no reference image needed)")
    print("       Faces are automatically grouped by person from video frames")
    return True

def main():
    print("=" * 60)
    print("YouTube Thumbnail Generator - Setup Check")
    print("=" * 60)

    all_ok = True

    # Core dependencies
    print("\nCore Dependencies:")
    all_ok &= check_import('cv2', 'OpenCV')
    all_ok &= check_import('numpy', 'NumPy')
    all_ok &= check_import('PIL', 'Pillow')
    all_ok &= check_import('requests', 'Requests')

    # Scene detection
    print("\nScene Detection:")
    all_ok &= check_import('scenedetect', 'PySceneDetect')

    # Face detection
    print("\nFace Detection:")
    all_ok &= check_import('insightface', 'InsightFace')

    # Transcription
    print("\nTranscription:")
    check_import('whisper', 'OpenAI Whisper (local)')

    # LLM APIs
    print("\nLLM APIs:")
    check_import('anthropic', 'Anthropic (Claude)')
    check_import('openai', 'OpenAI')
    check_import('google.generativeai', 'Google Generative AI')

    # Image generation
    print("\nImage Generation:")
    check_import('replicate', 'Replicate')

    # System tools
    print("\nSystem Tools:")
    all_ok &= check_ffmpeg()

    # GPU
    print("\nGPU Acceleration:")
    check_cuda()

    # API Keys
    check_api_keys()

    # Reference face
    print("\nConfiguration:")
    check_reference_face()

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("All core dependencies OK!")
        print("\nTo start processing videos:")
        print("  python main.py --dry-run    # Preview what will be processed")
        print("  python main.py              # Process all videos")
    else:
        print("Some dependencies are missing.")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements.txt")
    print("=" * 60)

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

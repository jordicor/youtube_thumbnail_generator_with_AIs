"""
CUDA Setup Module
=================
Import this module BEFORE any ONNX/InsightFace imports to ensure CUDA works correctly.

Usage:
    import cuda_setup  # This line must be FIRST
    import onnxruntime
    from insightface.app import FaceAnalysis
"""

import os
import sys


def setup_cuda_paths():
    """Add NVIDIA CUDA library paths to system PATH"""
    nvidia_base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")

    cuda_subdirs = [
        "cublas",
        "cudnn",
        "cuda_runtime",
        "cuda_nvrtc",
        "cufft",
        "cusparse",
        "curand",
        "cusolver",
        "nvjitlink",
    ]

    paths_added = []
    for subdir in cuda_subdirs:
        bin_path = os.path.join(nvidia_base, subdir, "bin")
        if os.path.exists(bin_path):
            os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
            paths_added.append(subdir)

    return paths_added


# Auto-setup on import
_cuda_paths = setup_cuda_paths()

if _cuda_paths:
    print(f"[cuda_setup] Added CUDA paths: {', '.join(_cuda_paths)}")

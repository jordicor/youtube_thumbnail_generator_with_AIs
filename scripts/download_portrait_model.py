"""Install the local portrait models (face landmarks, hair/skin/clothes segmentation) with checksum verification."""
import hashlib
import os
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.portrait_quality import MODEL_PATH, MODEL_SHA256, SEGMENTER_PATH, SEGMENTER_SHA256

MODELS = (
    (MODEL_PATH, MODEL_SHA256,
     "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
    (SEGMENTER_PATH, SEGMENTER_SHA256,
     "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/"
     "selfie_multiclass_256x256.tflite"),
)

if __name__ == "__main__":
    for path, checksum, url in MODELS:
        if path.exists():
            data = path.read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise RuntimeError(f"{path.name} checksum mismatch.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        print(path)

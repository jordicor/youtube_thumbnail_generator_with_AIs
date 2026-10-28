"""Download the body pose and segmentation models after verifying their checksums."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.body_references import download_model

if __name__ == '__main__':
    for path in download_model():
        print(path)

#!/usr/bin/env python3
"""
Download face swap models (Phase 2 — local face swap pipeline).
================================================================
Fetches the ONNX/PT model files used by the post-processing face swap
service:

  - models/inswapper/inswapper_128.onnx      (InsightFace InSwapper, ~554 MB)
  - models/gfpgan/GFPGANv1.4.pth             (Tencent GFPGAN restoration, ~333 MB)
  - models/occluder/xseg_{1,2,3}.onnx        (XSeg occluders, ~70 MB each)

Each model is downloaded only if not already present. SHA256 is verified
against expected hashes when known, so partial downloads fail fast.

Usage:
    python scripts/download_face_swap_models.py
    python scripts/download_face_swap_models.py --force      # re-download even if present
    python scripts/download_face_swap_models.py --inswapper-only
    python scripts/download_face_swap_models.py --gfpgan-only

Notes:
    - inswapper_128 was withdrawn from the official InsightFace repo. The
      URL below points to a widely-mirrored copy on HuggingFace. For a
      research/personal-use project (restoring an existing creator's channel
      backup) this is the established way to obtain the model. If you'd
      rather use ReSwapper or HyperSwap, swap the URL and the consumer
      service accepts ONNX models with matching I/O.
    - GFPGAN v1.4 is the official Tencent release on GitHub.
    - The XSeg occluders (DeepFaceLab's, exported to ONNX) come from
      FaceFusion's model releases on GitHub. They mark what covers a face
      so the swap is pasted only where the face shows.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"


@dataclass(frozen=True)
class ModelSpec:
    """A model file to download."""
    name: str
    target: Path
    url: str
    expected_sha256: str | None  # None disables hash check
    description: str
    approx_size_mb: int


MODELS = {
    "inswapper": ModelSpec(
        name="InSwapper 128",
        target=MODELS_DIR / "inswapper" / "inswapper_128.onnx",
        url="https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        expected_sha256="e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
        description="InsightFace InSwapper face swap model (128x128 output)",
        approx_size_mb=554,
    ),
    "gfpgan": ModelSpec(
        name="GFPGAN v1.4",
        target=MODELS_DIR / "gfpgan" / "GFPGANv1.4.pth",
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        expected_sha256="e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad",
        description="Tencent GFPGAN v1.4 face restoration model",
        approx_size_mb=333,
    ),
    **{
        f"xseg_{n}": ModelSpec(
            name=f"XSeg occluder {n}",
            target=MODELS_DIR / "occluder" / f"xseg_{n}.onnx",
            url=f"https://github.com/facefusion/facefusion-assets/releases/download/{release}/xseg_{n}.onnx",
            expected_sha256=sha256,
            description="XSeg face occluder: what covers a face stays in front of the swap",
            approx_size_mb=70,
        )
        for n, release, sha256 in (
            (1, "models-3.1.0", "c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3"),
            (2, "models-3.1.0", "cd9a0879eaf43841d765472cf1f8c330dbf9dcb03da0eace93e95f3bcc399042"),
            (3, "models-3.2.0", "48ccd7e8541e159a5a754ec9e62df2f12065f7df8f9af842c1750342c6533559"),
        )
    },
}


def _sha256_of(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _human_mb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 * 1024):.1f} MB"


def _download(url: str, target: Path) -> None:
    """Stream-download with a single-line progress indicator."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".part")

    # Clean up any stale partial from a previous failed attempt
    if tmp_path.exists():
        tmp_path.unlink()

    req = urllib.request.Request(
        url,
        headers={
            # Some mirrors block default urllib UA
            "User-Agent": "youtube-thumbnail-generator/face-swap-setup",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310
        total = int(response.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        last_print = 0
        with open(tmp_path, "wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                # Print progress at most ~50 times to keep stderr clean
                if total and downloaded - last_print >= max(1, total // 50):
                    pct = downloaded * 100 / total
                    sys.stderr.write(
                        f"\r  {_human_mb(downloaded)} / {_human_mb(total)}  ({pct:5.1f}%)"
                    )
                    sys.stderr.flush()
                    last_print = downloaded
        if total:
            sys.stderr.write("\n")

    tmp_path.replace(target)


def _ensure_model(spec: ModelSpec, force: bool) -> bool:
    """Download/verify a single model. Returns True if usable, False on failure."""
    target = spec.target
    print(f"\n[{spec.name}] -> {target}")
    print(f"  {spec.description}")

    if target.exists() and not force:
        size = target.stat().st_size
        print(f"  Already present ({_human_mb(size)}). Skipping download.")
        if spec.expected_sha256:
            print("  Verifying SHA256...")
            actual = _sha256_of(target)
            if actual != spec.expected_sha256:
                print(
                    f"  HASH MISMATCH:\n    expected {spec.expected_sha256}\n    got      {actual}",
                    file=sys.stderr,
                )
                print(
                    "  The local file is corrupted or a different build. Re-run with --force.",
                    file=sys.stderr,
                )
                return False
            print("  SHA256 OK.")
        return True

    print(f"  Downloading {spec.approx_size_mb} MB from:\n    {spec.url}")
    try:
        _download(spec.url, target)
    except urllib.error.HTTPError as e:
        print(f"  Download failed: HTTP {e.code} {e.reason}", file=sys.stderr)
        return False
    except urllib.error.URLError as e:
        print(f"  Download failed: {e.reason}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  Download failed: {e}", file=sys.stderr)
        return False

    if spec.expected_sha256:
        print("  Verifying SHA256...")
        actual = _sha256_of(target)
        if actual != spec.expected_sha256:
            print(
                f"  HASH MISMATCH:\n    expected {spec.expected_sha256}\n    got      {actual}",
                file=sys.stderr,
            )
            print(
                "  The downloaded file is corrupted or the mirror served a different build.\n"
                "  Delete the file and retry, or update the expected hash if the upstream changed.",
                file=sys.stderr,
            )
            return False
        print("  SHA256 OK.")

    print(f"  Saved ({_human_mb(target.stat().st_size)}).")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download models even if they already exist locally",
    )
    parser.add_argument(
        "--inswapper-only",
        action="store_true",
        help="Only download the InSwapper face swap model",
    )
    parser.add_argument(
        "--gfpgan-only",
        action="store_true",
        help="Only download the GFPGAN face restoration model",
    )
    args = parser.parse_args()

    if args.inswapper_only and args.gfpgan_only:
        print("--inswapper-only and --gfpgan-only are mutually exclusive", file=sys.stderr)
        return 2

    selected = list(MODELS.keys())
    if args.inswapper_only:
        selected = ["inswapper"]
    elif args.gfpgan_only:
        selected = ["gfpgan"]

    print(f"Models directory: {MODELS_DIR}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    failures = []
    for key in selected:
        ok = _ensure_model(MODELS[key], force=args.force)
        if not ok:
            failures.append(key)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("All requested models are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Download the INT16 model bundle from a GitHub release.

Downloads `int16_bundle_v1.0.0.pt` from the project's
GitHub release assets and places it in the `models/` directory.

The bundle is the only file distributed with this project. Microsoft's
upstream DCVC-RT checkpoints (cvpr2025_image.pth.tar / cvpr2025_video.pth.tar)
must be obtained separately from https://github.com/microsoft/DCVC.

Usage:
    python scripts/download_models.py

    # Override release URL (or set DCVC_BUNDLE_URL env var):
    python scripts/download_models.py --url https://github.com/.../releases/download/v1.0.0/int16_bundle_v1.0.0.pt

    # Force re-download even if file already exists:
    python scripts/download_models.py --force

    # Verify against a known SHA-256 hash:
    python scripts/download_models.py --sha256 <hex-digest>
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import get, load_config

# ---------------------------------------------------------------------------
# Configuration — update BUNDLE_URL after creating the GitHub release.
# Set in config/config.yaml under release.bundle_url, or via DCVC_BUNDLE_URL env var.
# ---------------------------------------------------------------------------
_cfg = load_config()
BUNDLE_URL = (
    os.environ.get("DCVC_BUNDLE_URL")
    or get(_cfg, "release", "bundle_url")
    or (
        "https://github.com/mathew-felix/deterministic-neural-video-codec"
        "/releases/download/v1.0.0/int16_bundle_v1.0.0.pt"
    )
)

BUNDLE_FILENAME = "int16_bundle_v1.0.0.pt"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "models"

MICROSOFT_DCVC_README = "https://github.com/microsoft/DCVC/blob/main/DCVC-family/README.md"


# ---------------------------------------------------------------------------
# Progress-aware download
# ---------------------------------------------------------------------------

class _TqdmProgressHook:
    """urllib reporthook that updates a tqdm bar."""

    def __init__(self, bar: "tqdm") -> None:
        self._bar = bar
        self._last_blocks = 0

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0 and self._bar.total != total_size:
            self._bar.total = total_size
        transferred = (block_num - self._last_blocks) * block_size
        self._bar.update(max(transferred, 0))
        self._last_blocks = block_num


def _download_with_progress(url: str, dest: Path) -> None:
    """Download *url* to *dest*, showing a progress bar when tqdm is available."""
    print(f"Downloading: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if _TQDM_AVAILABLE:
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=dest.name,
        ) as bar:
            urllib.request.urlretrieve(url, dest, reporthook=_TqdmProgressHook(bar))
    else:
        urllib.request.urlretrieve(url, dest)
        print(f"  saved to {dest}")


# ---------------------------------------------------------------------------
# SHA-256 verification
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(path: Path, expected: str) -> None:
    """Raise ValueError if the SHA-256 of *path* does not match *expected*."""
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for {path.name}:\n"
            f"  expected: {expected.lower()}\n"
            f"  actual:   {actual}"
        )
    print(f"  SHA-256 OK: {actual[:16]}...")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def _print_upstream_instructions() -> None:
    """Print instructions for obtaining Microsoft's FP16 checkpoints."""
    print()
    print("To build the INT16 bundle from scratch, you also need the upstream")
    print("DCVC-RT FP16 checkpoints from Microsoft:")
    print(f"  {MICROSOFT_DCVC_README}")
    print()
    print("Then run the export pipeline:")
    print("  python scripts/freeze_entropy_cdfs.py --model_path_i <image.pth.tar> \\")
    print("      --model_path_p <video.pth.tar> --output_path models/frozen_entropy_state.pt")
    print("  python scripts/export_int16_bundle.py --model_path_i <image.pth.tar> \\")
    print("      --model_path_p <video.pth.tar> \\")
    print("      --frozen_entropy_path models/frozen_entropy_state.pt \\")
    print("      --output_path models/int16_bundle_v1.0.0.pt")
    print()
    print("See docs/model_setup.md for the full step-by-step guide.")


def download(
    url: str,
    output_dir: Path,
    sha256: str | None,
    force: bool,
) -> Path:
    """Download the INT16 bundle to *output_dir*.

    Args:
        url: Direct download URL for the bundle file.
        output_dir: Directory where the bundle will be placed.
        sha256: Optional expected SHA-256 hex digest for verification.
        force: If True, re-download even if the file already exists.

    Returns:
        Path to the downloaded file.

    Raises:
        SystemExit: If the URL is a placeholder or the download fails.
        ValueError: If SHA-256 verification fails.
    """
    dest = output_dir / BUNDLE_FILENAME

    if not url or "github.com/.../releases/download/..." in url:
        # Placeholder or missing URL.
        print("ERROR: The bundle download URL has not been configured yet.")
        print()
        print("Set the DCVC_BUNDLE_URL environment variable or pass --url to point")
        print("to the actual GitHub release asset once you have created the release:")
        print()
        print("  Windows (PowerShell):")
        print("    $env:DCVC_BUNDLE_URL = 'https://github.com/.../releases/download/...'")
        print("    python scripts/download_models.py")
        print()
        print("  Linux / macOS:")
        print("    DCVC_BUNDLE_URL=https://... python scripts/download_models.py")
        _print_upstream_instructions()
        sys.exit(1)

    if dest.exists() and not force:
        print(f"Bundle already present: {dest}")
        if sha256:
            _verify_sha256(dest, sha256)
        else:
            digest = _sha256(dest)
            print(f"  SHA-256: {digest}")
            print("  Pass --sha256 <digest> to verify integrity.")
        return dest

    try:
        _download_with_progress(url, dest)
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        raise SystemExit(f"Download failed: {exc}") from exc

    print(f"  saved to {dest}")

    if sha256:
        _verify_sha256(dest, sha256)
    else:
        digest = _sha256(dest)
        print(f"  SHA-256: {digest}")
        print("  Record this digest to verify the bundle on other machines.")

    return dest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download the INT16 model bundle from a GitHub release."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=BUNDLE_URL,
        help=(
            "Direct URL to the INT16 bundle release asset. "
            "Defaults to DCVC_BUNDLE_URL env var or the built-in release URL."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the bundle will be saved (default: models/).",
    )
    parser.add_argument(
        "--sha256",
        type=str,
        default=None,
        help="Expected SHA-256 hex digest. Verification is skipped if not provided.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the bundle already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    dest = download(
        url=args.url,
        output_dir=args.output_dir,
        sha256=args.sha256,
        force=args.force,
    )
    print(f"\nBundle ready: {dest}")
    print("\nVerify the runtime loads it correctly:")
    print("  python scripts/check_setup.py --bundle_path models/int16_bundle_v1.0.0.pt")
    print("  python encode_mp4_to_bin.py --input_mp4 video.mp4 --frames 2 --check_only")


if __name__ == "__main__":
    main()

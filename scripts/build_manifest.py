#!/usr/bin/env python3
"""Build a calibration manifest JSON from a folder of video clips.

Scans a directory for .mp4 (and .mkv / .mov) files, infers a content-type
weight from the filename, and writes a manifest ready for
`scripts/calibrate_int16_bundle.py`.

Usage:
    # Scan calibrate_videos/ and write the default manifest:
    python scripts/build_manifest.py

    # Custom folder and output path:
    python scripts/build_manifest.py \
        --input_dir /path/to/clips \
        --output assets/manifests/calibration_manifest.generated.json

    # Preview without writing:
    python scripts/build_manifest.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import add_config_arg, get, load_config

DEFAULT_INPUT_DIR = ROOT / "calibrate_videos"
DEFAULT_OUTPUT = ROOT / "assets" / "manifests" / "calibration_manifest.generated.json"
MANIFEST_VERSION = "5.0"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

# ---------------------------------------------------------------------------
# Content-type keyword → weight mapping.
# Weight > 1 means the clip counts more toward the activation statistics.
# Night and sports clips tend to produce outlier activations and deserve
# higher weight so the calibrated scales do not clip during real inference.
# ---------------------------------------------------------------------------
_KEYWORD_WEIGHTS: list[tuple[list[str], float, str]] = [
    (["night", "dark", "low_light", "lowlight", "dim"],       1.8, "night/dark"),
    (["sport", "sports", "action", "fast", "motion", "race"], 1.6, "sports/fast-motion"),
    (["outdoor", "daylight", "day", "sunlight", "exterior"],  1.2, "outdoor/daylight"),
    (["indoor", "studio", "interior", "office"],               1.0, "indoor/studio"),
    (["talking", "head", "screen", "presentation", "chat"],   0.9, "talking-head/screen"),
    (["anim", "cartoon", "anime", "render", "cg"],             1.1, "animation"),
]
_DEFAULT_WEIGHT = 1.0
_DEFAULT_CATEGORY = "general"


def _infer_weight(stem: str) -> tuple[float, str]:
    """Return (weight, category) inferred from the filename stem."""
    lower = stem.lower()
    for keywords, weight, category in _KEYWORD_WEIGHTS:
        if any(kw in lower for kw in keywords):
            return weight, category
    return _DEFAULT_WEIGHT, _DEFAULT_CATEGORY


def _probe_duration(path: Path) -> float | None:
    """Return video duration in seconds using ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return None


def scan_clips(input_dir: Path) -> list[dict]:
    """Scan *input_dir* recursively for video files and build clip entries.

    Args:
        input_dir: Directory to scan.

    Returns:
        List of clip dicts with keys: name, path, weight, category,
        duration_sec (optional).
    """
    clips = []
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        weight, category = _infer_weight(path.stem)
        entry: dict = {
            "name": path.stem,
            "path": str(path.resolve().relative_to(ROOT)),
            "weight": weight,
            "category": category,
        }
        duration = _probe_duration(path)
        if duration is not None:
            entry["duration_sec"] = round(duration, 1)
        clips.append(entry)
    return clips


def build_manifest(clips: list[dict]) -> dict:
    """Return a manifest dict ready to be serialised as JSON."""
    return {
        "version": MANIFEST_VERSION,
        "clips": clips,
    }


def _print_table(clips: list[dict]) -> None:
    """Print a formatted preview of the clip list."""
    if not clips:
        print("  (no video files found)")
        return
    col_w = [40, 10, 20, 12]
    header = f"{'Name':<{col_w[0]}}  {'Weight':>{col_w[1]}}  {'Category':<{col_w[2]}}  {'Duration':>{col_w[3]}}"
    print(header)
    print("-" * (sum(col_w) + 6))
    for c in clips:
        dur = f"{c['duration_sec']:.1f}s" if "duration_sec" in c else "unknown"
        name = c["name"][:col_w[0]]
        print(f"{name:<{col_w[0]}}  {c['weight']:>{col_w[1]}.1f}  {c['category']:<{col_w[2]}}  {dur:>{col_w[3]}}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a calibration manifest from a folder of video clips."
    )
    add_config_arg(parser)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing calibration video clips (default: calibrate_videos/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output manifest JSON path (default: assets/manifests/calibration_manifest.generated.json).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview the detected clips without writing the manifest.",
    )

    known, _ = parser.parse_known_args()
    cfg = load_config(known.config)
    if get(cfg, "calibration", "videos_dir"):
        parser.set_defaults(input_dir=Path(get(cfg, "calibration", "videos_dir")))
    if get(cfg, "calibration", "manifest"):
        parser.set_defaults(output=Path(get(cfg, "calibration", "manifest")))

    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    if not args.input_dir.exists():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        print("Create the folder and add your calibration videos, then rerun.")
        sys.exit(1)

    clips = scan_clips(args.input_dir)

    print(f"Found {len(clips)} video file(s) in {args.input_dir}\n")
    _print_table(clips)

    if not clips:
        print("\nAdd .mp4 files to the folder and rerun.")
        sys.exit(1)

    # Summarise coverage
    categories = {}
    for c in clips:
        categories[c["category"]] = categories.get(c["category"], 0) + 1
    print("\nContent coverage:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} clip(s)")

    missing = [
        cat for cat, _, _ in _KEYWORD_WEIGHTS
        if not any(c["category"] == cat for c in clips)
    ]
    if missing:
        print("\nConsider adding clips for better coverage:")
        for cat in missing:
            print(f"  - {cat}")

    if args.dry_run:
        print("\n[dry-run] manifest not written.")
        return

    manifest = build_manifest(clips)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"\nManifest written to: {args.output}")
    print("\nNext step:")
    print(f"  python scripts/calibrate_int16_bundle.py \\")
    print(f"    --manifest {args.output} \\")
    print(f"    --bundle_path models/int16_bundle_v1.0.0.pt \\")
    print(f"    --output    models/int16_bundle_v1.0.0.pt \\")
    print(f"    --frames_per_clip 300 --qp 32")


if __name__ == "__main__":
    main()

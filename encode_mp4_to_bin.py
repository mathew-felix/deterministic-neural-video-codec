#!/usr/bin/env python3
"""Preflight and runtime switches for deterministic INT16 MP4 encoding.

Commit 12 introduces the user-facing switches for profiling and encode-only
P-frame execution. The full standalone packaging lands later in the narrative;
this entrypoint intentionally performs robust preflight checks and records the
effective runtime contract before invoking model-backed encoding.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INDEX_MAP = [0, 1, 0, 2, 0, 2, 0, 2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse encoder preflight and profiling options."""

    parser = argparse.ArgumentParser(
        description="Deterministic DCVC INT16 encoder preflight: MP4 -> .bin."
    )
    parser.add_argument("--input_mp4", type=str, default="test.mp4")
    parser.add_argument(
        "--bundle_path",
        type=str,
        default="models/int16_reference_bundle_v2_calibrated.pt",
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--qp_i", type=int, default=32)
    parser.add_argument("--qp_p", type=int, default=32)
    parser.add_argument("--reset_interval", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--disable_encode_only",
        action="store_true",
        help="Use the full encoder/decoder sync path instead of the fast P-frame mode.",
    )
    parser.add_argument(
        "--profile_pframe_stages",
        action="store_true",
        help="Enable stage timing through DCVC_PROFILE_INT16_PIPELINE.",
    )
    parser.add_argument(
        "--profile_output_json",
        type=str,
        default=None,
        help="Optional path for a future P-frame profile JSON artifact.",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Validate input, flags, ffprobe metadata, and model bundle presence.",
    )
    return parser.parse_args(argv)


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the repository root."""

    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command and raise with captured output on failure."""

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def probe_video(input_mp4: Path) -> dict[str, Any]:
    """Return basic video metadata using ffprobe."""

    payload = json.loads(
        run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(input_mp4),
            ]
        ).stdout
    )
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {input_mp4}")

    stream = streams[0]
    fps_expr = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1"
    try:
        fps = float(Fraction(fps_expr))
    except Exception:
        fps = 30.0
    nb_frames = stream.get("nb_frames")
    if nb_frames in (None, "", "N/A"):
        duration = stream.get("duration")
        nb_frames = int(round(float(duration) * fps)) if duration not in (None, "", "N/A") else None
    else:
        nb_frames = int(nb_frames)

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(fps),
        "nb_frames": nb_frames,
    }


def resolve_frame_count(requested_frames: int, available_frames: int | None) -> int:
    """Resolve the requested frame count against source metadata."""

    if requested_frames > 0:
        return min(requested_frames, available_frames) if available_frames else requested_frames
    if available_frames is None:
        raise RuntimeError("Unable to infer frame count from ffprobe; pass --frames explicitly.")
    return available_frames


def configure_runtime_flags(args: argparse.Namespace) -> dict[str, bool]:
    """Set environment flags used by the INT16 runtime and report their values."""

    encode_only = not args.disable_encode_only
    if encode_only:
        os.environ["DCVC_INT16_ENCODE_ONLY"] = "1"
    else:
        os.environ.pop("DCVC_INT16_ENCODE_ONLY", None)

    if args.profile_pframe_stages:
        os.environ["DCVC_PROFILE_INT16_PIPELINE"] = "1"
    else:
        os.environ.pop("DCVC_PROFILE_INT16_PIPELINE", None)

    return {
        "encode_only": encode_only,
        "profile_pframe_stages": bool(args.profile_pframe_stages),
    }


def build_preflight_report(args: argparse.Namespace) -> dict[str, Any]:
    """Validate local inputs and summarize the intended encode contract."""

    input_mp4 = resolve_path(args.input_mp4)
    bundle_path = resolve_path(args.bundle_path)
    output_dir = resolve_path(args.output_dir)
    if not input_mp4.exists():
        raise FileNotFoundError(f"Input MP4 not found: {input_mp4}")

    video_info = probe_video(input_mp4)
    frame_count = resolve_frame_count(args.frames, video_info["nb_frames"])
    flags = configure_runtime_flags(args)
    profile_output = (
        resolve_path(args.profile_output_json)
        if args.profile_output_json
        else output_dir / "pframe_profile.json"
    )

    return {
        "mode": "encode_preflight",
        "input_mp4": str(input_mp4),
        "bundle_path": str(bundle_path),
        "bundle_exists": bundle_path.exists(),
        "output_dir": str(output_dir),
        "profile_output_json": str(profile_output),
        "frames_requested": int(args.frames),
        "frames_resolved": int(frame_count),
        "width": video_info["width"],
        "height": video_info["height"],
        "fps": video_info["fps"],
        "source_frame_count": video_info["nb_frames"],
        "qp_i": int(args.qp_i),
        "qp_p": int(args.qp_p),
        "reset_interval": int(args.reset_interval),
        "device": args.device,
        "runtime_flags": flags,
    }


def encode_video(args: argparse.Namespace) -> dict[str, Any]:
    """Run preflight and stop before model-backed encoding when assets are absent."""

    report = build_preflight_report(args)
    if args.check_only:
        return report
    if not report["bundle_exists"]:
        raise FileNotFoundError(
            "INT16 bundle is required for encoding. "
            f"Expected: {report['bundle_path']}. "
            "Use --check_only to validate MP4 and runtime flags without model assets."
        )
    raise NotImplementedError(
        "Full MP4 encode orchestration is introduced in the standalone packaging commit."
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    try:
        report = encode_video(parse_args(argv))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

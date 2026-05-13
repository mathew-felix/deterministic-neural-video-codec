#!/usr/bin/env python3
"""Compute RGB PSNR and MS-SSIM between two MP4 files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.metrics import calc_msssim_rgb, calc_psnr


def _probe(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,duration,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())
    streams = json.loads(completed.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": None if stream.get("nb_frames") in (None, "N/A") else int(stream["nb_frames"]),
        "avg_frame_rate": stream.get("avg_frame_rate"),
    }


def _reader(path: Path, width: int, height: int) -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _read_frame(proc: subprocess.Popen, frame_bytes: int) -> bytes | None:
    data = proc.stdout.read(frame_bytes)
    if not data:
        return None
    if len(data) != frame_bytes:
        raise RuntimeError("Short frame read from FFmpeg.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PSNR/MS-SSIM between input and decoded MP4.")
    parser.add_argument("--reference_mp4", required=True, help="Original/reference MP4.")
    parser.add_argument("--decoded_mp4", required=True, help="Decoded/reconstructed MP4.")
    parser.add_argument("--frames", type=int, default=-1, help="Frames to compare. -1 compares all available frames.")
    parser.add_argument("--output_json", default=None, help="Optional metrics JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = Path(args.reference_mp4).resolve()
    decoded = Path(args.decoded_mp4).resolve()
    ref_info = _probe(reference)
    dec_info = _probe(decoded)
    if (ref_info["width"], ref_info["height"]) != (dec_info["width"], dec_info["height"]):
        raise SystemExit(f"Resolution mismatch: {ref_info} vs {dec_info}")

    width = ref_info["width"]
    height = ref_info["height"]
    frame_limit = args.frames
    if frame_limit <= 0:
        known = [v for v in (ref_info["frames"], dec_info["frames"]) if v is not None]
        frame_limit = min(known) if known else None

    frame_bytes = width * height * 3
    ref_proc = _reader(reference, width, height)
    dec_proc = _reader(decoded, width, height)
    psnr_values = []
    msssim_values = []
    samples = {}
    t0 = time.perf_counter()

    try:
        idx = 0
        while frame_limit is None or idx < frame_limit:
            ref_data = _read_frame(ref_proc, frame_bytes)
            dec_data = _read_frame(dec_proc, frame_bytes)
            if ref_data is None or dec_data is None:
                break
            ref_frame = np.frombuffer(ref_data, dtype=np.uint8).reshape(height, width, 3)
            dec_frame = np.frombuffer(dec_data, dtype=np.uint8).reshape(height, width, 3)
            ref_chw = np.transpose(ref_frame, (2, 0, 1))
            dec_chw = np.transpose(dec_frame, (2, 0, 1))
            psnr = float(calc_psnr(ref_frame, dec_frame))
            msssim = float(calc_msssim_rgb(ref_chw, dec_chw))
            psnr_values.append(psnr)
            msssim_values.append(msssim)
            if idx < 5:
                samples[str(idx)] = {"psnr_rgb": psnr, "ms_ssim_rgb": msssim}
            idx += 1
    finally:
        for proc in (ref_proc, dec_proc):
            if proc.poll() is None:
                proc.terminate()

    if not psnr_values:
        raise SystemExit("No frames were compared.")

    result = {
        "reference_mp4": str(reference),
        "decoded_mp4": str(decoded),
        "width": width,
        "height": height,
        "frames_compared": len(psnr_values),
        "average_psnr_rgb": float(np.mean(psnr_values)),
        "average_ms_ssim_rgb": float(np.mean(msssim_values)),
        "min_psnr_rgb": float(np.min(psnr_values)),
        "max_psnr_rgb": float(np.max(psnr_values)),
        "min_ms_ssim_rgb": float(np.min(msssim_values)),
        "max_ms_ssim_rgb": float(np.max(msssim_values)),
        "metric_runtime_sec": float(time.perf_counter() - t0),
        "sample_frames": samples,
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

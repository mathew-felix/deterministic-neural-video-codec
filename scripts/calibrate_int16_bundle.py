#!/usr/bin/env python3
"""Calibrate an INT16 model bundle by collecting activation statistics.

This script loads a pre-exported INT16 bundle, runs representative clips through
the encoder, and collects per-layer activation statistics. The output is a
refined bundle with activation-aware INT8 channel scales and clamp health data.

Usage:
    python scripts/calibrate_int16_bundle.py \
        --manifest assets/manifests/calibration_manifest.example.json \
        --bundle_path models/int16_reference_bundle_v2_calibrated.pt \
        --output models/int16_reference_bundle_v5_calibrated.pt \
        --frames_per_clip 300 \
        --qp 32

The manifest JSON should specify calibration clips:
    {
        "version": "5.0",
        "clips": [
            {"name": "daylight_outdoor", "path": "clips/daylight.mp4", "weight": 1.5},
            {"name": "night_scene",      "path": "clips/night.mp4",    "weight": 0.8}
        ]
    }
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for calibration."""
    parser = argparse.ArgumentParser(
        description="Calibrate INT16 bundle activation scales from representative clips."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to calibration manifest JSON.",
    )
    parser.add_argument(
        "--bundle_path",
        type=str,
        default="models/int16_reference_bundle_v2_calibrated.pt",
        help="Path to the input INT16 reference bundle.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/int16_reference_bundle_v5_calibrated.pt",
        help="Output path for the calibrated bundle.",
    )
    parser.add_argument(
        "--frames_per_clip",
        type=int,
        default=300,
        help="Number of frames to process per clip for statistics collection.",
    )
    parser.add_argument(
        "--qp",
        type=int,
        default=32,
        help="QP value to use during calibration.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.9,
        help="Percentile for activation range aggregation (default: 99.9).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for calibration inference.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load manifest and bundle, report structure, but skip inference.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(path_str)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and validate a calibration manifest JSON."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    if "clips" not in manifest:
        raise ValueError("Manifest must contain a 'clips' array.")
    if not isinstance(manifest["clips"], list) or not manifest["clips"]:
        raise ValueError("Manifest 'clips' must be a non-empty array.")
    for clip in manifest["clips"]:
        if not isinstance(clip, dict):
            raise ValueError(f"Clip entry must be an object: {clip}")
        if "path" not in clip:
            raise ValueError(f"Clip entry missing 'path': {clip}")
        clip.setdefault("name", Path(clip["path"]).stem)
        clip.setdefault("weight", 1.0)
        clip["weight"] = float(clip["weight"])
        if clip["weight"] <= 0:
            raise ValueError(f"Clip weight must be positive: {clip}")
    return manifest


class ActivationStatisticsCollector:
    """Collects per-layer activation statistics during INT16 inference.

    For each layer that is observed, this collector tracks:
    - Running min/max absolute values
    - Running histogram of absolute values (for percentile computation)
    - Sample count
    """

    def __init__(self, percentile: float = 99.9):
        self.percentile = percentile
        self.stats = defaultdict(lambda: {
            "count": 0,
            "abs_max": 0,
            "abs_values": [],
        })

    def observe(self, module_name: str, tensor) -> None:
        """Record activation statistics for a single forward pass.

        Args:
            module_name: Fully qualified module name (e.g., 'DMC.encoder.conv1').
            tensor: The INT16 activation tensor.
        """
        import torch

        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return

        stats = self.stats[module_name]
        abs_max = int(tensor.detach().abs().max().item())
        stats["abs_max"] = max(stats["abs_max"], abs_max)
        stats["count"] += 1

        # Sample a subset for percentile estimation (avoid OOM on long runs)
        flat = tensor.detach().abs().to(torch.int32).flatten()
        if flat.numel() > 10000:
            indices = torch.randperm(flat.numel(), device=flat.device)[:10000]
            flat = flat[indices]
        stats["abs_values"].append(flat.cpu())

    def compute_percentile_scales(self) -> dict[str, int]:
        """Compute per-layer activation scales from the collected statistics.

        Returns:
            Dictionary mapping module_name -> recommended INT8 activation scale.
        """
        import torch

        from src.layers.int16_backend import _pick_power_of_two_scale

        scales = {}
        for module_name, stats in self.stats.items():
            if stats["count"] == 0 or not stats["abs_values"]:
                scales[module_name] = 1
                continue

            all_abs = torch.cat(stats["abs_values"])
            if all_abs.numel() == 0:
                scales[module_name] = 1
                continue

            # Compute the percentile value
            sorted_vals = torch.sort(all_abs).values
            idx = min(
                int(len(sorted_vals) * (self.percentile / 100.0)),
                len(sorted_vals) - 1,
            )
            percentile_val = int(sorted_vals[idx].item())

            # Pick a power-of-two scale that covers this percentile
            scale = _pick_power_of_two_scale(percentile_val, limit=127)
            scales[module_name] = scale

        return scales

    def summary(self) -> list[dict[str, Any]]:
        """Return a summary of all observed layers."""
        return [
            {
                "module_name": name,
                "samples": stats["count"],
                "abs_max": stats["abs_max"],
            }
            for name, stats in sorted(self.stats.items())
        ]


def run_calibration_clip(
    p_frame_net,
    i_frame_net,
    clip_path: Path,
    frames: int,
    qp: int,
    device: str,
) -> int:
    """Run a single clip through the INT16 encoder for calibration.

    Args:
        p_frame_net: The DMC INT16 reference model.
        i_frame_net: The DMCI INT16 reference model.
        clip_path: Path to the MP4 clip.
        frames: Number of frames to process.
        qp: QP value for encoding.
        device: Torch device string.

    Returns:
        Number of frames actually processed.
    """
    import subprocess
    import tempfile

    import torch

    from src.layers.cuda_inference import replicate_pad
    from src.utils.transforms import ycbcr420_to_444_np
    from src.utils.video_reader import YUV420Reader

    # Probe video dimensions
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", str(clip_path),
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    stream_info = json.loads(result.stdout)["streams"][0]
    width, height = int(stream_info["width"]), int(stream_info["height"])

    with TemporaryDirectory(prefix="dcvc_calibration_") as temp_dir:
        temp_yuv = Path(temp_dir) / f"{clip_path.stem}.yuv"
        extract_cmd = [
            "ffmpeg", "-y", "-v", "error", "-i", str(clip_path),
            "-an", "-sn", "-dn", "-pix_fmt", "yuv420p",
            "-frames:v", str(frames), "-f", "rawvideo", str(temp_yuv),
        ]
        subprocess.run(extract_cmd, check=True)

        padding_r = (16 - (width % 16)) % 16
        padding_b = (16 - (height % 16)) % 16
        reader = YUV420Reader(str(temp_yuv), width, height)
        INDEX_MAP = [0, 1, 0, 2, 0, 2, 0, 2]
        processed = 0
        last_qp = 0

        try:
            with torch.no_grad():
                for frame_idx in range(frames):
                    y, uv = reader.read_one_frame()
                    if y is None or uv is None:
                        break

                    x = torch.from_numpy(ycbcr420_to_444_np(y, uv)).to(
                        device=device, dtype=torch.float32
                    ).unsqueeze(0) / 255.0
                    x_padded = replicate_pad(x, padding_b, padding_r)

                    if frame_idx == 0:
                        encoded = i_frame_net.compress(x_padded, qp)
                        p_frame_net.clear_dpb()
                        p_frame_net.add_ref_frame(None, encoded["x_hat"])
                    else:
                        fa_idx = INDEX_MAP[frame_idx % 8]
                        use_ada_i = 1 if (frame_idx % 32 == 1) else 0
                        if use_ada_i:
                            p_frame_net.prepare_feature_adaptor_i(last_qp)
                        curr_qp = p_frame_net.shift_qp(qp, fa_idx)
                        encoded = p_frame_net.compress(x_padded, curr_qp, encode_only=True)
                        last_qp = curr_qp

                    processed += 1
        finally:
            reader.close()
        return processed


def main() -> None:
    """Main calibration entry point."""
    args = parse_args()

    import torch

    from src.models.int16_reference import DMCIInt16Reference, DMCInt16Reference

    manifest_path = resolve_path(args.manifest)
    bundle_path = resolve_path(args.bundle_path)
    output_path = resolve_path(args.output)

    # Load and validate inputs
    manifest = load_manifest(manifest_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    print(f"[calibrate] Manifest: {manifest_path}")
    print(f"[calibrate] Bundle:   {bundle_path}")
    print(f"[calibrate] Output:   {output_path}")
    print(f"[calibrate] Clips:    {len(manifest['clips'])}")
    print(f"[calibrate] Frames/clip: {args.frames_per_clip}")
    print(f"[calibrate] QP:       {args.qp}")
    print(f"[calibrate] Percentile: {args.percentile}")

    if args.dry_run:
        print("[calibrate] Dry run — skipping inference.")
        for clip in manifest["clips"]:
            clip_path = resolve_path(clip["path"])
            print(f"  {clip['name']}: {clip_path} (exists={clip_path.exists()}, weight={clip['weight']})")
        return

    # Load models with activation observer
    device = torch.device(args.device)
    collector = ActivationStatisticsCollector(percentile=args.percentile)

    bundle_blob = torch.load(bundle_path, map_location="cpu", weights_only=False)
    bundle_models = bundle_blob["models"] if "models" in bundle_blob else bundle_blob

    i_frame_net = DMCIInt16Reference(bundle_models["i_frame_net"]).to(device).eval()
    p_frame_net = DMCInt16Reference(
        bundle_models["p_frame_net"],
        activation_observer=collector.observe,
    ).to(device).eval()

    # Run calibration clips
    total_frames = 0
    clip_results = []
    for clip_entry in manifest["clips"]:
        clip_path = resolve_path(clip_entry["path"])
        if not clip_path.exists():
            print(f"[calibrate] WARNING: Clip not found, skipping: {clip_path}")
            continue

        print(f"[calibrate] Processing: {clip_entry['name']} ({clip_path})")
        t0 = time.perf_counter()
        processed = run_calibration_clip(
            p_frame_net, i_frame_net, clip_path,
            args.frames_per_clip, args.qp, device,
        )
        elapsed = time.perf_counter() - t0
        total_frames += processed
        clip_results.append({
            "name": clip_entry["name"],
            "path": str(clip_path),
            "weight": clip_entry["weight"],
            "frames_processed": processed,
            "time_sec": round(elapsed, 2),
        })
        print(f"  → {processed} frames in {elapsed:.1f}s")

    # Compute calibrated scales
    scales = collector.compute_percentile_scales()
    print(f"\n[calibrate] Observed {len(scales)} layers across {total_frames} total frames.")

    # Save the calibrated bundle
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_meta = {
        "calibration_version": manifest.get("version", "unknown"),
        "percentile": args.percentile,
        "qp": args.qp,
        "total_frames": total_frames,
        "clips": clip_results,
        "layer_summary": collector.summary(),
    }
    bundle_blob["calibration"] = calibration_meta
    bundle_blob["activation_scales"] = {
        name: torch.tensor(scale, dtype=torch.int32)
        for name, scale in scales.items()
    }
    torch.save(bundle_blob, str(output_path))
    print(f"[calibrate] Saved calibrated bundle to {output_path}")

    # Write sidecar JSON
    sidecar_path = output_path.with_suffix(".calibration.json")
    with open(sidecar_path, "w", encoding="utf-8") as fp:
        json.dump(calibration_meta, fp, indent=2)
    print(f"[calibrate] Saved calibration sidecar to {sidecar_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Full INT16 model build pipeline.

Runs the four steps required to go from a raw DCVC-RT FP16 checkpoint
to a calibrated INT16 bundle ready for encode/decode:

  Step 1  freeze    Freeze rANS CDF tables from the FP16 checkpoint.
  Step 2  export    Quantize weights to INT16 and pack INT8 shadows.
  Step 3  manifest  Scan calibrate_videos/ and write the calibration manifest.
  Step 4  calibrate Run calibration clips and refine per-layer activation scales.

All paths and settings are read from config.yaml (copy from config.example.yaml).

Usage:
    # Run the full pipeline:
    python pipeline.py

    # Run a single step:
    python pipeline.py --step freeze
    python pipeline.py --step export
    python pipeline.py --step manifest
    python pipeline.py --step calibrate

    # Use a non-default config file:
    python pipeline.py --config path/to/my_config.yaml

    # Dry-run: print what would be executed without running anything:
    python pipeline.py --dry_run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import get, load_config

STEPS = ["freeze", "export", "manifest", "calibrate"]


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

def _step_freeze(cfg: dict, dry_run: bool, config_flag: list[str]) -> None:
    """Step 1: Freeze rANS CDF tables."""
    model_i  = get(cfg, "models", "checkpoint_i")
    model_p  = get(cfg, "models", "checkpoint_p")
    out      = get(cfg, "models", "frozen_entropy", default="models/frozen_entropy_state.pt")
    device   = get(cfg, "build", "device", default="cpu")
    fzt      = get(cfg, "build", "force_zero_thres")

    if not model_i:
        _fatal("models.checkpoint_i not set in config.yaml")

    cmd = [sys.executable, "scripts/freeze_entropy_cdfs.py",
           "--model_path_i", model_i,
           "--output_path",  out,
           "--device",       str(device),
           *config_flag]
    if model_p:
        cmd += ["--model_path_p", model_p]
    if fzt is not None:
        cmd += ["--force_zero_thres", str(fzt)]
    _run(cmd, dry_run)


def _step_export(cfg: dict, dry_run: bool, config_flag: list[str]) -> None:
    """Step 2: Export INT16 bundle with INT8 weight packing."""
    model_i   = get(cfg, "models", "checkpoint_i")
    model_p   = get(cfg, "models", "checkpoint_p")
    frozen    = get(cfg, "models", "frozen_entropy", default="models/frozen_entropy_state.pt")
    out       = get(cfg, "models", "bundle", default="models/int16_bundle_v1.0.0.pt")
    device    = get(cfg, "build", "device", default="cpu")
    fzt       = get(cfg, "build", "force_zero_thres")

    if not model_i:
        _fatal("models.checkpoint_i not set in config.yaml")

    frozen_path = Path(frozen)
    if not frozen_path.exists():
        _fatal(
            f"Frozen entropy file not found: {frozen_path}\n"
            "  Run step 'freeze' first:  python pipeline.py --step freeze"
        )

    cmd = [sys.executable, "scripts/export_int16_bundle.py",
           "--model_path_i",        model_i,
           "--frozen_entropy_path", frozen,
           "--output_path",         out,
           "--device",              str(device),
           *config_flag]
    if model_p:
        cmd += ["--model_path_p", model_p]
    if fzt is not None:
        cmd += ["--force_zero_thres", str(fzt)]
    _run(cmd, dry_run)


def _step_manifest(cfg: dict, dry_run: bool, config_flag: list[str]) -> None:
    """Step 3: Generate calibration manifest from calibrate_videos/."""
    cmd = [sys.executable, "scripts/build_manifest.py", *config_flag]
    if dry_run:
        cmd += ["--dry_run"]
    _run(cmd, dry_run=False)


def _step_calibrate(cfg: dict, dry_run: bool, config_flag: list[str]) -> None:
    """Step 4: Run calibration clips and refine activation scales."""
    manifest = get(cfg, "calibration", "manifest",
                   default="assets/manifests/calibration_manifest.generated.json")
    bundle   = get(cfg, "models", "bundle",
                   default="models/int16_bundle_v1.0.0.pt")
    frames   = get(cfg, "calibration", "frames_per_clip", default=300)
    qp       = get(cfg, "calibration", "qp", default=32)
    pct      = get(cfg, "calibration", "percentile", default=99.9)
    device   = get(cfg, "calibration", "device", default="cuda:0")

    if not Path(manifest).exists():
        _fatal(
            f"Manifest not found: {manifest}\n"
            "  Run step 'manifest' first:  python pipeline.py --step manifest"
        )
    if not Path(bundle).exists():
        _fatal(
            f"Bundle not found: {bundle}\n"
            "  Run step 'export' first:  python pipeline.py --step export"
        )

    cmd = [sys.executable, "scripts/calibrate_int16_bundle.py",
           "--manifest",       manifest,
           "--bundle_path",    bundle,
           "--output",         bundle,
           "--frames_per_clip", str(frames),
           "--qp",             str(qp),
           "--percentile",     str(pct),
           "--device",         str(device),
           *config_flag]
    if dry_run:
        cmd += ["--dry_run"]
    _run(cmd, dry_run=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STEP_FNS = {
    "freeze":    _step_freeze,
    "export":    _step_export,
    "manifest":  _step_manifest,
    "calibrate": _step_calibrate,
}


def _run(cmd: list[str], dry_run: bool) -> None:
    label = " ".join(cmd)
    if dry_run:
        print(f"  [dry-run] {label}")
        return
    print(f"\n  $ {label}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        _fatal(f"Step failed with exit code {result.returncode}")


def _fatal(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _print_header(step: str, index: int, total: int) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  Step {index}/{total}: {step.upper()}")
    print(f"{bar}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse pipeline arguments."""
    parser = argparse.ArgumentParser(
        description="Run the DCVC INT16 bundle build pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Steps: " + ", ".join(STEPS),
            "",
            "Examples:",
            "  python pipeline.py                    # full pipeline",
            "  python pipeline.py --step export      # single step",
            "  python pipeline.py --dry_run          # preview commands",
        ]),
    )
    parser.add_argument(
        "--step",
        choices=STEPS,
        default=None,
        help="Run a single named step instead of the full pipeline.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to YAML config file (default: config.yaml).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    """Pipeline entry point."""
    args = parse_args()
    cfg = load_config(args.config)

    if not cfg:
        print(
            "ERROR: config.yaml not found.\n"
            "\n"
            "Create it from the template and fill in your checkpoint paths:\n"
            "\n"
            "  Windows:  copy config.example.yaml config.yaml\n"
            "  Linux:    cp config.example.yaml config.yaml\n"
            "\n"
            "Then edit config.yaml and set at minimum:\n"
            "\n"
            "  models:\n"
            "    checkpoint_i: models/cvpr2025_image.pth.tar\n"
            "    checkpoint_p: models/cvpr2025_video.pth.tar\n"
            "\n"
            "Then rerun:  python pipeline.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    config_flag = ["--config", args.config] if args.config else []

    steps_to_run = [args.step] if args.step else STEPS
    total = len(steps_to_run)

    t_start = time.perf_counter()
    for i, step in enumerate(steps_to_run, 1):
        _print_header(step, i, total)
        _STEP_FNS[step](cfg, args.dry_run, config_flag)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    if not args.dry_run and (args.step is None or args.step == "calibrate"):
        bundle = get(cfg, "models", "bundle",
                     default="models/int16_bundle_v1.0.0.pt")
        print(f"\n  Bundle: {bundle}")
        print("\n  Verify the bundle:")
        print("    python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 2 --check_only")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the local runtime setup before running a full encode/decode."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import get, load_config


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _check_import(module: str) -> tuple[bool, str | None]:
    try:
        return importlib.util.find_spec(module) is not None, None
    except Exception as exc:
        return False, str(exc)


def _run(command: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return False, str(exc)
    output = (completed.stdout or completed.stderr or "").strip()
    return completed.returncode == 0, output.splitlines()[0] if output else ""


def _status(name: str, ok: bool, detail: str = "", required: bool = True) -> dict:
    state = "ok" if ok else ("fail" if required else "warn")
    return {"name": name, "status": state, "required": required, "detail": detail}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DCVC INT16 runtime setup.")
    parser.add_argument("--config", default=None, help="Config YAML path.")
    parser.add_argument("--bundle_path", default=None, help="Override INT16 bundle path.")
    parser.add_argument("--input_mp4", default=None, help="Optional MP4 path to validate.")
    parser.add_argument("--require_config", action="store_true", help="Fail if config/config.yaml is missing.")
    parser.add_argument("--require_cuda", action="store_true", help="Fail if CUDA is unavailable.")
    parser.add_argument("--require_bundle", action="store_true", help="Fail if the bundle is missing.")
    parser.add_argument(
        "--allow_missing_native",
        action="store_true",
        help="Warn instead of fail when native extensions are missing. Useful for non-CUDA CI.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    bundle_path = _resolve(args.bundle_path or get(cfg, "models", "bundle"))
    input_mp4 = _resolve(args.input_mp4 or get(cfg, "encode", "input_mp4"))
    checks = []

    checks.append(_status("python", sys.version_info >= (3, 10), sys.version.split()[0]))

    ffmpeg = shutil.which("ffmpeg")
    checks.append(_status("ffmpeg", ffmpeg is not None, ffmpeg or "Install FFmpeg and add it to PATH."))
    ffprobe = shutil.which("ffprobe")
    checks.append(_status("ffprobe", ffprobe is not None, ffprobe or "Install FFmpeg and add ffprobe to PATH."))

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        detail = f"torch {torch.__version__}, cuda_available={cuda_ok}"
        if cuda_ok:
            detail += f", device={torch.cuda.get_device_name(0)}"
        checks.append(_status("torch/cuda", cuda_ok or not args.require_cuda, detail, args.require_cuda))
    except Exception as exc:
        checks.append(_status("torch/cuda", False, f"PyTorch import failed: {exc}", True))

    ok, detail = _check_import("MLCodec_extensions_cpp")
    checks.append(
        _status(
            "MLCodec_extensions_cpp",
            ok,
            "installed" if ok else detail or "Install with: python -m pip install --no-build-isolation ./src/cpp",
            not args.allow_missing_native,
        )
    )

    ok, detail = _check_import("inference_extensions_cuda")
    checks.append(
        _status(
            "inference_extensions_cuda",
            ok,
            "installed" if ok else detail or "Optional. Install with: python -m pip install --no-build-isolation ./src/layers/extensions/inference",
            False,
        )
    )

    try:
        from src.layers.int16_cuda_ext import is_available as int16_cuda_available

        ok = bool(int16_cuda_available())
        detail = None
    except Exception as exc:
        ok = False
        detail = str(exc)
    checks.append(
        _status(
            "int16_cuda_ext",
            ok,
            "installed" if ok else detail or "Optional. Build with: python build_int16_cuda.py",
            False,
        )
    )

    config_path = _resolve(args.config or "config/config.yaml")
    config_ok = bool(config_path and config_path.exists())
    checks.append(
        _status(
            "config/config.yaml",
            config_ok,
            str(config_path) if config_path and config_path.exists() else "Copy config/config.example.yaml to config/config.yaml.",
            bool(args.require_config),
        )
    )

    bundle_required = bool(args.require_bundle)
    checks.append(
        _status(
            "int16 bundle",
            bool(bundle_path and bundle_path.exists()),
            str(bundle_path) if bundle_path else "No bundle path configured.",
            bundle_required,
        )
    )

    input_required = args.input_mp4 is not None
    input_ok = bool(input_mp4 and input_mp4.exists())
    detail = str(input_mp4) if input_mp4 else "No input MP4 configured."
    if input_ok and ffprobe:
        ok, probe = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames",
                "-of",
                "compact=p=0:nk=1",
                str(input_mp4),
            ]
        )
        detail = probe if ok else f"{input_mp4}: {probe}"
        input_ok = input_ok and ok
    checks.append(_status("input mp4", input_ok, detail, input_required))

    failed = [c for c in checks if c["required"] and c["status"] == "fail"]
    warnings = [c for c in checks if not c["required"] and c["status"] == "warn"]

    payload = {"ok": not failed, "failed": len(failed), "warnings": len(warnings), "checks": checks}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for check in checks:
            print(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")
        if failed:
            print("\nSetup is not ready. Fix required failures above.")
        else:
            print("\nRequired setup checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

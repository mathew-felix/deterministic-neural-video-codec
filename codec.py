#!/usr/bin/env python3
"""Small Python API for the deterministic INT16 codec runtime.

Use this module when you want to call the repo's encode/decode tools from
Python instead of shelling out manually:

    import codec

    result = codec.compress(
        input_mp4="video.mp4",
        output_dir="outputs",
        bundle_path="models/int16_bundle_v1.0.0.pt",
        qp=32,
    )
    codec.decompress(
        input_bin=result["bitstream_path"],
        bundle_path="models/int16_bundle_v1.0.0.pt",
    )
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def compress(
    input_mp4: str | Path,
    output_dir: str | Path,
    *,
    bundle_path: str | Path = "models/int16_bundle_v1.0.0.pt",
    qp: int = 32,
    frames: int = -1,
    device: str = "cuda:0",
    config: str | Path | None = None,
    python: str | Path | None = None,
) -> dict[str, Any]:
    """Encode an MP4 into a deterministic INT16 `.bin` bitstream."""
    py = str(python or sys.executable)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        py,
        str(ROOT / "encode_mp4_to_bin.py"),
        "--input_mp4",
        str(input_mp4),
        "--bundle_path",
        str(bundle_path),
        "--output_dir",
        str(out),
        "--frames",
        str(frames),
        "--qp_i",
        str(qp),
        "--qp_p",
        str(qp),
        "--device",
        device,
    ]
    if config is not None:
        cmd.extend(["--config", str(config)])
    _run(cmd)

    sidecars = [
        p
        for p in sorted(out.glob("*_q*.json"))
        if not p.name.endswith("_decode.json") and "_pframe_profile" not in p.name
    ]
    if not sidecars:
        raise FileNotFoundError(f"No encode sidecar JSON found in {out}")
    metrics = _load_json(sidecars[-1])
    return {
        "bitstream_path": metrics["bitstream_path"],
        "metrics_path": str(sidecars[-1].resolve()),
        "metrics": metrics,
    }


def decompress(
    input_bin: str | Path,
    *,
    output_mp4: str | Path | None = None,
    bundle_path: str | Path = "models/int16_bundle_v1.0.0.pt",
    device: str = "cuda:0",
    config: str | Path | None = None,
    python: str | Path | None = None,
) -> dict[str, Any]:
    """Decode a deterministic INT16 `.bin` bitstream into an MP4."""
    py = str(python or sys.executable)
    cmd = [
        py,
        str(ROOT / "decode_bin_to_mp4.py"),
        "--input_bin",
        str(input_bin),
        "--bundle_path",
        str(bundle_path),
        "--device",
        device,
    ]
    if output_mp4 is not None:
        cmd.extend(["--output_mp4", str(output_mp4)])
    if config is not None:
        cmd.extend(["--config", str(config)])
    _run(cmd)

    bin_path = Path(input_bin)
    mp4_path = Path(output_mp4) if output_mp4 else bin_path.with_name(f"{bin_path.stem}_decoded.mp4")
    decode_json = mp4_path.with_name(f"{mp4_path.stem}_decode.json")
    metrics = _load_json(decode_json) if decode_json.exists() else {}
    return {
        "output_mp4": str(mp4_path.resolve()),
        "metrics_path": str(decode_json.resolve()) if decode_json.exists() else None,
        "metrics": metrics,
    }


def roundtrip(
    input_mp4: str | Path,
    output_dir: str | Path,
    *,
    bundle_path: str | Path = "models/int16_bundle_v1.0.0.pt",
    qp: int = 32,
    frames: int = -1,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Encode and decode in one call."""
    enc = compress(
        input_mp4,
        output_dir,
        bundle_path=bundle_path,
        qp=qp,
        frames=frames,
        device=device,
    )
    dec = decompress(
        enc["bitstream_path"],
        bundle_path=bundle_path,
        device=device,
    )
    return {"encode": enc, "decode": dec}

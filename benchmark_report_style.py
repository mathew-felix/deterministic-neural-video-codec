#!/usr/bin/env python3
"""Summarize INT16 P-frame profile artifacts in report style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse profile summarizer arguments."""

    parser = argparse.ArgumentParser(
        description="Summarize DCVC INT16 P-frame profile JSON artifacts."
    )
    parser.add_argument("profile_json", type=str)
    parser.add_argument("--top", type=int, default=8)
    return parser.parse_args(argv)


def load_profile(path: Path) -> dict[str, Any]:
    """Load a JSON profile artifact."""

    return json.loads(path.read_text(encoding="utf-8"))


def summarize_profile(profile: dict[str, Any], top: int = 8) -> dict[str, Any]:
    """Build a compact stage summary from an encode profile artifact."""

    summary = profile.get("summary", profile)
    stage_summary = summary.get("stage_summary", {})
    rows = []
    for name, metrics in stage_summary.items():
        rows.append(
            {
                "stage": name,
                "avg_gpu_ms": float(metrics.get("avg_gpu_ms", 0.0)),
                "avg_cpu_ms": float(metrics.get("avg_cpu_ms", 0.0)),
                "samples": int(metrics.get("samples", 0)),
            }
        )
    rows.sort(key=lambda row: row["avg_gpu_ms"], reverse=True)
    return {
        "mode": "benchmark_report_style",
        "frames_profiled": int(summary.get("frames_profiled", 0)),
        "warmup_skip_frames": int(summary.get("warmup_skip_frames", 0)),
        "top_gpu_stages": rows[: max(top, 0)],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    report = summarize_profile(load_profile(Path(args.profile_json)), top=args.top)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

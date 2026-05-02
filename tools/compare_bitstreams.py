#!/usr/bin/env python3
"""Hash and optionally byte-compare deterministic codec bitstreams."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.equivalence import (  # noqa: E402
    compare_files_bytewise,
    infer_metrics_json_path,
    load_json_if_exists,
    sha256_file,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hash and optionally compare deterministic codec .bin bitstreams."
    )
    parser.add_argument(
        "bitstreams",
        nargs="+",
        help="One or more .bin files to hash.",
    )
    parser.add_argument(
        "--expect_equal",
        action="store_true",
        help="Byte-compare exactly two bitstreams and exit non-zero on mismatch.",
    )
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def build_file_summary(bitstream_path):
    if not bitstream_path.exists():
        raise FileNotFoundError(f"Bitstream not found: {bitstream_path}")
    metrics_path = infer_metrics_json_path(bitstream_path)
    metrics = None
    metrics_error = None
    if metrics_path.exists():
        try:
            metrics = load_json_if_exists(metrics_path)
        except Exception as exc:  # pylint: disable=broad-except
            metrics_error = str(exc)

    sha256 = sha256_file(bitstream_path)
    summary = {
        "path": str(bitstream_path),
        "size_bytes": int(bitstream_path.stat().st_size),
        "sha256": sha256,
        "metrics_json_path": str(metrics_path) if metrics_path.exists() else None,
        "equivalence_class": metrics.get("equivalence_class") if metrics is not None else None,
        "bitstream_sha256_in_metrics": metrics.get("bitstream_sha256") if metrics is not None else None,
        "bitstream_sha256_matches_metrics": (
            metrics.get("bitstream_sha256") == sha256
            if metrics is not None and metrics.get("bitstream_sha256") is not None
            else None
        ),
    }
    if metrics_error is not None:
        summary["metrics_json_error"] = metrics_error
    return summary


def main():
    args = parse_args()
    bitstreams = [resolve_path(path_str) for path_str in args.bitstreams]
    if args.expect_equal and len(bitstreams) != 2:
        raise SystemExit("--expect_equal requires exactly two bitstreams.")

    files = [build_file_summary(bitstream_path) for bitstream_path in bitstreams]
    unique_hashes = sorted({entry["sha256"] for entry in files})
    summary = {
        "tool": "compare_bitstreams",
        "file_count": int(len(files)),
        "all_sha256_equal": len(unique_hashes) == 1,
        "files": files,
    }

    if len(files) == 2:
        summary["pair_sha256_equal"] = files[0]["sha256"] == files[1]["sha256"]

    exit_code = 0
    if args.expect_equal:
        comparison = compare_files_bytewise(bitstreams[0], bitstreams[1])
        comparison["sha256_equal"] = files[0]["sha256"] == files[1]["sha256"]
        comparison["left_path"] = str(bitstreams[0])
        comparison["right_path"] = str(bitstreams[1])
        summary["comparison"] = comparison
        if not comparison["equal"]:
            exit_code = 1

    print(json.dumps(summary, indent=2))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

"""Tests for Commit 12 encode-only profiling control surfaces."""

import json
import tempfile
import unittest
from pathlib import Path

import benchmark_report_style
import encode_mp4_to_bin


class EncodeProfileCliTest(unittest.TestCase):
    def test_configure_runtime_flags_sets_encode_only_and_profile(self):
        args = encode_mp4_to_bin.parse_args(
            ["--check_only", "--profile_pframe_stages"]
        )
        flags = encode_mp4_to_bin.configure_runtime_flags(args)

        self.assertTrue(flags["encode_only"])
        self.assertTrue(flags["profile_pframe_stages"])

    def test_disable_encode_only_clears_fast_path_flag(self):
        args = encode_mp4_to_bin.parse_args(
            ["--check_only", "--disable_encode_only"]
        )
        flags = encode_mp4_to_bin.configure_runtime_flags(args)

        self.assertFalse(flags["encode_only"])
        self.assertFalse(flags["profile_pframe_stages"])

    def test_profile_summary_orders_gpu_hot_stages(self):
        profile = {
            "summary": {
                "frames_profiled": 3,
                "warmup_skip_frames": 1,
                "stage_summary": {
                    "entropy": {"avg_gpu_ms": 2.0, "avg_cpu_ms": 4.0, "samples": 3},
                    "encode_y": {"avg_gpu_ms": 9.0, "avg_cpu_ms": 9.5, "samples": 3},
                },
            }
        }

        summary = benchmark_report_style.summarize_profile(profile, top=1)

        self.assertEqual(summary["frames_profiled"], 3)
        self.assertEqual(summary["top_gpu_stages"][0]["stage"], "encode_y")

    def test_profile_cli_loads_json_artifact(self):
        profile = {
            "summary": {
                "frames_profiled": 1,
                "warmup_skip_frames": 0,
                "stage_summary": {
                    "decode_feature": {
                        "avg_gpu_ms": 3.5,
                        "avg_cpu_ms": 3.7,
                        "samples": 1,
                    }
                },
            }
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            loaded = benchmark_report_style.load_profile(path)

        self.assertEqual(loaded["summary"]["frames_profiled"], 1)


if __name__ == "__main__":
    unittest.main()

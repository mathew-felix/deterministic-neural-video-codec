"""Tests for byte-level bitstream equivalence utilities."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class BitstreamEquivalenceTest(unittest.TestCase):
    """Validate SHA-256 summaries and byte mismatch reporting."""

    def test_sha256_file_and_bytes_match(self):
        from src.utils.equivalence import sha256_bytes, sha256_file

        payload = b"deterministic-bitstream"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.bin"
            path.write_bytes(payload)

            self.assertEqual(sha256_file(path), sha256_bytes(payload))

    def test_bytewise_compare_reports_first_mismatch(self):
        from src.utils.equivalence import compare_files_bytewise

        with tempfile.TemporaryDirectory() as tmpdir:
            left = Path(tmpdir) / "left.bin"
            right = Path(tmpdir) / "right.bin"
            left.write_bytes(b"\x00\x01\x02\x03")
            right.write_bytes(b"\x00\x01\xff\x03")

            comparison = compare_files_bytewise(left, right, chunk_size=2)

        self.assertFalse(comparison["equal"])
        self.assertEqual(comparison["first_mismatch_offset"], 2)
        self.assertEqual(comparison["left_size_bytes"], 4)
        self.assertEqual(comparison["right_size_bytes"], 4)

    def test_compare_bitstreams_cli_accepts_equal_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            left = Path(tmpdir) / "left.bin"
            right = Path(tmpdir) / "right.bin"
            payload = b"\x10\x20\x30"
            left.write_bytes(payload)
            right.write_bytes(payload)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools" / "compare_bitstreams.py"),
                    str(left),
                    str(right),
                    "--expect_equal",
                ],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertTrue(summary["comparison"]["equal"])
        self.assertTrue(summary["comparison"]["sha256_equal"])

    def test_compare_bitstreams_cli_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            left = Path(tmpdir) / "left.bin"
            right = Path(tmpdir) / "right.bin"
            left.write_bytes(b"\x10\x20\x30")
            right.write_bytes(b"\x10\x21\x30")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools" / "compare_bitstreams.py"),
                    str(left),
                    str(right),
                    "--expect_equal",
                ],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 1)
        summary = json.loads(completed.stdout)
        self.assertFalse(summary["comparison"]["equal"])
        self.assertEqual(summary["comparison"]["first_mismatch_offset"], 1)


if __name__ == "__main__":
    unittest.main()

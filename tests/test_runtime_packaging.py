"""Packaging tests for the standalone encode/decode runtime."""

import importlib.util
import unittest
from pathlib import Path


class RuntimePackagingTest(unittest.TestCase):
    def test_standalone_entrypoints_exist(self):
        root = Path(__file__).resolve().parents[1]

        self.assertTrue((root / "encode_mp4_to_bin.py").exists())
        self.assertTrue((root / "decode_bin_to_mp4.py").exists())
        self.assertTrue((root / "bootstrap_runtime.py").exists())
        self.assertTrue((root / "scripts" / "download_models.ps1").exists())
        self.assertTrue((root / "scripts" / "download_models.sh").exists())

    def test_decode_entrypoint_imports(self):
        spec = importlib.util.find_spec("decode_bin_to_mp4")

        self.assertIsNotNone(spec)

    def test_transform_helpers_preserve_yuv420_shapes(self):
        import torch

        from src.utils.transforms import yuv_444_to_420

        yuv = torch.zeros((1, 3, 8, 10), dtype=torch.float32)
        y, uv = yuv_444_to_420(yuv)

        self.assertEqual(tuple(y.shape), (1, 1, 8, 10))
        self.assertEqual(tuple(uv.shape), (1, 2, 4, 5))


if __name__ == "__main__":
    unittest.main()

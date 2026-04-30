"""Bundle-format smoke tests for the INT16 reference layer."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class BundleLoadingSmokeTest(unittest.TestCase):
    """Validate that minimal bundle metadata can be serialized and restored."""

    def test_top_level_bundle_metadata_roundtrip(self):
        import torch

        payload = {
            "format_version": 1,
            "models": {
                "i_frame_net": {"model_type": "DMCI"},
                "p_frame_net": {"model_type": "DMC"},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "bundle.pt"
            torch.save(payload, bundle_path)
            restored = torch.load(bundle_path, map_location="cpu", weights_only=False)

        self.assertEqual(restored["format_version"], 1)
        self.assertEqual(restored["models"]["i_frame_net"]["model_type"], "DMCI")
        self.assertEqual(restored["models"]["p_frame_net"]["model_type"], "DMC")


if __name__ == "__main__":
    unittest.main()


"""Smoke tests for the INT16 backend contract."""

import importlib.util
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class Int16BackendSmokeTest(unittest.TestCase):
    """Validate lightweight INT16 backend invariants."""

    def test_manifest_roundtrip_preserves_quantization_scales(self):
        from src.layers.int16_backend import (
            Int16QuantConfig,
            export_int16_manifest,
            quant_config_from_manifest,
        )

        config = Int16QuantConfig(feature_scale=512, weight_scale=8192)
        manifest = export_int16_manifest(config)
        restored = quant_config_from_manifest(manifest)

        self.assertEqual(restored.feature_scale, 512)
        self.assertEqual(restored.weight_scale, 8192)
        self.assertEqual(restored.bias_scale, 512 * 8192)

    def test_feature_quantization_roundtrip_uses_int16_storage(self):
        import torch

        from src.layers.int16_backend import (
            Int16QuantConfig,
            feature_to_int16,
            int16_to_feature,
        )

        config = Int16QuantConfig(feature_scale=512)
        source = torch.tensor([-1.0, 0.0, 0.5, 1.0], dtype=torch.float32)
        quantized = feature_to_int16(source, config)
        restored = int16_to_feature(quantized, config)

        self.assertEqual(quantized.dtype, torch.int16)
        self.assertTrue(torch.allclose(source, restored, atol=1.0 / config.feature_scale))


if __name__ == "__main__":
    unittest.main()

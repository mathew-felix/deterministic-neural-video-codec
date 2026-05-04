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

    def test_conv2d_reference_fuses_residual_with_saturation(self):
        import torch

        from src.layers.int16_backend import Conv2dInt16Params, conv2d_int16_reference

        input_i16 = torch.tensor([[[[8, -8], [4, -4]]]], dtype=torch.int16)
        weight = torch.tensor([[[[1]]]], dtype=torch.int16)
        bias = torch.tensor([0], dtype=torch.int32)
        residual = torch.tensor([[[[32760, -32760], [3, -3]]]], dtype=torch.int16)
        params = Conv2dInt16Params(
            weight=weight,
            bias=bias,
            k2_layer=1,
            stride=1,
            padding=0,
            groups=1,
        )

        actual = conv2d_int16_reference(input_i16, params, residual=residual)
        expected = torch.tensor([[[[32767, -32768], [7, -7]]]], dtype=torch.int16)

        self.assertTrue(torch.equal(actual, expected))

    def test_conv2d_residual_shape_must_match_output_shape(self):
        import torch

        from src.layers.int16_backend import Conv2dInt16Params, conv2d_int16_reference

        input_i16 = torch.ones((1, 1, 4, 4), dtype=torch.int16)
        weight = torch.ones((2, 1, 3, 3), dtype=torch.int16)
        residual = torch.ones((1, 2, 4, 4), dtype=torch.int16)
        params = Conv2dInt16Params(
            weight=weight,
            bias=None,
            k2_layer=1,
            stride=1,
            padding=0,
            groups=1,
        )

        with self.assertRaisesRegex(RuntimeError, "Fused INT16 residual shape mismatch"):
            conv2d_int16_reference(input_i16, params, residual=residual)


if __name__ == "__main__":
    unittest.main()

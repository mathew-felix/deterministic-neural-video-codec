"""Tests for the experimental INT8 Tensor Core routing gate."""

import importlib.util
import os
import unittest
from unittest import mock


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class Int8RoutingTest(unittest.TestCase):
    """Validate that INT8 acceleration remains opt-in and layer-scoped."""

    def test_int8_tensor_cores_are_disabled_by_default(self):
        from src.layers.int16_backend import int8_tensor_cores_enabled

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(int8_tensor_cores_enabled())

    def test_single_layer_gate_overrides_default_allowlist(self):
        from src.layers.int16_backend import is_int8_layer_eligible

        with mock.patch.dict(os.environ, {"DCVC_INT8_SINGLE_LAYER": "DMC.decoder.conv2"}):
            self.assertTrue(is_int8_layer_eligible("DMC.decoder.conv2"))
            self.assertFalse(is_int8_layer_eligible("DMC.encoder.conv1"))

    def test_candidate_requires_1x1_stride1_group1_shape(self):
        import torch

        from src.layers.int16_backend import Conv2dInt16Params, is_int8_kernel_candidate

        candidate = Conv2dInt16Params(
            weight=torch.ones(4, 4, 1, 1, dtype=torch.int16),
            bias=None,
            weight_int8=torch.ones(4, 4, 1, 1, dtype=torch.int8),
            stride=1,
            padding=0,
            groups=1,
        )
        self.assertTrue(is_int8_kernel_candidate(candidate))

        candidate.padding = 1
        self.assertFalse(is_int8_kernel_candidate(candidate))

    def test_scale_from_max_abs_uses_power_of_two_divisor(self):
        from scripts.calibrate_int8_activation_scales import scale_from_max_abs

        self.assertEqual(scale_from_max_abs(127), 1)
        self.assertEqual(scale_from_max_abs(128), 2)
        self.assertEqual(scale_from_max_abs(512), 8)


if __name__ == "__main__":
    unittest.main()

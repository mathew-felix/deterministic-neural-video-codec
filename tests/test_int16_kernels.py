"""Parity tests for the optional INT16 CUDA kernels."""

import importlib.util
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class Int16CudaKernelParityTest(unittest.TestCase):
    """Compare CUDA extension outputs with the Python reference backend."""

    @classmethod
    def setUpClass(cls):
        import torch

        from src.layers.int16_cuda_ext import is_available

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if not is_available():
            raise unittest.SkipTest("INT16 CUDA extension did not build or load")
        cls.device = torch.device("cuda:0")

    def test_conv2d_kernel_matches_reference(self):
        import torch

        from src.layers.int16_backend import Conv2dInt16Params, conv2d_int16_reference
        from src.layers.int16_cuda_ext import conv2d_int16

        input_i16 = torch.tensor(
            [[[[300, -200, 500, 100], [0, 400, -100, 200], [600, -300, 100, 0], [200, 100, -400, 300]]]],
            dtype=torch.int16,
        )
        weight = torch.tensor(
            [[[[200, -100, 0], [100, 300, -200], [0, 100, 200]]]],
            dtype=torch.int16,
        )
        bias = torch.tensor([700], dtype=torch.int32)
        params = Conv2dInt16Params(
            weight=weight,
            bias=bias,
            k2_layer=8192,
            stride=1,
            padding=1,
            groups=1,
        )

        expected = conv2d_int16_reference(input_i16, params)
        actual = conv2d_int16(
            input_i16.to(self.device),
            weight.to(self.device),
            bias.to(self.device),
            stride=1,
            padding=1,
            groups=1,
            k2_layer=8192,
        ).cpu()

        self.assertTrue(torch.equal(actual, expected.to(torch.int16)))

    def test_optimized_1x1_kernel_with_residual_matches_reference(self):
        import torch

        from src.layers.int16_backend import Conv2dInt16Params, conv2d_int16_reference
        from src.layers.int16_cuda_ext import conv1x1_int16_gemm

        input_i16 = torch.arange(-18, 18, dtype=torch.int16).reshape(1, 4, 3, 3)
        weight = torch.tensor(
            [
                [1, -2, 0, 3],
                [0, 1, -1, 2],
                [2, 0, 1, -1],
                [-1, 3, 2, 0],
            ],
            dtype=torch.int16,
        )
        bias = torch.tensor([2, -3, 4, -5], dtype=torch.int32)
        residual = torch.full((1, 4, 3, 3), 7, dtype=torch.int16)
        params = Conv2dInt16Params(
            weight=weight.reshape(4, 4, 1, 1),
            bias=bias,
            k2_layer=1,
            stride=1,
            padding=0,
            groups=1,
        )

        expected = conv2d_int16_reference(input_i16, params, residual=residual)
        actual = conv1x1_int16_gemm(
            input_i16.to(self.device),
            weight.to(self.device),
            bias.to(self.device),
            residual=residual.to(self.device),
            k2_layer=1,
        ).cpu()

        self.assertTrue(torch.equal(actual, expected.to(torch.int16)))

    def test_optimized_1x1_kernel_with_post_scale_matches_reference(self):
        import torch

        from src.layers.int16_backend import Conv2dInt16Params, conv2d_int16_reference
        from src.layers.int16_cuda_ext import conv1x1_int16_gemm

        input_i16 = torch.arange(-18, 18, dtype=torch.int16).reshape(1, 4, 3, 3)
        weight = torch.tensor(
            [
                [1, -2, 0, 3],
                [0, 1, -1, 2],
                [2, 0, 1, -1],
                [-1, 3, 2, 0],
            ],
            dtype=torch.int16,
        )
        bias = torch.tensor([2, -3, 4, -5], dtype=torch.int32)
        residual = torch.full((1, 4, 3, 3), 7, dtype=torch.int16)
        post_scale = torch.tensor([256, 512, 1024, 768], dtype=torch.int16)
        params = Conv2dInt16Params(
            weight=weight.reshape(4, 4, 1, 1),
            bias=bias,
            k2_layer=1,
            stride=1,
            padding=0,
            groups=1,
        )

        expected = conv2d_int16_reference(
            input_i16,
            params,
            residual=residual,
            post_scale=post_scale,
        )
        actual = conv1x1_int16_gemm(
            input_i16.to(self.device),
            weight.to(self.device),
            bias.to(self.device),
            residual=residual.to(self.device),
            post_scale=post_scale.to(self.device),
            k2_layer=1,
        ).cpu()

        self.assertTrue(torch.equal(actual, expected.to(torch.int16)))

    def test_multiply_kernel_broadcasts_per_channel_like_reference(self):
        import torch

        from src.layers.int16_backend import multiply_int16 as multiply_reference
        from src.layers.int16_cuda_ext import multiply_int16

        input_i16 = torch.tensor(
            [[[[100, -100]], [[200, -200]], [[300, -300]]]],
            dtype=torch.int16,
        )
        scale = torch.tensor([256, 512, 1024], dtype=torch.int16)

        expected = multiply_reference(input_i16, scale.reshape(1, 3, 1, 1), 512)
        actual = multiply_int16(input_i16.to(self.device), scale.to(self.device), 512).cpu()

        self.assertTrue(torch.equal(actual, expected))

    def test_multiply_kernel_accepts_full_tensor_scale(self):
        import torch

        from src.layers.int16_backend import multiply_int16 as multiply_reference
        from src.layers.int16_cuda_ext import multiply_int16

        input_i16 = torch.tensor(
            [[[[100, -100]], [[200, -200]], [[300, -300]]]],
            dtype=torch.int16,
        )
        scale = torch.tensor(
            [[[[256, 512]], [[512, 1024]], [[1024, 256]]]],
            dtype=torch.int16,
        )

        expected = multiply_reference(input_i16, scale, 512)
        actual = multiply_int16(input_i16.to(self.device), scale.to(self.device), 512).cpu()

        self.assertTrue(torch.equal(actual, expected))

    def test_fused_depthwise_lut_kernel_matches_unfused_reference(self):
        import torch

        from src.layers.int16_backend import (
            INT16_MAX,
            INT16_MIN,
            Conv2dInt16Params,
            apply_lut_int16_reference,
            conv2d_int16_reference,
        )
        from src.layers.int16_cuda_ext import depthwise_conv3x3_lut_fused_int16

        input_i16 = torch.tensor(
            [
                [
                    [[-300, -200, -100, 0], [100, 200, 300, 400], [-400, -300, -200, -100], [0, 100, 200, 300]],
                    [[200, 100, 0, -100], [-200, -300, -400, -500], [500, 400, 300, 200], [100, 0, -100, -200]],
                ]
            ],
            dtype=torch.int16,
        )
        weight = torch.tensor(
            [
                [[[100, 0, -100], [200, 100, 0], [-100, 0, 100]]],
                [[[0, 100, 0], [100, -200, 100], [0, 100, 0]]],
            ],
            dtype=torch.int16,
        )
        bias = torch.tensor([100, -100], dtype=torch.int32)
        lut = torch.arange(INT16_MIN, INT16_MAX + 1, dtype=torch.int32).to(torch.int16)
        params = Conv2dInt16Params(
            weight=weight,
            bias=bias,
            k2_layer=8192,
            stride=1,
            padding=1,
            groups=2,
        )

        activated = apply_lut_int16_reference(input_i16, lut)
        expected = conv2d_int16_reference(activated, params)
        actual = depthwise_conv3x3_lut_fused_int16(
            input_i16.to(self.device),
            weight.to(self.device),
            bias.to(self.device),
            lut.to(self.device),
            stride=1,
            padding=1,
        ).cpu()

        self.assertTrue(torch.equal(actual, expected.to(torch.int16)))

    def test_add_kernel_saturates_like_reference(self):
        import torch

        from src.layers.int16_backend import add_int16 as add_int16_reference
        from src.layers.int16_cuda_ext import add_int16

        left = torch.tensor([32760, -32760, 10, -10], dtype=torch.int16)
        right = torch.tensor([20, -20, -3, 3], dtype=torch.int16)

        expected = add_int16_reference(left, right)
        actual = add_int16(left.to(self.device), right.to(self.device)).cpu()

        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()

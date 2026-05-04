"""Roundtrip tests for compact INT16 rANS entropy payloads."""

import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CPP_BUILD_DIR = PROJECT_ROOT / "src" / "cpp"
if str(CPP_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(CPP_BUILD_DIR))


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
@unittest.skipIf(
    importlib.util.find_spec("MLCodec_extensions_cpp") is None,
    "rANS extension has not been built",
)
class EntropyRoundtripTest(unittest.TestCase):
    """Validate that compact rANS streams replace tensor-dump payloads."""

    def _make_reference(self, force_zero_thres=None):
        import torch

        from src.layers.int16_backend import Int16QuantConfig
        from src.models.entropy_models import EntropyCoder, GaussianEncoder
        from src.models.int16_reference import Int16GaussianEntropyReference

        quant_cfg = Int16QuantConfig(feature_scale=256)
        entropy_coder = EntropyCoder()
        gaussian = GaussianEncoder()
        gaussian.update(entropy_coder, force_zero_thres=force_zero_thres)
        frozen_state = {"gaussian_encoder": gaussian.export_state()}
        reference = Int16GaussianEntropyReference(frozen_state, quant_cfg=quant_cfg)
        reference.bind_runtime_entropy(entropy_coder, gaussian.cdf_group_index)
        return entropy_coder, reference, quant_cfg, torch.device("cpu")

    @staticmethod
    def _torch_payload_size(payload) -> int:
        import torch

        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return len(buffer.getvalue())

    def test_gaussian_symbols_roundtrip_from_rans_bytes(self):
        import torch

        from src.layers.int16_backend import quantize_scalar_to_int16

        entropy_coder, reference, quant_cfg, device = self._make_reference()
        symbols = torch.tensor(
            [
                -2, -1, 0, 1, 2, -3, 3, 0,
                1, -1, 2, -2, 0, 3, -3, 1,
                0, 0, 1, -1, 2, -2, 3, -3,
                -1, 1, 0, 2, -2, 3, -3, 0,
            ],
            dtype=torch.int16,
        ).reshape(1, 2, 4, 4)
        scales = torch.full_like(
            symbols,
            quantize_scalar_to_int16(1.0, quant_cfg),
            dtype=torch.int16,
        )

        packed_stream = reference.build_indexes_encoder(symbols, scales)
        entropy_coder.reset()
        reference.encode_packed_stream(packed_stream)
        entropy_coder.flush()
        bit_stream = entropy_coder.get_encoded_stream()

        raw_payload_bytes = self._torch_payload_size(
            {"symbols": symbols, "scales": scales, "packed": packed_stream}
        )
        self.assertLess(len(bit_stream), raw_payload_bytes)

        entropy_coder.set_stream(bit_stream)
        decoded = reference.decode_and_get_y_from_stream(scales, device=device)
        self.assertTrue(torch.equal(decoded, symbols))

    def test_force_zero_skip_path_rebuilds_sparse_symbols_from_packed_stream(self):
        import torch

        from src.layers.int16_backend import quantize_scalar_to_int16

        _entropy_coder, reference, quant_cfg, device = self._make_reference(force_zero_thres=0.5)
        low_scale = quantize_scalar_to_int16(0.25, quant_cfg)
        high_scale = quantize_scalar_to_int16(1.0, quant_cfg)
        scales = torch.tensor(
            [
                low_scale, high_scale, low_scale, high_scale,
                high_scale, low_scale, high_scale, low_scale,
            ],
            dtype=torch.int16,
        ).reshape(1, 1, 2, 4)
        symbols = torch.tensor(
            [0, -2, 0, 1, 3, 0, -1, 0],
            dtype=torch.int16,
        ).reshape(1, 1, 2, 4)

        packed_stream = reference.build_indexes_encoder(symbols, scales)
        self.assertEqual(packed_stream.numel(), 4)

        indexes, skip_cond = reference.build_indexes_decoder(scales)
        decoded, decoded_indexes = reference.unpack_stream(
            packed_stream,
            scales.shape,
            skip_cond=skip_cond,
            expected_indexes=indexes,
        )

        self.assertTrue(torch.equal(decoded_indexes.to(device), indexes.to(device)))
        self.assertTrue(torch.equal(decoded, symbols))

    def test_stream_helper_preserves_compact_payload_bytes(self):
        from src.utils.stream_helper import NalType, read_header, read_ip_remaining, write_ip

        bit_stream = bytes([4, 2, 1, 0, 255, 16])
        with tempfile.TemporaryFile() as handle:
            written = write_ip(handle, is_i_frame=False, sps_id=3, qp=21, bit_stream=bit_stream)
            self.assertGreater(written, len(bit_stream))

            handle.seek(0)
            header = read_header(handle)
            qp, restored_stream = read_ip_remaining(handle)

        self.assertEqual(header["nal_type"], NalType.NAL_P)
        self.assertEqual(header["sps_id"], 3)
        self.assertEqual(qp, 21)
        self.assertEqual(restored_stream, bit_stream)

    def test_index_mismatch_reports_reset_frame_context(self):
        import torch

        from src.layers.int16_backend import quantize_scalar_to_int16

        _entropy_coder, reference, quant_cfg, _device = self._make_reference()
        scales = torch.full(
            (1, 1, 1, 2),
            quantize_scalar_to_int16(1.0, quant_cfg),
            dtype=torch.int16,
        )
        symbols = torch.tensor([1, -1], dtype=torch.int16).reshape(scales.shape)
        packed_stream = reference.build_indexes_encoder(symbols, scales)
        expected_indexes, skip_cond = reference.build_indexes_decoder(scales)
        mismatched_indexes = expected_indexes.clone()
        mismatched_indexes[0] += 1

        reference.set_debug_context(
            frame_idx=33,
            stage="decompress_prior_2x",
            stream_idx=0,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"frame_idx=33.*decompress_prior_2x.*stream_idx=0",
        ):
            reference.unpack_stream(
                packed_stream,
                scales.shape,
                skip_cond=skip_cond,
                expected_indexes=mismatched_indexes,
            )


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class PFrameResetStateTest(unittest.TestCase):
    """Guard the P-frame reset state transition that keeps entropy indexes aligned."""

    def test_use_ada_i_reset_rebuilds_matching_encoder_and_decoder_frames(self):
        import torch

        from src.models.int16_reference import DMCInt16Reference, Int16RefFrame

        reference = object.__new__(DMCInt16Reference)
        feature = torch.tensor([[[[7, -3], [2, 5]]]], dtype=torch.int16)
        reference.encoder_dpb = [
            Int16RefFrame(feature=feature.clone(), frame=None, poc=32),
        ]
        reference.decoder_dpb = [
            Int16RefFrame(feature=feature.clone(), frame=None, poc=32),
        ]

        calls = []

        def reconstruct_frame(feature_tensor, qp):
            calls.append((feature_tensor.clone(), qp))
            return (feature_tensor.to(torch.int32) + 1).to(torch.int16)

        reference.reconstruct_frame = reconstruct_frame

        reference.prepare_feature_adaptor_i(last_qp=21)
        reference.reset_ref_feature()

        enc_ref = reference.encoder_dpb[0]
        dec_ref = reference.decoder_dpb[0]
        self.assertIsNone(enc_ref.feature)
        self.assertIsNone(dec_ref.feature)
        self.assertTrue(torch.equal(enc_ref.frame, dec_ref.frame))
        self.assertTrue(DMCInt16Reference._dpb_heads_match(reference))
        self.assertEqual([qp for _feature_tensor, qp in calls], [21, 21])


if __name__ == "__main__":
    unittest.main()

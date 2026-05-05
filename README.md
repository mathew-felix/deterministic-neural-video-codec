# Deterministic Neural Video Codec

A deterministic INT16 runtime for [Microsoft DCVC-RT](https://arxiv.org/abs/2502.20762)
(CVPR 2025) that produces **byte-identical bitstreams** across heterogeneous
NVIDIA hardware (Jetson Orin Nano ARM ↔ x86 Windows workstation).

## Why INT16?

FP16 neural video codecs are fast but non-deterministic across GPU architectures.
Floating-point rounding differences corrupt the entropy coding context in
closed-loop recurrent codecs, causing cascading decoder crashes. This project
replaces all entropy-sensitive arithmetic with signed 16-bit integer operations,
guaranteeing cross-device bitstream consistency at the cost of speed.

## Status

| Area                     | Status                                    |
| :----------------------- | :---------------------------------------- |
| Cross-device determinism | ✅ Proven (Jetson ↔ x86, SHA-256 verified) |
| P-frame encode (laptop)  | ~178–197 ms/frame (~5 fps)               |
| Jetson encode            | ~2,134 ms/frame (0.47 fps)               |
| Quality vs FP16          | −1.28 dB PSNR (37.31 vs 38.59)          |
| INT8 Tensor Core path    | ❌ Abandoned (8.5–9 dB PSNR collapse)     |
| Test suite               | 8 test files, kernel parity + bitstream   |

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.x with CUDA
- FFmpeg (for MP4 ↔ YUV conversion)

### Setup

```powershell
pip install -r requirements.txt
python bootstrap_runtime.py
python build_int16_cuda.py          # optional, enables CUDA acceleration
```

### Encode

```powershell
python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 32 --output_dir outputs
```

### Decode

```powershell
python decode_bin_to_mp4.py --input_bin outputs\<name>.bin
```

### Verify Determinism

```powershell
python tools\compare_bitstreams.py outputs\run_a.bin outputs\run_b.bin --expect_equal
```

### Run Tests

```powershell
# CI smoke sequence

# Compile check
python -m py_compile src\layers\int16_backend.py src\models\int16_reference.py encode_mp4_to_bin.py

# Unit tests (no GPU, no model bundle)
python -m pytest tests\ -x -v --ignore=tests\test_int16_kernels.py -k "not cuda"

# Full test suite (requires CUDA and compiled extensions)
python -m pytest tests\ -x -v
```

## Documentation

- [Architecture](docs/architecture.md) — System design and kernel strategy
- [Determinism](docs/determinism.md) — Tier A determinism contract
- [Calibration](docs/calibration.md) — INT16 bundle calibration workflow
- [Performance](docs/performance.md) — Speed data, stage profiling, optimization status
- [Jetson](docs/jetson.md) — Edge deployment constraints and profiling guide
- [INT8 Pivot](docs/int8_pivot.md) — Why INT8 Tensor Cores were abandoned
- [Validation](docs/validation.md) — Encode/decode verification protocol
- [Release Checklist](docs/release_checklist.md) — Full validation matrix
- [Provenance](docs/provenance.md) — Upstream DCVC-RT attribution

## Architecture

```
source video (MP4)
  → YUV420 preprocessing + padding
  → DMCI I-Frame INT16 path
  → DMC P-Frame INT16 path (encode-only mode)
  → frozen rANS entropy coder
  → deterministic .bin bitstream
       ↕ cross-device transfer
  → rANS decode + INT16 reconstruction
  → output MP4
```

## Key Files

| File                           | Purpose                                 |
| :----------------------------- | :-------------------------------------- |
| `src/layers/int16_kernels.cu`  | Custom INT16 CUDA kernels               |
| `src/layers/int16_backend.py`  | Quantization contract, CUDA/Python path |
| `src/models/int16_reference.py`| I/P frame loops, DPB, profiler          |
| `src/models/entropy_models.py` | Frozen CDF entropy, rANS integration    |
| `encode_mp4_to_bin.py`         | CLI encoder                             |
| `decode_bin_to_mp4.py`         | CLI decoder                             |
| `tools/compare_bitstreams.py`  | SHA-256 / byte equivalence tool         |

## License

See [LICENSE](LICENSE).

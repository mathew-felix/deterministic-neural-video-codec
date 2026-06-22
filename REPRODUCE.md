# Reproducing Results

This guide reproduces the public local validation path for the deterministic
INT16 neural video codec runtime.

The committed measured artifact is:

```text
assets/metrics/video_full_local_2026-05-14.json
```

It records a full encode/decode run on a 2593-frame 1280x720, 30 fps video.
The result is local same-machine validation, not formal cross-device proof.

## Reference Environment

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| Python | 3.12.0 |
| PyTorch | 2.2.2+cu121 |
| PyTorch CUDA runtime | 12.1 |
| NVIDIA driver | 596.21 |
| Input | `video.mp4`, 1280x720, 30 fps, 2593 frames |
| Bundle | `models/int16_bundle_v1.0.0.pt` |

## Required Local Files

These files are intentionally not committed:

```text
models/int16_bundle_v1.0.0.pt
config/config.yaml
video.mp4
outputs/
```

Copy the config template before running:

```bash
cp config/config.example.yaml config/config.yaml
```

On Windows PowerShell:

```powershell
Copy-Item config\config.example.yaml config\config.yaml
```

## Install

```bash
python -m venv .venv
. .venv/bin/activate

python bootstrap_runtime.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python bootstrap_runtime.py
```

Validate the setup:

```bash
python scripts/check_setup.py --input_mp4 video.mp4 --require_config --require_cuda --require_bundle
python -m pytest tests/ -q
```

Expected test result from the measured local environment:

```text
35 passed
```

## Short Smoke Test

Use this for a quick recruiter/demo run:

```bash
python encode_mp4_to_bin.py --input_mp4 video.mp4 --frames 2 --output_dir outputs/video_smoke
python decode_bin_to_mp4.py --input_bin outputs/video_smoke/video_1280x720_30_2f_q32.bin
python tools/compare_bitstreams.py outputs/video_smoke/video_1280x720_30_2f_q32.bin
```

Expected files:

```text
outputs/video_smoke/video_1280x720_30_2f_q32.bin
outputs/video_smoke/video_1280x720_30_2f_q32.json
outputs/video_smoke/video_1280x720_30_2f_q32_decoded.mp4
outputs/video_smoke/video_1280x720_30_2f_q32_decoded_decode.json
```

## Full Validation Run

This reproduces the committed full-video result. It can take a long time on a
laptop GPU.

```bash
python encode_mp4_to_bin.py --input_mp4 video.mp4 --frames 2593 --output_dir outputs/video_full
python decode_bin_to_mp4.py \
  --input_bin outputs/video_full/video_1280x720_30_2593f_q32.bin \
  --output_mp4 outputs/video_full/video_1280x720_30_2593f_q32_decoded.mp4 \
  --keep_yuv
python scripts/quality_metrics.py \
  --reference_mp4 video.mp4 \
  --decoded_mp4 outputs/video_full/video_1280x720_30_2593f_q32_decoded.mp4 \
  --output_json outputs/video_full/quality.json
```

Measured local result:

| Metric | Value |
|---|---:|
| Frames encoded/decoded | 2593 |
| Bitstream bytes | 4,817,350 |
| Bitrate | 445.8789 kbps |
| BPP | 0.016127 |
| Average encode time | 223.5898 ms/frame |
| Average decode time | 207.8080 ms/frame |
| Average RGB PSNR | 28.8969 dB |
| Average RGB MS-SSIM | 0.933624 |

Bitstream SHA-256:

```text
d837ca00cc367c50a63d25ddff19d53a7cc7496cd66099ab73047249c4ad09ed
```

## Same-Machine Determinism Check

```bash
python encode_mp4_to_bin.py --input_mp4 video.mp4 --frames 32 --output_dir outputs/determinism_a
python encode_mp4_to_bin.py --input_mp4 video.mp4 --frames 32 --output_dir outputs/determinism_b
python tools/compare_bitstreams.py \
  outputs/determinism_a/video_1280x720_30_32f_q32.bin \
  outputs/determinism_b/video_1280x720_30_32f_q32.bin \
  --expect_equal
```

Measured local smoke result:

```text
SHA-256 equal: true
Bytewise equal: true
SHA-256: 0373a80d36a63f9327c730ee094a57c4954dcdf33f99a8a37cbe6641acdcaf42
```

## Cross-Device Validation

Cross-device validation should be run before publishing formal multi-GPU
determinism claims. The two machines must use the same equivalence class:

- Same input frames and preprocessing path
- Same INT16 bundle file and SHA-256
- Same QP values, frame count, and reset interval
- Same runtime flags
- Same source revision
- Compatible PyTorch/CUDA/native-extension behavior

Run the same encode on both machines and compare:

```bash
python tools/compare_bitstreams.py \
  outputs/machine_a/video_1280x720_30_32f_q32.bin \
  outputs/machine_b/video_1280x720_30_32f_q32.bin \
  --expect_equal
```

## Troubleshooting

**Missing config**

Copy `config/config.example.yaml` to `config/config.yaml`.

**Missing model bundle**

Download or build `models/int16_bundle_v1.0.0.pt`. The bundle is too large to
commit to git.

**Missing FFmpeg**

Install FFmpeg and make sure both `ffmpeg` and `ffprobe` are on `PATH`.

**Missing native extension**

Run:

```bash
python bootstrap_runtime.py
```

or install the required rANS extension manually:

```bash
python -m pip install --no-build-isolation ./src/cpp
```

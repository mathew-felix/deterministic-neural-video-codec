# Deterministic Neural Video Codec v1.0.0

This release provides the INT16 model bundle required to run the deterministic
neural video codec runtime and demo.

The project compresses an MP4 into a deterministic `.bin` bitstream, decodes it
back to MP4, and records SHA-256 hashes and quality metrics so the result can be
checked.

The runtime is derived from the public Microsoft DCVC / DCVC-RT project family.
This release is independent and is not affiliated with, endorsed by, or
sponsored by Microsoft.

## Included Asset

- `int16_bundle_v1.0.0.pt`

## Verified Bundle

- Size: `201,024,778` bytes
- SHA-256:

```text
c1fc2341d3faf28f16b8e77c0869aecddade674aa0b43be2b64c516f49a8554f
```

## Quick Start

```bash
python scripts/download_models.py --sha256 c1fc2341d3faf28f16b8e77c0869aecddade674aa0b43be2b64c516f49a8554f
python scripts/check_setup.py --input_mp4 video.mp4 --require_config --require_cuda --require_bundle
python encode_mp4_to_bin.py --input_mp4 video.mp4 --frames 2 --output_dir outputs/video_smoke
python decode_bin_to_mp4.py --input_bin outputs/video_smoke/video_1280x720_30_2f_q32.bin
```

## What This Release Enables

- Runs the public INT16 codec runtime.
- Encodes MP4 input into a `.bin` codec bitstream.
- Decodes the `.bin` bitstream back to MP4 for viewing.
- Verifies the downloaded model bundle with SHA-256.
- Supports the README demo, setup checker, CLI workflow, and reproducible local
  validation path.

## Known Limitations

- Requires an NVIDIA CUDA GPU for real encode/decode.
- Not real-time in the current tested setup.
- Native CUDA/C++ extensions require local build tools.
- Deterministic results require the same input, model bundle, runtime flags, and
  compatible CUDA/PyTorch environment.

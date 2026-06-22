# Model Setup

The runtime needs one model file:

```text
models/int16_bundle_v1.0.0.pt
```

The bundle is published in the GitHub Release instead of being tracked in git.
Most users should download the release bundle. Building it from upstream DCVC
checkpoints is only needed if you want to reproduce the model artifact yourself.

## Option A - Download The Release Bundle

Recommended for demos and normal setup:

```bash
python scripts/download_models.py --sha256 c1fc2341d3faf28f16b8e77c0869aecddade674aa0b43be2b64c516f49a8554f
```

This downloads:

```text
models/int16_bundle_v1.0.0.pt
```

Expected bundle:

| Field | Value |
|---|---:|
| Release tag | `v1.0.0` |
| File size | `201,024,778` bytes |
| SHA-256 | `c1fc2341d3faf28f16b8e77c0869aecddade674aa0b43be2b64c516f49a8554f` |

Verify setup:

```bash
python scripts/check_setup.py --input_mp4 video.mp4 --require_config --require_cuda --require_bundle
```

Run a two-frame smoke test:

```bash
python encode_mp4_to_bin.py --input_mp4 video.mp4 --frames 2 --output_dir outputs/video_smoke
python decode_bin_to_mp4.py --input_bin outputs/video_smoke/video_1280x720_30_2f_q32.bin
```

## Option B - Build The Bundle Yourself

Use this path only if you want to rebuild the model artifact from the upstream
Microsoft DCVC-RT checkpoints.

Download the DCVC-RT pretrained checkpoints from:

- Repository: [https://github.com/microsoft/DCVC](https://github.com/microsoft/DCVC)
- Checkpoint instructions:
  [DCVC-family/README.md](https://github.com/microsoft/DCVC/blob/main/DCVC-family/README.md)

You need:

- `cvpr2025_image.pth.tar`
- `cvpr2025_video.pth.tar`

Freeze entropy CDFs:

```bash
python scripts/freeze_entropy_cdfs.py \
  --model_path_i /path/to/cvpr2025_image.pth.tar \
  --model_path_p /path/to/cvpr2025_video.pth.tar \
  --output_path models/frozen_entropy_state.pt
```

Export the INT16 bundle:

```bash
python scripts/export_int16_bundle.py \
  --model_path_i /path/to/cvpr2025_image.pth.tar \
  --model_path_p /path/to/cvpr2025_video.pth.tar \
  --frozen_entropy_path models/frozen_entropy_state.pt \
  --output_path models/int16_bundle_v1.0.0.pt
```

Optional activation calibration can be run if you have representative local
video clips. See [calibration.md](calibration.md) for the detailed engineering
workflow.

## Why The Hash Matters

The encoder and decoder must use the same bundle. The SHA-256 hash lets you
confirm that the downloaded model file is exactly the expected release file.

If the bundle hash differs, deterministic bitstream comparisons are not valid.

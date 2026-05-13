# Deterministic Neural Video Codec

An INT16 runtime for codec engineers, ML compression researchers, and edge-video
engineers who need DCVC-RT-family neural video bitstreams to be reproducible
across NVIDIA GPU platforms.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-downloads)

---

## Table of Contents

1. [Overview](#overview)
2. [Demo](#demo)
3. [Features](#features)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Public Release Scope](#public-release-scope)
8. [Model Setup](#model-setup)
9. [Usage](#usage)
10. [Reproducing Results](#reproducing-results)
11. [Known Limitations](#known-limitations)
12. [Project Structure](#project-structure)
13. [Running Tests](#running-tests)
14. [Contributing](#contributing)
15. [License](#license)

---

## Overview

The upstream DCVC-RT codec uses FP16 arithmetic for fast neural video
compression. FP16 rounding behavior differs across GPU microarchitectures —
small numerical divergences accumulate through the entropy context and corrupt
reference frames when an encoder and decoder run on different GPU hardware. This
makes the FP16 runtime unsuitable for any workflow where the encode and decode
sides may not share the same GPU model.

This project replaces the FP16 inference path with a signed INT16 arithmetic
profile. Fixed power-of-two quantization scales make every multiply-accumulate
operation follow an explicit integer contract. The current local evidence shows
byte-identical same-machine bitstreams; cross-device validation should be run
before publishing broad multi-GPU claims.

The cost is a speed and quality trade-off versus the FP16 runtime. Measure the
FP16 reference under the same clip, QP, frame count, and preprocessing path
before publishing a precise PSNR gap for a release.

Local full-video validation on `video.mp4` encoded and decoded all 2593
1280x720 frames at QP 32, producing a 4,817,350-byte bitstream with SHA-256
`d837ca00cc367c50a63d25ddff19d53a7cc7496cd66099ab73047249c4ad09ed`,
28.8969 dB average RGB PSNR, and 0.933624 average RGB MS-SSIM.

This is not a Microsoft product and is not affiliated with, endorsed by, or
sponsored by Microsoft. See [docs/provenance.md](docs/provenance.md) for
upstream attribution.

---

## Demo

Original `video.mp4` versus decoded reconstruction from the deterministic INT16
codec bitstream:

![Original versus decoded reconstruction](assets/demo_before_after.gif)

Left: original input video. Right: decoded reconstruction.

| Artifact | Size |
|---|---:|
| Original MP4 | 84,089,033 bytes, 80.19 MiB |
| Deterministic codec bitstream | 4,817,350 bytes, 4.59 MiB |
| Decoded preview MP4 | 39,068,735 bytes, 37.26 MiB |

The `.bin` file is the compressed codec output. The decoded MP4 is included so
the reconstruction can be viewed in a normal media player.

| Metric | Value |
|---|---:|
| Size reduction versus original MP4 | 94.27% |
| Compression ratio versus original MP4 | 17.45x smaller |
| Full-video frames tested | 2593 |
| Resolution | 1280x720 |
| FPS | 30 |
| Average RGB PSNR | 28.8969 dB |
| Average RGB MS-SSIM | 0.933624 |

See [docs/demo.md](docs/demo.md) for the commands used to build the GIF and
portfolio MP4.

---

## Features

- Deterministic INT16 encode/decode path with SHA-256 bitstream validation
- Drop-in encode and decode command-line tools for MP4 input and `.bin` output
- SHA-256 bitstream comparison utility for cross-machine validation
- CUDA Graph support for P-frame forward pass acceleration (configurable)
- Preflight check mode that validates environment without running inference
- Per-frame tensor diagnostics via `--log_frame_stats`
- Calibration pipeline to build INT16 bundles from DCVC-RT FP16 checkpoints

---

## Requirements

- **OS:** Windows 10/11 or Linux (Ubuntu 20.04+)
- **Python:** 3.10 or newer. The pinned environment has been smoke-tested with
  Python 3.12.
- **PyTorch:** 2.x with CUDA support. `requirements.txt` installs the CUDA 12.1
  PyTorch wheel used by this project.
- **CUDA Toolkit:** 11.8 or newer for native extension builds. The PyTorch wheel
  provides CUDA runtime libraries, but compiling the local extensions still
  needs a working C++/CUDA build toolchain.
- **FFmpeg:** must be on `PATH` for video I/O
- **GPU:** NVIDIA GPU with compute capability 7.5 or newer (Turing+)
- **Native extensions:** real encode/decode requires the local rANS C++
  extension (`MLCodec_extensions_cpp`). The bootstrap step below installs it.
- **Determinism equivalence class:** deterministic claims require the same input
  frames, model bundle and SHA-256, QP values, frame count, reset interval,
  runtime flags, source revision, PyTorch/CUDA runtime, and native extension
  behavior.

> CPU-only PyTorch installations will pass syntax and unit tests but cannot run
> encode or decode inference.

---

## Installation

Clone the repository and set up a virtual environment. Then run the bootstrap
script; it installs Python dependencies, builds the required rANS extension, and
attempts to build the optional CUDA inference extensions.

```powershell
git clone https://github.com/<your-username>/deterministic-neural-video-codec.git
cd deterministic-neural-video-codec

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python bootstrap_runtime.py
```

On Linux or Jetson:

```bash
git clone https://github.com/<your-username>/deterministic-neural-video-codec.git
cd deterministic-neural-video-codec

python3 -m venv .venv
. .venv/bin/activate

python bootstrap_runtime.py
```

If you need to install pieces manually, use this order:

```powershell
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation .\src\cpp
python -m pip install --no-build-isolation .\src\layers\extensions\inference
python build_int16_cuda.py
```

After installation, run the setup validator:

```powershell
python scripts/check_setup.py --input_mp4 video.mp4 --require_cuda --require_bundle
```

The `src/cpp` install is required for real encode/decode. The inference CUDA
extension is optional; if it fails to build, the codec can fall back to the
PyTorch implementation, but it will be slower.

---

## Configuration

All scripts read from a single config file at `config/config.yaml`. Copy the
template:

```powershell
Copy-Item config\config.example.yaml config\config.yaml
```

On Linux:

```bash
cp config/config.example.yaml config/config.yaml
```

For downloaded INT16 bundle usage, edit these values:

```yaml
models:
  bundle: models/int16_bundle_v1.0.0.pt

encode:
  input_mp4: video.mp4
  frames: 2
  output_dir: outputs/video_smoke
  device: "cuda:0"
```

If you are building the bundle from Microsoft DCVC-RT checkpoints, also edit:

```yaml
models:
  checkpoint_i: models/cvpr2025_image.pth.tar
  checkpoint_p: models/cvpr2025_video.pth.tar

calibration:
  device: "cuda:0"
```

Every script reads this file automatically — no need to pass long argument
lists on every run. Any argument can still be overridden on the command line;
the priority is always **CLI flag > config/config.yaml > built-in default**.

`config/config.yaml` is gitignored so your local paths never end up in the repository.

---

## Model Setup

The INT16 model bundle (`models/int16_bundle_v1.0.0.pt`) is
not tracked in git. There are two ways to get it.

### Option A — Download from the GitHub release

```bash
python scripts/download_models.py
```

Verify it loaded correctly:

```bash
python encode_mp4_to_bin.py --input_mp4 video.mp4 --frames 2 --check_only
```

### Option B — Build from scratch

Use this if you want to produce the bundle yourself from the upstream Microsoft
DCVC-RT checkpoint. After setting up `config/config.yaml`, place your calibration
videos in `calibrate_videos/` and run:

```bash
python pipeline.py
```

This runs all four build steps in sequence with progress bars:

| Step | Script | What it does |
|---|---|---|
| 1 `freeze` | `scripts/freeze_entropy_cdfs.py` | Freeze rANS CDF tables from the FP16 checkpoint |
| 2 `export` | `scripts/export_int16_bundle.py` | Quantize weights to INT16, pack INT8 shadows |
| 3 `manifest` | `scripts/build_manifest.py` | Scan `calibrate_videos/` and write the manifest |
| 4 `calibrate` | `scripts/calibrate_int16_bundle.py` | Run clips, refine per-layer activation scales |

Run a single step if needed:

```bash
python pipeline.py --step calibrate
```

See [docs/model_setup.md](docs/model_setup.md) for clip selection guidelines
and calibration recommendations.

---

## Usage

With `config/config.yaml` set up, most runs need no arguments at all.

### Encode

```bash
python encode_mp4_to_bin.py
```

The input video, QP, output directory, and device are all read from
`config/config.yaml`. Override any value on the command line when needed:

```bash
python encode_mp4_to_bin.py --input_mp4 myvideo.mp4 --frames 64 --qp_p 21
```

Expected output:

```
outputs/
  test_1280x720_30_32f_q32.bin
  test_1280x720_30_32f_q32.json
```

### Decode

```bash
python decode_bin_to_mp4.py --input_bin outputs/test_1280x720_30_32f_q32.bin
```

### Compare Two Bitstreams

Run the same encode on two machines and verify the output is byte-identical:

```bash
python tools/compare_bitstreams.py \
  outputs/machine_a/test_1280x720_30_32f_q32.bin \
  outputs/machine_b/test_1280x720_30_32f_q32.bin \
  --expect_equal
```

### Preflight Check

Validate the local tools, extensions, CUDA visibility, bundle, config, and input
MP4 before running a full encode:

```bash
python scripts/check_setup.py --input_mp4 video.mp4 --require_cuda --require_bundle
```

Then validate that the bundle loads and the runtime flags are accepted without
running a full encode:

```bash
python encode_mp4_to_bin.py --input_mp4 video.mp4 --frames 2 --check_only
```

Then run a short end-to-end encode/decode smoke test:

```bash
python encode_mp4_to_bin.py --input_mp4 video.mp4 --frames 2 --output_dir outputs/video_smoke
python decode_bin_to_mp4.py --input_bin outputs/video_smoke/video_1280x720_30_2f_q32.bin
```

Compute quality metrics for a decoded MP4:

```bash
python scripts/quality_metrics.py \
  --reference_mp4 video.mp4 \
  --decoded_mp4 outputs/video_smoke/video_1280x720_30_2f_q32_decoded.mp4 \
  --output_json outputs/video_smoke/quality.json
```

### CUDA Graph Acceleration

Set `enable_pframe_graphs: true` in `config/config.yaml` under `encode:`, or pass
the flag explicitly:

```bash
python encode_mp4_to_bin.py --enable_pframe_graphs
```

> See [docs/jetson.md](docs/jetson.md) for memory constraints on small edge
> devices before enabling graphs.

---

## Reproducing Results

To validate determinism across two machines:

1. Complete [Installation](#installation) and [Configuration](#configuration)
   on both machines. Ensure both use the same bundle by checking its SHA-256:
   ```bash
   python scripts/download_models.py --sha256 <digest-from-release-notes>
   ```

2. Encode a clip on machine A:
   ```bash
   python encode_mp4_to_bin.py --output_dir outputs/run_a
   ```

3. Encode the same clip on machine B:
   ```bash
   python encode_mp4_to_bin.py --output_dir outputs/run_b
   ```

4. Transfer one `.bin` to the other machine and compare:
   ```bash
   python tools/compare_bitstreams.py \
     outputs/run_a/*.bin outputs/run_b/*.bin --expect_equal
   ```

The bitstreams must be byte-identical. Any mismatch indicates a bundle version
mismatch, a flags difference, or a runtime version difference. See
[docs/validation.md](docs/validation.md) for the full validation checklist.

---

## Known Limitations

- Real encode/decode requires an NVIDIA CUDA GPU, FFmpeg, the rANS C++
  extension, and a local INT16 model bundle.
- The INT16 path is built for deterministic auditability, not maximum FP16
  rate-distortion performance. Current documentation uses a measured local
  INT16 run; publish a matching FP16 run before making a precise quality-gap
  claim for a new release.
- Jetson Orin Nano-class devices are useful edge validation targets, but should
  not be presented as real-time 30 fps encode targets for this runtime.
- Cross-device determinism claims are valid only inside the equivalence class:
  same input frames, bundle SHA-256, QP, frame count, reset interval, flags,
  runtime versions, and source revision.
- Model bundles, large media files, and generated bitstreams are not tracked in
  git; release artifacts must be hashed and distributed separately.

---

## Project Structure

```text
deterministic-neural-video-codec/
├── .env.example               optional env-var overrides template
├── .github/
│   ├── workflows/ci.yml       compile + non-CUDA test CI
│   ├── ISSUE_TEMPLATE/        bug and feature templates
│   └── PULL_REQUEST_TEMPLATE.md
├── Makefile                   one-command shortcuts for setup/test/pipeline
├── README.md
├── REPRODUCE.md               step-by-step guide to reproduce published results
├── LICENSE
├── NOTICE
├── CONTRIBUTING.md
├── CHANGELOG.md
├── requirements.txt
├── config/
│   ├── config.example.yaml       config template — copy to config.yaml and edit
│   └── README.md                 config usage notes
├── pipeline.py                one-command full model build pipeline
├── encode_mp4_to_bin.py       command-line encoder (MP4 → .bin)
├── decode_bin_to_mp4.py       command-line decoder (.bin → MP4)
├── bootstrap_runtime.py       one-time runtime dependency setup
├── build_int16_cuda.py        builds optional native CUDA extension
├── src/
│   ├── config_loader.py       shared YAML config loader (used by all scripts)
│   ├── layers/                INT16 quantization kernels and backend
│   ├── models/                INT16 reference encoder and decoder models
│   ├── entropy/               entropy model wrappers
│   └── utils/                 video I/O, transforms, profiling utilities
├── tests/                     unit, parity, packaging, and bitstream tests
├── tools/
│   ├── compare_bitstreams.py  SHA-256 and byte-for-byte bitstream comparison
│   └── benchmark_report_style.py  structured benchmark report formatter
├── scripts/
│   ├── download_models.py         download INT16 bundle from GitHub release
│   ├── build_manifest.py          scan calibrate_videos/ and write manifest JSON
│   ├── freeze_entropy_cdfs.py     freeze rANS CDF tables from FP16 checkpoint
│   ├── export_int16_bundle.py     quantize weights and pack INT8 shadows
│   └── calibrate_int16_bundle.py  refine activation scales from calibration clips
├── docs/
│   ├── model_setup.md         how to obtain or build the INT16 bundle
│   ├── architecture.md        system design and INT16 arithmetic contract
│   ├── calibration.md         calibration workflow and clip selection guide
│   ├── validation.md          cross-device bitstream validation procedure
│   ├── jetson.md              Jetson Orin deployment notes and constraints
│   ├── determinism.md         determinism scope and equivalence class definition
│   └── provenance.md          upstream attribution and project identity
├── assets/
│   ├── manifests/             calibration manifest templates
│   └── metrics/               reproducibility metric examples
├── calibrate_videos/          local calibration clips (not tracked in git)
└── models/                    local model bundles (not tracked in git)
```

---

## Running Tests

Run the full non-CUDA test suite:

```powershell
python -m pytest tests\ -x -v --ignore=tests\test_int16_kernels.py -k "not cuda"
```

Run syntax and import checks:

```powershell
python -m py_compile `
  src\layers\int16_backend.py `
  src\models\int16_reference.py `
  encode_mp4_to_bin.py `
  decode_bin_to_mp4.py
```

Run CUDA tests after building the native extension:

```powershell
python -m pytest tests\ -x -v
```

---

## Setup Troubleshooting

**`ModuleNotFoundError: No module named 'MLCodec_extensions_cpp'`**

The rANS C++ extension was not installed. Run:

```powershell
python -m pip install --no-build-isolation .\src\cpp
```

or rerun the full bootstrap:

```powershell
python bootstrap_runtime.py
```

**`cannot import cuda implementation for inference, fallback to pytorch`**

The optional CUDA inference extension is not installed. The codec can still run
with the PyTorch fallback, but it will be slower. To build the extension:

```powershell
python -m pip install --no-build-isolation .\src\layers\extensions\inference
```

**`config/config.yaml not found`**

Copy the template and edit the input video, model bundle, frame count, and
device:

```powershell
Copy-Item config\config.example.yaml config\config.yaml
```

**`Input MP4 not found: test.mp4`**

The default config uses `test.mp4`. Either rename your clip, edit
`config/config.yaml`, or pass the input explicitly:

```powershell
python encode_mp4_to_bin.py --input_mp4 video.mp4 --frames 2 --check_only
```

**Native extension build fails**

Install a C++ compiler, CUDA Toolkit, and matching PyTorch CUDA wheel. On
Windows this usually means Visual Studio Build Tools plus the CUDA Toolkit. On
Linux, install the CUDA Toolkit and a compatible `g++`.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow, code
style requirements, and determinism contract that all changes must preserve.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE)
for the full text.

Portions of this work are derived from the
[Microsoft DCVC](https://github.com/microsoft/DCVC) project, copyright
Microsoft Corporation, licensed under the MIT License. See
[NOTICE](NOTICE) for the full upstream attribution.

# Deterministic Neural Video Codec

An INT16 runtime for a DCVC-RT-family neural video codec that produces
byte-identical bitstreams across different NVIDIA GPU architectures.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-downloads)

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Public Release Scope](#public-release-scope)
7. [Model Setup](#model-setup)
8. [Usage](#usage)
9. [Reproducing Results](#reproducing-results)
10. [Project Structure](#project-structure)
11. [Running Tests](#running-tests)
12. [Contributing](#contributing)
13. [License](#license)

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
operation produce the same integer result regardless of the GPU used, giving
bit-exact bitstreams and lossless cross-device decoding.

The cost is a modest speed reduction and a ~1.3 dB PSNR gap versus FP16. The
trade-off is acceptable when reproducibility across hardware matters more than
peak compression efficiency.

This is not a Microsoft product and is not affiliated with, endorsed by, or
sponsored by Microsoft. See [docs/provenance.md](docs/provenance.md) for
upstream attribution.

---

## Features

- Bit-exact encode/decode across tested NVIDIA GPU generations (Turing, Ampere, Ada)
- Drop-in encode and decode command-line tools for MP4 input and `.bin` output
- SHA-256 bitstream comparison utility for cross-machine validation
- CUDA Graph support for P-frame forward pass acceleration (configurable)
- Preflight check mode that validates environment without running inference
- Per-frame tensor diagnostics via `--log_frame_stats`
- Calibration pipeline to build INT16 bundles from DCVC-RT FP16 checkpoints

---

## Requirements

- **OS:** Windows 10/11 or Linux (Ubuntu 20.04+)
- **Python:** 3.10 or newer
- **PyTorch:** 2.x with CUDA support (`torch`, `torchvision`)
- **CUDA Toolkit:** 11.8 or newer (for building the optional native extension)
- **FFmpeg:** must be on `PATH` for video I/O
- **GPU:** NVIDIA GPU with compute capability 7.5 or newer (Turing+)

> CPU-only PyTorch installations will pass syntax and unit tests but cannot run
> encode or decode inference.

---

## Installation

Clone the repository and set up a virtual environment:

```powershell
git clone https://github.com/<your-username>/deterministic-neural-video-codec.git
cd deterministic-neural-video-codec

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python bootstrap_runtime.py
python build_int16_cuda.py
```

On Linux or Jetson:

```bash
git clone https://github.com/<your-username>/deterministic-neural-video-codec.git
cd deterministic-neural-video-codec

python3 -m venv .venv
. .venv/bin/activate

pip install -r requirements.txt
python bootstrap_runtime.py
python build_int16_cuda.py
```

---

## Configuration

All scripts read from a single config file at `config/config.yaml`. Copy the
template and edit the two lines that point to your local DCVC-RT checkpoints:

```bash
cp config/config.example.yaml config/config.yaml
```

Minimum edits inside `config/config.yaml`:

```yaml
models:
  checkpoint_i: models/cvpr2025_image.pth.tar   # path to Microsoft I-frame checkpoint
  checkpoint_p: models/cvpr2025_video.pth.tar   # path to Microsoft P-frame checkpoint

calibration:
  device: "cuda:0"   # GPU to use for calibration and inference
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
python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 2 --check_only
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

Validate that the bundle loads and the runtime is functional without running
a full encode:

```bash
python encode_mp4_to_bin.py --check_only
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

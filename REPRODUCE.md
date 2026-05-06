# Reproducing Results

This document describes the exact steps to reproduce the published results for
the Deterministic Neural Video Codec INT16 runtime. Follow it on a clean
machine to confirm that encoding produces byte-identical bitstreams and matches
the reported PSNR and bitrate figures.

---

## Reference Environment

Results were produced on the following hardware and software:

| Component | Value |
|---|---|
| GPU | NVIDIA (Turing or Ampere architecture) |
| CUDA Toolkit | 12.1 |
| PyTorch | 2.2.2+cu121 |
| Python | 3.12 |
| OS | Windows 10 / Ubuntu 20.04 |
| Resolution | 1280 × 720 |
| Frame rate | 30 fps |
| QP (I / P) | 32 / 32 |

Cross-device validation was performed by encoding on one GPU architecture and
decoding on a different GPU architecture. Both runs produced the same SHA-256
bitstream digest.

Reference metrics are in `assets/metrics/validation_results.example.json` and
`assets/metrics/validation_cross_device.example.csv`. Replace the placeholder SHA-256
values with the digest from your own run.

---

## Required Files

Before running anything, confirm these files are in place:

```
deterministic-neural-video-codec/
├── models/
│   └── int16_bundle_v1.0.0.pt          ← INT16 calibrated bundle (see Step 2)
├── config.yaml                          ← copied from config.example.yaml
└── test.mp4                             ← any 1280×720 30fps test clip
```

`models/` and `test.mp4` are gitignored and must be obtained separately.
See the naming convention in `models/README.md`.

---

## Step 1 — Install Dependencies

```bash
git clone https://github.com/felixmathew/deterministic-neural-video-codec.git
cd deterministic-neural-video-codec

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1

pip install -r requirements.txt
python bootstrap_runtime.py
python build_int16_cuda.py
```

Verify the install:

```bash
python -m py_compile src/layers/int16_backend.py src/models/int16_reference.py
python -m pytest tests/ -x -v --ignore=tests/test_int16_kernels.py -k "not cuda"
```

Expected: all tests pass.

---

## Step 2 — Obtain the Model Bundle

**Option A — Download from the GitHub release (recommended):**

```bash
python scripts/download_models.py --sha256 <digest-from-release-notes>
```

Place the downloaded file at `models/int16_bundle_v1.0.0.pt` and update
`config.yaml` to point to it:

```yaml
models:
  bundle: models/int16_bundle_v1.0.0.pt
```

**Option B — Build from scratch:**

1. Download the DCVC-RT FP16 checkpoints from
   [github.com/microsoft/DCVC](https://github.com/microsoft/DCVC):
   - `cvpr2025_image.pth.tar` → `models/cvpr2025_image.pth.tar`
   - `cvpr2025_video.pth.tar` → `models/cvpr2025_video.pth.tar`

2. Copy and edit the config:
   ```bash
   cp config.example.yaml config.yaml
   # Set models.checkpoint_i and models.checkpoint_p to the paths above
   ```

3. Place at least 6 calibration clips (one per content type) in
   `calibrate_videos/`. See `assets/manifests/calibration_manifest.example.json`
   for the required content types and recommended weights.

4. Run the full pipeline:
   ```bash
   python pipeline.py
   ```
   This runs freeze → export → manifest → calibrate in sequence.
   Runtime: approximately 30 minutes per 300 frames per calibration clip
   on a mid-range GPU.

5. The output bundle is written to `models/` using the name set in
   `config.yaml` under `models.bundle`.

---

## Step 3 — Set Up config.yaml

```bash
cp config.example.yaml config.yaml
```

Minimum required edits:

```yaml
models:
  bundle: models/int16_bundle_v1.0.0.pt   # path to your bundle

encode:
  input_mp4: test.mp4
  frames: 300
  qp_i: 32
  qp_p: 32
  output_dir: outputs/reproduce
  device: "cuda:0"
```

---

## Step 4 — Encode the Reference Clip

```bash
python encode_mp4_to_bin.py
```

Expected output in `outputs/reproduce/`:

```
test_1280x720_30_300f_q32.bin
test_1280x720_30_300f_q32.json
```

The JSON sidecar contains per-frame metrics, total bits, and the bundle
SHA-256 used for the encode.

---

## Step 5 — Verify the Bitstream

Print the SHA-256 of your encoded bitstream:

```bash
python -c "
import hashlib, pathlib, glob
files = sorted(glob.glob('outputs/reproduce/*.bin'))
for f in files:
    d = pathlib.Path(f).read_bytes()
    print(hashlib.sha256(d).hexdigest(), f)
"
```

Compare this digest against the reference digest published in the GitHub
release notes. They must match exactly on any supported GPU.

---

## Step 6 — Cross-Device Validation (two machines)

To validate that encode on GPU A decodes correctly on GPU B:

**Machine A — encode:**
```bash
python encode_mp4_to_bin.py --output_dir outputs/machine_a
```

**Machine B — copy the `.bin` file, then decode:**
```bash
python decode_bin_to_mp4.py --input_bin outputs/machine_a/test_1280x720_30_300f_q32.bin
```

**Either machine — compare bitstreams:**
```bash
python tools/compare_bitstreams.py \
  outputs/machine_a/test_1280x720_30_300f_q32.bin \
  outputs/machine_b/test_1280x720_30_300f_q32.bin \
  --expect_equal
```

Expected: `MATCH — bitstreams are byte-identical`.

---

## Step 7 — Check Clamp Health

Run with frame diagnostics to verify no activation overflow occurred:

```bash
python encode_mp4_to_bin.py --log_frame_stats --output_dir outputs/health_check
```

A healthy bundle produces zero clamp overflow events. Non-zero values mean
the calibration clips did not cover the dynamic range of your test content.
Re-calibrate with more representative clips if this occurs.

---

## Expected Metrics

These are the reference values for QP 32 on 1280×720 30fps content:

| Metric | Value |
|---|---|
| PSNR (INT16) | ~37.31 dB |
| PSNR (FP16 reference) | ~38.59 dB |
| PSNR gap | ~1.28 dB |
| Bitstream SHA-256 | see release notes |
| Cross-device match | byte-identical |

The ~1.28 dB PSNR gap versus FP16 is a known, permanent characteristic of
post-hoc INT16 calibration. It is the trade-off for deterministic
cross-device bitstreams. Closing the gap requires Quantization-Aware Training.

---

## Troubleshooting

**`models/int16_bundle_v1.0.0.pt` not found**
Run `python scripts/download_models.py` or check `config.yaml` bundle path.

**`config.yaml not found`**
Run `cp config.example.yaml config.yaml` and fill in model paths.

**CUDA out of memory during calibration**
Reduce `frames_per_clip` in `config.yaml` or use fewer calibration clips.

**Bitstream does not match reference**
Check that `config.yaml` points to the exact same bundle version and that
no flags differ between runs (`qp_i`, `qp_p`, `reset_interval`, `frames`).

**Tests fail after install**
Rebuild the native extension: `python build_int16_cuda.py`

# Model Setup

The INT16 runtime requires a calibrated model bundle (`int16_bundle_v1.0.0.pt`)
that is **not tracked in git**. There are two ways to get it:

- **Option A (recommended):** Download from the GitHub release — one command,
  no extra setup required.
- **Option B:** Build from scratch from the upstream Microsoft DCVC-RT checkpoint
  — needed if you want to reproduce or re-calibrate the bundle yourself.

---

## Option A — Download from GitHub Release

```bash
python scripts/download_models.py
```

This downloads the INT16 bundle from the project's GitHub release, saves it to
`models/`, and prints its SHA-256 digest. Pass `--sha256 <digest>` to verify
integrity against a known hash:

```bash
python scripts/download_models.py --sha256 <hex-digest>
```

After downloading, verify the runtime loads it:

```bash
python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 2 --check_only
```

---

## Option B — Build from Scratch

Use this path if you want to reproduce the INT16 bundle from the upstream
Microsoft checkpoint, or if you want to run your own calibration.

#### Step 1 — Obtain the DCVC-RT Checkpoint

Download the DCVC-RT pretrained checkpoints from the official Microsoft DCVC
repository:

- Repository: [https://github.com/microsoft/DCVC](https://github.com/microsoft/DCVC)
- Checkpoint download links are in
  [DCVC-family/README.md](https://github.com/microsoft/DCVC/blob/main/DCVC-family/README.md)

You need two `.pth.tar` files:
- `cvpr2025_image.pth.tar` — I-frame model checkpoint
- `cvpr2025_video.pth.tar` — P-frame (video) model checkpoint

Place both files anywhere on your local machine; the paths are passed as
arguments in Step 2.

---

### Step 2 — Build the INT16 Bundle

The bundle export pipeline has three stages. Run them in order from the project
root with your virtual environment active.

#### 2a. Freeze Entropy CDFs

```bash
python scripts/freeze_entropy_cdfs.py \
  --model_path_i /path/to/cvpr2025_image.pth.tar \
  --model_path_p /path/to/cvpr2025_video.pth.tar \
  --output_path models/frozen_entropy_state.pt
```

On Windows (PowerShell):

```powershell
python scripts/freeze_entropy_cdfs.py `
  --model_path_i C:\path\to\cvpr2025_image.pth.tar `
  --model_path_p C:\path\to\cvpr2025_video.pth.tar `
  --output_path models/frozen_entropy_state.pt
```

This captures the rANS CDF tables so they remain identical across all future
encode and decode sessions.

#### 2b. Export the INT16 Bundle

```bash
python scripts/export_int16_bundle.py \
  --model_path_i /path/to/cvpr2025_image.pth.tar \
  --model_path_p /path/to/cvpr2025_video.pth.tar \
  --frozen_entropy_path models/frozen_entropy_state.pt \
  --output_path models/int16_bundle_v1.0.0.pt
```

This quantizes all weights to INT16 and embeds the frozen entropy state.

#### 2c. Activation Calibration (optional but recommended)

If you have representative video clips, run activation calibration to refine
the per-layer INT8 channel scales:

```bash
python scripts/calibrate_int16_bundle.py \
  --manifest assets/manifests/calibration_manifest.example.json \
  --bundle_path models/int16_bundle_v1.0.0.pt \
  --output models/int16_bundle_v1.0.0.pt \
  --frames_per_clip 300 \
  --qp 32
```

Edit `assets/manifests/calibration_manifest.example.json` to point to your
local clips before running. See [calibration.md](calibration.md) for a full
guide including clip selection, clamp health checks, and known limitations.

---

## Verify the Bundle

After completing the steps above, run the preflight check to confirm the bundle
loads correctly without running a full encode:

```powershell
python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 2 --check_only
```

Expected output:

```
[preflight] bundle loaded: models/int16_bundle_v1.0.0.pt
[preflight] OK
```

Then run a short smoke encode to verify end-to-end output:

```powershell
python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 32 --output_dir outputs/smoke
python decode_bin_to_mp4.py --input_bin outputs/smoke/<bitstream>.bin
```

---

## Bundle Integrity

Record the SHA-256 digest of your bundle and check it into a sidecar file. This
lets you confirm that both encoder and decoder machines are using the same
weights:

```powershell
python -c "import hashlib, pathlib; d=pathlib.Path('models/int16_bundle_v1.0.0.pt').read_bytes(); print(hashlib.sha256(d).hexdigest())"
```

Store the printed hash in `models/bundle_sha256.txt` and commit that file.
If the hash does not match between two machines, the bitstreams they produce
will not be byte-identical.

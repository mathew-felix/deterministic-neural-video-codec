# Calibration Workflow

## Overview

The INT16 runtime uses a calibrated model bundle to map FP16 network weights and
activations to a signed 16-bit integer grid. This calibration process determines
the quantization scales that minimize reconstruction error while preserving the
deterministic bitstream contract.

The current production bundle is `int16_bundle_v1.0.0.pt`.

## Quantization Contract

All INT16 arithmetic uses power-of-two scales defined in `Int16QuantConfig`:

| Parameter        | Default | Purpose                                      |
| :--------------- | :------ | :------------------------------------------- |
| `feature_scale`  | 512     | Maps floating-point activations to INT16      |
| `weight_scale`   | 8192    | Maps floating-point weights to INT16          |
| `bias_scale`     | 4194304 | `feature_scale × weight_scale`               |

These scales are frozen at export time and must not change between encoder and
decoder. Any mismatch causes entropy context corruption and decoder crashes.

## Bundle Export Pipeline

```
FP16 DCVC-RT checkpoint (.pth)
  → scripts/freeze_entropy_cdfs.py       (freeze rANS CDF tables)
  → scripts/export_int16_bundle.py       (quantize weights, pack INT8 shadows)
  → scripts/calibrate_int16_bundle.py    (collect activation statistics, refine scales)
  → int16_bundle_v1.0.0.pt
```

### Step 1: Freeze Entropy CDFs

```powershell
python scripts/freeze_entropy_cdfs.py `
  --model_path_i <i_frame_checkpoint.pth> `
  --model_path_p <p_frame_checkpoint.pth> `
  --output_path models/frozen_entropy_state.pt
```

This captures the Gaussian entropy model's CDF tables so they remain identical
across all future encode/decode sessions.

### Step 2: Export INT16 Bundle

```powershell
python scripts/export_int16_bundle.py `
  --model_path_i <i_frame_checkpoint.pth> `
  --model_path_p <p_frame_checkpoint.pth> `
  --frozen_entropy_path models/frozen_entropy_state.pt `
  --output_path models/int16_bundle_v1.0.0.pt
```

This quantizes all weights to INT16 using `weight_scale=8192`, packs INT8
shadow copies for eligible 1×1 convolutions, and embeds the frozen entropy state.

### Step 3: Activation Calibration

```powershell
python scripts/calibrate_int16_bundle.py `
  --manifest assets/manifests/calibration_manifest.example.json `
  --bundle_path models/int16_bundle_v1.0.0.pt `
  --output models/int16_reference_bundle_v5_calibrated.pt `
  --frames_per_clip 300 `
  --qp 32
```

This runs representative clips through the INT16 encoder and collects per-layer
activation statistics. The weighted 99.9th-percentile aggregation determines
activation clamp boundaries that prevent overflow without compressing the useful
dynamic range.

Before launching a long calibration sweep, validate the manifest and bundle
paths without running inference:

```powershell
python scripts/calibrate_int16_bundle.py `
  --manifest assets/manifests/calibration_manifest.example.json `
  --bundle_path models/int16_bundle_v1.0.0.pt `
  --dry_run
```

The example manifest is a schema and content-distribution template. Replace the
`clips/*.mp4` paths with local calibration clips; the repository intentionally
does not track source media.

## Clamp Health Monitoring

After calibration, verify that the bundle does not produce any clamp overflows
during normal encoding:

```powershell
python encode_mp4_to_bin.py `
  --input_mp4 test.mp4 `
  --frames 300 `
  --log_frame_stats `
  --output_dir outputs/calibration_check
```

The `--log_frame_stats` flag enables per-frame tensor diagnostics. A healthy
bundle produces:

| Metric       | Expected Value |
| :----------- | :------------- |
| `dmci_total` | 0              |
| `dmc_total`  | 0              |

Non-zero values indicate that activations exceeded the INT16 range during
quantization. This does not break determinism, but it degrades reconstruction
quality and may cause strict PSNR gate failures.

## Calibration Clip Selection

A production calibration sweep should cover representative content instead of a
single easy clip:

| Content Type              | Target Property                           |
| :------------------------ | :---------------------------------------- |
| Outdoor daylight          | High dynamic range, strong gradients      |
| Indoor studio             | Low noise, predictable activations        |
| Sports / fast motion      | Large inter-frame differences, high entropy |
| Night / dark scene        | Low-light outlier activation tails        |
| Talking head / screen     | Uniform regions, low inter-frame entropy  |
| Animation                 | Saturated colors, sharp edges, no grain   |

For each type, collect activation statistics over at least 300 frames at the
target QP. Use weighted 99.9th-percentile aggregation with weights matched to
the intended deployment distribution.

## Known Limitations

- **Low-light PSNR gate failures:** Night-scene clips produce outlier activation
  tails that fail strict automated PSNR numeric gates, even though visual
  reconstruction quality is intact. This is a fundamental limitation of post-hoc
  calibration on recurrent architectures.

- **Quality gap vs FP16:** INT16 averages ~37.31 dB PSNR vs FP16's ~38.59 dB,
  a ~1.28 dB permanent deficit. Closing this gap requires Quantization-Aware
  Training (QAT), not better calibration.

- **Bundle version lock:** Changing the calibrated bundle changes the bitstream.
  Both encoder and decoder must use the same bundle version for Tier A
  determinism. Document the bundle SHA-256 in sidecar metadata.

## Evidence Artifacts

Small, reviewable examples live under source control:

| Artifact | Purpose |
| :------- | :------ |
| `assets/manifests/calibration_manifest.example.json` | Six-content-type calibration manifest template |
| `assets/metrics/validation_cross_device.example.csv` | Cross-device validation table shape |

Large calibration clips, raw YUV files, checkpoints, and generated bundles stay
outside git. Track only sidecar summaries, SHA-256 digests, and manifest
metadata needed to reproduce a result.

# Validation

## Commit 14 Runtime Packaging

Commit 14 packages the deterministic INT16 runtime as a standalone command-line
surface:

- `encode_mp4_to_bin.py` converts MP4 input to the deterministic `.bin`
  container when a calibrated INT16 bundle is present.
- `decode_bin_to_mp4.py` decodes a generated `.bin` back to MP4 and writes
  decode metrics next to the output.
- `bootstrap_runtime.py` installs Python dependencies, builds the rANS
  extension, and attempts optional CUDA extension preloading.
- `tools/compare_bitstreams.py` remains the Tier A byte/SHA-256 equivalence
  checker.
- `scripts/download_models.*` documents and automates local model placement
  without tracking checkpoints in git.

## Required Local Assets

The repository intentionally does not track videos, model bundles, generated
bitstreams, YUV files, or decoded MP4s. Full encode/decode validation requires:

```text
models/int16_reference_bundle_v2_calibrated.pt
```

The local smoke clip is also ignored by git. Use `--check_only` to validate the
MP4 and effective runtime flags when the model bundle is absent:

```powershell
python encode_mp4_to_bin.py --input_mp4 .\test.mp4 --frames 2 --check_only
```

## Full Runtime Check

When the model bundle is available:

```powershell
python encode_mp4_to_bin.py --input_mp4 .\test.mp4 --frames 2 --output_dir outputs\smoke
python decode_bin_to_mp4.py --input_bin outputs\smoke\<name>.bin
python tools\compare_bitstreams.py outputs\smoke\<run_a>.bin outputs\smoke\<run_b>.bin --expect_equal
```

The encode sidecar JSON records an `equivalence_class` object containing git
metadata, PyTorch/CUDA device metadata, relevant `DCVC_*` environment values,
model-bundle checksum metadata, runtime flags, and video/QP settings.

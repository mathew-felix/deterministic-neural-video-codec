# Validation

## Runtime Packaging

The deterministic INT16 runtime is packaged as a standalone command-line
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

## Same-Machine Determinism Check

Two identical encodes on the same machine must produce the same bitstream:

```powershell
python encode_mp4_to_bin.py --input_mp4 .\test.mp4 --frames 32 --output_dir outputs\run_a
python encode_mp4_to_bin.py --input_mp4 .\test.mp4 --frames 32 --output_dir outputs\run_b
python tools\compare_bitstreams.py outputs\run_a\*.bin outputs\run_b\*.bin --expect_equal
```

Expected result: `all_sha256_equal: true`, exit code 0.

## Cross-Device Determinism Protocol

This is the core validation that justifies the INT16 approach.

### Step 1: Encode on Device A (e.g., Jetson)

```bash
python encode_mp4_to_bin.py \
    --input_mp4 test.mp4 \
    --frames 1390 \
    --output_dir outputs/jetson_run
```

Save the output `.bin` file and its sidecar `.json` (which contains the
`bitstream_sha256`).

### Step 2: Transfer Bitstream to Device B (e.g., x86 laptop)

Copy only the `.bin` file. Do not copy the model bundle or the MP4 — Device B
must use its own local copy of the same bundle version.

### Step 3: Decode on Device B

```powershell
python decode_bin_to_mp4.py --input_bin outputs\jetson_run\<name>.bin
```

If the decode completes without error, the bitstream is valid across devices.

### Step 4: Verify SHA-256

Compare the `bitstream_sha256` from the Jetson sidecar JSON against the hash
of the transferred `.bin` file:

```powershell
python tools\compare_bitstreams.py outputs\jetson_run\<name>.bin --expect_equal
```

Or manually verify:

```powershell
certutil -hashfile outputs\jetson_run\<name>.bin SHA256
```

### Step 5: Re-encode on Device B and Compare

To verify full byte-equality (both directions):

```powershell
python encode_mp4_to_bin.py --input_mp4 .\test.mp4 --frames 1390 --output_dir outputs\laptop_run
python tools\compare_bitstreams.py outputs\jetson_run\*.bin outputs\laptop_run\*.bin --expect_equal
```

Expected result: `pair_sha256_equal: true`.

## Encode-Only vs Full-Mode Equivalence

The encode-only mode (`DCVC_INT16_ENCODE_ONLY=1`, default) skips redundant
decoder-side reconstruction during P-frame encoding. This must produce
identical bitstreams to the full encode+decode sync mode:

```powershell
python encode_mp4_to_bin.py --input_mp4 .\test.mp4 --frames 32 --output_dir outputs\full --disable_encode_only
python encode_mp4_to_bin.py --input_mp4 .\test.mp4 --frames 32 --output_dir outputs\eo
python tools\compare_bitstreams.py outputs\full\*.bin outputs\eo\*.bin --expect_equal
```

## Full Runtime Check

When the model bundle is available:

```powershell
python encode_mp4_to_bin.py --input_mp4 .\test.mp4 --frames 2 --output_dir outputs\smoke
python decode_bin_to_mp4.py --input_bin outputs\smoke\<name>.bin
python tools\compare_bitstreams.py outputs\smoke\<run_a>.bin outputs\smoke\<run_b>.bin --expect_equal
```

## Equivalence Metadata

The encode sidecar JSON records an `equivalence_class` object containing:

- Git commit hash and dirty state
- PyTorch version and CUDA version
- GPU device name and compute capability
- Model bundle file path and SHA-256
- All `DCVC_*` environment variable values
- Video dimensions, FPS, and frame count
- QP settings and reset interval
- Runtime flags (encode-only, CUDA graphs, async entropy, profiling)

Two encodes should produce identical bitstreams if and only if their
equivalence class objects match on all fields except timing data.

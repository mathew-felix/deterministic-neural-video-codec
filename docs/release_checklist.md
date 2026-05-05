# Release Validation Matrix

## Pre-Merge Checklist

Every optimization commit must pass the following checks before merging.

### CI Smoke Sequence

- [ ] **Compile check:**
  ```powershell
  python -m py_compile src\layers\int16_backend.py src\models\int16_reference.py encode_mp4_to_bin.py
  ```

- [ ] **Unit tests without GPU or model bundle:**
  ```powershell
  python -m pytest tests\ -x -v --ignore=tests\test_int16_kernels.py -k "not cuda"
  ```

### Tier 1: Build Integrity

- [ ] **Python syntax:**
  ```powershell
  python -m py_compile src\layers\int16_backend.py
  python -m py_compile src\models\int16_reference.py
  python -m py_compile encode_mp4_to_bin.py
  python -m py_compile decode_bin_to_mp4.py
  ```

- [ ] **CUDA extension build:**
  ```powershell
  python build_int16_cuda.py
  ```

- [ ] **rANS extension build:**
  ```powershell
  cd src\cpp && python setup.py build_ext --inplace
  ```

### Tier 2: Unit Tests

- [ ] **Kernel parity tests** (requires CUDA):
  ```powershell
  python -m pytest tests\test_int16_kernels.py -x -v
  ```

- [ ] **INT16 backend tests** (CPU-only):
  ```powershell
  python -m pytest tests\test_int16_backend.py -x -v
  ```

- [ ] **Bundle loading smoke test:**
  ```powershell
  python -m pytest tests\test_bundle_loading.py -x -v
  ```

- [ ] **Entropy roundtrip test:**
  ```powershell
  python -m pytest tests\test_entropy_roundtrip.py -x -v
  ```

- [ ] **Bitstream equivalence test:**
  ```powershell
  python -m pytest tests\test_bitstream_equivalence.py -x -v
  ```

- [ ] **INT8 routing gate test:**
  ```powershell
  python -m pytest tests\test_int8_routing.py -x -v
  ```

- [ ] **Runtime packaging test:**
  ```powershell
  python -m pytest tests\test_runtime_packaging.py -x -v
  ```

### Tier 3: Functional Validation (Requires Model Bundle)

- [ ] **Preflight check:**
  ```powershell
  python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 2 --check_only
  ```

- [ ] **Same-machine bitstream SHA-256 equality** (two identical encodes):
  ```powershell
  python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 32 --output_dir outputs\run_a
  python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 32 --output_dir outputs\run_b
  python tools\compare_bitstreams.py outputs\run_a\*.bin outputs\run_b\*.bin --expect_equal
  ```

- [ ] **Encode-only vs full-mode equivalence:**
  ```powershell
  python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 32 --output_dir outputs\full --disable_encode_only
  python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 32 --output_dir outputs\eo
  python tools\compare_bitstreams.py outputs\full\*.bin outputs\eo\*.bin --expect_equal
  ```

- [ ] **Decode roundtrip:**
  ```powershell
  python decode_bin_to_mp4.py --input_bin outputs\run_a\<name>.bin
  ```

### Tier 4: Cross-Device Validation (Requires Jetson + x86)

- [ ] **Jetson encode → x86 decode:**
  Encode on Jetson, transfer `.bin` to x86, decode on x86. Verify SHA-256
  equality between Jetson's encode sidecar JSON and the `.bin` file hash.

- [ ] **x86 encode → Jetson decode:**
  Same process in reverse direction.

### Tier 5: Performance Regression

- [ ] **P-frame wall-clock time** does not regress beyond 5% vs baseline:
  ```powershell
  python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 64 --profile_pframe_stages
  ```
  Check `avg_frame_encode_time_ms` in the output JSON.

- [ ] **Calibration clamp health** remains at zero:
  ```powershell
  python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 300 --log_frame_stats
  ```
  Verify `dmci_total=0` and `dmc_total=0` in logged output.

## Validation Evidence

All validation evidence should be captured as JSON sidecar files alongside the
bitstreams. The `equivalence_class` object in each sidecar records:

- Git commit hash and dirty state
- PyTorch version and CUDA version
- GPU device name and compute capability
- Model bundle SHA-256
- All `DCVC_*` environment variable values
- QP settings and frame count
- Runtime flags (encode-only, CUDA graphs, async entropy)

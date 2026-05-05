# Deterministic Neural Video Codec: 15-Commit Repository Plan

## Purpose

This plan reconstructs the finished `dcvc_portal` work into a professional
15-commit engineering history inside `deterministic-neural-video-codec/`.
The goal is not to dump the final code in one import. The goal is to show the
major engineering decisions that led from upstream DCVC-RT to a deterministic
INT16 neural video codec that can run across NVIDIA Jetson and Windows x86
environments with auditable bitstream behavior.

Primary evidence reviewed:

- Project logs: `plan*.md`, `report*.md`, `DCVC_PORTAL_SUMMARY.md`,
  `weekly_meeting_dcvc_rt_int16.md`, `jetson_p15_1.md`
- Backend design: `DCVC/int16_backend_plan.md`
- Final runtime: `dcvc-rt-int16/`
- Standalone intermediate runtime: `dcvc_p15/`
- Encoder-only package: `encode_p10/`
- Upstream/reference trees: `DCVC/`, `dcvc_oiginal/`, and DCVC-family codec
  sources

## Narrative Thesis

The project is a systems-and-numerics reconstruction of DCVC-RT. The original
FP16 path is fast, but floating point is not a valid interoperability contract
for cross-device neural video coding. The 15 commits should communicate this
arc:

1. Establish a clean product repo and deterministic scope.
2. Import only the runtime pieces needed for a real codec.
3. Build a fixed-point INT16 reference path with frozen entropy state.
4. Replace slow reference ops with CUDA kernels.
5. Prove rANS bitstreams and local determinism.
6. Explore INT8 Tensor Cores, then reject them for the gold profile.
7. Fix the long-run P-frame reset bug.
8. Recover speed with profiling, encode-only execution, and residual fusion.
9. Package the standalone runtime with equivalence tooling.
10. Validate Jetson-to-laptop behavior and document calibration limits.

## Engineering Rules For Every Commit

- Use Conventional Commits: `feat:`, `fix:`, `perf:`, `test:`, `docs:`.
- Keep artifacts out of git. Store checkpoints, `.pt`, `.pth`, `.bin`, videos,
  YUV files, and large profiler outputs outside tracked source.
- Prefer structured metrics in `assets/` only when they are small CSV/JSON
  summaries or manifests, not raw generated media.
- Python additions should use type hints and Google-style docstrings.
- CUDA changes should document hardware assumptions such as compute capability,
  memory pressure, graph capture constraints, and stream synchronization.
- Each correctness-sensitive commit needs a validation command or a documented
  expected artifact path.

## Target Repository Layout

```text
deterministic-neural-video-codec/
  assets/
    metrics/
    manifests/
  docs/
    architecture.md
    calibration.md
    determinism.md
    int8_pivot.md
    jetson.md
    performance.md
    validation.md
  scripts/
    calibrate_int16_bundle.py
    download_models.ps1
    download_models.sh
  src/
    cpp/
    layers/
    models/
    utils/
  tests/
    test_bitstream_equivalence.py
    test_bundle_loading.py
    test_entropy_roundtrip.py
    test_int16_backend.py
    test_int16_kernels.py
  tools/
    compare_bitstreams.py
  encode_mp4_to_bin.py
  decode_bin_to_mp4.py
  benchmark_report_style.py
  bootstrap_runtime.py
  build_int16_cuda.py
```

`tools/` is added intentionally even though Commit #1 only created the requested
five baseline directories. It is the right home for user-facing diagnostic
utilities that are not experiment scripts.

## 15-Commit Story

### Phase I: Scaffolding & Deterministic Contract

#### Commit 1: `feat: initialize project scaffold`

Status: already created as `b41d398`.

Intent:
- Establish professional hierarchy.
- Add Python/CUDA `.gitignore`.
- Pin initial runtime dependencies.
- Keep README minimal with project title and initialization status.

Files:
- `.gitignore`
- `requirements.txt`
- `README.md`
- `src/.gitkeep`, `tests/.gitkeep`, `docs/.gitkeep`, `scripts/.gitkeep`,
  `assets/.gitkeep`

Decision shown:
- The repository starts as a clean product repo, not a copy of the research
  workspace.

#### Commit 2: `docs: define deterministic codec scope and provenance`

Intent:
- Add the formal project contract before importing code.
- State the upstream DCVC/DCVC-RT provenance and the unofficial derivative
  status.
- Define Tier A determinism: byte-identical `.bin` or matching SHA-256 under
  fixed input, model bundle, QP, flags, device class, and environment.

Files:
- `docs/provenance.md`
- `docs/determinism.md`
- `docs/architecture.md`
- `README.md`

Decision shown:
- FP16 is treated as the fast deployment path, while INT16 is the cross-device
  interoperability path.

Evidence:
- `DCVC_PORTAL_SUMMARY.md`
- `weekly_meeting_dcvc_rt_int16.md`
- `dcvc-rt-int16/README.md`

### Phase II: Baseline Codec Runtime

#### Commit 3: `feat: import dcvc codec utilities and rans entropy coder`

Intent:
- Bring in the minimal runtime utilities needed for real encode/decode:
  stream helpers, video IO, metrics, common deterministic setup, and rANS C++
  bindings.
- Add extension build support for the arithmetic coder.

Files:
- `src/utils/common.py`
- `src/utils/metrics.py`
- `src/utils/stream_helper.py`
- `src/utils/video_reader.py`
- `src/utils/video_writer.py`
- `src/cpp/setup.py`
- `src/cpp/py_rans/*`

Decision shown:
- The project starts with valid codec infrastructure, not tensor dumps.

Validation:
- `python -m py_compile src/utils/*.py`
- `python src/cpp/setup.py build_ext --inplace`

#### Commit 4: `feat: add frozen entropy bundle and int16 reference backend`

Intent:
- Add frozen entropy CDF export/load support.
- Add the INT16 quantization contract: power-of-two feature and weight scales,
  signed rounding, clamp behavior, LUTs, packed runner descriptors, and Python
  reference execution.
- Add model wrappers for DMCI/DMC INT16 I-frame and P-frame loops.

Files:
- `scripts/freeze_entropy_cdfs.py`
- `scripts/export_int16_bundle.py`
- `src/layers/int16_backend.py`
- `src/models/common_model.py`
- `src/models/image_model.py`
- `src/models/video_model.py`
- `src/models/entropy_models.py`
- `src/models/int16_reference.py`
- `tests/test_int16_backend.py`
- `tests/test_bundle_loading.py`

Decision shown:
- Correctness is established in an inspectable reference path before CUDA
  performance work.

Evidence:
- `DCVC/int16_backend_plan.md`
- `report.md`

Validation:
- `python -m py_compile src/layers/int16_backend.py src/models/int16_reference.py`
- Bundle load smoke test with CPU map location.

#### Commit 5: `feat: wire compact rans bitstreams into int16 compression`

Intent:
- Replace raw tensor-dump outputs with compact rANS bitstreams.
- Close I-frame and P-frame entropy loops so encode-side and decode-side
  reconstructed symbols match.

Files:
- `src/models/int16_reference.py`
- `src/models/entropy_models.py`
- `src/utils/stream_helper.py`
- `tests/test_entropy_roundtrip.py`

Decision shown:
- The runtime becomes a real codec once the entropy payload is compact,
  decodable, and deterministic.

Evidence:
- `plan2.md`: tensor dump had approximately 170x bpp inflation.
- `report.md`: packed entropy streams rebuild correctly on decode.

Validation:
- 32-frame smoke should land around `bpp=0.020`.

### Phase III: CUDA Acceleration

#### Commit 6: `feat: add int16 cuda extension and kernel parity tests`

Intent:
- Add PyTorch CUDA extension loading with Python fallback.
- Implement first-pass kernels for conv2d, LUT lookup, scale-index lookup,
  clamp reciprocal, add-multiply, and shared arithmetic surfaces.
- Add unit tests comparing CUDA output to the Python reference.

Files:
- `build_int16_cuda.py`
- `src/layers/int16_cuda_ext.py`
- `src/layers/int16_kernels.cu`
- `src/layers/int16_kernels_bind.cpp`
- `tests/test_int16_kernels.py`

Decision shown:
- CUDA is optional for import but required for the performance story.

Evidence:
- `plan.md`
- `plan2.md`

Validation:
- `python build_int16_cuda.py`
- `python -m pytest tests/test_int16_kernels.py`

#### Commit 7: `perf: optimize int16 convolutions with tiling and fused depthwise paths`

Intent:
- Replace naive global-memory convolution with shared-memory tiled kernels.
- Add optimized 1x1 path after the tested INT16 `cublasGemmEx` route failed.
- Add depthwise 3x3 and fused WSiLU LUT path for DCVC-RT depthwise blocks.

Files:
- `src/layers/int16_kernels.cu`
- `src/layers/int16_kernels_bind.cpp`
- `src/layers/int16_cuda_ext.py`
- `src/layers/int16_backend.py`

Decision shown:
- Real hardware behavior guides the implementation; the plan adapts when a
  nominal cuBLAS path fails.

Evidence:
- `report4.md`: `cublasGemmEx` status 7.
- Direct 720p timing improved from `2714.6 ms` I-frame and `1106.5 ms`
  P-frame to `1999.5 ms` and `1060.0 ms`, while RD stayed stable.

Validation:
- Kernel tests match Python reference bit-for-bit.
- 32-frame RD stays near `bpp=0.020039`, `PSNR=37.307 dB`.

#### Commit 8: `perf: add fixed-shape cuda graphs and local bitstream checks`

Intent:
- Add graph capture for stable-shape network subgraphs.
- Keep CPU entropy coding outside graph capture.
- Add same-machine SHA-256 and byte-comparison tooling.

Files:
- `src/models/int16_reference.py`
- `src/layers/cuda_inference.py`
- `src/utils/equivalence.py`
- `tools/compare_bitstreams.py`
- `tests/test_bitstream_equivalence.py`
- `docs/architecture.md`

Decision shown:
- Graph capture is useful, but determinism claims must be backed by bitstream
  equality rather than speed numbers alone.

Evidence:
- `plan4.md`
- `report5.md`
- `report16.md`

Validation:
- Graph replay equals non-graph output.
- Two fixed-input encodes produce the same SHA-256.

### Phase IV: The INT8 Pivot and Correctness Fix

#### Commit 9: `perf: prototype int8 tensor-core route behind deterministic gates`

Intent:
- Add INT8 weight packing, activation casting, cublas INT8 GEMM, per-layer
  routing, and calibration hooks.
- Keep INT8 disabled unless explicitly requested.

Files:
- `src/layers/int16_backend.py`
- `src/layers/int16_kernels.cu`
- `src/layers/int16_cuda_ext.py`
- `scripts/export_int16_bundle.py`
- `scripts/calibrate_int8_activation_scales.py`
- `docs/int8_pivot.md`

Decision shown:
- The fastest available NVIDIA path was tested seriously, with entropy-critical
  layers protected by routing policy.

Evidence:
- `plan4.md`
- `plan5.md`
- `report5.md`: INT8 plus graphs reached approximately `353.5 ms` I-frame and
  `280.3 ms` P-frame but RD collapsed.
- `report6.md`: pure INT16 stayed at `bpp=0.020039`; mixed precision reached
  `bpp=0.036917` to `0.039353`.

#### Commit 10: `docs: record int8 failure and reinstate pure-int16 gold profile`

Intent:
- Preserve the INT8 route as an engineering pivot rather than deleting the
  evidence.
- Document per-channel scaling, activation scaling, and exact-descale attempts.
- State that the gold interoperable profile is pure INT16.

Files:
- `docs/int8_pivot.md`
- `docs/determinism.md`
- `src/layers/int16_backend.py`

Decision shown:
- Speed does not win when it perturbs the closed-loop entropy and temporal
  reconstruction path.

Evidence:
- `plan6.md`
- `plan7.md`
- `plan8.md`
- `DCVC_PORTAL_SUMMARY.md`

#### Commit 11: `fix: align p-frame reset state across encoder and decoder`

Intent:
- Add frame-level diagnostics for symbols, indexes, DPB tensors, and reset
  frames.
- Fix the `use_ada_i` reset path so encoder and decoder rebuild `ctx`, `ctx_t`,
  reference features, and DPB state identically.
- Keep P-frame CUDA graphs opt-in until reset-path behavior is proven safe.

Files:
- `src/models/int16_reference.py`
- `src/models/entropy_models.py`
- `encode_mp4_to_bin.py`
- `tests/test_entropy_roundtrip.py`

Decision shown:
- Deterministic arithmetic is not enough; deterministic state transitions are
  also required.

Evidence:
- `plan10.md`
- `report11.md`: 300-frame int16 round trip passed after the reset fix.

Validation:
- 32-frame regression.
- 64-frame smoke past old failure.
- 300-frame full int16 round trip.

### Phase V: Optimization, Packaging, and Validation

#### Commit 12: `perf: profile p-frame pipeline and add encode-only mode`

Intent:
- Add low-overhead CUDA event and CPU wall-time profiling.
- Use profiler evidence to cancel the planned C++ orchestration rewrite.
- Add encode-only P-frame mode that skips redundant decoder-side reconstruction
  during encoding while preserving bitstream identity.

Files:
- `src/models/int16_reference.py`
- `encode_mp4_to_bin.py`
- `benchmark_report_style.py`
- `docs/performance.md`

Decision shown:
- The performance bottleneck is GPU-side INT16 work, not Python dispatch.
- The biggest safe speedup comes from removing redundant codec work.

Evidence:
- `report13.md`: Python overhead was less than `1 ms`; clean GPU P-frame encode
  was about `296-431 ms`.
- `report14.md`: encode-only reduced P-frame encode to `194-200 ms` with zero
  PSNR delta across checked frames.

Validation:
- Full mode vs encode-only bitstream equality on fixed inputs.

#### Commit 13: `perf: fuse residual addition into int16 cuda kernels`

Intent:
- Pass optional residual tensors into conv/descale kernels.
- Remove PyTorch ATen residual adds and extra global GPU memory reads/writes.
- Integrate residual fusion through the INT16 backend runners.

Files:
- `src/layers/int16_kernels.cu`
- `src/layers/int16_kernels_bind.cpp`
- `src/layers/int16_cuda_ext.py`
- `src/layers/int16_backend.py`

Decision shown:
- The remaining wins are memory-traffic and kernel-boundary wins, not Python
  rewrites.

Evidence:
- `report15.md`: NN forward improved from about `181.5 ms` to `175.3 ms`,
  with total encode pipeline near `178.0 ms`.

Validation:
- Kernel parity tests.
- Bitstream equality against pre-fusion encode on fixed inputs.

#### Commit 14: `feat: package standalone encode/decode runtime with equivalence metadata`

Intent:
- Add MP4 encode/decode entrypoints, runtime bootstrap, extension build helper,
  model download placeholders, profiling output, and equivalence metadata in
  sidecar JSON.
- Add direct bitstream comparison tool for SHA-256 and byte equality.

Files:
- `encode_mp4_to_bin.py`
- `decode_bin_to_mp4.py`
- `benchmark_report_style.py`
- `bootstrap_runtime.py`
- `build_int16_cuda.py`
- `scripts/download_models.sh`
- `scripts/download_models.ps1`
- `tools/compare_bitstreams.py`
- `src/utils/equivalence.py`
- `docs/validation.md`

Decision shown:
- The research implementation becomes a portable runtime with reproducible
  environment capture.

Evidence:
- `encode_p10/`
- `dcvc_p15/`
- `dcvc-rt-int16/`
- `report16.md`
- `weekly_meeting_dcvc_rt_int16.md`

Validation:
- `python bootstrap_runtime.py`
- `python encode_mp4_to_bin.py --frames 2 --output_dir outputs/smoke`
- `python decode_bin_to_mp4.py --input_bin outputs/smoke/<name>.bin`
- `python tools/compare_bitstreams.py run_a.bin run_b.bin --expect_equal`

#### Commit 15: `docs: publish jetson validation calibration and release checklist`

Intent:
- Add Jetson deployment notes, memory constraints, calibration workflow, final
  performance summary, and release validation matrix.
- Document async entropy prep as an opt-in prototype: locally reduced CPU wait,
  not a default end-to-end speed win.
- Update README from initialization status to the final evidence-based project
  overview.

Files:
- `README.md`
- `docs/calibration.md`
- `docs/jetson.md`
- `docs/performance.md`
- `docs/release_checklist.md`
- `docs/validation.md`
- `scripts/calibrate_int16_bundle.py`
- `assets/metrics/validation_gpu.example.csv`
- `assets/metrics/plan19_summary.example.json`
- `assets/manifests/calibration_manifest.example.json`

Decision shown:
- The final repo is a professional systems artifact with known tradeoffs,
  validation evidence, and reproducible operating constraints.

Evidence:
- `jetson_p15_1.md`
- `dcvc-rt-int16/report18.md`
- `dcvc-rt-int16/report19.md`
- `report16.md`
- `report17.md`

Metrics to cite:
- Laptop optimized P-frame encode: approximately `178-197 ms/frame` depending
  benchmark mode.
- Laptop decode: approximately `166-175 ms/frame`.
- Jetson full sequence encode: `1390` frames, `3044.333 s`,
  `2134.768 ms/frame`, `0.468 fps`.
- Laptop decode of Jetson bitstream: `166.365 ms/frame`, `6.011 fps`.
- Jetson and laptop output totals: `16364584` bits, `2045573` bytes,
  `353.19246043165464 kbps`.
- Calibration clamp health remained stable with `dmci_total=0` and
  `dmc_total=0`, while strict low-light PSNR gates remained a known limitation.

Validation matrix:
- Python compile.
- CUDA extension build.
- Kernel parity tests.
- Bundle load smoke.
- Same-machine bitstream SHA-256 equality.
- Encode-only vs full-mode equivalence.
- Full-sequence laptop encode/decode.
- Jetson encode -> laptop decode.
- Calibration pass with clamp-health summary.

## Suggested Next Three Commits

The next three logical commits after the existing scaffold are:

1. `docs: define deterministic codec scope and provenance`
2. `feat: import dcvc codec utilities and rans entropy coder`
3. `feat: add frozen entropy bundle and int16 reference backend`

These establish the technical/legal context, the real bitstream foundation, and
the fixed-point reference contract before the CUDA acceleration work begins.

## Open Risks To Preserve In The Story

- INT16 is deterministic but slower than the FP16 deployment path.
- CUDA graph capture improves fixed-shape execution but can create memory
  pressure on Jetson-class devices.
- INT8 Tensor Core acceleration is fast but not acceptable for the gold
  deterministic/RD profile without retraining or a different architecture.
- Calibration can eliminate clamp failures while still failing strict
  float-reference PSNR gates on difficult low-light clips.
- Async entropy prep reduces a local CPU bubble but is not yet a default
  end-to-end speed win.


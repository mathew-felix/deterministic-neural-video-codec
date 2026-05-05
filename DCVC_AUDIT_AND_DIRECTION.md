# DCVC-RT INT16 — Full Codebase Audit & Direction Report

## 1. What This Project Is

The project converts **Microsoft DCVC-RT** (CVPR 2025, arXiv:2502.20762) — a fast FP16 neural video codec — into a **deterministic INT16 runtime** that produces byte-identical bitstreams across heterogeneous hardware (Nvidia Jetson Orin Nano 8GB ARM ↔ x86 Windows workstation). FP16 fails this requirement because floating-point rounding differences across GPU architectures corrupt the entropy coding context, causing cascading decoder crashes.

### Architecture Pipeline

```
source video (MP4)
  → YUV420 preprocessing + padding
  → DMCI I-Frame INT16 path        (src/models/image_model.py, int16_reference.py)
  → DMC P-Frame INT16 path         (src/models/int16_reference.py, encode-only mode)
  → frozen rANS entropy coder      (src/models/entropy_models.py)
  → deterministic .bin bitstream
       ↕ cross-device transfer
  → rANS decode + INT16 reconstruction
  → output MP4
```

**CUDA acceleration stack:**

| File | Lines | Role |
| :--- | :--- | :--- |
| `src/layers/int16_kernels.cu` | 1448 | All custom INT16 CUDA kernels |
| `src/layers/int16_backend.py` | 1195 | Quantization contract, CUDA/Python fallback |
| `src/models/int16_reference.py` | 2217 | I/P frame loops, profiler, DPB reset, async entropy |
| `src/models/entropy_models.py` | 331 | Frozen CDF entropy, rANS integration |
| `encode_mp4_to_bin.py` | 663 | CLI encode entry point |
| `decode_bin_to_mp4.py` | 276 | CLI decode entry point |
| `tools/compare_bitstreams.py` | — | Tier A SHA256/byte equivalence tool |

---

## 2. What Is Working Correctly

### Proven and Validated

- **Cross-device determinism (byte-exact):** Jetson encode + laptop decode on 1390-frame 1280×720 clip. Both sides produce exactly 16,364,584 bits / 353.19 kbps. SHA256-confirmed on the full bitstream.
- **P-frame reset bug fixed:** `use_ada_i` reset path (encoder/decoder DPB divergence) that was crashing at frame 33 of 300-frame runs is resolved in `prepare_feature_adaptor_i`.
- **rANS entropy coding:** Compact bitstreams (~0.020 bpp), lossless round-trip proven.
- **Encode-only mode:** Skips redundant decoder-sync during encoding, saves ~100 ms/frame with zero quality delta (PSNR bit-identical across all tested QPs).
- **INT16 CUDA kernel suite:** Tiled conv2d, blocked 1×1 (`cublasGemmEx` INT16 failed on Ampere — custom fallback proven correct), depthwise + WSiLU fused, residual fusion into conv epilogue.
- **CUDA graph capture:** Fixed-shape P-frame subgraph graphs working on desktop (disabled on Jetson due to OOM).
- **Standalone `dcvc_p15` package:** Self-contained, builds own extensions, runs independently of parent `DCVC/` tree.
- **Equivalence tooling:** `compare_bitstreams.py` with sidecar JSON environment capture (git hash, PyTorch/CUDA versions, device metadata, flags).
- **Stage profiling:** Structured per-frame JSON profiling artifact for laptop-vs-Jetson comparison.
- **Test suite:** 8 test files covering kernel parity, bundle loading, entropy roundtrip, bitstream equivalence, INT8 routing gates, and runtime packaging.

### Performance Journey (P-Frame Encode, 1280×720)

```
FP16 baseline:                   ~16 ms/frame
INT16 start (plan 2):          ~1107 ms/frame
After tiled kernels (plan 3):   ~700 ms/frame
After CUDA graphs (plan 4):     ~280 ms  (INT8 RD collapsed — reverted)
Pure INT16 + encode-only:      ~194–200 ms/frame
+ residual fusion (current):   ~178–197 ms/frame  ← current state
```

---

## 3. Issues and Limitations

### A. Speed — Far from Real-Time

| Path | Laptop P-Frame | Jetson P-Frame | Real-Time Target |
| :--- | :--- | :--- | :--- |
| FP16 upstream | ~16 ms | unknown | 33 ms (30 fps) |
| INT16 current | ~178–197 ms | ~2134 ms | 33 ms (30 fps) |
| Gap vs real-time | ~6× too slow | ~65× too slow | — |

- ~11× additional laptop speedup still needed to reach 30 fps real-time.
- Jetson is 10.85× slower than laptop (fewer SMs, lower memory bandwidth), running at 0.468 fps.
- I-frame encode is ~353 ms (CUDA graph enabled) — no dedicated I-frame graph optimization exists.

### B. Async Entropy Prep Is Net Negative (Report 17 — Latest State)

Phase 13.1 collapsed the `compress_prior_2x` CPU bubble from **123 ms → 4 ms** (local win, −96.6%), but caused a **+14.93% wall-clock regression (+31 ms/frame)** end-to-end.

The change is Tier A safe but not performance-positive. The likely cause is that the async side-stream is not genuinely overlapping with main-stream GPU work, and the D2H pinned-copy overhead and `ready_event.synchronize()` stall recover the same serialized bubble with added management cost.

**Status:** REVERTED. The `is_async_entropy_prep_active()` method now unconditionally returns `False` to resolve the regression.

### C. INT8 Tensor Core Path Permanently Closed

- INT8 acceleration caused **8.5 dB PSNR collapse** with global per-layer scale.
- Even with per-channel weight and activation scaling, a single "safe" diagnostic layer (`DMC.decoder.conv2`) still failed with a **9.0 dB drop**.
- DCVC-RT's closed-loop temporal prediction architecture amplifies small local quantization errors across P-frame chains. No amount of post-hoc calibration can fix this without retraining.
- `INT8_ELIGIBLE_LAYERS` now contains only 1 layer in code — effectively dead code for production use.

### D. Quality Gap vs FP16

- **INT16 = ~37.31 dB** vs **FP16 = ~38.59 dB** (~1.28 dB permanent deficit at 300 frames).
- Low-light/night video clips fail strict FP16 PSNR numeric gates even with v4 calibration bundles (weighted 99.9th-percentile aggregation). Visual structure is intact and determinism is never broken, but strict automated gating fails.
- Calibration clamping is stable (`dmci_total=0`, `dmc_total=0`) — the quality gap is quantization error, not overflow.

### E. Commit 15 Delivered

`repo_plan.md` defines a 15-commit story. The final Commit 15 has been delivered, establishing the required documentation, calibration logic, and validation evidence:

| Completed File | Purpose |
| :--- | :--- |
| `docs/calibration.md` | Calibration workflow + clamp health documentation |
| `docs/jetson.md` | Jetson deployment constraints, OOM notes, graph disable |
| `docs/release_checklist.md` | Final validation matrix |
| `docs/performance.md` | Async entropy status, Jetson vs laptop comparison |
| `scripts/calibrate_int16_bundle.py` | Calibration automation script |
| `assets/metrics/` | GPU validation CSV and Plan 19 JSON summary |
| `assets/manifests/` | 6-content-type calibration manifest |

### F. Plan 13 Phases 13.2–13.5

| Phase | Focus | Status |
| :--- | :--- | :--- |
| 13.0 | Lock equivalence tooling into CI | Done (`compare_bitstreams.py`) |
| 13.1 | Async entropy prep CPU bubble | **Done (Reverted)** |
| 13.2 | Kernel fusion in major subgraphs | Not started |
| 13.3 | Enlarge CUDA graph coverage for P steady-state | Not started |
| 13.4 | Jetson-specific kernel/build/occupancy pass | Not started |
| 13.5 | ~~FP16 accumulation with INT16 lattice (lab spike)~~ | **CANCELLED** (Fundamentally non-deterministic) |

### G. Other Gaps

- **Single-clip validation only:** All determinism and RD evidence uses one 1280×720 30 fps clip (`test.mp4`). No multi-resolution, multi-fps, or multi-content testing.
- **No CI:** Tests exist but no automated pipeline runs them on commit.
- **Model download script is a placeholder:** `scripts/download_models.sh` / `.ps1` documents the bundle path but has no actual download URL. The bundle cannot be obtained from a fresh clone.
- **No I-frame CUDA graph:** The I-frame path (~353 ms) lacks graph capture optimization.
- **Serialized pipeline:** P-frame closed-loop recurrence (DPB state) prevents any inter-frame parallelism.

---

## 4. What Direction to Take

### Immediate — Unblock Current State

**1. Add pytest CI smoke sequence** to `README.md`:

```powershell
# Compile check
python -m py_compile src\layers\int16_backend.py src\models\int16_reference.py encode_mp4_to_bin.py

# Unit tests (no GPU, no model bundle)
python -m pytest tests\ -x -v --ignore=tests\test_int16_kernels.py -k "not cuda"
```

Most tests already skip gracefully when CUDA or the extension is unavailable.

---

### Short-Term Performance — Plan 13.2–13.4 (~1–3 months)

**4. Kernel fusion in the 5 dominant stages (Plan 13.2 Pillar A).**

These consume 76% of steady-state P-frame GPU time:

| Stage | GPU ms | % of total |
| :--- | :--- | :--- |
| `reconstruct_frame_enc` | 41.0 | 21.2% |
| `decode_feature_enc` | 36.0–38.4 | 18.6–19.9% |
| `res_prior_param_decoder_enc` | 32.2–33.3 | 16.6–17.2% |
| `extract_context_enc` | 33.1–39.0 | 17.1–20.2% |
| `encode_y` | 30.2–30.9 | 15.6–16.0% |

For each stage, build a fusion plan identifying which operations are pointwise (scale, clamp, activation, add) and fuse them into the conv epilogue or prologue in `int16_kernels.cu`, using the same pattern proven in report15 (fused residual). Validate each change with `compare_bitstreams.py --expect_equal` on at least a 32-frame proof before committing. A 20% mean reduction across these five stages saves ~25 ms/frame.

**5. Narrow-channel specialist kernels (Plan 13.2, Pillar A sub-item).**

Current `BLOCK_C=8` is designed for wider channel counts. Layers with 48, 64, or 96 channels waste SM occupancy due to incomplete tiles. Add dedicated kernel variants with tighter block shapes (`BLOCK_C=4` or `BLOCK_C=6`) for these widths. Use Nsight Compute occupancy analysis to verify the benefit before merging.

**6. Jetson-specific profiling and build audit (Plan 13.4, Pillar D).**

Run the same `--profile_pframe_stages` JSON collection on Jetson and compare stage percentages against the laptop table. If Jetson shows different stage ordering (e.g., `hyper_encode` or `feature_adaptor_enc` dominating), the bottleneck on Orin is different from desktop and requires different fusions.

Build audit steps:
- Confirm `-arch sm_87` is used during extension compilation on Jetson.
- Run `ncu` to check register spill count per kernel; reduce with `__launch_bounds__` if spills are significant.
- Check shared memory tile sizes in `conv2d_int16_tiled_kernel`; reduce tile dimensions if Orin's L1 is smaller than desktop assumption.

**7. Enlarge CUDA graph coverage (Plan 13.3, Pillar C).**

After 13.2 reduces dynamic CPU work inside the capture window, extend `DCVC_ENABLE_INT16_PFRAME_GRAPHS` to capture more of the steady-state P-frame NN forward in a single graph. Exclude the rANS and entropy-packing sections (they remain architecturally outside graph capture). Validate with `compare_bitstreams.py` before and after graph expansion on ≥ 32 frames.

---

### Medium-Term Research — 3–12 Months

**8. ~~FP16 accumulation with INT16 lattice — Plan 13.5, Pillar E (lab spike).~~ [CANCELLED]**

> [!WARNING]
> **Research Finding:** This hypothesis is mathematically unsound for cross-hardware equivalence and has been cancelled.
> 
> NVIDIA Tensor Cores perform accumulations using parallel reduction trees baked into hardware logic. These trees vary substantially between SM architectures (e.g., Ampere on Jetson vs. Ada/Hopper on workstations). Because floating-point addition is non-associative (`(a + b) + c != a + (b + c)`), the reduction order produces varying partial sums.
> 
> Even with rounding to an INT16 grid, if the cumulative floating-point variation between a Jetson and an x86 GPU straddles a rounding boundary (e.g., `2.499` vs `2.501`), they will round to different integers. In DCVC's closed-loop recurrent architecture, a single bit difference permanently corrupts the entropy context and crashes the decoder. **Any FP accumulation inside the deterministic envelope will inevitably break cross-hardware equivalence.**

**9. Multi-clip calibration for bundle v5.**

The `v2_calibrated` bundle was generated from a limited clip set. A v5 sweep should cover:

| Content type | Target property |
| :--- | :--- |
| Outdoor daylight | High dynamic range, strong gradients |
| Indoor studio | Low noise, predictable activations |
| Sports / fast motion | Large inter-frame differences, high entropy |
| Night / dark scene | Low-light outlier activation tails |
| Talking head / screen content | Uniform regions, low inter-frame entropy |
| Animation | Saturated colors, sharp edges, no film grain |

For each type, collect statistics over ≥ 300 frames at QP 32. Use weighted 99.9th-percentile aggregation with content-type weights tuned to the target deployment distribution. After exporting the bundle, run `compare_bitstreams.py` on the same clip with v2_calibrated vs v5 — the bitstreams will differ (different scales), but both should be Tier A internally (same bundle → same output). Quantify PSNR gate failures by content type.

**10. Multi-resolution validation.**

Test at 480p, 1080p, and 4K. INT16 kernel tiling parameters (tile width/height, `BLOCK_C`) may not be optimal at non-720p resolutions. Check whether the `ec_part` split threshold (currently `height * width > 1280 * 720`) is appropriate for larger resolutions. Verify that `replicate_pad` alignment to 16-pixel macroblock boundaries works correctly at all tested resolutions.

---

### Long-Term Research — >12 Months

**11. Quantization-aware training (QAT). [STRONGLY RECOMMENDED]**

> [!TIP]
> **Research Finding:** QAT is the industry standard for deploying learned video compression (LVC) to edge hardware. Post-Training Quantization (PTQ) almost always introduces severe PSNR drops on recurrent video codecs. QAT with Straight-Through Estimators (STE) is required to close the performance gap.

The ~1.28 dB quality gap between INT16 and FP16 is a post-hoc quantization cost. The correct fix is to retrain a DCVC-RT variant with INT16 constraints baked into the training loop (fake-quantize operations during forward pass, straight-through estimators for gradients). A natively trained INT16 model could close most of the quality gap and potentially allow INT8 weights on some layers without entropy corruption — because the network would have learned to keep quantization-sensitive activations within a safe range.

This requires access to the training infrastructure, the original training dataset, and significant compute. **Given the cancellation of FP16 accumulation spikes, QAT is now the primary path to resolving quality gaps without breaking determinism.**

**12. Architecture redesign for deterministic edge deployment.**

DCVC-RT's closed-loop recurrent architecture (P-frame temporal prediction via DPB, closed-loop entropy context) is fundamentally hostile to fixed-point quantization and hardware acceleration:
- Any quantization error in the reconstructed frame feeds into the next P-frame prediction and compounds over time.
- The recurrent DPB state serializes the pipeline — no inter-frame parallelism is possible.
- The entropy context is tightly coupled to the reconstruction path, so quantization noise propagates through entropy coding as well.

A future deterministic edge codec design should consider: decoupled entropy context (separate from the reconstruction loop), non-recurrent or short-horizon prediction (reduce error propagation), smaller channel counts (fewer INT16 multiplications per frame), and explicit quantization checkpoints in the architecture (quantize intermediate features to defined precision at specific points rather than hoping post-hoc calibration is sufficient).

**13. Practical Jetson real-time path.**

The 10.85× speed gap means the current INT16 path will not reach real-time on Jetson without a fundamentally different approach. Three viable strategies in increasing engineering cost:

- **(a) Hybrid mode:** FP16 for encoding on Jetson (fast, non-deterministic), INT16 reserved for cross-device decode verification only. The Jetson runs FP16 encode but periodically re-encodes a reference clip in INT16 to confirm the model bundle is consistent with the laptop's decoder.
- **(b) Lighter model:** Use a DCVC-RT variant with reduced channel counts (e.g., half-width channels) trained specifically for edge deployment. Accept a modest quality reduction in exchange for lower compute.
- **(c) Deep Jetson kernel tuning:** After Plan 13.4 provides Nsight profiles, invest in SM87-specific kernel variants (register allocation, shared memory layout, warp scheduling) to close the gap from the hardware side. This can realistically provide 2–3× improvement but likely not 10×.

---

## 5. Summary Table

| Area | Status | Priority |
| :--- | :--- | :--- |
| Cross-device determinism | Working, proven (Jetson ↔ x86) | Maintain |
| P-frame performance (laptop) | ~197 ms/frame, ~5 fps | Plan 13.2–13.3 |
| I-frame performance | ~353 ms/frame | No graph optimization yet |
| Jetson encode speed | ~2134 ms/frame, 0.47 fps | Plan 13.4 |
| Async entropy regression | Reverted (+31 ms resolved) | Done |
| INT8 Tensor Core path | Abandoned (8.5–9 dB PSNR drop) | Closed |
| Low-light PSNR gates | Failing strict gates; visual quality intact | Multi-clip calibration |
| Commit 15 docs/scripts | Completed | Done |
| Plan 13.2–13.4 | Not started | Immediate priority |
| CI / automated test runner | No pipeline; tests exist | Low-effort, add now |
| Quality gap vs FP16 | ~1.28 dB | QAT (long-term) |
| Jetson real-time | Not achievable with current approach | Hybrid/lighter model (long-term) |

---

## 6. Evidence References

| Document | Key content |
| :--- | :--- |
| `DCVC_PORTAL_SUMMARY.md` | Executive rollup of all phases |
| `plan.md` – `plan13.md` | Per-phase engineering plans |
| `report.md` – `report17.md` | Per-phase experimental results and metrics |
| `weekly_meeting_dcvc_rt_int16.md` | Laptop vs Jetson throughput, cross-device summary |
| `jetson_p15_1.md` | Full Jetson 1390-frame encode + cross-decode validation |
| `repo_plan.md` | 15-commit story and target file layout |
| `artifacts/plan13_phase131_*/` | Before/after bitstreams for Plan 13.1 async regression |

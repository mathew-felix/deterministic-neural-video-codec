# Performance

## Commit 12 Scope

This commit introduces the profiling and encode-only control surface for the
INT16 P-frame path. The model runtime already exposes three important switches:

- `DCVC_INT16_ENCODE_ONLY=1` skips redundant decoder-side reconstruction during
  P-frame encoding.
- `DCVC_PROFILE_INT16_PIPELINE=1` enables CUDA-event and CPU wall-clock stage
  timing in `DMCInt16Reference`.
- `DCVC_ENABLE_INT16_PFRAME_GRAPHS=1` remains opt-in because reset-frame state
  transitions must be validated independently from graph replay.

## Why Encode-Only Is Safe

The P-frame encoder already reconstructs the quantized latent and local frame
state needed for the next prediction step. The decoder-side reconstruction
inside the encoder is redundant when the entropy payload is lossless and the
packed INT16 symbols rebuild to the same indexes. Commit 11 guards the reset
state that makes this assumption valid across `use_ada_i` frames.

## Profiling Boundary

The profiler measures the neural-network side of the INT16 path and reports
per-stage GPU and CPU timings. Dynamic entropy packing and CPU rANS handoff
remain outside whole-frame CUDA graph capture. Treat profiler artifacts as
local performance evidence, not as determinism evidence.

Tier A determinism still requires byte-identical `.bin` files or matching
SHA-256 digests for the same input, model bundle, QP, flags, and environment.

## Local Smoke Command

The clean repository does not track model bundles. Use preflight mode to verify
the local MP4, frame-count resolution, and effective runtime flags without
adding media or checkpoint files to git:

```powershell
python encode_mp4_to_bin.py --input_mp4 .\test.mp4 --frames 2 --check_only
```

If the local clip has a different filename, pass it explicitly. Full encoding
requires `models/int16_reference_bundle_v2_calibrated.pt`.

## Commit 13 Residual Fusion

The INT16 backend now routes residual additions through the convolution call
site instead of launching a separate tensor add after every residual block.
For eligible layers this saves one intermediate tensor write and one later read
from global GPU memory. The CUDA extension accepts an optional residual tensor
for both the generic `conv2d_int16` entrypoint and the optimized `1x1` path.

The fallback Python reference applies the same contract:

- convolution accumulation uses the existing INT16 quantization scale,
- the residual tensor is added after requantization,
- the final value is clamped to signed INT16 range,
- residual shape must match the convolution output shape exactly.

The shape check is intentionally performed before CUDA dispatch. A residual
shape mismatch would otherwise be a silent memory-indexing risk in a fused
kernel, and silent residual drift would invalidate bitstream equivalence.

## Performance Journey (P-Frame Encode, 1280×720)

```
FP16 baseline:                   ~16 ms/frame
INT16 start (plan 2):          ~1107 ms/frame
After tiled kernels (plan 3):   ~700 ms/frame
After CUDA graphs (plan 4):     ~280 ms  (INT8 RD collapsed — reverted)
Pure INT16 + encode-only:      ~194–200 ms/frame
+ residual fusion (current):   ~178–197 ms/frame  ← current state
```

## Laptop vs Jetson Comparison

| Metric                  | x86 Laptop (RTX)       | Jetson Orin Nano       |
| :---------------------- | :--------------------- | :--------------------- |
| P-frame encode (avg)    | ~178–197 ms/frame      | ~2,134 ms/frame        |
| I-frame encode          | ~353 ms (graph)        | ~3,044 ms              |
| Throughput              | ~5 fps                 | 0.468 fps              |
| Decode speed            | ~166 ms/frame          | Not measured           |
| Speed ratio             | baseline               | 10.85× slower          |
| CUDA graphs             | Enabled (stable)       | Disabled (OOM risk)    |
| Real-time gap           | ~6× too slow           | ~65× too slow          |

The Jetson row comes from the full 1390-frame sequence described in
`docs/jetson.md`: 3044.333 seconds total encode time, 2134.768 ms/frame, and
0.468 fps. The x86 row is the optimized laptop profile after encode-only mode,
CUDA graph coverage for stable submodules, and residual fusion.

### Dominant P-Frame Stages (Laptop, Steady State)

| Stage                        | GPU ms   | % of total |
| :--------------------------- | :------- | :--------- |
| `reconstruct_frame_enc`      | 41.0     | 21.2%      |
| `decode_feature_enc`         | 36.0–38.4 | 18.6–19.9% |
| `res_prior_param_decoder_enc`| 32.2–33.3 | 16.6–17.2% |
| `extract_context_enc`        | 33.1–39.0 | 17.1–20.2% |
| `encode_y`                   | 30.2–30.9 | 15.6–16.0% |

These five stages consume ~76% of steady-state P-frame GPU time and are the
primary targets for kernel fusion in Plan 13.2.

## Async Entropy Prep — Reverted

Plan 13.1 attempted to overlap entropy preparation with GPU kernels using
CUDA streams. While it reduced CPU wait time (`compress_prior_2x` from
123 ms → 4 ms, a −96.6% local improvement), it caused a net **+14.93%
(+31 ms/frame)** wall-clock regression end-to-end:

| Metric                  | Before       | After        | Delta        |
| :---------------------- | :----------- | :----------- | :----------- |
| `compress_prior_2x` CPU | 123.94 ms    | 4.27 ms      | −119.67 ms   |
| Avg frame encode time   | 208.78 ms    | 239.95 ms    | **+31.18 ms** |
| Bitstream SHA256        | matched      | matched      | Tier A safe  |

Nsight Systems comparison of `--enable_async_entropy_prep` off vs on showed the
async side-stream was not genuinely overlapping with main-stream GPU work. The
D2H pinned-copy overhead and `ready_event.synchronize()` stall recovered the
same serialized bubble with added management cost.

The `is_async_entropy_prep_active()` method now unconditionally returns `False`.
The implementation is preserved as reference code with the `--enable_async_entropy_prep`
CLI flag remaining for future experimentation.

## Next Optimization Targets (Plan 13.2–13.4)

| Phase | Focus                                  | Expected Gain       |
| :---- | :------------------------------------- | :------------------ |
| 13.2  | Kernel fusion in 5 dominant stages     | ~25 ms/frame        |
| 13.2  | Narrow-channel specialist kernels      | ~5–10 ms/frame      |
| 13.3  | Expanded CUDA graph coverage           | ~10–15 ms/frame     |
| 13.4  | Jetson-specific kernel/build audit     | 2–3× on Jetson      |

## Source-Controlled Evidence

The repository tracks only small evidence summaries:

| Artifact | Contents |
| :------- | :------- |
| `assets/metrics/plan19_summary.example.json` | Plan 13/19 performance milestones, async entropy revert, cross-device metrics |
| `assets/metrics/validation_gpu.example.csv` | Example validation table for x86 and Jetson runs |
| `assets/manifests/calibration_manifest.example.json` | Six-class calibration clip manifest template |

Raw traces, YUV files, encoded bitstreams, model checkpoints, and generated
bundles remain local artifacts. Record their paths and SHA-256 values in sidecar
JSON rather than committing the binary payloads.

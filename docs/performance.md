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

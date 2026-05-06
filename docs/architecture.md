# Architecture

## System Goal

The system is organized as a deterministic INT16 neural video codec runtime.
The architecture separates the fast FP16 research baseline from the INT16
interoperability profile so that performance experiments do not blur the
bitstream contract.

## Runtime Flow

```text
source video
  -> preprocessing and padding
  -> DMCI I-frame INT16 path
  -> DMC P-frame INT16 path
  -> frozen entropy model and rANS stream writer
  -> deterministic .bin bitstream
  -> rANS stream reader
  -> INT16 decode and local reconstruction
  -> output frames / MP4 mux
```

## Repository Modules

- `src/utils/`: video IO, metrics, stream helpers, deterministic environment
  setup, and equivalence metadata.
- `src/cpp/`: rANS entropy-coder extension and future native helpers.
- `src/layers/`: INT16 quantization contract, CUDA extension loader, CUDA
  kernels, and backend runners.
- `src/models/`: DMCI and DMC model wrappers, entropy models, and INT16
  reference runtime.
- `scripts/`: model export, entropy freezing, calibration, download, and
  operational utilities.
- `tests/`: parity, bundle-load, entropy-loop, and bitstream-equivalence tests.
- `docs/`: design decisions, hardware constraints, and validation protocol.
- `assets/`: small metrics, manifests, and documentation evidence only.

## Hardware Assumptions

The target runtime is designed for CUDA-capable NVIDIA devices, including
laptop GPUs and Jetson Orin-class edge hardware. CUDA extensions should degrade
to slower Python/PyTorch reference paths when they cannot be built, but the
performance profile assumes successful extension loading.

Jetson-class devices are memory constrained relative to desktop GPUs. CUDA
Graphs and preallocated activation pools must therefore remain configurable
rather than hard-coded as always-on behavior.

## CUDA Kernel Strategy

The INT16 backend keeps the Python reference path as the correctness oracle,
then routes eligible CUDA tensors through native kernels. The optimized path
prioritizes three hot surfaces:

- 1x1 convolutions use a blocked INT16 kernel with optional residual fusion
  because these layers dominate channel mixing in the codec networks.
- General convolutions use shared-memory tiling to reduce repeated global
  memory loads while preserving the same signed rounding and clamp contract as
  the reference implementation.
- Depthwise 3x3 WSiLU-style blocks can fuse LUT activation with convolution,
  avoiding a separate activation tensor write before the depthwise pass.

These kernels are performance accelerators only. If the extension cannot be
built for the local PyTorch/CUDA/toolchain combination, imports must still
succeed and the runtime must fall back to the deterministic reference path.

## CUDA Graphs And Equivalence Checks

The INT16 reference runtime includes fixed-shape CUDA graph caches for stable
subgraphs. Graph capture is a launch-overhead optimization; dynamic entropy
coding, mutable DPB state, and reset-frame behavior remain outside any
determinism claim unless the produced `.bin` stream is checked directly.

Local equivalence checks live in `tools/compare_bitstreams.py` and
`src/utils/equivalence.py`. The tool reports SHA-256 digests, sibling metrics
JSON metadata when present, and optional byte-for-byte comparison results.

## Experimental INT8 Route

The runtime contains a gated INT8 Tensor Core prototype for eligible 1x1
convolutions. It is disabled by default and documented as an engineering pivot
in `docs/int8_pivot.md`. Pure INT16 remains the default interoperability
profile; INT8 runs require separate bitstream-equivalence and quality evidence.

## Engineering Boundary

This repository will not track checkpoints, generated bitstreams, raw video,
YUV data, or large profiler dumps. Source commits should reference these
artifacts through paths, checksums, small manifests, or structured summaries.

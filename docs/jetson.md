# Jetson Deployment Guide

## Target Hardware

| Spec               | Value                                     |
| :------------------ | :--------------------------------------- |
| Device              | NVIDIA Jetson Orin Nano 8 GB             |
| GPU                 | Ampere SM87, 1024 CUDA cores             |
| Memory              | 8 GB unified (shared CPU/GPU)            |
| Compute Capability  | 8.7                                      |
| JetPack             | 6.x (L4T)                               |
| PyTorch             | 2.x (ARM wheel from NVIDIA)             |
| CUDA                | 12.x (bundled with JetPack)             |

## Cross-Device Determinism

The INT16 runtime produces byte-identical bitstreams across Jetson (ARM) and
x86 Windows workstations. This has been validated on a 1390-frame 1280×720
clip:

| Metric             | Jetson Encode          | x86 Decode            |
| :----------------- | :--------------------- | :-------------------- |
| Total bits         | 16,364,584             | 16,364,584            |
| Total bytes        | 2,045,573              | 2,045,573             |
| Bitrate            | 353.19 kbps            | 353.19 kbps           |
| SHA-256            | Matched                | Matched               |

## Memory Constraints

The Jetson Orin Nano has only 8 GB of unified memory shared between CPU and
GPU. This creates several constraints:

### CUDA Graphs — Disabled on Jetson

> **Do NOT enable `DCVC_ENABLE_INT16_PFRAME_GRAPHS=1` on Jetson.**

CUDA graph capture pins all intermediate tensors in memory for the lifetime of
the graph. On 8 GB unified memory, this causes OOM during P-frame steady-state
encoding. The environment variable `DCVC_ENABLE_INT16_PFRAME_GRAPHS` defaults
to disabled and must remain so on Jetson.

### Model Bundle Loading

The INT16 bundle (`~200–300 MB`) plus the rANS entropy state plus runtime
intermediate tensors consume a significant fraction of available memory. To
minimize pressure:

- Use `torch.load(..., map_location="cpu")` then `.to(device)` — do not load
  directly to CUDA.
- Clear the DPB (`p_frame_net.clear_dpb()`) between clip encodes.
- Close the YUV reader after encoding to release file-mapped memory.

### Batch Size

Only `batch_size=1` is supported. The kernel tile sizes and CUDA graph capture
shapes assume single-image batches.

## Build Requirements

### Extension Compilation

The CUDA extension must be compiled with the correct architecture flag:

```bash
# On Jetson Orin Nano (SM 8.7)
TORCH_CUDA_ARCH_LIST="8.7" python build_int16_cuda.py
```

Verify the architecture flag in the build log:

```bash
nvcc --version  # Should show CUDA 12.x
python -c "import torch; print(torch.cuda.get_device_capability())"  # (8, 7)
```

### rANS Extension

```bash
cd src/cpp && python setup.py build_ext --inplace
```

The rANS C++ extension builds on ARM without modification. Ensure `g++` and
`python3-dev` headers are available.

## Performance

### Current State

| Metric                  | Jetson Orin Nano       | x86 Laptop (RTX)      |
| :---------------------- | :--------------------- | :-------------------- |
| P-frame encode          | ~2,134 ms/frame        | ~178–197 ms/frame     |
| I-frame encode          | ~3,044 ms/frame        | ~353 ms/frame         |
| Throughput              | 0.468 fps              | ~5 fps                |
| Speed ratio             | 10.85× slower          | baseline              |

### Profiling

To identify Jetson-specific bottlenecks, collect per-stage profiles:

```bash
python encode_mp4_to_bin.py \
  --input_mp4 test.mp4 \
  --frames 32 \
  --profile_pframe_stages \
  --output_dir outputs/jetson_profile
```

Compare the stage breakdown against the laptop profile. Jetson may show
different dominant stages due to lower SM count and memory bandwidth.

### Kernel Tuning Checklist

After profiling, check for register spills and occupancy issues:

```bash
# Analyze the INT16 convolution kernel
ncu --target-processes all \
    --set full \
    --kernel-name conv2d_int16_tiled_kernel \
    python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 2

# Check register count and spills
ncu --metrics launch__registers_per_thread \
    python encode_mp4_to_bin.py --input_mp4 test.mp4 --frames 2
```

If register spills are significant, consider:

- Adding `__launch_bounds__(MAX_THREADS, MIN_BLOCKS)` to hot kernels.
- Reducing shared memory tile dimensions in `conv2d_int16_tiled_kernel` if
  Orin's L1 capacity is insufficient for the desktop tile size.
- Using `BLOCK_C=4` instead of `BLOCK_C=8` for narrow channel layers (48, 64,
  96 channels) to improve SM occupancy.

## Real-Time Path — Strategic Options

The 10.85× speed gap means the current INT16 path will not reach real-time
(33 ms/frame at 30 fps) on Jetson. Three viable strategies:

### Option A: Hybrid Mode (Lowest Engineering Cost)

Run FP16 encode on Jetson for speed. Use INT16 only for periodic cross-device
verification. The Jetson produces FP16 bitstreams for real-time use and
periodically re-encodes a reference clip in INT16 to confirm bundle consistency.

### Option B: Lighter Model (Medium Cost)

Train a DCVC-RT variant with reduced channel widths (e.g., half-width) for edge
deployment. Accept a modest quality reduction in exchange for lower compute. This
requires access to the training infrastructure.

### Option C: Deep Kernel Tuning (Highest Cost)

Invest in SM87-specific kernel variants after Plan 13.4 Nsight profiles.
Realistic improvement: 2–3×, but unlikely to close the full 10× gap.

## Environment Variables

| Variable                              | Default | Effect on Jetson                    |
| :------------------------------------ | :------ | :---------------------------------- |
| `DCVC_ENABLE_INT16_PFRAME_GRAPHS`     | `0`     | Must stay `0` — OOM risk           |
| `DCVC_INT16_ENCODE_ONLY`              | `1`     | Keep `1` — saves ~100 ms/frame     |
| `DCVC_PROFILE_INT16_PIPELINE`         | `0`     | Set `1` for profiling only         |
| `DCVC_INT16_ASYNC_ENTROPY_PREP`       | `0`     | Reverted — net negative            |

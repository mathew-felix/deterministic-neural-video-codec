# Provenance

## Project Identity

Deterministic Neural Video Codec is an independent engineering reconstruction
of a deterministic INT16 runtime for a DCVC-RT-family neural video codec. It is
not a Microsoft product and is not affiliated with, endorsed by, or sponsored
by Microsoft.

## Upstream Lineage

The engineering work is derived from the public DCVC codebase and the DCVC-RT
research direction. The source project family provides the neural codec
architecture, entropy-coding structure, model checkpoints, and fast FP16
deployment context that this repository remasters into a deterministic INT16
systems narrative.

Relevant upstream references:

- Microsoft DCVC public repository: `https://github.com/microsoft/DCVC`
- DCVC-RT paper: `https://arxiv.org/abs/2502.20762`
- DCVC project page: `https://dcvccodec.github.io/`

## Local Research Evidence

This repository is reconstructed from the `dcvc_portal` workspace, including:

- Project plans and reports documenting INT16 backend design, CUDA kernel
  optimization, INT8 Tensor Core experiments, P-frame reset debugging, and
  Jetson-to-laptop validation.
- The final standalone runtime represented by `dcvc-rt-int16/`.
- Intermediate standalone and encoder-focused packages represented by
  `dcvc_p15/` and `encode_p10/`.

## Boundary

Large model weights, video files, YUV assets, generated bitstreams, and raw
profiling outputs are not source code for this repository. They should remain
external artifacts referenced by path, checksum, or small manifests.


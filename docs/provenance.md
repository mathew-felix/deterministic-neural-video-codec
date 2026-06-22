# Provenance

## Project Identity

Deterministic Neural Video Codec is an independent software engineering project
that packages and validates a deterministic INT16 runtime for a DCVC-RT-family
neural video codec. It is not a Microsoft product and is not affiliated with,
endorsed by, or sponsored by Microsoft.

## Upstream Lineage

The engineering work is derived from the public DCVC codebase and DCVC-RT
implementation lineage. The source project family provides the neural codec
architecture, entropy-coding structure, model checkpoints, and fast FP16
deployment context used by this INT16 runtime.

Relevant upstream references:

- Microsoft DCVC public repository: `https://github.com/microsoft/DCVC`
- DCVC-RT technical reference: `https://arxiv.org/abs/2502.20762`
- DCVC project page: `https://dcvccodec.github.io/`

## Boundary

Large model weights, video files, YUV assets, generated bitstreams, and raw
profiling outputs are not source code for this repository. They should remain
external artifacts referenced by path, checksum, or small manifests.


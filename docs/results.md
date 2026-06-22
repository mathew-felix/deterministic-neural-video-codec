# Results

This page records verified local numbers only. Do not add placeholder values to
this file.

## Full `video.mp4` Validation

Artifact: `assets/metrics/video_full_local_2026-05-14.json`

Environment: NVIDIA GeForce RTX 3070 Ti Laptop GPU, Python 3.12.0,
PyTorch 2.2.2+cu121, CUDA 12.1 runtime, driver 596.21.

| Field | Value |
|---|---:|
| Tests | 35 passed |
| Frames | 2593 |
| Resolution | 1280x720 |
| FPS | 30 |
| QP I/P | 32 / 32 |
| Bitstream bytes | 4,817,350 |
| Bitrate | 445.87890474354026 kbps |
| BPP | 0.016126985848652353 |
| Average encode | 223.58980925285022 ms/frame |
| Average decode | 207.80797111878198 ms/frame |
| Average RGB PSNR | 28.896943432512362 dB |
| Average RGB MS-SSIM | 0.9336242754667393 |

Bitstream SHA-256:

```text
d837ca00cc367c50a63d25ddff19d53a7cc7496cd66099ab73047249c4ad09ed
```

Bundle SHA-256:

```text
c1fc2341d3faf28f16b8e77c0869aecddade674aa0b43be2b64c516f49a8554f
```

## Data Folder Smoke Test

Verified on June 22, 2026 with
`../data/test/multi_cow_night_1.mp4`. This is a short software smoke test, not
a benchmark.

| Field | Value |
|---|---:|
| Source frames available | 606 |
| Frames encoded/decoded | 2 |
| Resolution | 1280x720 |
| FPS | 30 |
| QP I/P | 32 / 32 |
| Bitstream bytes | 5,343 |
| Decoded MP4 bytes | 34,238 |
| Bitrate | 641.16 kbps |
| BPP | 0.0231901042 |
| Average encode | 210.6320 ms/frame |
| Average decode | 128.9049 ms/frame |
| Average RGB PSNR | 36.7375 dB |
| Average RGB MS-SSIM | 0.974147 |

Bitstream SHA-256:

```text
2c663656b79b7bc0cbc0cad12e7a59e8303fcaeb164cfa93c9e003a1bbe1643a
```

## Scope

These numbers prove that the local runtime can encode, decode, and measure the
complete test clip on the listed RTX 3070 Ti Laptop GPU environment. They do
not prove cross-device determinism. Cross-device proof requires matching
bitstream SHA-256 values from at least two machines inside the same equivalence
class.

## Same-Machine Determinism Smoke

Two separate 32-frame encodes of `video.mp4` on the same machine produced
byte-identical bitstreams.

| Field | Value |
|---|---:|
| Frames | 32 |
| Bitstream bytes | 91,438 |
| SHA-256 equal | true |
| Bytewise equal | true |

SHA-256:

```text
0373a80d36a63f9327c730ee094a57c4954dcdf33f99a8a37cbe6641acdcaf42
```

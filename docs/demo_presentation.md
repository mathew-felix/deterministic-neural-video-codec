# Demo Presentation Guide

Use this guide for a portfolio page, GitHub README walkthrough, or a short
screen recording. The goal is to show the engineering clearly without asking a
recruiter to install CUDA or run a long encode.

## 10-Second Recruiter Version

> This is a video compression project. I built the INT16 runtime that compresses
> an MP4 into a `.bin` bitstream, decodes it back to MP4, and verifies the result
> with hashes and quality metrics. It is derived from Microsoft DCVC/DCVC-RT;
> my work is the software engineering layer: runtime packaging, CLI tools,
> setup checks, tests, and reproducible validation.

Show:

1. The README project snapshot.
2. The before/after GIF.
3. The measured results table.

Stop there for a recruiter.

## 45-Second Screen Recording

Record the screen in this order:

1. Open the top of `README.md`.
2. Scroll to the architecture diagram.
3. Scroll to the demo GIF.
4. Show the short smoke-test commands.
5. Open `assets/demo_terminal_output.txt`.
6. End on the measured results table.

Suggested narration:

> This project turns a DCVC-RT-family neural video codec into a runnable
> software package. The input is an MP4. FFmpeg prepares frames, PyTorch and CUDA
> run the INT16 codec path, rANS writes a deterministic `.bin` bitstream, and
> the decoder reconstructs an MP4. The important engineering result is that the
> repo includes CLI tools, setup checks, tests, hashes, environment metadata,
> and quality metrics instead of only a model script.

## 2-Minute Engineer Version

Use this when an interviewer asks how it works.

1. Start with the problem:
   FP16 neural codec arithmetic can diverge across hardware, which is bad when
   encoder and decoder machines differ.

2. Explain the implementation:
   The runtime uses a signed INT16 arithmetic profile, fixed quantization
   scales, CUDA/PyTorch model wrappers, FFmpeg video I/O, and rANS entropy
   coding.

3. Show the command flow:
   `check_setup.py`, `encode_mp4_to_bin.py`, `decode_bin_to_mp4.py`, and
   `tools/compare_bitstreams.py`.

4. Show evidence:
   The committed local result encoded and decoded 2593 frames at 1280x720,
   produced a 4,817,350-byte bitstream, and measured 28.8969 dB RGB PSNR and
   0.933624 RGB MS-SSIM.

5. State the boundary:
   The committed evidence is local full-video validation plus same-machine
   determinism. Cross-device claims require a separate validation table with
   matching hashes.

## What To Show

- `assets/demo_before_after.gif`
- `assets/demo_terminal_output.txt`
- `assets/metrics/video_full_local_2026-05-14.json`
- `docs/results.md`
- `codec.py`

## What Not To Show First

- Long calibration details
- Experimental INT8/QAT work
- Full-video encode running live
- Large local output folders
- Unverified cross-device claims

## Portfolio Caption

```text
Deterministic INT16 neural video codec runtime built with PyTorch, CUDA,
FFmpeg, and rANS entropy coding. Encodes MP4 video into a compact,
hash-checkable bitstream and decodes it back to MP4 with reproducible metrics.
```

## Interview Close

> The project is intentionally not presented as a production video product. It
> is a systems engineering showcase: runtime packaging, deterministic numeric
> behavior, native extensions, video I/O, entropy coding, and reproducible test
> evidence.

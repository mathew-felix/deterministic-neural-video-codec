# Determinism

## Deterministic Contract

The gold profile for this repository is an INT16 codec path whose encoded
bitstream is reproducible under a fixed equivalence class. A deterministic
claim is valid only when the following inputs are fixed:

- Source video frames and preprocessing path.
- Model bundle and bundle checksum.
- Frozen entropy state.
- QP settings and frame count.
- Runtime flags, including graph, encode-only, and experimental paths.
- CUDA device class, driver/runtime versions, and PyTorch build.
- Codec source revision and dirty-state status.

## Tier A Evidence

Tier A determinism means one of the following:

- Two `.bin` bitstreams are byte-identical.
- Two `.bin` bitstreams have the same SHA-256 digest and the compared files are
  known to be the complete codec outputs for the same equivalence class.

Visual similarity, PSNR closeness, matching total byte counts, or successful
decode alone are useful diagnostics, but they are not Tier A proof.

## Why INT16

The FP16 path is the speed-oriented deployment path. It is appropriate for
throughput, but floating-point execution is not a sufficient interoperability
contract across heterogeneous devices. Small floating-point differences can
alter entropy contexts, reconstructed reference state, and later P-frame
prediction state.

The INT16 path makes rounding, clamping, scale-index lookup, local
reconstruction, and entropy context regeneration explicit. That is the basis
for cross-device codec behavior.

## Known Risk Areas

- P-frame reset frames must rebuild encoder and decoder DPB state identically.
- Entropy CDFs must be frozen and bundled rather than rebuilt implicitly.
- CUDA Graph capture must not hide mutable state or reset-path differences.
- Experimental INT8 Tensor Core paths can be fast but may perturb
  reconstruction and entropy contexts enough to break rate-distortion quality.
- Jetson-class devices may need different graph and memory settings even when
  bitstream semantics remain the same.


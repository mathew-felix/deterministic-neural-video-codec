# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- `docs/model_setup.md`: step-by-step guide for obtaining the DCVC-RT checkpoint
  and building the INT16 bundle locally without redistributing model weights.
- `NOTICE` file with upstream Microsoft DCVC MIT license attribution.
- `CONTRIBUTING.md` with contribution workflow and determinism contract.

### Changed
- `README.md`: expanded with Model Setup section, project structure, full
  installation and usage instructions, and Apache 2.0 + NOTICE attribution.
- `LICENSE`: filled copyright year (2025).

---

## [1.0.0] — 2025-05-01

### Added
- Initial INT16 runtime for DCVC-RT-family neural video codec.
- `encode_mp4_to_bin.py`: command-line encoder producing `.bin` bitstreams.
- `decode_bin_to_mp4.py`: command-line decoder from `.bin` to MP4.
- `tools/compare_bitstreams.py`: SHA-256 and byte-for-byte bitstream comparison
  utility for cross-device validation.
- `src/layers/int16_backend.py`: INT16 quantization kernels with power-of-two
  scales.
- `src/models/int16_reference.py`: INT16 P-frame and I-frame reference models
  with optional CUDA Graph acceleration.
- `scripts/calibrate_int16_bundle.py`: activation statistics collection and
  scale refinement for INT16 bundles.
- `assets/manifests/calibration_manifest.example.json`: calibration manifest
  template covering six content types.
- `docs/architecture.md`, `docs/calibration.md`, `docs/validation.md`,
  `docs/jetson.md`, `docs/provenance.md`: public documentation.
- Apache 2.0 `LICENSE`.

# Release Checklist

Use this checklist before publishing a model bundle or making public
determinism claims.

## Bundle

- Build or obtain `models/int16_bundle_v1.0.0.pt`.
- Compute the bundle SHA-256:

```bash
python - <<'PY'
import hashlib
from pathlib import Path
p = Path("models/int16_bundle_v1.0.0.pt")
h = hashlib.sha256()
with p.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print(h.hexdigest())
PY
```

- Upload the bundle to a GitHub Release.
- Set `release.bundle_url` in `config/config.example.yaml` or document that
  users must pass `--url` to `scripts/download_models.py`.
- Record the bundle SHA-256 in release notes, README, and validation artifacts.

## Local Smoke Test

```bash
python scripts/check_setup.py --input_mp4 video.mp4 --require_cuda --require_bundle
python encode_mp4_to_bin.py --input_mp4 video.mp4 --frames 32 --output_dir outputs/release_smoke_a
python encode_mp4_to_bin.py --input_mp4 video.mp4 --frames 32 --output_dir outputs/release_smoke_b
python tools/compare_bitstreams.py outputs/release_smoke_a/*.bin outputs/release_smoke_b/*.bin --expect_equal
```

## Quality Metrics

```bash
python decode_bin_to_mp4.py --input_bin outputs/release_smoke_a/<bitstream>.bin
python scripts/quality_metrics.py --reference_mp4 video.mp4 --decoded_mp4 outputs/release_smoke_a/<decoded>.mp4 --output_json outputs/release_smoke_a/quality.json
```

## Cross-Device Evidence

- Run the same encode command on each target GPU generation.
- Verify identical `.bin` SHA-256 values with `tools/compare_bitstreams.py`.
- Save hardware, driver, CUDA, PyTorch, bundle SHA-256, command flags, latency,
  bitrate, PSNR, MS-SSIM, and bitstream SHA-256 in a JSON artifact.

## CI

- CI runs Python compile checks, `scripts/check_setup.py --json`, and non-CUDA
  tests.
- CUDA parity and encode/decode tests require a local NVIDIA machine.

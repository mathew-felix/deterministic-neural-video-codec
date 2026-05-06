# models/

This directory holds model files that are **not tracked in git**. All `.pt`,
`.pth`, and `.tar` files are gitignored. The directory itself is kept in the
repository via this README.

---

## File Naming Convention

| File | Source | Description |
|---|---|---|
| `int16_bundle_v<VERSION>.pt` | GitHub release / `pipeline.py` | Calibrated INT16 bundle — the file required to run encode and decode |
| `cvpr2025_image.pth.tar` | [microsoft/DCVC](https://github.com/microsoft/DCVC) | Upstream DCVC-RT I-frame FP16 checkpoint |
| `cvpr2025_video.pth.tar` | [microsoft/DCVC](https://github.com/microsoft/DCVC) | Upstream DCVC-RT P-frame FP16 checkpoint |
| `frozen_entropy_state.pt` | `scripts/freeze_entropy_cdfs.py` | Intermediate: frozen rANS CDF tables (build step 1) |
| `bundle_sha256.txt` | generated after build | SHA-256 digest of the bundle — commit this file |

### Bundle versioning

The INT16 bundle filename encodes the release version:

```
int16_bundle_v1.0.0.pt    ← initial public release
int16_bundle_v1.1.0.pt    ← re-calibrated with more clips
int16_bundle_v2.0.0.pt    ← breaking change (new architecture or quantization scheme)
```

Use semantic versioning:
- **Patch** (`v1.0.x`): no change to bitstream, documentation or bug fix only
- **Minor** (`v1.x.0`): re-calibrated bundle, same architecture — bitstreams differ but decoder is compatible
- **Major** (`vX.0.0`): architecture or quantization contract change — old bitstreams cannot be decoded

`config.yaml` points to the bundle by filename. Both encoder and decoder must
use the same bundle version for bitstream equivalence.

---

## How to Obtain the Bundle

**Download from GitHub release:**
```bash
python scripts/download_models.py
```

**Build from scratch:**
```bash
python pipeline.py
```

See `REPRODUCE.md` for the full step-by-step guide and
`docs/model_setup.md` for details on each pipeline step.

---

## Recording the Bundle SHA-256

After building or downloading the bundle, record its SHA-256 and commit the
digest file:

```bash
python -c "
import hashlib, pathlib
for name in sorted(pathlib.Path('models').glob('int16_bundle_*.pt')):
    digest = hashlib.sha256(name.read_bytes()).hexdigest()
    print(digest, name.name)
    pathlib.Path('models/bundle_sha256.txt').write_text(f'{digest}  {name.name}\n')
"
```

Commit `models/bundle_sha256.txt`. This file is the integrity anchor — both
encoder and decoder machines should verify their bundle matches this digest
before comparing bitstreams.

---

## What Must NOT Go Here

- Training checkpoints not derived from the DCVC-RT public release
- Private or proprietary model weights
- Large raw calibration data or YUV files

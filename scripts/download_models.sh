#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
PRIMARY_README_URL="https://github.com/microsoft/DCVC/blob/main/README.md"
PRIMARY_SHARE_URL="https://1drv.ms/f/c/2866592d5c55df8c/Esu0KJ-I2kxCjEP565ARx_YB88i0UnR6XnODqFcvZs4LcA?e=by8CO8"
BACKUP_SHARE_URL="https://1drv.ms/f/c/2866592d5c55df8c/EozfVVwtWWYggCitBAAAAAABbT4z2Z10fMXISnan72UtSA?e=BID7DA"

EXPECTED_REQUIRED=(
  "int16_reference_bundle_v2_calibrated.pt"
)
EXPECTED_RECOMMENDED=(
  "cvpr2025_image.pth.tar"
  "cvpr2025_video.pth.tar"
)
EXPECTED_OPTIONAL=(
  "frozen_entropy.pt"
)

mkdir -p "$MODELS_DIR"

all_present=true
for name in "${EXPECTED_REQUIRED[@]}"; do
  if [[ ! -f "$MODELS_DIR/$name" ]]; then
    all_present=false
  fi
done

if [[ "$all_present" == true ]]; then
  echo "Models already present in: $MODELS_DIR"
  for name in "${EXPECTED_REQUIRED[@]}" "${EXPECTED_RECOMMENDED[@]}" "${EXPECTED_OPTIONAL[@]}"; do
    if [[ -f "$MODELS_DIR/$name" ]]; then
      echo "  found: $name"
    fi
  done
  exit 0
fi

if [[ -n "${DCVC_MODEL_SOURCE_DIR:-}" ]]; then
  source_dir="${DCVC_MODEL_SOURCE_DIR%/}"
  echo "Copying model files from: $source_dir"
  for name in "${EXPECTED_REQUIRED[@]}" "${EXPECTED_RECOMMENDED[@]}" "${EXPECTED_OPTIONAL[@]}"; do
    if [[ -f "$source_dir/$name" ]]; then
      cp -f "$source_dir/$name" "$MODELS_DIR/$name"
      echo "  copied: $name"
    fi
  done
fi

if [[ -n "${DCVC_MODEL_URL_PREFIX:-}" ]]; then
  url_prefix="${DCVC_MODEL_URL_PREFIX%/}"
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is not available, so automatic download cannot run."
  else
    echo "Downloading model files from: $url_prefix"
    for name in "${EXPECTED_REQUIRED[@]}" "${EXPECTED_RECOMMENDED[@]}" "${EXPECTED_OPTIONAL[@]}"; do
      echo "  downloading: $name"
      curl -fL "$url_prefix/$name" -o "$MODELS_DIR/$name"
    done
  fi
fi

if [[ -f "$MODELS_DIR/int16_reference_bundle_v2_calibrated.pt" ]]; then
  echo "Model files are ready in: $MODELS_DIR"
  for name in "${EXPECTED_RECOMMENDED[@]}" "${EXPECTED_OPTIONAL[@]}"; do
    if [[ -f "$MODELS_DIR/$name" ]]; then
      echo "  found: $name"
    else
      echo "  missing (optional): $name"
    fi
  done
  exit 0
fi

cat <<EOF
Automatic download is unreliable for the official OneDrive shares.

Download the checkpoints from:
  - $PRIMARY_README_URL
  - Primary folder: $PRIMARY_SHARE_URL
  - Backup folder:  $BACKUP_SHARE_URL

Place these filenames in:
  $MODELS_DIR

Required:
  - int16_reference_bundle_v2_calibrated.pt

Recommended:
  - cvpr2025_image.pth.tar
  - cvpr2025_video.pth.tar

Optional:
  - frozen_entropy.pt

If you already have the files elsewhere, rerun with:
  DCVC_MODEL_SOURCE_DIR=/path/to/model/folder bash scripts/download_models.sh

If you have direct file URLs, rerun with:
  DCVC_MODEL_URL_PREFIX=https://example.com/models bash scripts/download_models.sh
EOF

exit 1

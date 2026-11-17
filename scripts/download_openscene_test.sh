#!/usr/bin/env bash
# Download OpenScene v1.1 TEST camera blobs for NAVSIM navtest (no lidar).
set -euo pipefail
DEST="${DEST:-/data2/navsim}"
DL="$DEST/downloads"
URL_BASE="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera"
NPROC="${NPROC:-6}"
mkdir -p "$DL" "$DEST/sensor_blobs"
cd "$DL"

download_one() {
  local i="$1"
  local f="openscene_sensor_test_camera_${i}.tgz"
  if [[ -f "${f}.ok" ]]; then
    echo "[skip] $f"
    return 0
  fi
  echo "[dl] $f"
  wget -c --tries=40 --timeout=90 --retry-connrefused \
    -O "$f" "${URL_BASE}/${f}"
  touch "${f}.ok"
  echo "[ok] $f $(du -h "$f" | awk '{print $1}')"
}

export -f download_one
export URL_BASE
printf "%s\n" {0..31} | xargs -I {} -P "$NPROC" bash -c 'download_one "$@"' _ {}

echo "[extract] sequential extract into $DEST"
for i in {0..31}; do
  f="openscene_sensor_test_camera_${i}.tgz"
  marker="${f}.extracted"
  if [[ -f "$marker" ]]; then
    echo "[skip extract] $f"
    continue
  fi
  echo "[extract] $f"
  tar -xzf "$f" -C "$DEST"
  touch "$marker"
done

if [[ -d "$DEST/openscene-v1.1/sensor_blobs" ]]; then
  mkdir -p "$DEST/sensor_blobs"
  if [[ -d "$DEST/sensor_blobs/test" ]]; then
    echo "[merge] sensor_blobs already exists, rsync"
    rsync -a "$DEST/openscene-v1.1/sensor_blobs/" "$DEST/sensor_blobs/test/"
  else
    mv "$DEST/openscene-v1.1/sensor_blobs" "$DEST/sensor_blobs/test"
  fi
  rm -rf "$DEST/openscene-v1.1"
fi

echo "[done] cameras"
du -sh "$DEST/sensor_blobs/test" || true
find "$DEST/sensor_blobs/test" -type f 2>/dev/null | wc -l

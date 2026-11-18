#!/usr/bin/env bash
# Download OpenDriveVLA-0.5B + RnB (stanfordasl/nuscenes-waypoints-model) teacher weights.
# Draft heads ship in weights/; teachers are ~10GB and are pulled from Hugging Face.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${TRAJSPEC_TEACHER_DIR:-$ROOT/teachers}"
mkdir -p "$OUT"

if command -v hf >/dev/null 2>&1; then
  HF=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli download)
else
  echo "Install huggingface_hub: pip install -U 'huggingface_hub[cli]'"
  exit 1
fi

echo "[1/2] OpenDriveVLA-0.5B -> $OUT/OpenDriveVLA-0.5B"
echo "      (gated: accept terms at https://huggingface.co/OpenDriveVLA/OpenDriveVLA-0.5B and hf auth login)"
"${HF[@]}" OpenDriveVLA/OpenDriveVLA-0.5B --local-dir "$OUT/OpenDriveVLA-0.5B"

echo "[2/2] RnB waypoints (Qwen3-VL-4B) -> $OUT/nuscenes-waypoints-model"
"${HF[@]}" stanfordasl/nuscenes-waypoints-model --local-dir "$OUT/nuscenes-waypoints-model"

cat <<EOF

Done. Export on the new machine:

  export TRAJSPEC_OPENDRIVE_CKPT=$OUT/OpenDriveVLA-0.5B
  export TRAJSPEC_RNB_CKPT=$OUT/nuscenes-waypoints-model
  export TRAJSPEC_OPENDRIVE_REPO=/path/to/OpenDriveVLA   # clone DriveVLA/OpenDriveVLA
  export TRAJSPEC_NUSC_ROOT=/path/to/nuscenes

Draft heads (already in repo):
  weights/draft_opendrive_nusc.pt
  weights/draft_waypoints_nusc.pt
EOF

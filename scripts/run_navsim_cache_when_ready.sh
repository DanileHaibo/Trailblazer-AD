#!/usr/bin/env bash
# After OpenScene cameras are extracted, cache AutoVLA teacher on navtest (2x L20).
set -euo pipefail
ROOT=/home/xinhchen2/hhb/TrajSpecDec
PY=/data2/conda_env/qwendrive/bin/python
OUT0=$ROOT/results/cache_autovla_navtest_g0.pt
OUT1=$ROOT/results/cache_autovla_navtest_g1.pt
LOG0=$ROOT/results/cache_autovla_navtest_g0.log
LOG1=$ROOT/results/cache_autovla_navtest_g1.log

wait_for_cameras() {
  local marker=/data2/navsim/downloads/openscene_sensor_test_camera_31.tgz.extracted
  echo "[wait] cameras $marker"
  while [[ ! -f "$marker" ]]; do
    sleep 60
  done
  # also wait until extract loop finished
  while ! grep -q '\[done\] cameras' "$ROOT/results/download_openscene_test.log"; do
    sleep 30
  done
}

wait_for_cameras
echo "[cache] start dual GPU"
CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/scripts/cache_autovla_navsim.py" \
  --start 0 --stride 2 --out "$OUT0" --temperature 0.01 > "$LOG0" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PY" "$ROOT/scripts/cache_autovla_navsim.py" \
  --start 1 --stride 2 --out "$OUT1" --temperature 0.01 > "$LOG1" 2>&1 &
pid1=$!
echo "gpu0=$pid0 gpu1=$pid1"
wait $pid0
wait $pid1
"$PY" - <<'PY'
import torch
from pathlib import Path
root = Path("/home/xinhchen2/hhb/TrajSpecDec/results")
a = torch.load(root/"cache_autovla_navtest_g0.pt", map_location="cpu", weights_only=False)
b = torch.load(root/"cache_autovla_navtest_g1.pt", map_location="cpu", weights_only=False)
rows = a["rows"] + b["rows"]
seen=set(); uniq=[]
for r in rows:
    if r["token"] in seen:
        continue
    seen.add(r["token"]); uniq.append(r)
out = root/"cache_autovla_navtest.pt"
torch.save({"rows": uniq, "split": "navtest", "hidden_size": a.get("hidden_size", 2048), "horizon": 10}, out)
print(f"[merge] {out} n={len(uniq)}")
PY
echo "[cache] merged, train draft"
CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/scripts/run_navsim_specdec.py" \
  > "$ROOT/results/specdec_autovla_navsim.log" 2>&1
echo "[done] specdec"

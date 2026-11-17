#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/xinhchen2/hhb/TrajSpecDec
LOG0=$ROOT/results/logs/cache_rnb_g0.log
LOG1=$ROOT/results/logs/cache_rnb_g1.log
PY=/data2/conda_env/unidrivevla/bin/python
cd "$ROOT"

echo "[watch] waiting for cache shards" >&2
while true; do
  done0=0; done1=0
  grep -q '^\[done\]' "$LOG0" 2>/dev/null && done0=1 || true
  grep -q '^\[done\]' "$LOG1" 2>/dev/null && done1=1 || true
  if [[ "$done0" == 1 && "$done1" == 1 ]]; then
    break
  fi
  if ! pgrep -f 'scripts/cache_rnb_nusc.py' >/dev/null; then
    echo "[watch] cache processes exited before [done] done0=$done0 done1=$done1" >&2
    tail -n 40 "$LOG0" "$LOG1" >&2 || true
    exit 1
  fi
  sleep 60
done

echo "[watch] merging" >&2
"$PY" -u scripts/merge_rnb_cache.py \
  "$ROOT/results/cache_rnb_nusc_val_g0.pt" \
  "$ROOT/results/cache_rnb_nusc_val_g1.pt" \
  --out "$ROOT/results/cache_rnb_nusc_val.pt"

echo "[watch] train+score" >&2
CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/run_rnb_specdec.py \
  --cache "$ROOT/results/cache_rnb_nusc_val.pt" \
  --ckpt "$ROOT/results/specdec_rnb_heldout.pt" \
  --out "$ROOT/results/rnb_official_compare.json"

echo "[watch] live latency n=12" >&2
CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/run_live_specdec_rnb.py \
  --n 12 --warmup 2 \
  --ckpt "$ROOT/results/specdec_rnb_heldout.pt" \
  --out "$ROOT/results/live_specdec_rnb.json" \
  || echo "[watch] live failed (scores already written)" >&2
echo "[watch] finished" >&2

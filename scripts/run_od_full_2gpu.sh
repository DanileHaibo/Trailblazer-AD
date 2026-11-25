#!/usr/bin/env bash
# Dual-GPU full OpenDrive val (6019). Each shard is exclusive on its card.
# On CUDA fault a shard aborts+checkpoints; the other keeps going.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/logs

OD_PY="${OD_PY:-/data2/conda_env/unidrivevla/bin/python}"
# 6019 = 3010 + 3009
N0=3010
N1=3009

gpu_ok() {
  local i="$1"
  nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits >/dev/null 2>&1
}

for i in 0 1; do
  gpu_ok "$i" || { echo "[abort] GPU$i unhealthy"; exit 1; }
  used=$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print int($1)}')
  if [[ "$used" -gt 2000 ]]; then
    echo "[abort] GPU$i not exclusive (used=${used}MiB) — refuse dual launch"
    exit 1
  fi
done

echo "[$(date -Is)] launching OD full dual-GPU shards"

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$OD_PY" -u scripts/run_live_specdec_opendrive.py \
  --n "$N0" --warmup 2 --start 0 --shard 0 --num-shards 2 --save-every 50 \
  --out results/live_od_full_shard0.json \
  > results/logs/live_od_full_shard0.log 2>&1 &
PID0=$!
echo $PID0 > results/logs/live_od_full_shard0.pid

CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$OD_PY" -u scripts/run_live_specdec_opendrive.py \
  --n "$N1" --warmup 2 --start 0 --shard 1 --num-shards 2 --save-every 50 \
  --out results/live_od_full_shard1.json \
  > results/logs/live_od_full_shard1.log 2>&1 &
PID1=$!
echo $PID1 > results/logs/live_od_full_shard1.pid

echo "[$(date -Is)] shard0 pid=$PID0 on GPU0; shard1 pid=$PID1 on GPU1"

rc0=0 rc1=0
wait "$PID0" || rc0=$?
wait "$PID1" || rc1=$?
echo "[$(date -Is)] shards done rc0=$rc0 rc1=$rc1"

"$OD_PY" scripts/merge_live_od_shards.py \
  results/live_od_full_shard0.json results/live_od_full_shard1.json \
  --out results/live_specdec_opendrive_full_6019.json \
  | tee results/logs/live_od_full_merge.log

echo "[$(date -Is)] merged -> results/live_specdec_opendrive_full_6019.json"
exit $(( rc0 != 0 ? rc0 : rc1 ))

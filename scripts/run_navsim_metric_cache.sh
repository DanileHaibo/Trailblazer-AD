#!/usr/bin/env bash
# NAVSIM metric cache for navtest (logs + maps only; no cameras needed).
set -euo pipefail
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data2/navsim/maps"
export NAVSIM_EXP_ROOT="/data2/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/home/xinhchen2/hhb/AutoVLA/navsim"
export OPENSCENE_DATA_ROOT="/data2/navsim"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
PY="${PY:-/data2/conda_env/autovla/bin/python}"
mkdir -p "$NAVSIM_EXP_ROOT"
cd "$NAVSIM_DEVKIT_ROOT"
"$PY" navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  worker=sequential \
  cache.cache_path=/data2/navsim/navtest_metric_cache

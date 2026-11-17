#!/usr/bin/env bash
# Slim NAVSIM + nuplan-devkit env for JSON export, metric cache, PDMS scoring.
# AutoVLA GPU inference stays in qwendrive.
set -euo pipefail
PREFIX="${PREFIX:-/data2/conda_env/autovla}"
NAVSIM_ROOT="${NAVSIM_ROOT:-/home/xinhchen2/hhb/AutoVLA/navsim}"

source /home/xinhchen2/miniconda3/etc/profile.d/conda.sh
if [[ ! -x "$PREFIX/bin/python" ]]; then
  conda create -y -p "$PREFIX" python=3.9 pip
fi
"$PREFIX/bin/python" -m pip install -U pip setuptools wheel
"$PREFIX/bin/python" -m pip install \
  "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit@nuplan-devkit-v1.2" \
  hydra-core==1.2.0 omegaconf==2.3.0 \
  numpy==1.23.4 pandas scipy scikit-learn==1.2.2 \
  shapely geopandas pyogrio pyproj fiona rasterio rtree affine \
  pyquaternion casadi==3.6.7 control==0.9.1 \
  opencv-python==4.9.0.80 pillow matplotlib tqdm lz4 \
  pytorch-lightning==2.2.1 torchmetrics \
  --extra-index-url https://download.pytorch.org/whl/cpu
cd "$NAVSIM_ROOT"
"$PREFIX/bin/python" -m pip install -e . --no-warn-conflicts
"$PREFIX/bin/python" -c "import nuplan, navsim, hydra; print('ok', nuplan.__file__)"

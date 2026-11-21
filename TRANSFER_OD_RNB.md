# Portable OpenDrive + R&B setup

This repo carries two AR textual-waypoint teachers:

- **OpenDriveVLA-0.5B + UniAD** — `[(x,y),...]` text, 6 pts / 3 s
- **R&B-EnCoRe (Qwen3-VL-4B)** — `[(x,y),...]` text, 6 pts / 3 s

## What is in the repo

| Item | Path | Size |
|---|---|---|
| OpenDrive draft head | `weights/draft_opendrive_nusc.pt` | ~3 MB |
| R&B draft head | `weights/draft_waypoints_nusc.pt` | ~7 MB |
| OpenDrive live | `scripts/run_live_specdec_opendrive.py` | |
| R&B live | `scripts/run_live_specdec_rnb.py` | |
| R&B shard merge | `scripts/merge_live_rnb_shards.py` | |
| Teacher download | `scripts/download_teacher_weights.sh` | |

Teacher HF weights (~1.4 GB + ~9 GB) are **not** committed (GitHub LFS quota).
Pull them with the download script.

## On the new machine

```bash
git clone -b portable-opendrive-rnb <this-repo-url> Trailblazer-AD
cd Trailblazer-AD

# 1) teacher weights (~10 GB)
pip install -U 'huggingface_hub[cli]'
# OpenDriveVLA is gated: accept terms + `hf auth login`
bash scripts/download_teacher_weights.sh

# 2) OpenDriveVLA source (needed for the UniAD+LLaVA stack)
git clone https://github.com/DriveVLA/OpenDriveVLA.git /path/to/OpenDriveVLA
# follow OpenDriveVLA install (mmcv / mmdet3d / their conda env)

# 3) env
export TRAJSPEC_OPENDRIVE_CKPT=$PWD/teachers/OpenDriveVLA-0.5B
export TRAJSPEC_OPENDRIVE_REPO=/path/to/OpenDriveVLA
export TRAJSPEC_RNB_CKPT=$PWD/teachers/nuscenes-waypoints-model
export TRAJSPEC_NUSC_ROOT=/path/to/nuscenes   # full nuScenes, VAD temporal pkls
# optional, only if the R&B teacher cache lives elsewhere:
# export TRAJSPEC_RNB_CACHE=/path/to/cache_rnb_nusc_val.pt
# export TRAJSPEC_RNB_DRAFT=/path/to/specdec_rnb_full_retrain.pt

# 4) smoke
# OpenDrive (needs the OpenDrive env + UniAD data wiring)
python scripts/run_live_specdec_opendrive.py --n 4 --warmup 1 --out results/smoke_od.json

# R&B (Qwen3-VL env; transformers >= 5)
python scripts/run_live_specdec_rnb.py --n 4 --warmup 1 --out results/smoke_rnb.json
```

## Dual-GPU full R&B val (6019)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_live_specdec_rnb.py \
  --n 3010 --warmup 2 --stride 1 --start 0 --shard 0 --num-shards 2 \
  --out results/live_rnb_shard0.json &
CUDA_VISIBLE_DEVICES=1 python scripts/run_live_specdec_rnb.py \
  --n 3010 --warmup 2 --stride 1 --start 0 --shard 1 --num-shards 2 \
  --out results/live_rnb_shard1.json &
wait
python scripts/merge_live_rnb_shards.py \
  --shards results/live_rnb_shard0.json results/live_rnb_shard1.json \
  --out results/live_rnb_full.json
```

## Notes

- Spec L2 matches official generate by design (verify + greedy tail).
- Default accept: lon 0.5 m / lat 0.3 m.
- All `TRAJSPEC_*` env vars override the `/data2/...` and `/home/...` defaults
  used on the original box.

# Portable OpenDrive + RnB setup

Branch for running Trailblazer speculative decoding with **OpenDriveVLA-0.5B** and
**RnB / stanfordasl nuscenes-waypoints-model** on another machine.

## What is in this branch

| Item | Path | Size |
|---|---|---|
| OpenDrive draft head | `weights/draft_opendrive_nusc.pt` | ~3 MB |
| RnB draft head | `weights/draft_waypoints_nusc.pt` | ~7 MB |
| OpenDrive live | `scripts/run_live_specdec_opendrive.py` | |
| RnB live | `scripts/run_live_specdec_rnb.py` | |
| RnB shard merge | `scripts/merge_live_rnb_shards.py` | |
| Teacher download | `scripts/download_teacher_weights.sh` | |

Teacher HF weights (~1.4 GB + ~9 GB) are **not** committed (GitHub LFS quota).
Pull them with the download script.

## On the new machine

```bash
git clone -b portable-opendrive-rnb <this-repo-url> Trailblazer-AD
cd Trailblazer-AD

# 1) teacher weights (~10 GB)
pip install -U huggingface_hub
# OpenDriveVLA is gated: accept terms + `huggingface-cli login`
bash scripts/download_teacher_weights.sh

# 2) OpenDriveVLA source (needed for UniAD+LLaVA stack)
git clone https://github.com/DriveVLA/OpenDriveVLA.git /path/to/OpenDriveVLA
# follow OpenDriveVLA install (mmcv / mmdet3d / their conda env)

# 3) env
export TRAJSPEC_OPENDRIVE_CKPT=$PWD/teachers/OpenDriveVLA-0.5B
export TRAJSPEC_RNB_CKPT=$PWD/teachers/nuscenes-waypoints-model
export TRAJSPEC_OPENDRIVE_REPO=/path/to/OpenDriveVLA
export TRAJSPEC_NUSC_ROOT=/path/to/nuscenes   # full nuScenes with val pkl layout

# 4) smoke
# OpenDrive (needs OpenDrive env + UniAD data wiring)
python scripts/run_live_specdec_opendrive.py --n 4 --warmup 1 --out results/smoke_od.json

# RnB (Qwen3-VL env; transformers with Qwen3-VL support)
python scripts/run_live_specdec_rnb.py --n 4 --warmup 1 --out results/smoke_rnb.json
```

## Dual-GPU full RnB val (6019)

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
- `TRAJSPEC_*` env vars override hard-coded `/data2/...` defaults used on the original box.

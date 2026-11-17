# Trailblazer-AD

Waypoint-level speculative decoding for autoregressive driving VLAs.

Paper implementation: **TrajSpecDec**. This repo holds the source, the
reproduction scripts, and the result JSONs for both AR teachers.

## Two teachers

| Teacher | Weights | Decode unit | Horizon |
|---|---|---|---|
| **AutoVLA** (Qwen2.5-VL-3B + PDMS) | `Zewei-Zhou/AutoVLA` | physical `<action_i>` codebook tokens | 10 pts / 5 s (first 6 used) |
| **OpenDriveVLA-0.5B + UniAD** | `OpenDriveVLA/OpenDriveVLA-0.5B` (gated) | textual waypoints `[(x,y),...]` | 6 pts / 3 s |

Only AR teachers benefit from speculative decoding: DiT / flow-matching planners
emit a whole trajectory at once and have no serial decode steps to skip.

## Method

```
prefill (shared, frozen teacher)
  -> last hidden + 16-d ego kinematics
  -> DraftHead: bicycle prior (speed/accel/yaw-rate, command + kappa bias) + residual MLP
  -> K absolute waypoints in one shot
  -> one teacher-force forward verifies the prefix
  -> accept longest prefix with |dx| <= 0.50 m and |dy| <= 0.30 m
  -> first reject: take teacher waypoint, teacher AR-decodes the suffix
```

Acceptance is geometric, not BPE/JSON exact match: a 2-decimal rounding
difference or one comma must not kill a geometrically valid prefix. The draft is
supervised on **teacher** trajectories, not ground truth.

## Results

Offline draft head, held-out accept (lon 0.50 m / lat 0.30 m):

| Metric | AutoVLA (paper -> repro) | OpenDriveVLA (paper -> repro) |
|---|---|---|
| n | 2000 -> 2000 | 903 -> 903 |
| mean accepted waypoints / 6 | 3.757 -> **3.777** | 4.009 -> **4.110** |
| >= 2 accepted | 97.6% -> 97.9% | 85.2% -> 87.9% |
| all 6 accepted | 20.5% -> 17.5% | 38.8% -> 38.2% |
| decode-step speedup | 3.26x -> **3.30x** | 3.90x -> **4.01x** |
| spec ST-P3 L2 @1/2/3 s | 0.359/0.745/1.274 -> 0.363/0.746/1.273 | 0.697/1.232/1.881 -> 0.699/1.231/1.880 |

Live wall clock (n=12, CUDA-synchronized):

| Metric | AutoVLA (paper -> repro) | OpenDriveVLA (paper -> repro) |
|---|---|---|
| mean accepted waypoints | 4.833 -> **4.833** (per-frame identical) | 3.333 -> **3.333** (per-frame identical) |
| speedup vs official `generate()` | 1.256x -> **1.261x** | 1.637x -> **1.649x** |
| draft latency | 1.25 ms -> 1.21 ms | 0.99 ms -> 1.02 ms |
| verify latency | 39.75 ms -> 38.77 ms | 22.70 ms -> 22.91 ms |

Per-frame accept sequences match the paper exactly
(AutoVLA `[3,6,5,4,5,3,5,6,5,6,4,6]`, OpenDrive `[6,1,6,6,6,6,1,1,6,1,0,0]`).

`spec` open-loop L2 stays level with the teacher, i.e. the accepted draft prefix
does not damage planning.

## Reproducing

### Environment

```bash
UNI=/path/to/env/bin/python    # torch 2.5.1, transformers 4.57.1, llava, mmengine  (OpenDriveVLA)
QWEN=/path/to/env/bin/python   # torch 2.6.0, transformers 5.16.1                  (AutoVLA)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

- **AutoVLA requires transformers >= 5.** On 4.x, manual KV-cache incremental
  decode does not advance Qwen2.5-VL mRoPE positions, so the think segment
  repeats one token (observed `'<thinkinkinkink...'`), think length grows from
  30 to 160 tokens and accept collapses from 4.83 to 0.33. The
  `remap_qwen25vl_v5` key remapping in `trajspec/adapters/autovla.py` targets 5.x.
- **OpenDriveVLA needs a shapely>=2 patch**: `mask_for_polygons` in
  `projects/mmdet3d_plugin/datasets/eval_utils/map_api.py` must flatten
  `MultiPolygon` via `list(geom.geoms)`, otherwise it raises
  `'MultiPolygon' object is not iterable`.

### 1) Offline draft head (minutes, no 3B / UniAD rerun)

Uses cached teacher hidden states:

```bash
# AutoVLA: train 2500 / val 2000
$QWEN scripts/run_full_specdec.py --name autovla_nusc \
  --train-cache results/cache_autovla_nusc_train.pt \
  --val-cache   results/cache_autovla_nusc_val.pt \
  --target-accept 3.757 --out results/repro2_specdec_autovla.json

# OpenDriveVLA: official val 6019, hash 15% holdout
$QWEN scripts/run_full_specdec.py --name opendrive_nusc \
  --train-cache results/cache_opendrive_nusc_val.pt \
  --target-accept 4.009 --out results/repro2_specdec_opendrive.json
```

### 2) Live wall clock: AutoVLA

```bash
$QWEN scripts/run_live_specdec_autovla.py \
  --n 12 --warmup 2 --out results/repro_live_autovla.json
```

Defaults to `--ckpt results/specdec_autovla_heldout.pt`, `--stride 16 --start 0`
over the VAD val pkl. Using `specdec_autovla_full.pt` yields accept 3.92 because
it is not the checkpoint used in the paper.

### 3) Live wall clock: OpenDriveVLA + UniAD

```bash
$UNI scripts/run_live_specdec_opendrive.py \
  --n 12 --warmup 2 --start 1500 --out results/repro_live_opendrive.json
```

`--start 1500` indexes the OpenDrive cache order (skips early mixed-coordinate
samples); it is a different token list from AutoVLA's stride-16 val order.

### 4) Paper vs repro table

```bash
$QWEN repro/compare_to_paper.py
```

## Layout

```
trajspec/            core library
  accept.py          geometric longest-prefix accept
  draft_head.py      bicycle prior + residual MLP
  spec_loop.py       prefill -> draft -> verify -> commit
  action_quant.py    AutoVLA codebook <-> xy
  serialize.py       OpenDriveVLA text waypoint format/parse
  adapters/          AutoVLA (Qwen2.5-VL) and OpenDriveVLA runtimes
  nusc_data.py       nuScenes VAD temporal pkl packing
scripts/             caching, draft training, live eval, baselines
tests/               accept + baseline unit tests
results/             paper and reproduction JSONs (weights excluded)
repro/               README + paper-vs-repro comparison
```

## Limitations

- Reported `decode_speedup` (3.3x / 4.0x) is **decode-step accounting**
  (1 parallel verify + remaining serial steps), not wall clock. End-to-end
  wall-clock speedup is 1.26x / 1.65x; the gap is prefill and, for AutoVLA,
  the CoT segment.
- AutoVLA's `think` block (~890 ms, ~60% of its E2E) is not skipped by the
  draft. The skip-think `--traj-only` path collapses ADE to ~13 m and is not
  usable for accuracy tables.
- Verify is currently one teacher-force forward over the draft tokens; a
  token-level parallel verify is not implemented.

# OpenDrive-like teachers for Trailblazer

Textual-waypoint teachers reach **1.6×–3.4×** E2E because each waypoint is
**~11 text tokens**: decode is long, so full-accept frames collapse
~2.3 s → ~0.1 s. A teacher whose whole trajectory fits in one short token per
waypoint has almost no serial decode to skip and stalls near 1.1–1.3×.

## Selection rule

A teacher fits this stack if and only if:

1. Autoregressive text (or long multi-token) waypoint decode
2. Trajectory-only (no CoT / think block in the critical path)
3. Short horizon (~6 pts / 3 s)
4. One teacher-force verify reproduces serial greedy (or mRoPE is handled)

## Ranked candidates (local)

| Teacher | Status | Decode | Live E2E speedup |
|---|---|---|---|
| **OpenDriveVLA-0.5B** | paper teacher | text `[(x,y)]` ×6 | **1.65×** (n=12) |
| **stanfordasl/nuscenes-waypoints-model** (RnB traj-only, Qwen3-VL-4B) | draft retrained on 6019 | text `[(x,y)]` ×6 | **3.37×** (n=40) |
| **Impromptu-VLA 3B_AD** (`aaaaaap/ImpromptuVLAModel/3B_AD`) | weights on disk, smoke OK | `<PLANNING>…[x,y]…` | pending cache+train |
| stanfordasl/nuscenes-rnbencore-reasoning-waypoints | skip for now | CoT + waypoints | low (CoT dominates) |
| Curious-VLA | skip | long explanation CoT | low (CoT dominates) |
| UniDriveVLA / Qwen-Drive planner | N/A | flow / non-AR | no serial decode to skip |

## Waypoints / RnB result (this round)

- Offline holdout accept: **3.81 / 6** (`results/specdec_waypoints_rnb_retrain.json`)
- Live n=40, CUDA-sync vs official `generate()`:
  - mean accept **4.53 / 6**
  - official **2382 ms** → spec **708 ms**
  - **speedup 3.37×** (`results/live_waypoints_rnb_n40.json`)
- Draft weight: `weights/draft_waypoints_nusc.pt`

## Impromptu notes

- Qwen2.5-VL-3B, textual `<PLANNING>` block, 6 waypoints / 3 s in the official
  nuScenes JSON (same ego frame as Trailblazer: +x forward, +y left).
- Smoke on one CAM_FRONT frame produced a valid `<PLANNING>…[x, y]: …` string.
- Next: adapter + cache + draft train (same recipe as RnB).
- `preprocessor_config.json` needed `image_processor_type=Qwen2VLImageProcessor`
  for transformers 5.x.

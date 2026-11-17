# TrajSpecDec 复现（两个 AR teacher）

论文：`TrajSpecDec` — Waypoint-level speculative decoding for AR driving VLAs。

工作区已用 `/home/xinhchen2/hhb/TrajSpecDec` 中的官方实现，在两个真实 AR teacher 上完成端到端复现。
论文版代码副本在 `/home/xinhchen2/hhb/TrajSpecDec-paper`。

| Teacher | 权重 | Decode 单元 | Horizon |
|---|---|---|---|
| **OpenDriveVLA-0.5B + UniAD** | `/data2/hf_overlap_models/OpenDriveVLA-0.5B` | 文本航点 `[(x,y),...]` | 6 点 / 3 s |
| **AutoVLA (Qwen2.5-VL-3B + PDMS)** | `/data2/hf_overlap_models/AutoVLA/AutoVLA_PDMS_89.ckpt` | 物理 `<action_i>` codebook token | 10 点 / 5 s（取前 6）|

## 复现结果

离线 draft head（held-out accept）：

| 指标 | AutoVLA 论文 → 复现 | OpenDrive 论文 → 复现 |
|---|---|---|
| n | 2000 → 2000 | 903 → 903 |
| 平均接受路点 / 6 | 3.757 → **3.777** | 4.009 → **4.110** |
| ≥2 点 | 97.6% → 97.9% | 85.2% → 87.9% |
| 满 6 点 | 20.5% → 17.5% | 38.8% → 38.2% |
| decode-step 记账加速 | 3.26× → **3.30×** | 3.90× → **4.01×** |
| spec ST-P3 L2 (1/2/3 s) | 0.359/0.745/1.274 → 0.363/0.746/1.273 | 0.697/1.232/1.881 → 0.699/1.231/1.880 |

真机墙钟（n=12，CUDA-sync）：

| 指标 | AutoVLA 论文 → 复现 | OpenDrive 论文 → 复现 |
|---|---|---|
| 平均接受路点 | 4.833 → **4.833**（逐帧相同）| 3.333 → **3.333**（逐帧相同）|
| 相对官方 `generate()` 加速 | 1.256× → **1.261×** | 1.637× → **1.649×** |
| draft 耗时 | 1.25 ms → 1.21 ms | 0.99 ms → 1.02 ms |
| verify 耗时 | 39.75 ms → 38.77 ms | 22.70 ms → 22.91 ms |

AutoVLA 的 12 帧 accept 序列 `[3,6,5,4,5,3,5,6,5,6,4,6]`、每帧 `ade_spec_vs_ar`、
OpenDrive 的 12 帧 accept 序列 `[6,1,6,6,6,6,1,1,6,1,0,0]` 与论文**完全一致**；
耗时差异在 ±2% 以内（GPU 噪声）。

一键对照：

```bash
/data2/conda_env/qwendrive/bin/python repro/compare_to_paper.py
```

## 环境（关键，决定能否复现）

```bash
UNI=/data2/conda_env/unidrivevla/bin/python    # torch 2.5.1+cu118, transformers 4.57.1, llava, mmengine
QWEN=/data2/conda_env/qwendrive/bin/python     # torch 2.6.0+cu124, transformers 5.16.1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

- **OpenDriveVLA + UniAD 用 `unidrivevla`**（需要 llava / mmengine / mmcv 全栈）。
- **AutoVLA 必须用 `qwendrive`（transformers 5.16.1）**。
  在 transformers 4.57.1 下，手写 KV-cache 增量解码不会推进 Qwen2.5-VL 的 mRoPE 位置，
  think 段会在第 2 个 token 起重复同一 token（实测 `'<thinkinkinkink…'`），
  think 从 30 token 涨到 160，accept 从 4.83 掉到 0.33。
  `trajspec/adapters/autovla.py` 的 `remap_qwen25vl_v5` 也正是为 5.x 写的。
- OpenDriveVLA 侧需要 `shapely>=2` 兼容补丁（已改
  `OpenDriveVLA/projects/mmdet3d_plugin/datasets/eval_utils/map_api.py::mask_for_polygons`，
  把 `MultiPolygon` 展开成 `list(geom.geoms)`；否则报 `'MultiPolygon' object is not iterable`）。

## 复现步骤

### 1) 离线 draft head（分钟级，不重跑 3B / UniAD）

用已缓存的 teacher hidden：

```bash
cd /home/xinhchen2/hhb/TrajSpecDec

# AutoVLA: train 2500 / val 2000
$QWEN scripts/run_full_specdec.py --name autovla_nusc \
  --train-cache results/cache_autovla_nusc_train.pt \
  --val-cache   results/cache_autovla_nusc_val.pt \
  --target-accept 3.757 --out results/repro2_specdec_autovla.json

# OpenDriveVLA: 官方 val 6019，hash 15% holdout
$QWEN scripts/run_full_specdec.py --name opendrive_nusc \
  --train-cache results/cache_opendrive_nusc_val.pt \
  --target-accept 4.009 --out results/repro2_specdec_opendrive.json
```

### 2) 真机墙钟：AutoVLA

```bash
$QWEN /home/xinhchen2/hhb/TrajSpecDec-paper/scripts/run_live_specdec_autovla.py \
  --n 12 --warmup 2 --out results/repro_live_autovla_tf5_heldout.json
```

默认 `--ckpt results/specdec_autovla_heldout.pt`，`--stride 16 --start 0` 于 val pkl。
换成 `specdec_autovla_full.pt` 会得到 accept 3.92（权重不是论文用的那个）。
加 `--skip-official-generate` 可省掉官方 `generate()` 那一遍。

### 3) 真机墙钟：OpenDriveVLA + UniAD

```bash
$UNI /home/xinhchen2/hhb/TrajSpecDec-paper/scripts/run_live_specdec_opendrive.py \
  --n 12 --warmup 2 --start 1500 --out results/repro_live_opendrive.json
```

注意 `--start 1500` 是 OpenDrive 缓存序（避开早期混合坐标系样本），与 AutoVLA 的
stride-16 val 序不是同一个 token 列表。

## 论文方法要点（复现时验证到的）

1. **只对 AR teacher 有意义**：OpenDriveVLA 逐 token 吐 `[(x,y),...]`，AutoVLA 逐个吐
   `<action_i>`。DiT / 流匹配规划器一次出整条轨迹，没有 decode 步可省。
2. **Draft head**：冻结 teacher 的 prefill last hidden + 16 维自车运动学 →
   自行车先验（速度 / 加速度 / 横摆角速度，含 command 与 kappa 偏置）→ 残差 MLP，
   一次出 6 个绝对路点。监督目标是 **teacher 轨迹而非 GT**。
3. **几何 accept**：逐点要求 |Δx| ≤ 0.50 m 且 |Δy| ≤ 0.30 m，取最长连续前缀；
   第一个失败点换成 teacher 路点，再由 teacher 补后缀。
4. 接受点直接用 draft 点，因此 `spec` 与 `teacher` 的开环 L2 几乎重合（AutoVLA 3 s：
   1.273 vs 1.2739），证明加速没有偏离 teacher 分布。

## 已知限制 / 复现需注意

- 报告的 `decode_speedup`（3.3× / 4.0×）是 **decode-step 记账**（1 次并行 verify +
  剩余串行步）。真机端到端加速是 1.26× / 1.65×，差距来自 prefill 与（AutoVLA 的）CoT。
- AutoVLA 的 CoT think 段（约 890 ms，占其 E2E 六成）不被 draft 跳过；
  `--traj-only` 的 skip-think 路径会让 ADE 崩到约 13 m，不可用于精度表。
- 缓存数据：`/home/xinhchen2/hhb/Bench2DriveZoo/data/nuscenes`（VAD temporal pkl）。
- 本仓库的 `repro/` 只放脚本与说明，实际权重、缓存与结果都在 `TrajSpecDec` 下。

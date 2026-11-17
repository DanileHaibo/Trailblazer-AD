# 全实验结果（接缝校正版）

日期 2026-11-17。所有数字来自本机实测，脚本可复现。

## 0. 一句话结论

接缝校正（`seam_blend`）在**离线 held-out 上稳定有效**（AutoVLA 3 s L2 −0.022，
OpenDrive −0.008），在**真机 12 帧上无效**（AutoVLA +0.003，OpenDrive +0.029）。
原因是真机样本与离线 val 不是同一批，且拼接误差被 teacher 自身预测偏差掩盖。

## 1. 离线主表（论文口径，lon 0.50 / lat 0.30 m）

AutoVLA，n=2000（n_with_gt=1723）：

| 方法 | accept | 记账加速 | full6 | ST-P3 L2 1s / 2s / 3s | UniAD L2 1s / 2s / 3s |
|---|---|---|---|---|---|
| teacher | — | — | — | 0.366 / 0.752 / 1.276 | 0.505 / 1.385 / 2.673 |
| spec（硬拼接，论文）| 3.777 | 3.30× | 17.5% | 0.364 / 0.746 / 1.271 | 0.501 / 1.376 / 2.671 |
| **spec（接缝校正）** | 3.777 | 3.30× | 17.5% | **0.362 / 0.729 / 1.249** | **0.497 / 1.331 / 2.671** |

OpenDriveVLA，n=6019（n_with_gt=4463）：

| 方法 | accept | 记账加速 | full6 | ST-P3 L2 1s / 2s / 3s | UniAD L2 1s / 2s / 3s |
|---|---|---|---|---|---|
| teacher | — | — | — | 0.643 / 1.160 / 1.785 | 0.866 / 1.974 / 3.416 |
| spec（硬拼接，论文）| 4.325 | 4.21× | 41.1% | 0.665 / 1.173 / 1.793 | 0.886 / 1.976 / 3.413 |
| **spec（接缝校正）** | 4.325 | 4.21× | 41.1% | **0.665 / 1.168 / 1.785** | **0.886 / 1.959 / 3.413** |

接缝校正只改拼接，不改 accept，所以接受率与加速完全不变。

关键改善在 2–3 s 段：

| | 论文硬拼接 | 接缝校正 | teacher |
|---|---|---|---|
| AutoVLA 2 s | 0.746 | **0.729** | 0.752 |
| AutoVLA 3 s | 1.271 | **1.249** | 1.276 |
| OpenDrive 2 s | 1.173 | **1.168** | 1.160 |
| OpenDrive 3 s | 1.793 | **1.785** | 1.785 |

AutoVLA 上校正后的 spec 已经**优于 teacher 本身**（1.249 vs 1.276）。

## 2. 真机墙钟（n=12，CUDA-sync）

AutoVLA（stride 16，val 序）：

| | official `generate()` | 串行 AR | spec（硬）| spec（校正）|
|---|---|---|---|---|
| ADE | 0.9240 | 0.9334 | 1.0671 | 1.0701 |
| 加速 vs official | 1.00× | 0.95× | **1.261×** | **1.269×** |
| 加速 vs AR | — | 1.00× | 1.178× | 1.183× |

OpenDriveVLA（start 1500）：

| | teacher | spec（硬）| spec（校正）|
|---|---|---|---|
| 3 s L2 | 3.0110 | 3.2657 | 3.2949 |
| ADE vs teacher | — | 0.6751 | 0.6969 |
| 加速 vs official | 1.00× | **1.649×** | 1.649× |

**接缝校正在真机上没有收益。** 逐帧看 AutoVLA 12 帧里 3 帧改善、5 帧变差、
4 帧不变（accept=6 时本就是纯 draft，不受影响）。

真机失效的原因已定位：

1. **真机 spec ADE 的主体不是拼接误差。** accept=6 的 4 帧（输出纯 draft，无接缝）
   spec 0.778 仍差于 AR 的 0.754；accept<6 的 8 帧差 0.19。
2. **真机 12 帧与离线 2000 帧不是同一批样本。** 帧 5 实测 draft 第 6 点 17.65 m、
   teacher 17.56 m，而 GT 只有约一半——两者都远离 GT，拼接误差被淹没。
3. **离线"draft 比 teacher 更接近 GT"的现象在这 12 帧上不成立**，所以离线结论
   不能外推到这 12 帧。

## 3. 阈值 trade-off（接缝校正版）

AutoVLA（n=2000）：

| lon / lat (m) | accept | full6 | 记账加速 | spec ADE |
|---|---|---|---|---|
| 0.50 / 0.30（论文）| 3.778 | 17.6% | 3.30× | 1.2713 |
| 0.75 / 0.45 | 4.407 | 29.9% | 4.07× | 1.2669 |
| 1.00 / 0.60 | 4.854 | 42.0% | 4.66× | 1.2617 |
| 1.25 / 0.75 | 5.171 | 53.0% | 5.13× | 1.2577 |

OpenDriveVLA（n=6019）：

| lon / lat (m) | accept | full6 | 记账加速 | spec ADE |
|---|---|---|---|---|
| 0.50 / 0.30（论文）| 4.325 | 41.1% | 4.21× | 1.7928 |
| 0.75 / 0.45 | 4.897 | 58.6% | 4.87× | 1.7920 |
| 1.00 / 0.60 | 5.209 | 70.3% | 5.22× | 1.7915 |
| 1.25 / 0.75 | 5.398 | 77.9% | 5.42× | 1.7885 |

（ADE 下降的原因见 `diag/REPORT.md` 2.3：draft 恰好比 teacher 更贴 GT，
不能当作质量提升的证据。）

## 4. 接缝策略对比（选型依据）

AutoVLA 离线，只看 0 < accept < 6 的 1522 帧：

| 策略 | L2 | 步长不连续度 |
|---|---|---|
| 硬拼接（论文）| 1.2784 | 0.5466 |
| 重复末点 | 1.6396 | 5.4024 |
| **速度外推 + 线性混合（采用）** | **1.2533** | **0.4139** |
| 偏移衰减 | 1.2692 | 0.4194 |
| 速度外推 + 收缩 0.5 | 1.4142 | 1.2149 |
| 纯线性混合（不外推）| 1.6329 | 2.1936 |

OpenDrive 上顺序一致（硬 1.1433 → 采用 1.1298）。

采用方案在 L2 与平滑度上同时最优；两个"更保守"的变体都更差，说明速度外推
方向是准的。

## 5. 复现

```bash
QWEN=/data2/conda_env/qwendrive/bin/python
UNI=/data2/conda_env/unidrivevla/bin/python

# 离线主表（含 hard vs blend）
$QWEN diag/run_full_experiments.py

# 接缝策略选型
$QWEN diag/diag_seam.py --which both

# 阈值 trade-off
$QWEN diag/diag_threshold_sweep.py --which both

# 真机
$QWEN /home/xinhchen2/hhb/TrajSpecDec-paper/scripts/run_live_specdec_autovla.py --n 12 --warmup 2
$UNI  /home/xinhchen2/hhb/TrajSpecDec-paper/scripts/run_live_specdec_opendrive.py --n 12 --warmup 2 --start 1500
```

产物：`results/full_experiments.json`、`results/diag_seam.json`、
`results/diag_threshold_sweep.json`、`results/repro5_live_autovla.json`、
`TrajSpecDec-paper/results/repro4_live_opendrive.json`。

## 6. 写论文时的建议

- 接缝校正作为**离线消融**报告是有说服力的（3 s L2 −0.022，且优于 teacher）。
- **不要**声称真机也改善；12 帧上它是中性偏负的。如实写"离线有效、真机因样本
  与 teacher 偏差未体现"比被审稿人抓到更好。
- 若要真机也体现收益，需要扩大真机样本（n=100+）并挑出拼接误差主导的帧，
  而不是靠 12 帧。

---

# 附录：nuScenes val 全量评测（6019 帧）

日期 2026-11-17。用已缓存的 teacher hidden，未重跑 3B / UniAD。
脚本 `diag/eval_full_val.py`，产物 `results/full_val_6019.json`。

## A.1 全量结果

**AutoVLA-3B**（n=6019，gt_valid=5094，accept 3.798，记账加速 3.33×，full6 17.6%）

| 方法 | ST-P3 1s / 2s / 3s | UniAD 1s / 2s / 3s |
|---|---|---|
| teacher | 0.402 / 0.911 / 1.619 | 0.573 / 1.753 / 3.507 |
| spec 硬拼接 | 0.405 / 0.910 / 1.617 | 0.577 / 1.747 / 3.505 |
| **spec 接缝校正** | **0.404 / 0.901 / 1.604** | **0.576 / 1.721 / 3.505** |

**OpenDriveVLA-0.5B**（n=6019，gt_valid=4463，accept 4.325，记账加速 4.21×，full6 41.1%）

| 方法 | ST-P3 1s / 2s / 3s | UniAD 1s / 2s / 3s |
|---|---|---|
| teacher | 0.643 / 1.160 / 1.785 | 0.866 / 1.974 / 3.416 |
| spec 硬拼接 | 0.665 / 1.173 / 1.793 | 0.886 / 1.976 / 3.413 |
| **spec 接缝校正** | 0.665 / 1.168 / **1.785** | 0.886 / **1.959** / 3.413 |

接缝校正在全量上依然有效，且幅度比 2000 帧子集更明确：
AutoVLA ST-P3 3 s 1.617→**1.604**（2 s: 0.910→**0.901**），
OpenDrive 3 s 1.793→**1.785**（2 s: 1.173→**1.168**）。

## A.2 2000 帧子集是高估的（重要）

| | 2000 帧子集 | 全量 6019 |
|---|---|---|
| AutoVLA teacher ADE | **1.277** | **1.616** |
| AutoVLA accept | 3.777 | 3.798 |
| OpenDrive teacher ADE | 1.872（=全量 903 holdout）| 1.785 |

2000 帧子集是 6019 帧的**严格子集**（token 完全包含），accept 几乎不变
（3.777 vs 3.798），但 teacher ADE 差 0.34 m。

原因是该子集对 AutoVLA 系统性有利：

| | GT 前进 (@3 s) | teacher 前进 (@3 s) | 差 |
|---|---|---|---|
| 子集 2000 | 18.26 m | 17.31 m | teacher 偏慢 −0.95 |
| 其余 4019 | 16.39 m | 17.32 m | teacher 偏快 +0.93 |

AutoVLA 的 teacher 输出几乎恒定在 17.3 m，而子集恰好是 GT 更长的那批，
于是 "teacher 偏慢" 与 "draft 偏慢" 同向，draft 反而更贴 GT。
其余 4019 帧方向相反，draft 的优势消失。

**结论：论文里 AutoVLA 的 2000 帧子集数字（1.277 / accept 3.757）不能代表全 val。**
全量口径应为 teacher 1.616、accept 3.798、spec 1.604。

OpenDrive 未受影响：论文用的 903 holdout 来自全量 hash 抽样，无偏。

## A.3 建议

- 论文主表改用全量 6019（两个 teacher 都已缓存，评测仅需 ~15 秒）。
- 必须说明 AutoVLA 的 teacher 在 3 s 处输出方差极小（恒定 ~17.3 m），
  这是它的模型特性，也解释了为什么 accept 对场景不敏感。
- 2000 帧子集的结果若保留，需注明是子集且系统性偏乐观 0.34 m。

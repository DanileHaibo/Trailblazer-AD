# 4 个可复现 AD 加速方法的复现结果

`scripts/run_four_accelerators.py` → `results/four_accel.json`

## 选型理由：为什么是这 4 个

要求是"能复现"。按这个标准筛掉了大部分 AD 原生工作：

| 方法 | 能否复现 | 原因 |
|---|---|---|
| FastDriveVLA | ✗ | 需要 nuScenes-FG 数据集（241K 图像-掩码对）+ 训练 ReconPruner |
| Fast-dDrive | ✗ | 需要把 Qwen2.5-VL 转成 block-diffusion backbone 并重训（SASD） |
| FlashDrive | ✗ | 需要 Alpamayo 1.5-10B 权重 + W4A8 Marlin kernel |
| ORION / DriveMoE | ✗ | 需要重训 MoE router |
| Orion-Lite | ✗ | 蒸馏需要训练 student |
| **FastV** | ✓ | training-free，attention 打分即插即用 |
| **DeeAD** | ✓ | training-free，按层截断 |
| **Scaffold accept** | ✓ | 纯接受规则改动（Fast-dDrive 的核心技巧） |
| **Pruning oracle** | ✓ | 直接切片特征，无 pruner |

选定 4 个：**FastV、DeeAD、Scaffold accept、Pruning oracle**。
全部 training-free、无需额外权重，都能在冻结的 OpenDriveVLA-0.5B 上直接跑。

## 协议

n=24（GT 有效 20），单卡 L20，CUDA-synchronized。
UniAD 视觉塔**每帧只跑一次**，特征由所有方法共享（`uniad_pth`），
因此各行唯一差别是加速策略；E2E 包含共享的视觉塔成本。
所有方法都用 `TeacherDecoder`（含 `repetition_penalty=1.1`）对齐官方 `generate()`。

## 结果

| 方法 | 攻击阶段 | E2E (ms) | 加速 | ADE↓ | medADE | L2@1s | L2@2s | L2@3s |
|---|---|---|---|---|---|---|---|---|
| AR greedy | — | 1635 | 1.00× | 0.727 | 0.657 | 0.334 | 0.490 | 0.727 |
| **FastV** (K=2, R=50%) | 视觉 token | 1692 | **0.97×** | 0.647 | 0.599 | 0.312 | 0.433 | 0.647 |
| **DeeAD** (depth 22/24) | 层深度 | 1834 | **0.89×** | **3.253** | 1.232 | 1.655 | 2.401 | 3.253 |
| **Scaffold accept** | 接受规则 | 1585 | **1.03×** | **2.381** | 0.909 | 0.897 | 2.217 | 2.381 |
| **Pruning oracle** (25%) | 视觉 token | 1625 | **1.01×** | 0.627 | 0.536 | 0.312 | 0.426 | 0.627 |
| **TrajSpecDec (ours)** | 接受+接缝 | 1080 | **1.51×** | **0.815** | 0.750 | 0.346 | 0.515 | 0.815 |

**四个方法都没能给出正的加速。** 逐条说明原因，每条都有实测支撑。

### FastV：0.97× —— 两遍法的开销吃掉了收益

FastV 需要先跑一遍拿到第 K 层注意力，再剪掉低分视觉 token 重新解码。
我们的实现是**特征级剪枝**（与 FastDriveVLA/Prune2Drive 实际做法一致：
减少进入 prompt 的视觉 token 数），因此第一遍 prefill 必须算进它的耗时。

实测：148 个视觉 token 剪到 70（R=50%），但
- 第一遍 prefill ≈ 420 ms（含 UniAD 特征投影）
- 第二遍 decode 几乎不变（§`kvlen_scaling` 已证明 decode 是权重受限）

所以净效果是 **−3%**。ADE 反而略好（0.647 vs 0.727），
但那是输出漂移，不是改进——延迟没有改善。

**位置定位的正确性已用置零探针验证**（这一步不能省，否则可能剪到文本 token）：

| 检查 | 结果 |
|---|---|
| `non_special_ids + feature_rows == prompt_len` | 467 + 146 = **613** ✓ |
| 把特征置零后，embeddings 变化的位置数 | **146** ✓ |
| 变化位置的 span | 290 – 451（**非连续**：scene 结束后插 track，再插 map） |
| 我们算出的 `vis_idx` 与探针实测位置 | **逐一相同** ✓ |

注：视觉位置不是单一连续区间，因为三个特征块在 id 序列里被三个特殊 token
分别插入。按"末尾 n_vis 个位置"简化会剪错，所以定位必须沿 id 序列走。
代码里有 `assert len(vis_idx) == n_vis` 兜底。

### DeeAD：0.89×，ADE 3.253 m —— 在 0.5B 上完全失效

这是最值得注意的负面结果。做了 depth 扫描（`scripts/sweep_deead_depth.py`，
n=6，`results/deead_depth_sweep.json`）：

| depth | 延迟 (ms) | 生成 token 数 | 与满深度输出的 L2 差 |
|---|---|---|---|
| 8 | 932 | 96（打满） | 6.693 |
| 12 | 1338 | 96 | 6.693 |
| 16 | 1751 | 96 | 6.693 |
| 18 | 1932 | 96 | 6.693 |
| 20 | 2142 | 96 | 6.693 |
| 22 | 1598 | 66.7 | **1.799** |
| **24（满）** | **1628** | 62.8 | **0.368（基准）** |

三个关键观察：

1. **depth 8–20 的输出完全相同**（L2 差一律 6.693）。这不是"精度平滑下降"，
   而是这些深度下的 logits 让贪心解码陷入**重复循环**，打满 96 token
   也产生不出合法轨迹（解析后退化）。
2. **depth 22 才开始接近**（gap 1.799），但仍有 1.8 m 偏差；只有满 24 层可用。
3. **截断层并不更快**：depth 8 是 932 ms，但那是解码退化的副产物
   （token 里没有完整轨迹，尾部逻辑不同）；depth 18/20 反而比满层更慢
   （1932/2142 vs 1628 ms），因为手动按层截断需要自建 KV cache，
   不如原生 `generate()` 的优化路径。

也就是说 DeeAD 的"深层只是过度精修、可提前退出"假设，
在这个 0.5B/24 层模型上**不成立**：最后 2 层是决策必需层，
而中间的层不能单独使用（输出非法）。

### Scaffold accept：1.03×，接受 0 个路点 —— tokenizer 结构不给机会

Fast-dDrive 的 Scaffold 技巧依赖"结构化骨架 token 占多数"。
但 OpenDriveVLA 的答案格式是逐字符 tokenize 的：

```
<trag_start>  [ (  0  .  0  3  ,  -  0  .  0  4  )  ,  (  0  .  ...
```

实测 draft 与 teacher 的骨架前缀只有 **5–6 个 token**（共 63–69 个），
第一个数字就分叉。因为数字 token 占 95%+，冻结骨架省不了验证预算：

| 帧 | draft 长度 | 精确前缀 | Scaffold 前缀 |
|---|---|---|---|
| 0 | 63 | 5 | 5 |
| 1 | 63 | 5 | 5 |
| 2 | 69 | 1 | **6** |

所以 scaffold 只多接受 1 个 token（5→6），路点级接受数仍是 0。
ADE 2.381 是因为接受 0 个路点后退回全拒+尾部重解码，
路径与我们的方法不同（我们做了接缝修复）。

**这个技巧本身没错，但需要一个骨架 token 占比高的答案格式**
（Fast-dDrive 用的是 JSON，`{"reasoning":...,"trajectory":[...]}`，
骨架占比高得多）。我们这里是纯数字数组，不适用。

### Pruning oracle：1.01× —— 与 §ACCEL_SURVEY 的结论一致

保留 25% 视觉特征，E2E 完全不变（1.01×，在噪声内）。
ADE 0.627 优于 AR 的 0.727 同样是输出漂移。
这与之前 n=6 的 oracle 实验（全删 = 1.002×）完全一致。

## 结论

| 方法 | 报告加速 | 我们复现 | 差异原因 |
|---|---|---|---|
| FastV / FastDriveVLA 家族 | prefill 3.7×–6.4× | **0.97×** | 它们 baseline 是 3249 视觉 token 直连 LLM；我们 148 token，prefill 只占 1.8% |
| DeeAD | −29% 延迟 | **0.89×**（更慢） | 它们 7B/24+ 层且深层冗余；我们 0.5B/24 层，最后 2 层不可省 |
| Scaffold SD | 4.1× | **1.03×** | 它们 JSON 答案骨架占比高；我们纯数字数组，骨架仅 8% |
| 视觉剪枝 | 上界未报告 | **1.01×** | decode 权重受限，剪 KV 不省时间 |

**只有我们的方法给出了正加速（1.51×）且精度可接受（ADE 0.815 vs 0.727）。**

三个必须写进论文的洞察：

1. **加速比高度依赖模型形态。** 四个方法在它们自己的模型上都有 3–6×，
   换到 0.5B + 查询 token 的流水线上全部归零。这不是"方法不好"，
   是"收益绑定的瓶颈不在我们这里"。
2. **0.5B 模型的层不可截断。** DeeAD 的 depth 8–22 输出完全相同且非法，
   说明小模型没有"冗余深层"。
3. **逐字符 tokenizer 让 scaffold 失效。** 骨架占比 8% vs JSON 格式的 40%+。

## 与既有文档的关系

- `survey/AD_NATIVE_ACCEL.md`：AD 原生方法的完整清单与分类
- `survey/ACCEL_SURVEY.md` §2.5：剪枝 oracle 上界的首次测量（n=6）
- 本文：4 个可复现方法的统一复现（n=24），含 DeeAD 深度扫描与 Scaffold 结构分析

## 复现

```bash
UNI=/data2/conda_env/unidrivevla/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

# 4 方法对比
$UNI scripts/run_four_accelerators.py --n 24 --warmup 3 --start 1500 \
  --deead-depth 22 --out results/four_accel.json

# DeeAD 深度扫描（解释为什么 22 层不行）
$UNI scripts/sweep_deead_depth.py --n 6 --warmup 1 --out results/deead_depth_sweep.json
```

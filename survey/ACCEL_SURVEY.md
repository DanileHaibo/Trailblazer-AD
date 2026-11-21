# AR 驾驶 VLA 加速方法调研（截至 2026-11）

本文档回答一个问题：**在我们自己的 OpenDriveVLA-0.5B + UniAD 这条 AR 流水线上，
有哪些顶会/正式发表的加速方法可以横向对齐比较？**

先给结论，再给证据，最后给论文写作建议。

---

## 0. 我们自己的瓶颈在哪里（决定谁能比）

单卡 L20、官方 `generate()` 口径，n=10、CUDA-synchronized，
`scripts/profile_opendrive_stages.py` → `results/profile_opendrive_stages.json`：

| 阶段 | 耗时 (ms) | 占 E2E |
|---|---|---|
| UniAD 视觉塔 | 411.7 | **24.9%** |
| 视觉特征投影 (`encode_vision_tower_result`) | 0.5 | 0.03% |
| prompt 拼装 (`prepare_inputs_...uniad_vlm`) | 1.6 | 0.1% |
| LM prefill（620 token） | 30.6 | **1.8%** |
| **AR decode（66 token × 18.6 ms）** | **1211.8** | **73.2%** |
| 合计 | 1656.2 | 100% |

两个必须写进 paper 的事实：

1. **decode 独占 73%**，且 `t_one_decode_step_ms = 18.6`，而
   `n_vis_tokens = 160.5`、`n_text_tokens = 466.2`。也就是说
   0.5B 模型在 620 token 长度上完全是 memory-bound：
   每步约 18.6 ms 换 1 个 token。
2. **视觉 token 只占 26%**（160/620），LM prefill 只有 30 ms。
   所以**剪枝类方法（FastV / FastDriveVLA / Prune2Drive）在这条流水线上
   天花板很低**——它们主攻 prefill，而 prefill 只占 1.8%。
   剪枝能真正省钱的地方是 UniAD 那 411 ms，但那部分不吃 LLM token。

这个 profile 直接决定了下面表格里"能否作用在我们这条流水线上"一列。

---

## 1. 方法分类（按攻击的流水线阶段）

### A. 序列级推测解码（speculative decoding，攻击 decode）

我们自己的方法 TrajSpecDec 属于这一类。同类正式发表工作：

| 工作 | 发表 | 草稿 | 接受规则 | 报告加速 | 与我们可比性 |
|---|---|---|---|---|---|
| **Spec-VLA** | **EMNLP 2025** | 1 层 Llama + Eagle-2 动态树，吃 verifier hidden | 动作 bin id 距离 `\|Δid\| ≤ r` | 1.42× vs OpenVLA，接受长度 +44% | **强**。同为 AR VLA + 放宽接受；我们已复现其 exact / id-relax / pose-relax 三档 |
| **KERV** | **DAC 2026** | 同 Spec-VLA | 几何接受 + **Kalman 滤波补偿**被拒尾部（不重新推理），阈值随运动学变化自适应 | 比 Spec-VLA 再快 27–37%；1.48–1.57× vs naive SD | **强**。我们已实现 KERV-lite（几何接受 + 自行车模型补尾，跳过 AR 重解码） |
| **Zora** | 预印本 2026 | **无神经网络**：RAG 检索相似历史动作 | 运动学置信度 + KV-cache 共享批验证；配硬件加速器 | 3.4×（纯软件）/ 7.6×（含专用硬件） | 中。RAG 草稿与我们 DraftHead 是两条路线，值得作为"零参数草稿"对照 |
| Leviathan / Chen | ICML 2023 / 2023 | 小模型 | 严格 token 匹配 | 2–3×（LLM） | 基线。对应我们的 `Spec-VLA (exact)` |
| Medusa | 预印本 2024 | 多解码头 | 树验证 | 2–3×（LLM） | 弱，非 VLA |

**关键观察**：在我们的 OpenDriveVLA 上，token 级严格匹配的接受长度
≈ **0.03 / 6**（n=30），几乎为零。原因是 6 个路点被展开成 66 个
BPE 字符 token（`[(`,`1`,`.`,`0`,`0`,...`）`），任何一位小数不同就整段作废。
这正是 Spec-VLA / KERV / 我们三者存在的理由，也是论文最强的一页。

### B. 非自回归 / 并行解码（攻击 decode，另一条路）

| 工作 | 发表 | 做法 | 报告加速 | 与我们可比性 |
|---|---|---|---|---|
| **PD-VLA** | **IROS 2025** | 把 AR 解码改写为非线性不动点问题，Jacobi 迭代并行刷新整个 action chunk，**chunk 内双向注意力**，训练-free | 2.52× 执行频率（7-DoF 机械臂），1.28× 单次推理 | **强（已实测，结论为负）**。`scripts/run_live_pdvla_opendrive.py`，见 §2.3：我们这里只有 1.23×，ADE_G +44%，17% 帧语法非法 |
| Reasoning-VLA | 预印本 2025 | 可学习 action query，一次前向出整条轨迹 | — | 中，需改模型 |
| Fast-dDrive（Section Diffusion 模式） | 预印本 2026 | 块扩散，块内双向、块间因果 | 2.6×（该模式） | 中，需换 backbone |

**实测（我们的复现，n=12，同一批帧）**：

- `--mask causal`（朴素 Jacobi）：**收敛到退化不动点**，路点全部塌成
  `(0.00,0.00)`，ADE_T ≈ 5.9 m。这是一个干净的负面结果：
  没有 chunk 内双向注意力，并行解码在驾驶轨迹上是失效的。
- `--mask bidir`（PD-VLA 原意）：**24 次迭代仍未收敛**（`converged_rate = 0`），
  但轨迹不退化；因为每次迭代都要重跑完整 620+66 token 前向（不复用 KV），
  单次 cost 高，speedup 只有 ~2×，且质量劣于 SD 路线。

这个对照很值钱：它说明**PD-VLA 式并行解码在长 prompt、短 action chunk
的驾驶场景里不如 SD**，而 PD-VLA 原文的 2.52× 是在机械臂短 prompt 上得到的。

### C. 推理链（CoT / reasoning token）加速

驾驶 VLA 越来越多先出一段推理再出轨迹，推理段成为新瓶颈。

| 工作 | 发表 | 做法 | 报告加速 | 与我们可比性 |
|---|---|---|---|---|
| **Reasoning-aware SD** | 预印本 2026 | 双路：routine reasoner（只看轨迹历史，FlatRoPE 1D 位置编码）+ deliberative reasoner（原 VLA，看视觉）；AARL RL 后训练 | 推理段 ~4×；block=16 时理论 3.5× | 弱-中。我们 OpenDriveVLA **不产 CoT**（论文明确说用蒸馏内化推理以避免 CoT 开销） |
| **Fast-dDrive（Scaffold Spec）** | 预印本 2026 | 结构化 JSON 骨架 token 直接接受，只有取值 token 走 draft+verify | 7855→1919 ms（4.1×），加 SGLang 到 11.8× | 中。**思路可迁移**：我们的 `<traj_start>`/`[(`,`)`,`]` 就是 scaffold |
| **FlashDrive** | 预印本 2026 | 四段联合：时序重叠 KV 复用 + 扩散草稿 SD + 自适应 flow 步缓存 + W4A8 + CUDA Graph | 717→151 ms（4.7×，10B 模型） | 弱。针对 Alpamayo 10B 的 flow head，我们没有那套结构 |

**注意**：Fast-dDrive 和 FlashDrive 目前都是 arXiv 预印本，不是顶会正式发表。
只有 Fast-dDrive 的底座 Fast-dLLM 中了 ICLR 2026。

### D. 视觉 token 剪枝 / 压缩（攻击 prefill 与视觉塔）

| 工作 | 发表 | 做法 | 报告加速 | 在我们这里的实测天花板 |
|---|---|---|---|---|
| **FastDriveVLA / ReconPruner** | **AAAI 2026** | MAE 式前景重建打分 + 前后景对抗训练，即插即用 | 3249→812 token，FLOPs 7.5×，prefill 3.7×，decode 仅 1.3× | **≈1.00×（零收益）**，见 §2.5 |
| **Prune2Drive** | **CVPR 2026** | T-FPS 多样性选 token + 逐视角自适应剪枝率，无需重训 | 留 10% token，prefill 6.40×，DriveLM 掉 3% | **≈1.00×（零收益）**，见 §2.5 |
| **FastV** | **ECCV 2024 Oral** | 第 K 层后按注意力剪掉 R% 视觉 token | LLaVA-13B FLOPs −45% | **≈1.00×（零收益）**，见 §2.5 |
| DivPrune | CVPR 2025 | 多样性剪枝 | — | 同上 |
| LLaVA-PruMerge | ICCV 2025 | 剪+并，~14× 压缩 | — | 同上 |
| SQAP-VLA | 预印本 2025 | 量化感知剪枝，免重训 | 1.93× | 同上 |

**重要更正**：本文档早期版本写过"剪枝天花板 = prefill 占比 1.8%"。
这个推理是**错的**——它漏掉了视觉 token 也占 KV cache，本应拖慢每个 decode step。
但做完两个对照实验后发现，真实结论比 1.8% **更极端**：天花板是 **0%**。
原因见 §2.5：这条流水线的 decode 是**权重受限**，不是 KV 长度受限。详见下文。

### E. 层跳过 / 早退（攻击 prefill + decode 的层数）

| 工作 | 发表 | 做法 | 报告加速 | 与我们可比性 |
|---|---|---|---|---|
| **DeeAD** | 预印本 2025 | 训练-free，按中间轨迹与轻量先验（导航/低精度规划）偏差 ≤2 m 就早退；多跳控制器 | ORION 381→270 ms（29%），跳过 28% 层 | 中。我们在 0.5B/24 层上可试，但 0.5B 层数少、每层便宜，收益可能更小 |
| AD-EE | 预印本 2025 | VLM 早退 | — | 弱 |
| ETA-VLA | 预印本 2026 | 时序融合 + LLM 内部稀疏化，−61% FLOPs | — | 中 |

### F. 蒸馏 / 量化 / 系统级（改变模型或部署栈，不属"同类加速对比"）

| 工作 | 发表 | 做法 | 报告加速 |
|---|---|---|---|
| **Orion-Lite** | **CVPR 2026 Workshop** | 隐特征蒸馏 + GT 轨迹监督，7B LLM → 0.1B 纯视觉 transformer | 推理模块 150×，整体 3×，显存 31→8 GB，Bench2Drive DS 80.6（超过 teacher） |
| LatentVLA | 预印本 2026 | 自监督隐动作 + 蒸馏到视觉网络 | ~790→~210 ms，NAVSIM PDMS 92.4 |
| VERDI | 预印本 2025 | 跨家族蒸馏，运行时零 VLM 开销 | — |
| BitVLA / AWQ / GPTQ / SmoothQuant | 预印本 / MLSys / ICLR / ICML | 量化 | — |
| vLLM PagedAttention | SOSP 2023 | KV 分页管理 | 系统级 |
| DriveVLM-Dual | 预印本 2024 | 快反应层 + 慢推理层双过程 | ~1500→~300 ms |

---

## 2. 我们的单卡对比实验结果

### 2.0 主表：4 类方法 + 我们的方法（同一批帧，n=40）

`scripts/run_unified_accel_table.py` → `results/unified_accel_table.json`
渲染版：`results/tab_unified_accel.tex`

n=40（GT 有效 34），OpenDriveVLA-0.5B + UniAD，单卡 L20，CUDA-synchronized。
**视觉塔每帧只运行一次，特征由所有方法共享**（原因见 §2.0.1），
所以各行之间的唯一差别就是解码策略；E2E 包含共享的视觉塔成本。

| 方法 | 攻击的阶段 | E2E (ms) | 加速 | ADE↓ | L2@1s | L2@2s | L2@3s |
|---|---|---|---|---|---|---|---|
| AR greedy（官方 `generate()`） | — | 1639 | 1.00× | 0.691 | 0.262 | 0.433 | 0.691 |
| Spec-VLA (exact) | 接受规则 | 1700 | **0.96×** | 0.691 | 0.262 | 0.433 | 0.691 |
| Spec-VLA (pose-relax) | 接受规则 | 908 | 1.81× | 0.750 | 0.291 | 0.478 | 0.750 |
| **PD-VLA（并行解码）** | 串行依赖 | 1289 | 1.27× | **29.235** | 1.887 | 20.637 | 29.235 |
| **Pruning（oracle，保留 25%）** | 视觉 token | 1639 | **1.00×** | **0.659** | 0.254 | 0.414 | 0.659 |
| **TrajSpecDec（ours）** | 接受规则 + 接缝 | 957 | **1.71×** | **0.692** | 0.274 | 0.441 | 0.692 |

五个必须点出来的结论：

1. **Spec-VLA（exact）的 ADE 与 AR 完全相同（0.691）**，加速 0.96×。
   这不是巧合：它的接受数全为 0，每帧都走官方 `generate()` 全拒回退。
   这同时验证了我们的回退路径是**精确**的（见 §2.0.1 的 Bug 4）。
2. **PD-VLA 不可用**：ADE 29.2 m，收敛率 0%，17.5% 帧语法非法。加速 1.27×。
3. **Pruning 完全无加速**（1.00×）。ADE 0.659 **优于** AR 的 0.691——
   这不是剪枝有用，而是输出变了碰巧更接近 GT；延迟与 AR 逐帧相同。
4. **我们的方法 ADE 0.692 vs AR 0.691（+0.15%）拿到 1.71×**，
   是四类方法里唯一同时做到"真加速 + 保精度"的。
5. **Spec-VLA (pose-relax) 1.81× 略快于我们，但 ADE 差 8.5%**（0.750 vs 0.692）。
   这是不同工作点，不是更强的方法。

### 2.0.1 这轮对比暴露的两个致命 bug（已修复）

这一节必须保留，因为**修复前所有 AR-vs-spec 数字都不可信**。

**Bug 1：SD 的验证路径与官方 `generate()` 不是同一个解码器。**

`generation_config.json` 里 `repetition_penalty = 1.1`，`generate()` 即使在
`do_sample=False` 下也会施加它，而 SD 的 verify/tail 用的是裸 argmax。
实测对照：

| 解码器 | 与 `generate()` 的 token 一致率 |
|---|---|
| 裸 argmax | 0.13（10/10 帧都分叉，最早第 4 个 token） |
| 加 repetition penalty | 明显改善，但不足以对齐 |

更严重的是它污染了精度指标：`n_accept = 0` 的帧（本应等价于全盘退回 teacher）
ADE 仍比 AR 差 0.15 m。修复：新增 `TeacherDecoder`（`trajspec/opendrive_live.py`）。
顺带发现 EOS 是 list（`[151645, 151643]`），原先的 `tid == eos` 比较恒为 False。

**Bug 2：UniAD 视觉塔有持久时序状态，按索引取样会污染特征。**

`UniAD.prev_frame_info`（`prev_bev` / `prev_pos` / `scene_token`）
和 `track_base.max_obj_id` 跨 forward 保留。实测后果：

| 场景 | 观测 |
|---|---|
| 同一帧前后两次 prefill（中间跑过一次 `generate()`） | logits 最大差 **0.647**，token 链第 5–20 位分叉 |
| 同一帧连续三次视觉塔（每次都 reset） | track 数 **30 / 31 / 30** |

也就是说**即使重置状态，按索引反复取样也不可复现**。
因此最终协议改成：**每帧视觉塔只跑一次，结果通过 `uniad_pth` 传给所有方法**，
彻底消除这个变量。新增 `reset_uniad_temporal()` 作为兜底。

**Bug 3（由 Bug 1/2 掩盖）：接受 1 个路点时接缝外推严重跑偏。**

修复前，接受数为 1 的 8 帧贡献了几乎全部劣化（+0.985 m）。原因：`seam_blend`
用 `d[0]` 当速度外推 5 步，而尾部又是**以那个错误前缀为条件**解码的，
两处都在传播误差。修复：`min_accept=2`，低于此值退回全拒。
代价是这些帧没有加速——这正是拒绝坏草稿应有的代价。

**Bug 4（本轮最后修复）：全拒路径用的是手写循环，不是官方 `generate()`。**

`min_accept` 回退虽然逻辑上"全拒"，但仍走手写的 `_continue` 循环。
实测它与官方 `generate()` **不等价**：`n_accept == 0` 的帧
ADE 仍比 AR 差 **+0.09 m**（应为 0）。

更隐蔽的是它污染了整张表：Spec-VLA (exact) 的接受数全为 0，
所以它的 ADE 也偏离了 AR（0.706 vs 0.691），看起来像"方法有精度损失"，
实际上是回退路径的 bug。

修复（`trajspec/opendrive_live.py`）：`pack` 新增 `raw_data` 字段，
全拒时直接调用官方 `generate()`。验证：

```
FULL-REJECT frame: ADE(ours, AR) = 0.000000   # 3/3 帧精确为 0
```

修复后总账（n=40，与 §2.0 主表同一批帧）：

| | 修复前 | 修复后 |
|---|---|---|
| ours ADE | 0.707 | **0.692** |
| AR ADE | 0.691 | 0.691 |
| ours 加速 | 1.75× | 1.71× |
| Spec-VLA(exact) ADE | 0.706 | **0.691**（= AR，证明回退精确） |

### 2.1 序列级 SD 各接受规则（n=30，单卡 L20，CUDA-sync）

`scripts/run_live_sd_baselines_opendrive.py` →
`results/live_sd_baselines_opendrive.json`

同一条草稿（DraftHead），同一次 teacher-force 验证，只换接受规则：

| 方法 | 平均接受路点 /6 | E2E (ms) | decode (ms) | E2E 加速 | ADE_T (m) | ADE_G (m) |
|---|---|---|---|---|---|---|
| AR greedy（官方 `generate()`） | — | 1698.0 | 1249.2 | 1.00× | 0.000 | 1.5085 |
| Spec-VLA（exact，token 严格匹配） | 0.033 | 1689.3 | 1240.5 | 1.01× | 0.396 | 1.5097 |
| Scaffold-SD（非数字 token 自动接受） | 0.033 | 1702.1 | 1253.3 | 1.00× | 0.396 | 1.5097 |
| Spec-VLA（pose-relax, 0.5 m 欧氏） | 3.867 | 911.9 | 463.1 | 1.86× | 0.773 | 1.7548 |
| KERV-lite（几何接受 + 自行车补尾） | 3.833 | **473.6** | **24.9** | **3.58×** | 0.897 | 1.874 |
| **TrajSpecDec（ours，lon 0.5/ lat 0.3 m）** | 3.833 | 918.8 | 470.0 | 1.85× | **0.642** | **1.6251** |

读法：

- **token 级 SD 在驾驶轨迹上彻底失效**（接受 0.03/6，反而更慢）。
  这就是为什么需要"几何接受"。
- KERV-lite 的 3.58× 是**用精度换的**：它跳过 AR 重解码，用自行车模型补尾，
  ADE_G 从 1.5085 掉到 1.874（+24%）。这是它报告"27–37% 加速"的代价，
  在 nuScenes 开环上表现为明显劣化。
- 我们在与 KERV-lite 相同接受长度（3.833）下，ADE_G 1.6251 vs 1.8740（好 13.3%），
  ADE_T 0.6422 vs 0.8966（好 28.4%），但 E2E 慢一倍（918.8 vs 473.6 ms）。
  **这是我们最强的定位**：同等接受率下质量显著更高；同等速度下
  （对比 pose-relax 911.9 ms）我们精度全面更好（ADE_G 1.6251 vs 1.7548，
  ADE_T 0.6422 vs 0.7729）。
- `p50` 差异极大（pose-relax / ours 的 decode p50 = 27 ms，均值 463 ms）：
  因为一旦满 6 点接受就没有尾部，否则要串行再解 60 多个 token。
  尾部是长尾，ours 的 E2E `p90 = 1696.6 ms`，已经等于 AR 基线（1698.0 ms）。
  这是必须披露的弱点。

### 2.2 按接受长度分桶（ours，n=40）——延迟双峰与精度来源

`results/unified_accel_table.json` 逐帧记录的再聚合（与 §2.0 主表同一批帧）：

| 接受路点 | 帧数 | AR ADE | ours ADE | Δ | ours E2E (ms) |
|---|---|---|---|---|---|
| 0 | 2 | 0.6035 | 0.6035 | **+0.0000** | 1684 |
| 1 | 8 | 0.6680 | 0.6680 | **+0.0000** | 1698 |
| 2 | 6 | 0.7760 | 0.7735 | −0.0025 | 1285 |
| 4 | 1 | 0.7616 | 0.7373 | −0.0243 | 886 |
| 5 | 1 | 0.3062 | 0.4218 | +0.1156 | 690 |
| 6 | 16 | 0.7007 | 0.6980 | −0.0027 | **476** |

三条必须写进论文的结论：

1. **接受 0/1 的帧偏差恰为 0.0000。** 这两组共 10 帧（占 29%）
   走的是官方 `generate()` 全拒回退，输出与 AR 逐位相同。
   这既是正确性证据，也意味着**这些帧拿不到任何加速**（1684–1698 ms，
   与 AR 的 1639 ms 相比反而多付了草稿+验证成本）。
2. **延迟是双峰的**：满接受的 16 帧（40%）是 **476 ms（3.4×）**，
   接受 0/1 的 10 帧（29%）等于 AR。均值 1.71× 掩盖了这个分布。
   必须在论文里给这张分桶表，否则 p90 会暴露问题。
3. **满接受时精度不降反升**（0.6980 vs AR 0.7007）。
   这与早期 n=30 的结果（满接受 ADE 2.067 劣于 teacher 1.5085）方向相反，
   原因是早期用的是"vs teacher"口径且未修全拒路径；
   以 GT 为准的当前数据表明草稿在满接受帧上并不比 teacher 差。

### 2.3 非自回归并行解码（PD-VLA 式，n=12）

`scripts/run_live_pdvla_opendrive.py` → `results/live_pdvla_opendrive.json`

PD-VLA 原意是"chunk 内双向注意力 + Jacobi 迭代到不动点"。实测结果：

| 配置 | 收敛率 | 迭代数 | 语法可解析率 | ADE_T (m) | ADE_G (m) | E2E (ms) | 加速 |
|---|---|---|---|---|---|---|---|
| AR greedy | — | — | 1.00 | 0.00 | 3.572 | 1647.7 | 1.00× |
| `--mask causal`（朴素 Jacobi） | 1.00 | 6 | 1.00 | 5.90 | — | — | — |
| `--mask bidir`（PD-VLA 原意） | **0.00** | 24（上限） | **0.833** | 6.21 | **5.133** | 1342.9 | **1.23×** |

三个必须报告的失败模式：

1. **不带 chunk 内双向注意力时，迭代收敛到退化不动点**——6 个路点
   全部塌成 `(0.00,0.00)`，`ADE_T ≈ 5.9 m`。干净的负面结果：
   驾驶轨迹的并行解码**依赖** chunk 内全连接。
2. **带上双向注意力后反而不收敛**（24 次上限全部用完，`converged_rate = 0`）。
   每个位置同时依赖其它位置，不动点可能根本不存在。
3. **语法无保证**：17% 的帧（2/12）不动点里出现 `7..` 这类非法数字，
   解析直接失败。串行 AR 解码由词表合法性隐式保证，并行解码没有。
   表中解析失败记为硬失败（轨迹置零）。

**代价**：1.23× 加速换来 `ADE_G` 从 3.572 → 5.133（**+44%**），
而且延迟方差极小（p50≈p90≈1345 ms）说明它不是"偶发变慢"，
是**稳定地慢且稳定地错**。

**结论**：PD-VLA 式并行解码在我们这条"长 prompt（620 token）、
短 action chunk（66 token）"流水线上完全不可用。PD-VLA 原文 2.52×
是在机械臂短 prompt 上得到的，不可直接迁移；它也没有报告轨迹级
几何误差（机械臂只报成功率）。

### 2.4 阶段剖析小结

见第 0 节表格。

### 2.5 剪枝类方法的实测天花板：**零**（n=6，oracle 上界）

这是本次调研中最重要的负面结果，它推翻了纯靠 profile 推断出来的结论。

**先纠正一个推理错误**：仅看"prefill 只占 1.8%"就断言"剪枝天花板 1.8%" 是错的。
视觉 token 也在 KV cache 里，本该拖慢每个 decode step，而 decode 占 73%。
正确的估计应该是 `160/620 × 73% ≈ 19%`。为了判定哪个对，做了两个实验。

#### 实验一：decode 单步耗时是否随 prompt 长度变化？

`scripts/measure_kvlen_scaling.py` → `results/kvlen_scaling_opendrive.json`
（截断 prompt 尾部，测单步 decode，各长度取 3 次最小值）

| prompt 长度 | prefill (ms) | **decode 单步 (ms)** |
|---|---|---|
| 625（完整，n=3 帧 625–638） | 33.6 | **23.39** |
| 500 | 32.3 | 23.69 |
| 420 | 31.6 | 23.72 |
| 300 | 31.0 | 23.63 |
| 200 | 29.9 | 23.48 |

线性拟合：**斜率 −0.00064 ms / prompt token**，截距 23.86 ms。

**decode 是权重受限（weight-bound），不是 KV 长度受限。**
prompt 从 625 砍到 200（少 68%），单步耗时反而在噪声内不变
（23.39 → 23.48，差 +0.09 ms）。
所以"少 160 个视觉 token 能省 19%"这个推断不成立。

#### 实验二：oracle 剪枝直接测 E2E

`scripts/run_oracle_prune_opendrive.py` → `results/oracle_prune_opendrive.json`
（不做任何打分，直接把视觉特征切片。**这是任意剪枝策略的性能上界**）

| 保留比例 | 视觉 token | E2E (ms) | 加速 | 输出与未剪枝一致率 |
|---|---|---|---|---|
| 1.000（对照） | 148 | 1594.4 | 1.000× | 1.00 |
| 0.500 | 74 | 1613.2 | **0.988×** | 0.00 |
| 0.250 | 36 | 1599.3 | **0.997×** | 0.00 |
| 0.125 | 18 | 1607.7 | **0.992×** | 0.00 |
| **0.000（全删）** | **0** | **1590.7** | **1.002×** | 0.00 |

**把视觉 token 一个不剩全删掉，E2E 也只快 0.2%（在噪声内）。**

#### 结论

1. **FastDriveVLA / Prune2Drive / FastV 这条线在 OpenDriveVLA-0.5B 上
   端到端收益为零。** 它们的收益全在视觉塔内部（我们这里是 UniAD 那 411 ms），
   而那部分由 3D 检测/地图头决定，不吃 LLM 的视觉 token。
2. **它们报告的大幅加速来自完全不同的模型形态**：Impromptu-VLA 是
   3249 个视觉 token / 1596² 分辨率的多相机 ViT 直连 LLM，prefill 真的占大头；
   我们是 148 个 UniAD 查询 token，LLM prefill 只有 30 ms。
3. **写论文时不要说"剪枝天花板 1.8%"，要说"实测 0%"**，并给出
   这个 oracle 上界实验。这比 profile 推断强得多，因为它连
   "更好的打分函数" 这种反驳都堵住了。

### 2.6 草稿来源对比：零参数 vs 神经（n=903 holdout）

Zora（arXiv 2611.11098）主张用 RAG 检索替代神经网络草稿。
`scripts/eval_drafter_zoo_opendrive.py` → `results/drafter_zoo_opendrive.json`

同一接受规则（lon 0.5 / lat 0.3 m）、同一 holdout（903 帧）、同一接缝修复：

| 草稿器 | 参数 | 接受/6 | 满6率 | 零接受率 | decode 步数 | 记账加速 | ADE_T | ADE_G |
|---|---|---|---|---|---|---|---|---|
| bicycle（物理先验） | 0 | 2.833 | 21.9% | 9.4% | 3.39 | 1.77× | 0.244 | 1.944 |
| **rag (k=1)** | 0 | **0.548** | 2.7% | **76.2%** | 5.48 | **1.10×** | 0.087 | 1.915 |
| rag (k=3) | 0 | 0.557 | 2.4% | 74.3% | 5.47 | 1.10× | 0.097 | 1.916 |
| rag (k=5) | 0 | 0.574 | 2.4% | 73.9% | 5.45 | 1.10× | 0.096 | 1.919 |
| **neural（ours）** | 19M | **4.009** | **38.8%** | 10.1% | **2.38** | **2.52×** | 0.167 | **1.874** |

读法：

- **RAG 草稿在这条流水线上彻底失败**（接受 0.55/6，76% 帧全拒，
  记账加速只有 1.10×，基本等于没加速）。加大 k 完全无效。
- **原因**：驾驶轨迹由自车运动学强决定、由视觉弱决定。检索到"运动学相似"
  的历史帧，其未来轨迹在该时刻的**转向意图**上仍然可以完全不同
  （同一个 8 m/s、零偏航率的状态，可能是直行也可能是即将入弯）。
  RAG 的 `ADE_T = 0.087` 最低，正说明它检索到的轨迹"平均而言很接近"，
  但**逐点几何一致性**不足以跨过 0.5/0.3 m 的接受阈值。
- **这反而是对我们有利的证据**：单纯物理先验（bicycle，2.833）已经能拿到
  1.77×，说明轨迹确实主要由运动学决定；但从 2.833 到 4.009 的这一段，
  必须靠**学习**（读 prefill hidden）才能补上。这正是 DraftHead 存在的价值。
- 注意 ADE_G 一列：neural 最好（1.874），bicycle 最差（1.944），
  RAG（1.915）居中。加速和精度在这里方向一致，没有互斥。

---

## 3. 论文写作建议

### 3.1 建议的对比表结构

分三块，不要混成一张：

1. **序列级 SD 接受规则**（Spec-VLA exact / id-relax / pose-relax、
   Scaffold-SD、KERV-lite、ours）：同草稿同验证，只换规则 → 归因干净。
   这是我们已有的 `results/tab_sd_baselines.tex`。
2. **不同草稿来源**（DraftHead vs RAG-Zora vs 扩散草稿 Fast-dDrive 式）：
   目前只有我们和 RAG 可做，建议至少补一个"零参数检索草稿"作为上界/下界。
3. **正交加速项**（剪枝、早退、量化）：不要再写"理论上界"，
   直接报 §2.5 的 **oracle 实测 1.00×**。这个实验比任何 profile 推断都有力，
   因为它连"换个更好的打分函数"这种反驳都堵死了。

### 3.1.1 建议新增的两张表（都是我们独有的证据）

**表 A：剪枝天花板（oracle 上界）**

| 保留视觉 token | E2E 加速 | 输出一致率 |
|---|---|---|
| 100%（148 tok） | 1.000× | — |
| 0%（0 tok） | **1.002×** | 0.00 |

配一句："我们不做剪枝，但我们证明了在这个 teacher 上剪枝没用。"
这会把一篇可能的"缺少剪枝基线"审稿意见，转成我们的一个贡献。

**表 B：草稿来源对比**

| 草稿器 | 参数 | 接受/6 | 记账加速 | ADE_G |
|---|---|---|---|---|
| bicycle | 0 | 2.833 | 1.77× | 1.944 |
| rag (k=1/3/5) | 0 | 0.55–0.57 | 1.10× | 1.915 |
| neural（ours） | 19M | **4.009** | **2.52×** | **1.874** |

这张表同时回答两个质疑：
"为什么不直接用物理模型"（bicycle 只能到 2.833）
和"为什么不直接用检索"（RAG 只有 0.55）。

### 3.2 必须主动披露的四点

1. **token 级 SD 在我们的 teacher 上接受长度 ≈ 0**。这是真实测量，
   不是我们方法弱，而是驾驶轨迹的 tokenization（66 个 BPE 字符 token
   表示 6 个路点）让严格匹配失去意义。这一条同时解释了
   Spec-VLA 为什么必须放宽，以及我们为什么用米制阈值。
2. **KERV-lite 的 3.58× 建立在 ADE 劣化 24% 之上**。
   不要把它当作"更强的方法"，要当作"不同工作点"。
3. **我们自己的双峰延迟与长尾**：17% 的帧接受长度为 0，
   E2E 1709.6 ms 反而比 AR 基线慢；53% 的帧满接受，472.5 ms。
   `p90 = 1696.6 ms`。建议补"按接受长度分桶的延迟分布"图（见 §2.2）。
4. **满接受时精度由草稿决定**（ADE_G 2.067 vs teacher 1.5085），
   整体均值 1.6251 是被 0 接受帧的 0.555 拉回来的。这个抵消效应
   必须明说，否则审稿人一旦自己分桶就会认为我们在误导。

### 3.3 引用清单（按发表状态分级）

**已正式发表（可称顶会）**

- Spec-VLA — EMNLP 2025
- KERV — DAC 2026
- PD-VLA — IROS 2025
- FastDriveVLA — AAAI 2026
- Prune2Drive — CVPR 2026
- FastV — ECCV 2024 (Oral)
- DivPrune — CVPR 2025；LLaVA-PruMerge — ICCV 2025
- Orion-Lite — CVPR 2026 Workshop
- vLLM — SOSP 2023；FlashAttention — NeurIPS 2022
- AWQ — MLSys 2024；GPTQ — ICLR 2023；SmoothQuant — ICML 2023
- Fast-dLLM v1/v2 — ICLR 2026
- OpenDriveVLA — AAAI 2026（我们的 teacher #1）
- R&B-EnCoRe — 我们的 teacher #2（trajectory-only waypoints）

**预印本（不可称顶会）**

- Fast-dDrive (arXiv 2605.23163)、FlashDrive (arXiv 2608.12932)、
  Reasoning-aware SD (arXiv 2606.31160)、Zora (arXiv 2611.11098)、
  DeeAD (arXiv 2511.20720)、LatentVLA (arXiv 2601.05611)、
  Reasoning-VLA、VERDI、BitVLA、SQAP-VLA、ETA-VLA、AD-EE

---

## 4. 可复现命令

环境（`/data2/conda_env/` 下实测可用的解释器）：

| 环境 | torch | transformers | 用途 |
|---|---|---|---|
| `unidrivevla` | 2.5.1+cu118 | 4.57.1 | **本文全部 OpenDriveVLA 实验**（llava + mmengine） |
| `qwendrive` | 2.6.0+cu124 | 5.16.1 | R&B-EnCoRe（Qwen3-VL，需 transformers ≥5） |
| `morphx` | 2.6.0+cu124 | 4.57.1 | 备选 |
| `spec_openvla` / `spec_nora` / `spec_pi05` | 2.2 / 2.4 / 2.10 | 4.40 / 4.50 / 4.58 | 其他 VLA 复现线 |

已知坏环境（不要用）：`safebench`、`safebench_test`（缺 `libcudnn.so.9`）、
`vista`（缺 `urllib3`）、`TCP`（缺 `sympy`）、`maptr`（无 torch 或无 transformers）。

```bash
UNI=/data2/conda_env/unidrivevla/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

# 阶段剖析（谁占时间）
$UNI scripts/profile_opendrive_stages.py --n 10 --warmup 2 --start 1500 \
  --out results/profile_opendrive_stages.json

# 序列级 SD 各接受规则（n=30）
$UNI scripts/run_live_sd_baselines_opendrive.py --n 30 --warmup 3 --start 1500 \
  --out results/live_sd_baselines_opendrive.json

# PD-VLA 式并行解码（n=12）
$UNI scripts/run_live_pdvla_opendrive.py --n 12 --warmup 1 --start 1500 \
  --mask bidir --max-iter 24 --out results/live_pdvla_opendrive.json

# 剪枝天花板：decode 单步是否随 prompt 长度变化
$UNI scripts/measure_kvlen_scaling.py --n 3 --warmup 1 --start 1500 \
  --out results/kvlen_scaling_opendrive.json

# 剪枝天花板：oracle 视觉 token 切片（任意剪枝策略的上界）
$UNI scripts/run_oracle_prune_opendrive.py --n 6 --warmup 2 --start 1500 \
  --out results/oracle_prune_opendrive.json

# 草稿来源对比：bicycle / RAG / neural（903 帧 holdout，很快，7 秒）
$UNI scripts/eval_drafter_zoo_opendrive.py --k 1 \
  --out results/drafter_zoo_opendrive.json
```

## 5. 数据出处

| 文件 | 内容 |
|---|---|
| `results/profile_opendrive_stages.json` | 五段耗时 + prompt/token 构成，n=10 |
| `results/live_sd_baselines_opendrive.json` | 6 种接受规则，n=30，含逐帧记录 |
| `results/live_pdvla_opendrive.json` | PD-VLA 并行解码，n=12，含 `parse_ok` 逐帧标记 |
| `results/kvlen_scaling_opendrive.json` | decode 单步 vs prompt 长度，n=3 |
| `results/oracle_prune_opendrive.json` | 剪枝 oracle 上界，n=6 |
| `results/drafter_zoo_opendrive.json` | bicycle / RAG / neural 草稿对比，n=903 |
| `results/tab_sd_baselines.tex` | 两个 teacher 的 SD 基线表（论文用，n=12 口径，需按本文新数据更新） |

> 注：`tab_sd_baselines.tex` 里的 OpenDrive 数字来自 n=12（AR 1640 ms / 1.65×），
> 本文的 n=30 重测为 AR 1698.0 ms / 1.85×。写论文时统一用同一批 n，
> 不要混用。

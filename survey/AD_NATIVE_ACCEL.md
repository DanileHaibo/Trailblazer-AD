# 面向自动驾驶的 VLA 加速方法（AD 原生）

只收 **专门针对自动驾驶 VLA** 的工作。机器人操作 / 通用 VLA 的方法
（Spec-VLA、KERV、PD-VLA、Zora、SwiftVLA、SQAP-VLA、TRIAGE 等）
一律排除，仅在第 6 节列为跨域参考。

判定标准（三条全满足才算 AD 原生）：

1. 论文的实验平台是驾驶数据集或驾驶仿真（nuScenes / NAVSIM / Bench2Drive /
   Waymo / WOD-E2E / CARLA），不是 LIBERO / SimplerEnv / 机械臂；
2. 输出是**轨迹**或驾驶决策，不是 7-DoF 动作块；
3. 效率主张是在驾驶指标（L2 / ADE / PDMS / DS / RFS）下报告的。

---

## 1. 先看瓶颈在哪，才知道哪类方法有意义

OpenDriveVLA-0.5B + UniAD，单卡 L20，官方 `generate()`，n=10，CUDA-sync
（`scripts/profile_opendrive_stages.py`）：

| 阶段 | ms | 占 E2E |
|---|---|---|
| UniAD 视觉塔 | 411.7 | 24.9% |
| 视觉特征投影 | 0.5 | 0.03% |
| prompt 拼装 | 1.6 | 0.1% |
| **LM prefill（620 token）** | **30.6** | **1.8%** |
| **AR decode（66 token × 18.6 ms）** | **1211.8** | **73.2%** |

两个非直觉的实测结论，决定了下面每一类方法在我们这里的价值：

**（1）decode 是权重受限，不是 KV 长度受限。**
截断 prompt 测单步：

| prompt 长度 | decode 单步 (ms) |
|---|---|
| 625（完整） | 23.39 |
| 420 | 23.72 |
| 200 | 23.48 |

斜率 −0.00064 ms/token。prompt 砍 68%，单步不变。

**（2）视觉 token 全删，E2E 也不变。** oracle 切片：

| 保留视觉 token | E2E 加速 |
|---|---|
| 100%（148 tok） | 1.000× |
| 50% | 0.988× |
| 25% | 0.997× |
| **0%** | **1.002×** |

所以"剪视觉 token"这一类在 OpenDriveVLA 上端到端收益为零。
但注意：**这不代表剪枝方法本身没用**，而是它们的收益绑在
"多相机 ViT 直连 LLM"这种形态上（见 §3.1）。

---

## 2. AD 原生加速方法的六条技术路线

### 路线 A：视觉 token 剪枝 / 压缩（attack: 视觉塔 + prefill）

AD 原生、且是本领域**最主要**的一类。

| 工作 | 发表 | AD 平台 | 做法 | 报告收益 |
|---|---|---|---|---|
| **FastDriveVLA / ReconPruner** | **AAAI 2026** | nuScenes（Impromptu-VLA） | MAE 式前景重建打分 + 前景/背景对抗训练；即插即用，同视觉编码器可跨模型复用 | 3249→812 token，FLOPs 7.5×，**prefill 3.7×，decode 仅 1.3×**；L2 31.80 cm（未剪 31.83） |
| **Prune2Drive** | **CVPR 2026** | DriveLM / DriveLMM-o1 | T-FPS（余弦距离最远点采样）保多样性 + 逐视角自适应剪枝率；**无需重训、无需 attention map** | 留 10% token，**prefill 6.40×**，FLOPs 13.4%，DriveLM 掉 3% |
| **ORION / QT-Former** | **ICCV 2025** | Bench2Drive | query-based 压缩：512 scene query + 600 perception query + 16 history query 压多视角特征，配 memory bank | 架构级压缩，DS 77.47 / SR 54.62%（主要贡献是闭环性能，压缩是手段） |
| **DriveMoE（Vision MoE）** | **CVPR 2026** | Bench2Drive | router 按驾驶上下文**只选相关相机视角**，其余不送进 VLM；Action MoE 用技能专家替换稠密 FFN | 减少每步视觉 token；主报告是闭环性能与 mode averaging，效率为副产品 |

**关键观察**：这三篇的加速都集中在 **prefill**（3.7× / 6.4×），
而 prefill 在"多相机高分辨率 ViT 直连 LLM"的模型里确实占大头
（FastDriveVLA 的 baseline：3249 视觉 token，prefill 187 ms/token）。
一旦换成 **UniAD 查询 token（148 个）+ 0.5B LM**，prefill 只剩 30 ms，
这条路线就没有收益空间了（§1 已实测：oracle 全删 = 1.002×）。

**与我们的关系**：正交但无收益。可以写进论文，但只能作为
"我们验证了它在这个 teacher 上不可用"，不能当作可叠加的加速项。

---

### 路线 B：推理链（CoT）压缩 / 自适应深度（attack: reasoning tokens）

AD 原生，因为驾驶 VLA 越来越多先出一段推理。

| 工作 | 发表 | AD 平台 | 做法 | 报告收益 |
|---|---|---|---|---|
| **Reasoning-aware Speculative Decoding** | 预印本 2026 | Alpamayo-R1 | 双路 reasoner：routine（只看轨迹历史，**FlatRoPE** 1D 位置编码把注意力从视觉引开）+ deliberative（原 VLA 看视觉）；AARL 做 RL 后训练 | 推理段耗时 **~4×**；block=16 且 L=4.6 时理论上 3.5×，routine 路承担 78% CoC token |
| **DeeAD** | 预印本 2025 | ORION + Bench2Drive | 训练-free 早退：中间轨迹与轻量先验（导航 / 低精度规划）偏差 ≤δ 就退出；多跳控制器按分数变化率跳层 | δ=2.0 m 时跳 **28% 层**，381 → 270 ms（**−29%**），L2 0.67 → 1.13 m（碰撞率持平） |
| **ETA-VLA** | 预印本 2026 | 驾驶 | 时序融合 + LLM 内部稀疏化（ILSA） | −61% FLOPs |

**关键观察**：CoT 压缩那条线（自适应推理 + RL 惩罚 CoT 长度）是**训练期**
改行为，不是推理期加速。我们选的两个 teacher 都是 trajectory-only：
OpenDriveVLA 论文明确说用蒸馏内化推理以避免 CoT 开销，R&B 的
traj-only 变体也没有 reasoning block。所以：

- 不存在「跳过 CoT」这类可挖开销，整条 E2E 就是 prefill + 解码；
- 反过来这也让 SD 的收益更干净：省下的串行步数直接体现在墙钟上，
  不需要先把 think 段拿掉。

---

### 路线 C：非自回归 / 扩散式解码（attack: 串行依赖）

AD 原生，且是 2026 年最活跃的方向。

| 工作 | 发表 | AD 平台 | 做法 | 报告收益 |
|---|---|---|---|---|
| **Fast-dDrive（Scaffold Spec Decoding）** | 预印本 2026（NVIDIA） | WOD-E2E / nuScenes | block-diffusion VLA：块内双向、块间因果；**结构化 JSON 骨架 token 冻结**（scaffold 直接接受），只有取值 token 走 draft+verify；SASD 训练 | 7855 → 1919 ms（**4.1×**，单 H100）；接 SGLang+FP8 到 **11.8×**（608 TPS）；ADE@5s 1.982（AR 2.083） |
| **FlashDrive** | 预印本 2026 | Alpamayo 1.5-10B | 四段联合：时序重叠 **streaming KV 复用** + 扩散草稿 SD（block=8，接受长度 5.6）+ flow 速度场自适应步缓存 + W4A8 + CUDA Graph | 717 → 151 ms（**4.7×**），1.4 Hz → 6.6 Hz；minADE@6.4s 仅变 0.08 m |
| **Reasoning-VLA** | 预印本 2025 | 驾驶 | 可学习 action query，一次前向出整条轨迹 | — |
| DriveVLA-W0（Action Expert） | **ICLR 2026** | NAVSIM v1/v2 | 轻量 MoE Action Expert（500M）与大 VLA backbone 解耦，Joint Attention 共享 KV，**一次前向出轨迹** | H200 延迟 117.8 → 74.3 ms（**63.1%**），PDMS 85.6 → 88.4 |

**关键观察**：这一类的加速比（4.1× / 4.7×）**远高于**推测解码类，
因为它们同时干掉了"串行"和"每步都要跑一遍大模型"两件事。
Fast-dDrive 的 Scaffold 思想对我们有直接借鉴价值：
`<traj_start>` / `[(` / `),(` / `]` / `<traj_end>` 就是天然的 scaffold，
可以无条件接受，只对数字 token 做 draft/verify。

**与我们的关系**：需要换 backbone（block diffusion）或改架构（Action Expert），
不是纯推理期插件。但 **Scaffold 这个局部技巧可以直接搬到我们的 SD 上**。

---

### 路线 D：推理期 KV / 缓存复用（attack: prefill 重复计算）

| 工作 | 发表 | AD 平台 | 做法 | 报告收益 |
|---|---|---|---|---|
| **FlashDrive（streaming KV）** | 预印本 2026 | Alpamayo | 相邻帧视觉高度重叠 → 跨帧复用 KV cache，只算新增部分 | 属于其四段联合加速的一部分 |
| **Fast-dDrive（shared-prefix KV）** | 预印本 2026 | WOD-E2E | N 条随机 rollout 从**同一份 shared-prefix KV** 分叉，再平均；低成本 test-time scaling | 用极小代价压预测方差 |
| **DriveVLA-W0（Joint Attention）** | **ICLR 2026** | NAVSIM | Action Expert 与 backbone 各自算 Q/K/V 后拼接做联合注意力，避免重复 prefill | 见路线 C |

**关键观察**：这条路线的前提是**连续帧推理**（同一个 scene 相邻时刻）。
我们的评测是逐帧独立（且已发现 UniAD 的时序状态会让逐帧不可复现，
见 `survey/ACCEL_SURVEY.md` §2.0.1），所以这条线在当前的离线评测协议下
无法体现收益；要测必须在连续的 scene 上跑。

---

### 路线 E：蒸馏 / 轻量化（attack: 整个模型）

| 工作 | 发表 | AD 平台 | 做法 | 报告收益 |
|---|---|---|---|---|
| **Orion-Lite** | **CVPR 2026 Workshop** | Bench2Drive | 隐特征蒸馏 + GT 轨迹监督；把 7B LLM 换成 0.1B 纯视觉 transformer（1 个 planning query + 6 层 decoder 做 cross-attention 瓶颈） | 推理模块 **150×**，整体 **3×**，显存 31 → 8 GB；**DS 80.6 超过 teacher** |
| **DriveVLA-W0（Action Expert）** | **ICLR 2026** | NAVSIM + 7000 万帧内部数据 | 轻量 MoE action expert 替代大 backbone 出动作 | 见路线 C |
| **LatentVLA** | 预印本 2026 | NAVSIM / nuScenes | 自监督 latent action 预测（无需语言标注）+ 蒸馏到视觉网络 | ~790 → ~210 ms；NAVSIM PDMS 92.4 |
| **VERDI** | 预印本 2025 | 驾驶 | 跨家族蒸馏，把 VLM 推理抽成模块化 AD 系统，运行时零 VLM 开销 | — |

**关键观察**：Orion-Lite 是这批里**收益最大**的（整体 3×，显存 31→8 GB，
且精度不降反升）。代价是要训练一个 student，且之后不再是"VLA"了。

**与我们的关系**：不构成竞争，而是**替代**。如果审稿人问"为什么不直接蒸馏",
这是标准答案。可以作为上限参考。

---

### 路线 F：系统级 / 部署栈（attack: 执行效率，不改算法）

| 工作 | 发表 | AD 平台 | 做法 | 报告收益 |
|---|---|---|---|---|
| **FlashDrive（W4A8 + CUDA Graph）** | 预印本 2026 | Alpamayo 10B | ParoQuant 量化权重到 4bit、激活 8bit，Marlin kernel；action expert 保持 BF16；CUDA Graph 编译 + 算子融合 | 显存 31.6 → 18.3 GB；再降 14%（176.0 → 151.4 ms） |
| **Fast-dDrive（SGLang + FP8）** | 预印本 2026 | WOD-E2E | 接 SGLang serving，FP8 kernel | 4.1× → **11.8×**，TPS 51.6 → 608.5 |
| **DriveVLM-Dual** | 预印本 2024 | 真实部署 | 快反应层 + 慢推理层双过程架构 | ~1500 → ~300 ms |

**关键观察**：这一类的加速比在四类里最高（11.8×），但**完全依赖工程栈**，
换机器就没了。我们的 L20 + PyTorch eager 环境离这个很远。

---

## 3. 汇总：按"攻击哪一段"排优先级

### 3.1 各路线在我们这条流水线上的实际空间

| 路线 | 代表工作 | 攻击的阶段 | 报告收益 | **在我们这里的实测/预估** |
|---|---|---|---|---|
| A 剪枝 | FastDriveVLA (AAAI'26), Prune2Drive (CVPR'26) | prefill | 3.7×–6.4× prefill | **1.002×（oracle 全删实测，零收益）** |
| B CoT 压缩 | Reasoning-aware SD, adaptive reasoning + RL | reasoning | −67% / 4× | **两个 teacher 都无 CoT，不适用** |
| C 非自回归 | Fast-dDrive, FlashDrive, DriveVLA-W0 | 串行依赖 | 4.1×–4.7× | 需换 backbone；**但 Scaffold 技巧可直接搬** |
| D KV 复用 | FlashDrive, Fast-dDrive | 跨帧重复 prefill | 含在 C 内 | 需连续 scene 评测，当前协议测不出 |
| E 蒸馏 | Orion-Lite (CVPR'26 W) | 整模型 | 3×（整体） | 是替代而非叠加 |
| F 系统 | FlashDrive W4A8, SGLang | 执行栈 | 最高 11.8× | 依赖工程栈 |

### 3.2 结论

对我们（OpenDriveVLA-0.5B + UniAD，AR，轨迹级输出）：

1. **只有路线 C 的局部技巧（Scaffold）、和推理期 SD 类**能真正起作用；
2. 路线 A 在 AD 领域最热，但在我们这个模型形态上**零收益**，
   这是可以写进论文的负面结果（oracle 实验已做）；
3. 路线 B 在 trajectory-only teacher 上不适用：没有 CoT 可压，
   省下的串行步数只能靠推理期 SD 拿到；
4. 路线 E/F 是不同赛道，不该混进同一张对比表。

---

## 4. 发表状态分级（写引用时不要搞错）

**已正式发表（可称顶会）**

| 工作 | 会议 |
|---|---|
| FastDriveVLA | AAAI 2026 |
| Prune2Drive | CVPR 2026 |
| DriveMoE | CVPR 2026 |
| ORION | ICCV 2025 |
| DriveVLA-W0 | ICLR 2026 |
| Orion-Lite | CVPR 2026 Workshop |
| OpenDriveVLA（我们的 teacher） | AAAI 2026 |

**预印本（不可称顶会）**

| 工作 | arXiv |
|---|---|
| Fast-dDrive | 2605.23163（NVIDIA，2026-05） |
| FlashDrive | 2608.12932 |
| Reasoning-aware Speculative Decoding | 2606.31160 |
| DeeAD | 2511.20720 |
| LatentVLA | 2601.05611 |
| Reasoning-VLA / VERDI / ETA-VLA / AD-EE | — |

---

## 5. 与我们方法的关系（论文里怎么说）

| 方法 | 关系 | 建议写法 |
|---|---|---|
| FastDriveVLA / Prune2Drive | **正交但无收益** | 报告 oracle 上界实验（1.002×），说明不是我们没测，是这条路线在此 teacher 上无空间 |
| Fast-dDrive Scaffold | **可直接吸收** | 我们的 `<traj_start>`/`[(`/`]` 是天然 scaffold，可作为 accepting rule 的一个扩展 |
| DriveVLA-W0 / Orion-Lite | **替代路线** | 明确说明它们是换模型，不是推理期加速，不构成同表对比 |
| FlashDrive / Fast-dDrive 系统部分 | **不同评测条件** | 它们报的是 H100 + SGLang + FP8，我们是 L20 + PyTorch eager，不可直接比绝对值 |

---

## 6. 被排除的跨域工作（列表见 `ACCEL_SURVEY.md`）

以下工作的实验平台是机器人操作，**不属于"专门面向自动驾驶"**，
但方法可借鉴，引用时须注明是跨域：

Spec-VLA（EMNLP'25，OpenVLA/LIBERO）、KERV（DAC'26，LIBERO）、
PD-VLA（IROS'25，7-DoF 机械臂）、Zora（预印本，LIBERO）、
SwiftVLA（CVPR'26，π0/Jetson）、SQAP-VLA（预印本，Simpler）、
TRIAGE（预印本，LIBERO/SimplerEnv）、FastV（ECCV'24，通用 VLM）、
DivPrune、LLaVA-PruMerge、Medusa、vLLM。

我们自己在 `ACCEL_SURVEY.md` 里已经复现了其中三个（Spec-VLA 的三种接受规则、
PD-VLA 的并行解码、Zora 式 RAG 草稿），**结论应表述为"跨域方法的迁移实验"**，
而不是"与 AD 方法的对比"。

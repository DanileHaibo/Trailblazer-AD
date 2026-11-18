# Skip-layer 与 Pruning：技术谱系

这两类方法常被混为一谈，但它们作用在不同的"维度"上，且适用的前提条件完全不同。

```
一个 Transformer 的计算量 ≈ 层数(深度) × 序列长度 × 宽度
                                ↑            ↑        ↑
                          skip-layer      pruning   量化
```

- **skip-layer**（跳层/早退）= 减少**深度**
- **pruning**（剪枝）= 减少**序列长度**（token 剪枝）或**宽度/层数**（结构化剪枝）

---

## 一、Skip-layer（动态深度）

### 1.1 两个基本范式，差别很大

| | Early Exit（早退） | Layer Skipping（跳层） |
|---|---|---|
| 做什么 | 在第 k 层就输出，**丢弃所有后续层** | **选择性跳过**某些层，其余照跑 |
| 决策粒度 | 每个 token 一个退出点 | 每层/每 token 一个"是否执行" |
| 误差分布 | 误差**集中在上层** | 误差**均匀分布** |
| 最终 hidden 相似度 | 低 | **高** |
| 对生成任务 | 差（误差累积） | **更好** |

这个对比有硬数据支撑：`Dynamic layer selection in decoder-only transformers`
(arXiv 2410.20022) 直接测了两种方法在 24 层 decoder-only LM 上的表现，
结论是 **pre-trained 模型对 layer skipping 的鲁棒性显著高于 early exit**。
原因是早退把误差全压在上层，导致 next-token 预测变差；
跳层把误差摊平，虽然 KV cache 精度略低，但 token 预测更准。

**这直接解释了我们的 DeeAD 复现为什么失败。** DeeAD 属于 early-exit 范式，
我们在 0.5B/24 层上测到 depth 8–20 的输出**完全相同且非法**，
正是"误差集中在上层"的极端表现。

### 1.2 Early Exit 家族

| 工作 | 发表 | 机制 | 关键点 |
|---|---|---|---|
| **CALM** | NeurIPS 2022 | 中间层预测 + 置信度阈值决定是否退出 | 需要**校准**阈值以满足质量保证；训练时对所有层加权监督 |
| **DeeAD** | 预印本 2025 | 训练-free，按中间轨迹与轻量先验的偏差 ≤δ 早退 | 用**物理可行性**而非置信度做判据；ORION 上 −29% 延迟 |
| **AD-EE** | 预印本 2025 | VLM 早退，面向驾驶 | — |
| **LayerSkip** | ACL 2024 | 见下（它同时是训练法） | Meta |

CALM 的一个重要结论：**静态层数分配是浪费的**。
它的 oracle 实验显示，只用 **1.5 层**就能保持全模型性能（在部分预测上），
而静态基线需要 3–7 层。也就是说绝大多数 token 根本不需要深层计算。

### 1.3 LayerSkip：目前最完整的方案（Meta, ACL 2024）

它不是纯推理技巧，而是**训练 + 推理联合设计**。三件事：

**(1) Layer Dropout（训练）**
```
x_{l+1,t} = x_{l,t} + M(p_{l,t}) · f_l(x_{l,t})
```
`M(p)` 是 Bernoulli(1-p)。关键是 dropout 率**随层数指数增长**：
```
D(l) = exp(l·ln2/(L-1)) - 1        # 第 0 层 0，最后一层 1
```
即浅层几乎不丢、深层大量丢 —— 逼模型不要依赖深层。

**(2) Early Exit Loss（训练）**
所有层共享同一个 LM head，每一层都直接受语言建模损失监督。
这一条是**前提**：否则中间层的 hidden 根本无法被 lm_head 正确 unembed。

**(3) Self-Speculative Decoding（推理）**
```
前 E 层  → 草稿（draft）
后 L-E 层 → 验证与纠正（verify）
```
因为草稿和验证用**同一个模型**，KV cache 和激活可复用 ——
内存占用比"小模型草稿"方案低得多。

报告加速：CNN/DM 摘要 **2.16×**、代码 1.82×、TOPv2 语义解析 2.0×。

**重要限制**（HF 官方博客明确写了）：
> 只有用这个 recipe 训练过的 checkpoint 才能拿到加速。
> 对任意 decoder-only 模型设 `assistant_early_exit` 只是能跑，
> 但中间层 logits 无法正确 unembed，且不会变快。

### 1.4 Mixture-of-Depths (MoD)

与 early exit 不同，MoD 是**稀疏激活**：
每个 block 只让 top-k 个 token 通过（其余走残差直连）。

| 特性 | 说明 |
|---|---|
| 计算量 | 固定预算（top-k 比例事先定），不是自适应 |
| tensor 形状 | **静态**，对硬件友好（不像早退那样动态变长） |
| 训练 | 需要从头训练 router |

与 CALM 的对比很关键：CALM 是"per-token 自适应退出"，MoD 是"per-token 固定预算路由"。
MoD 的静态形状让它在 GPU 上更容易真正变快。

### 1.5 Router-Tuning：让 MoD 变得实用（EMNLP 2025）

MoD / LayerSkip 的最大障碍是**训练成本**（要从头训或大规模继续预训练）。
Router-Tuning 只微调 router：

- 冻结主模型，只训一个判断"这层要不要跳过"的小 router
- 支持 token-level 和 sequence-level 两种粒度
- 效果：**21% 加速，性能只掉 0.2%**

这是目前"训练成本 vs 收益"最划算的一条。

### 1.6 静态层剪枝（不是推理期跳层，是改模型）

如果连推理期的动态判断都不想要，可以直接**删层**：

| 工作 | 指标 | 效果 |
|---|---|---|
| **ShortGPT** | Block Influence = 层输入与输出的余弦相似度 | LLaMA2-13B 删 10 层（25%），MMLU 55.0→52.2 |
| **ShortGPT-gen** | 同上 + 输入跳层、生成走全层 | 训练-free，GSM8K 28.89→26.91 |
| **LaCo** | 把相似层**合并**而非删除 | 梯度式合并，比硬删好 |
| **Reassessing Layer Pruning** | 系统对比各种指标 | **结论颠覆**：直接删最后 25% 层 + 微调，比复杂指标更好 |

`Reassessing Layer Pruning in LLMs`(arXiv 2411.15558) 的结论值得记住，
因为它花了数千 GPU 小时做基准：

> 1. **倒序剪枝**（直接删末尾几层）比很多复杂指标更好
> 2. **LoRA 不如部分层微调**（只训最后几层 + lm_head）
> 3. **迭代剪枝无益**，一次性剪更好

也就是说：这一领域的很多"精妙指标"，实际不如"删最后 25% 再微调"。

---

## 二、Pruning

需要先区分三个完全不同的东西，它们的名字都叫 pruning：

| 类型 | 剪什么 | 减少的计算 | 代表 |
|---|---|---|---|
| **Token pruning** | 序列中的 token | 序列长度（O(N²)） | FastV, DivPrune, FastDriveVLA |
| **Depth pruning** | 整个 Transformer block | 层数 | ShortGPT, LaCo |
| **Width / structured pruning** | 注意力头、FFN 通道 | 宽度 | 各 LLM 压缩工作 |

下面主要讲 **token pruning**，因为在驾驶 VLA 上它最热。

### 2.1 Token pruning 的两大流派（重要，且互相批评）

| 流派 | 判据 | 代表 | 弱点 |
|---|---|---|---|
| **Importance-based**（重要性） | 保留注意力高的 token | FastV, SparseVLM, HiRED, PyramidDrop | 保留的 token **彼此相似**，冗余高 |
| **Redundancy-based**（冗余/多样性） | 保留彼此差异大的 token | DivPrune, DART | 会选到**背景边缘**，丢细节 |

这两个流派的互相批评很有信息量（引自 VLM-Pruner）：

> Importance-driven 方法倾向于保留同一个焦点物体周围的多个相似局部区域，
> 造成大量冗余。在某些情况下甚至**不如随机剪枝**（DART 论文的发现）。
>
> Redundancy-reduction 方法选出的 token 分散但不完整，抓不住物体的细粒度细节。
> 因为背景边缘与主体相似度低，DivPrune 容易选到边缘背景 token。

**结论：两者都不完美，所以近期工作都在做 hybrid。**

### 2.2 具体方法

| 工作 | 发表 | 判据 | 备注 |
|---|---|---|---|
| **FastV** | ECCV 2024 Oral | 第 K 层后，保留**从文本 token 收到的注意力**高的视觉 token | 最经典的 baseline |
| **SparseVLM** | 2025 | 文本-to-视觉注意力，**text-aware**（问题自适应） | 批评 FastV 是 text-agnostic |
| **PyramidDrop** | 2025 | 逐层渐进丢弃（浅层保留全部，深层越丢越多） | 利用"视觉冗余随层数增长" |
| **VisPruner** | 2025 | 两阶段：先按 [CLS] 注意力选重要，再按余弦相似度去冗余 | hybrid |
| **DivPrune** | CVPR 2025 | 贪心最大化**最小成对距离**（多样性） | 多样性最强（erank 21.84） |
| **LLaVA-PruMerge** | ICCV 2025 | [CLS] 注意力 + key 相似度聚类，**剪+并** | 多样性最弱（erank 10.91） |
| **FastDriveVLA / ReconPruner** | AAAI 2026 | **MAE 前景重建**打分 + 前后景对抗训练 | 驾驶专用；即插即用 |
| **Prune2Drive** | CVPR 2026 | **T-FPS**（余弦距离最远点采样）+ 逐视角自适应剪枝率 | 驾驶专用；无需 attention map |

### 2.3 一个被低估的细节：注意力类方法可能不如随机

DART / VLM-Pruner 都指出：FastV 这类纯注意力方法，
在某些数据集上**不比随机剪枝好，甚至更差**。

原因：注意力高的 token 高度聚集在同一个物体上，
剪掉其他区域后信息覆盖不全。这也解释了为什么
VisPruner 这类"注意力 + 多样性"的 hybrid 能稳定胜出。

### 2.4 驾驶场景的两个特殊考虑

FastDriveVLA 和 Prune2Drive 都做了通用方法没做的事：

**FastDriveVLA 的核心论点**：
> 通用 VLM 的剪枝依赖 visual token 相似性 或 visual-text 注意力，
> 但两者在驾驶场景下表现都差。

它的替代方案是**前景重建**：用 MAE 式像素重建训练一个 pruner，
让它学会给"前景（车、人、车道）"高分。这是把**驾驶领域的先验**
（人开车只看前景）编进 pruner，而不是用通用的注意力启发式。

**Prune2Drive 的两个创新**：
1. **T-FPS**：用余弦距离的**最远点采样**保证跨视角的语义+空间覆盖
   （直接针对"注意力方法保留聚集 token"的问题）
2. **逐视角自适应剪枝率**：不同相机分配不同预算（前视保留多、侧视少）

---

## 三、两者在我们模型上的实测（都已做过）

| 方法 | 类别 | 我们实测 | 归零原因 |
|---|---|---|---|
| FastV (K=2, R=50%) | token pruning | **0.97×** | 两遍法 prefill 开销吃掉收益 |
| Pruning oracle（保留 25%） | token pruning 上界 | **1.01×** | decode 是权重受限，剪 KV 不省时间 |
| DeeAD（depth 22/24） | early exit | **0.89×**，ADE 3.25 | 0.5B 无冗余深层，depth 8–20 输出完全相同 |

详见 `survey/FOUR_ACCEL_REPRO.md`。

**为什么全归零**（两条根本原因）：

1. **我们的 decode 是权重受限，不是 KV 长度受限。**
   实测 prompt 从 625 token 砍到 200，单步耗时 23.39→23.48 ms 不变。
   所以 token pruning 省不到 decode 时间。

2. **我们的 0.5B 模型没有冗余层。**
   DeeAD 扫描显示 depth 8–20 的输出**完全相同**（gap 6.693），
   只有满 24 层可用 —— 与 ShortGPT 在 LLaMA2-13B 上
   "删 25% 层只掉 5%" 的结论形成鲜明对比。
   小模型没有大模型那种层冗余。

---

## 四、如果我们还要在这个方向做，哪条路最有希望

按"前提是否满足"排序：

| 方案 | 需要什么 | 我们的状态 | 预期 |
|---|---|---|---|
| **LayerSkip 式自推测解码** | 用层 dropout + early exit loss **重训** | 可做（有训练脚本） | 高。同时省深度和步数 |
| **Router-Tuning** | 只训 router（成本最低） | 可做 | 中高。21% 加速/0.2% 掉点 |
| **ShortGPT 式静态删层** | 一次剪枝 + 微调最后几层 | 可做 | 未知。需先测 BI 分布 |
| FastV / FastDriveVLA 类 | 已测 | **零收益** | ✗ |
| DeeAD 式 early exit | 已测 | **失败** | ✗ |

**最值得试的是 LayerSkip 的自推测解码**，因为它同时解决两个问题：
- 前 E 层做草稿 → 减少 decode 步数（与我们 TrajSpecDec 同向）
- 后 L−E 层验证 → 复用 KV cache（比独立草稿模型省内存）

而且它与我们的方法**可以叠加**：LayerSkip 加速"每步"，TrajSpecDec 减少"步数"。

**但必须先解决一个前提**：LayerSkip 需要重训。我们目前是冻结 teacher 做推理期加速，
这是根本性差异。要试的话需要先确认是否有训练算力和数据 pipeline。

---

## 五、一句话总结

> **skip-layer 减少"层数"，pruning 减少"token 数"；
> 前者被误差分布问题困扰（早退≠跳层），后者被 KV 长度是否真的影响延迟困扰。
> 两者都属于"减少计算量"，而**我们的瓶颈是减少不了的计算量**（权重加载）。**

这就是为什么这两类方法在我们模型上全部归零 ——
它们假设瓶颈是 FLOPs，而我们的实测显示瓶颈是内存带宽。

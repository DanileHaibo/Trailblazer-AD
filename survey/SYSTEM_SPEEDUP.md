# 能拿到大加速的地方在哪：系统层 > 算法层

## 1. 微基准：decode 单步的真实成本结构

`scripts/bench_decode_micro.py` → `results/decode_microbench.json`
裸 Qwen2-0.5B（OpenDriveVLA 的 LM），L20，620 token prompt：

| 配置 | 1 token | 620 token prefill |
|---|---|---|
| eager fp32 | **16.68 ms** | 26.41 ms |
| bf16 decoder stack | 14.76 ms | 16.15 ms |
| **compiled bf16** | **1.68 ms** | **8.94 ms** |

**compiled 把 decode 单步从 16.68 → 1.68 ms，9.9×。**

拆开那 16.68 ms：

| 组件 | 耗时 |
|---|---|
| 24 层 decoder | 16.81 ms（≈0.7 ms/层） |
| lm_head（151682 词表） | 0.75 ms |
| embed_tokens | 0.02 ms |

关键判据：**1 个 token 的前向（16.8 ms）比 620 个 token 的 prefill（26.4 ms）
慢不了多少。** 这只能是 kernel 启动开销主导——24 层 × 约 6 个 kernel × 约 4 μs，
再加每步 Python 开销。算力完全不是瓶颈。

所以在这个模型上：

- **算法层**（少解码 token）的天花板 = decode 占比 73%
- **系统层**（每步更快）的天花板 = **9.9×**，且与算法层**相乘**

这解释了为什么所有公开的 4–12× 驾驶 VLA 加速论文都是"算法 + 系统"联合设计：
单靠算法拿不到那个量级。

## 2. 为什么之前没发现

我们的 harness 一直用 `model.float()`（fp32），沿袭自 OpenDriveVLA 官方脚本，
而官方那样写是**被迫的**：

```
RuntimeError: "modulated_deformable_im2col_gpu" not implemented for 'BFloat16'
```

UniAD 的 mmcv deformable-conv 没有 bf16 kernel。但**这只约束视觉塔**，
而视觉塔每帧只跑一次（411 ms）。LM 是 24 层纯 Qwen2，完全可以 bf16。

正确的混合精度分工：

| 组件 | dtype | 理由 |
|---|---|---|
| UniAD 视觉塔 | **fp32** | mmcv 算子无 bf16 kernel |
| mm_projector_* | fp32 | 消费 fp32 特征 |
| LM decoder（24 层 + norm） | **bf16** | 纯 Qwen2，无自定义算子 |
| lm_head | bf16 | 同上 |

边界：`prepare_inputs...` 输出的 prompt embeddings 在进 LM 前转 bf16；
logits 出 LM 后转回 fp32（接受规则和解析需要）。

## 3. 已验证的部分与未验证的部分

必须区分开，不要混报。

### 已确证（微基准，多次重复）

| 结论 | 数据 |
|---|---|
| compile 对 decode 单步 9.9× | 16.68 → 1.68 ms |
| compile 对 prefill 2.95× | 26.41 → 8.94 ms |
| bf16 单独只带来 1.13× | 16.68 → 14.76 ms |

注：compile 的这两行是**纯前向微基准**（`input_ids` 随机、无 KV cache 复用），
用于定位开销结构；它们不等于 E2E 加速，见 §3 的估算。

### 已验证但需注意

| 结论 | 数据 | 注意 |
|---|---|---|
| bf16 使 prefill 32 → 21 ms | `results/mixed_precision_bench.json` | 完整 pipeline；E2E 1.08× |
| **LM-only bf16 输出不是 bit-exact** | token 数相同（62–63），但个别坐标有 2 位小数差异 | 见下表 |

fp32 vs bf16 的逐帧对比（同一帧、同一条 prompt，只改 LM 精度）：

| fp32 | bf16 |
|---|---|
| `[(0.02,2.29),(0.03,4.42),(0.03,6.35),(0.03,8.00)` | `[(0.02,2.29),(0.03,4.38),(0.03,6.28),(0.03,7.98)` |
| `[(0.02,2.05),(0.04,3.93),(0.06,5.65),(0.08,7.20)` | `[(0.02,2.04),(0.04,3.92),(0.06,5.65),(0.08,7.20)` |
| `[(0.01,1.84),(0.02,3.55),(0.03,5.08),(0.04,6.42)` | `[(0.01,1.84),(0.02,3.55),(0.03,5.07),(0.04,6.34)` |

**不能说 bf16 是 lossless。** 逐字符 tokenize 的数值输出对浮点误差敏感
（第 3 位小数抖动就会改变后续 token），这与"图像分类 top-1 不变"是两回事。
必须在表里报 ADE 差异，而不是声称输出一致。

### 尚未验证（不要写进论文）

| 项 | 状态 | 原因 |
|---|---|---|
| E2E 层 compile 加速比 | **尝试过，失败** | 见下 |
| CUDA Graph 捕获 | **未测** | `StaticCache` + bf16 捕获失败 |
| compile × 推测解码的组合加速 | **未测** | 依赖上两项 |

#### E2E compile 的失败记录（`scripts/bench_compile_e2e.py`）

按 serving engine 的标准做法（固定 shape、static cache、in-place mask 更新）尝试了四条路，全部失败：

| 尝试 | 结果 |
|---|---|
| `mode="reduce-overhead"` + 动态 KV cache | `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run` —— CUDA Graph 的 buffer 复用与 KV cache 增长冲突 |
| `mode="default"` + 自然长度 | 每步序列长度变 1，guards 反复重编译，10 分钟未出一帧 |
| 固定 L_PAD + `StaticCache` | 首个 token 正确（diff 4.8e-5），但 decode 阶段 cache 偏移按 `L_PAD` 写入，输出退化成 `0.0.0.00.0.0...` |
| `mode="default"` + 自然长度 + 正确 mask | 首帧 **191 秒**（编译），第二帧 1057 ms —— **无加速** |

**而且这四条路上手动解码循环都与官方 `generate()` 不一致**
（`ADE_vs_fp32` = 0.49–0.68，本该为 0）。这是比加速更严重的问题：
说明我复现的解码循环缺少 `generate()` 的某个内部步骤，
在修好这个之前，任何"加速比"都无意义。

**诚实结论**：微基准的 9.9× 是"单步前向"的真实数字，
但把它变成 E2E 加速需要的工程量（固定 shape + CUDA Graph + 与官方解码器逐 bit 对齐）
超出本轮范围。这条路线**方向正确但未完成**。

**诚实表述**：微基准的 9.9× 是"单步前向"的加速，不是 E2E。
E2E 上界可以通过下式估算：

```
E2E 时间 = prefill(30ms) + n_tokens × step(16.7ms) + 视觉塔(411ms)
```

若 step 降到 1.68 ms 且 prefill 降到 8.9 ms：

```
fp32 eager : 30 + 64×16.7 + 411 = 1510 ms
compiled   :  9 + 64×1.68 + 411 = 527 ms   → 2.9×
```

**注意视觉塔（411 ms）成了新瓶颈，占 compiled 版的 78%。**
这也解释了 FastDriveVLA/Prune2Drive 为什么先攻视觉 token——
在 system-optimized 的栈上，视觉塔才是主要开销。

## 4. 下一步（按预期收益排序）

| # | 动作 | 预期 | 难度 |
|---|---|---|---|
| 1 | 让 compile 在自回归循环里稳定（固定 prompt 长度、pad 到固定 shape、用 static cache） | E2E 2.5–3× | 中 |
| 2 | CUDA Graph 捕获稳态 decode | 在 1 上再 +20~40% | 中 |
| 3 | 把 UniAD 视觉塔也降精度（fp16 而非 bf16，mmcv 可能支持） | 411 → 约 250 ms | 低 |
| 4 | 在 1+2 的基础上叠加 TrajSpecDec | 组合，见下 | 低 |

第 4 项的估算：如果单步 1.68 ms、视觉塔 411 ms，
那么 TrajSpecDec 省下的 40 个 token ≈ 67 ms，占 E2E 的 13%——
**系统优化反而会稀释算法加速的占比**。这是必须提前想清楚的定位问题：
在未优化的栈上我们是 1.75×，在优化后的栈上可能只有 1.1×。

## 5. 对我们论文的影响

三种可能的定位，需要明确选一个：

**(a) 在 naive 栈上比**（当前做法）
1.75× vs AR 1.00×，但审稿人会问"为什么不 compile/bf16？"

**(b) 在 optimized 栈上比**
公平，但我们的相对优势被稀释到约 1.1×；
需要论证"decode 步数减少"的价值（减少 40 个 token 是真实的计算节省）。

**(c) 双向报告**
主表在 optimized 栈上（公平），附一张 naive 栈的表说明算法增量。
并在正文明确：**算法加速与系统加速相乘，我们的贡献是可叠加的那部分。**

推荐 (c)，且必须在 Limitations 里写明视觉塔是优化后的主要瓶颈。

# 怎么复现"加速论文"：方法学与踩过的坑

这份文档不是文献综述，而是**复现加速论文时容易犯的错误**清单，
全部来自我们在这个仓库里的实测教训。

---

## 1. 先分清三种"加速"，它们不能混在一起报

| 类型 | 定义 | 典型来源 | 我们的实测 |
|---|---|---|---|
| **记账加速**（accounting） | 只数"解码步数"变少 | 推测解码论文常报 | decode-step 3.3×/4.0×，但 wall-clock 1.26×/1.65× |
| **算法加速**（algorithm） | 同样的栈，少做事 | 接受规则、早退、剪枝 | 我们 1.51–1.75× |
| **系统加速**（system） | 同样的算法，每步更快 | bf16、`torch.compile`、CUDA Graph、量化、SGLang | **compile 单步 9.9×** |

**混报是复现中最常见的错误。** 论文写 "4.1×" 时，必须问清是哪一种。
Fast-dDrive 的 4.1× 是"块扩散 + scaffold 接受"（算法）叠加
单 H100（系统）；它的 11.8× 是再加 SGLang + FP8（系统）。
把 11.8× 当作算法贡献是错的。

我们自己的 README 里就犯过这个错：早期报 `decode_speedup = 3.3×`
实际是记账口径，wall-clock 只有 1.26×。已在 Limitations 里注明。

---

## 2. 复现之前必须先测的四件事

### 2.1 基线是不是"官方默认精度"

这是本轮最大的发现。

OpenDriveVLA 官方评测用 **bf16**。我们的脚本沿用了 `model.float()`（fp32），
因为 UniAD 的 `modulated_deformable_im2col_gpu` **没有 bf16 kernel**：

```
RuntimeError: "modulated_deformable_im2col_gpu" not implemented for 'BFloat16'
```

于是我们一直在 fp32 下测。后果：

| 配置 | 1 token | 620 token prefill |
|---|---|---|
| eager fp32 | **16.68 ms** | 26.41 ms |
| bf16 decoder | 14.76 ms | 16.15 ms |
| compiled bf16 | **1.68 ms** | 8.94 ms |

fp32 的 decode 是 bf16 的 1.13 倍慢，而 **compile 后快 9.9 倍**。
在 fp32 eager 下测出来的"算法加速比"，会被这个固定开销大幅稀释。

**教训**：报加速比之前，先确认基线精度与原论文一致，并测一遍
"同一个算法在更快的栈上"是多少。

### 2.2 decode 到底受什么限制

一个 0.5B 模型解 1 个 token 花 16.7 ms 是不合理的。拆开看：

| 组件 | 耗时 |
|---|---|
| 24 层 decoder | 16.81 ms |
| lm_head（151682 词表） | 0.75 ms |
| embed | 0.02 ms |

24 层 = 16.8 ms ≈ 每层 0.7 ms，而 620 token 的 prefill 总共才 26 ms。
**这说明是 kernel 启动开销主导，不是算力。**
所以能拿到的最大加速在系统层（融合 kernel、CUDA Graph），
而不是"少解码几个 token"。

我们用这个判断否掉了一整类方法（见 §4）。

### 2.3 模型是否有隐藏的持久状态

OpenDriveVLA 的 UniAD 视觉塔有 `prev_frame_info`（`prev_bev` / `prev_pos` /
`scene_token`）和 `track_base.max_obj_id`，**跨 forward 保留**，只在一个场景
切换时清空。按索引取样（stride、非连续列表、或者跑完 `generate()` 再 prefill）
都会让它看到错误的 BEV：

| 场景 | 观测 |
|---|---|
| 同一帧前后两次 prefill（中间跑过一次 `generate()`） | logits 最大差 **0.647**，token 链第 5–20 位分叉 |
| 同一帧连续三次视觉塔（每次都 reset 状态） | track 数 **30 / 31 / 30** |

**结论：不可复现。** 最终协议改成**每帧视觉塔只跑一次，特征通过 `uniad_pth`
传给所有方法**。

**教训**：任何"有记忆"的模块（BEV、memory bank、tracker、KV cache）
都会让"逐帧独立评测"这个假设失效。复现第一步就该查这个。

### 2.4 验证路径是否与官方解码器一致

OpenDriveVLA 的 `generation_config.json` 里 `repetition_penalty = 1.1`。
HF 的 `generate()` **即使在 `do_sample=False` 下也会施加它**，
而我们的推测解码 verify 路径用的是裸 argmax。

后果：与 `generate()` 的 token 一致率只有 **0.13**（10/10 帧都分叉，
最早第 4 个 token）。更隐蔽的是，`n_accept = 0` 的帧（本该等价于原样退回
teacher）ADE 仍比 AR 差 0.15 m。

修复：`TeacherDecoder`（`trajspec/opendrive_live.py`）。
顺带发现 `eos_token_id` 是 **list**（`[151645, 151643]`），
原来的 `tid == eos` 恒为 False。

**教训**：`generation_config.json` 是一份隐藏的"解码约定"。
复现任何推测解码/并行解码工作前，先把它打印出来逐项核对。

---

## 3. 测量的常见错误

参考 [On Evaluating Performance of LLM Inference Systems](https://arxiv.org/html/2507.09019)、
[Lossless but Not Free](https://arxiv.org/html/2607.17283)、
[SPEED-Bench](https://arxiv.org/html/2604.09557) 的结论，结合我们的实测：

| 错误 | 后果 | 我们的处理 |
|---|---|---|
| 不做 warmup | 首次推理含 kernel 编译，慢 10–100× | 每方法 `--warmup 2~3` 帧 |
| 不 `torch.cuda.synchronize()` | 只测到 launch 时间，严重低估 | 所有计时函数前后同步 |
| 只报均值 | 双峰分布被掩盖 | 同时报 p50 / p90 |
| 单样本 | 无法看出方差 | n=24–40 |
| 用随机 token 当 prompt | SD 的接受率完全失真 | 用真实 nuScenes 帧 |
| 只报"每步加速" | 掩盖 prefill / tail 开销 | 报 E2E wall clock |

**双峰分布是最容易被审稿人抓的点。** 我们的方法：

| 接受路点 | 帧数 | E2E |
|---|---|---|
| 6（满） | 16 (53%) | **472 ms** |
| 1 | 6–7 | ~1460 ms |
| 0（全拒） | 2–5 | **1675 ms（比 AR 还慢）** |

"平均 1.75×"背后是"一半帧 3.5×，六分之一帧倒退"。
只报均值会被认定为误导。

---

## 4. 判断"这个加速能不能迁移到我的模型"的检查表

复现别人的加速方法前，先回答这五个问题。任一为否，加速比很可能归零。

| 检查项 | 为什么 | 我们的实例 |
|---|---|---|
| **① 我的瓶颈是不是它攻击的部位？** | 剪枝论文攻 prefill；我们 prefill 只占 1.8% | oracle 全删视觉 token = **1.002×** |
| **② 我的 token 结构支持它的技巧吗？** | Scaffold 需要骨架 token 占比高 | 我们是逐字符 tokenize，骨架仅 5–6/63 = **8%**；Fast-dDrive 的 JSON 是 40%+ |
| **③ 我的模型有"冗余层"吗？** | DeeAD 依赖深层只做精修 | 0.5B/24 层：depth 8–20 输出**完全相同**且非法（gap 6.693），只有满 24 层可用 |
| **④ decode 是算力受限还是带宽受限？** | 决定剪 KV 有没有用 | 权重受限：prompt 620→200 token，单步 23.39→23.48 ms |
| **⑤ 有没有跨帧状态？** | 决定 KV 复用能否生效 | 有（UniAD `prev_bev`），且逐帧不可复现，需改成单帧协议 |

---

## 5. 我们四个方法的复现结果（作为反面案例）

全部 training-free、无需重训，都在冻结的 OpenDriveVLA-0.5B 上跑：

| 方法 | 原文报告 | 我们复现 | 归零原因（对应 §4） |
|---|---|---|---|
| FastV / FastDriveVLA | prefill 3.7–6.4× | **0.97×** | ① 它们 baseline 3249 视觉 token，我们 148 |
| DeeAD | −29% 延迟 | **0.89×**（更慢） | ③ 0.5B 无冗余层；且手写截断不如原生 generate |
| Scaffold accept | 4.1× | **1.03×** | ② 骨架仅 8%，纯数字数组 |
| Pruning oracle | （未报告上界） | **1.01×** | ①④ decode 权重受限 |

详细的逐帧数据、深度扫描、探针验证见 `survey/FOUR_ACCEL_REPRO.md`。

---

## 6. 复现一个加速论文的标准流程

```
1. 打印 generation_config.json           # 解码约定（§2.4）
2. 测阶段耗时 profile                    # 瓶颈在哪（§2.2）
3. 检查模块持久状态                      # 可复现性（§2.3）
4. 确认基线精度/栈与原论文一致            # 公平性（§2.1）
5. 复现"同一个算法在更快栈上"的收益        # 分离算法与系统（§1）
6. warmup + sync + 多次 + 报分布          # 测量（§3）
7. 用 §4 检查表预判能否迁移
```

对应脚本：

| 步骤 | 脚本 |
|---|---|
| 阶段 profile | `scripts/profile_opendrive_stages.py` |
| 持久状态 / 解码约定诊断 | `scripts/diag_verify_drift.py` |
| dtype / 系统瓶颈 | `scripts/bench_decode_micro.py`、`scripts/bench_mixed_precision.py` |
| 统一对比 | `scripts/run_four_accelerators.py`、`scripts/run_unified_accel_table.py` |
| 深度扫描等超参 | `scripts/sweep_deead_depth.py` |

---

## 7. 一句话总结

> **加速比不是一个方法的属性，而是"方法 × 模型形态 × 瓶颈位置 × 测量栈"的联合属性。**

所以复现加速论文时，最有价值的问题不是"我能不能跑出它的数字"，
而是"**它的收益绑定在哪个前提上，我的模型满足这个前提吗**"。
我们四个方法全部归零，恰恰提供了四条清晰的前提边界，这比强行调参
凑出一个好看的数字更有写进论文的价值。

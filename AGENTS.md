# AGENTS.md

## 项目概述

ZO-vLLM: 在vLLM上实现LOZO（零阶优化）的LoRA适配器验证框架

## 环境配置

- Python: 3.12 (`.venv`)
- GPU: 仅使用GPU 5
- 模型: `facebook/opt-2.7b` (fp16训练，不支持bf16)
- vLLM: vllm-ZO子模块 (`third_party/vllm/`)
- 包管理: uv
- **LOZO baseline**: 独立虚拟环境 (`third_party/LOZO/large_models/.venv`)

## 关键配置

```bash
# 环境变量
VLLM_BATCH_INVARIANT=1          # 批量不变性，确保跨batch size结果一致
CUDA_VISIBLE_DEVICES=5          # 仅使用GPU 5

# vLLM引擎参数
gpu_memory_utilization=0.5      # GPU内存利用率
max_lora_rank=16                # LoRA最大rank

# LOZO参数
eps=1e-3                        # 扰动步长（论文对齐）
rank=8 或 16                    # LoRA rank
U, V ~ N(0,1)                   # 随机矩阵分布
```

## 构建命令

```bash
# 激活环境
source .venv/bin/activate

# 构建vLLM
MAX_JOBS=16 NVCC_THREADS=4 VLLM_TARGET_DEVICE=cuda pip install -e third_party/vllm --no-build-isolation
```

## vLLM批处理优化

**重要经验**：利用vLLM的批量处理特性可以显著减少调度开销。预先准备所有adapter，一次性发送请求让vLLM自己调度。

## Phase 1 结论

**vLLM fake-LoRA 路径在真实 LOZO 多层扰动场景下完全可靠。**

所有 sign mismatch 都发生在低信号区（|delta_lozo| < 0.005）。高信号区达到 100% sign match。

| 区域 | 样本数 | 占比 | Sign Match |
|------|--------|------|------------|
| High-signal (\|delta\| >= 0.005) | 2020 | 78.9% | **100.0%** |
| Low-signal (\|delta\| < 0.005) | 540 | 21.1% | 69.4% |

详见 `phase1_results.md`

## 注意事项

1. **不要使用bf16**：OPT-2.7B是fp16训练的，bf16会损失精度（尾数7位 vs fp16的10位）
2. **VLLM_BATCH_INVARIANT=1**：必须设置，否则跨batch size结果不一致
3. **多模块adapter格式**：vLLM需要多模块adapter格式才能正确加载
4. **PEFT lora_alpha = r**：scaling factor = 1

## 文件结构

```
scripts/
├── persistent_test.py      # 主测试脚本（批量处理版本）
├── phase1_verify.py        # Phase 1验证
├── phase1_official.py      # Phase 1官方对齐验证
├── batch_test.py           # 批量测试CLI版本
└── test_batch_invariance.py # 批量不变性测试

phase1_results/
└── phase1_raw_*.json       # Phase 1原始数据

phase2/
├── memory_lora_loader.py    # 内存LoRA Mock框架
├── lozo_controller.py       # LOZO控制器（master weights, U/V采样, V cache）
├── temp_lora_runtime.py     # 临时LoRA slots（plus/minus扰动）
├── vllm_scorer.py           # vLLM loss计算（prompt_logprobs，返回avg）
├── weight_sync.py           # 权重同步到vLLM（packed qkv处理）
├── module_map.py            # HF/vLLM/LoRA模块名映射
├── IMPLEMENTATION_NOTES.md  # 实现简化说明（1D/embedding跳过）
├── run_baseline_helper.py   # LOZO baseline helper（subprocess调用）
├── test_step_a.py           # Step A测试（in-memory LoRA scoring）
├── test_step_b.py           # Step B测试（weight sync）
├── test_step_c.py           # Step C测试（LOZO closure）
├── test_training_loop.py    # 完整训练loop测试
├── test_gradient_alignment.py        # 梯度对齐基础测试
├── test_gradient_alignment_detail.py # 详细对比（U/V + loss + c）
├── test_gradient_alignment_vllm.py   # 使用baseline U/V对比
├── test_multi_step_alignment.py      # 多步trajectory对齐
├── test_lozo_baseline_alignment.py   # 单步完整对比
├── test_real_lozo_baseline_side_by_side.py # 并行运行LOZO
└── test_memory_lora_*.py    # 内存LoRA测试（Phase 2 Milestone 1-2）

logs/
└── *.log                   # 运行日志

results/
└── *.json                  # 结果文件
└── baseline_gradient_data.json  # baseline U/V + loss数据
└── baseline_trajectory.json      # baseline多步trajectory
└── vllm_trajectory.json          # vLLM多步trajectory
```

## Phase 2 Milestone 1-2 完成 ✅

**实现方案**：Mock safetensors 文件读取，让 vLLM 从内存加载 LoRA

### Mock 覆盖点

| 函数 | Mock逻辑 |
|------|----------|
| `safetensors.safe_open()` | 返回 `FakeSafeFile(memory_tensors)` |
| `os.path.isfile()` | `/memory_lora_cpu/*` 返回 True |
| `os.path.isabs()` | `/memory_lora_cpu/*` 返回 True |
| `os.path.exists()` | `/memory_lora_cpu/*` 返回 True |
| `builtins.open()` | adapter_config.json 返回 StringIO(JSON) |

### API

```python
from phase2.memory_lora_loader import register_memory_lora_cpu
from vllm.lora.request import LoRARequest

# 注册内存LoRA
path = register_memory_lora_cpu(lora_id, config, tensors)

# 使用
llm.generate(prompts, lora_request=LoRARequest("name", lora_id, path))
```

### 测试结果

| 测试 | 结果 | 关键数据 |
|------|------|----------|
| 结果对齐 | ✅ PASS | Token IDs 100%一致 |
| 速度对比 | ✅ PASS | 注册速度 **42-189x 更快** |
| 多LoRA并发 | ✅ PASS | 顺序/batch处理正确 |

### 性能数据

| LoRA数量 | 文件写入 | 内存注册 | 速度up |
|---------|---------|---------|--------|
| 1 | 0.004s | 0.00008s | **42x** |
| 10 | 0.04s | 0.0002s | **189x** |
| 50 | 0.2s | 0.001s | **143x** |

### 注意事项

1. **GPU版本暂缓**：CPU版本已足够快（LoRA tensor <1MB，拷贝开销极小）
2. **不需要LRU**：LOZO训练每次只用一个扰动LoRA，内存占用小
3. **类型兼容**：Mock函数需处理 `str` 和 `pathlib.Path` 类型
4. **context manager**：`FakeSafeFile` 必须实现 `__enter__`/`__exit__`

## 下一步

- [x] Phase 2 Milestone 1: 内存LoRA Mock框架（CPU版本）
- [x] Phase 2 Milestone 2: 对齐验证 + 速度对比 + 多LoRA测试
- [x] Phase 2 Milestone 3: LOZO训练loop实现
- [ ] **Phase 2 Milestone 4: 梯度对齐验证**（当前阻塞）
  - loss_base 已对齐 ✓
  - c 值不对齐 ❌（sign和magnitude都不同）
- [ ] Sample-level batch invariance 验证
- [ ] 记录正式结果表格用于论文

## Phase 2 Milestone 3 完成 ✅

**完整 LOZO 训练 loop 实现**：

### 核心组件

| 组件 | 功能 | 关键实现 |
|------|------|----------|
| `LOZOController` | Master weights + U/V采样 + V cache | 仅管理 Linear 2D weights，过滤 embeddings/1D params |
| `TempLoRARuntime` | In-memory LoRA slots | plus/minus扰动，PEFT格式 |
| `VLLMScorer` | Loss计算 | prompt_logprobs |
| `WeightSync` | 权重同步 | unwrap_lora_module + packed qkv_proj |

### Weight Sync 关键逻辑

```python
# LoRA wrapper -> base_layer
def unwrap_lora_module(module):
    if hasattr(module, "base_layer"):
        return module.base_layer
    return module

# qkv_proj packed weight: [3*hidden_size, hidden_size]
# Slice: q[0:H], k[H:2H], v[2H:3H]
packed_weight = param.data.clone()
packed_weight[0:hidden_size, :] = tensor_q  # q_proj
packed_weight[hidden_size:2*hidden_size, :] = tensor_k  # k_proj
packed_weight[2*hidden_size:3*hidden_size, :] = tensor_v  # v_proj
param.data.copy_(packed_weight)
```

### 测试结果

```
Initial loss: 3.6736
Final loss: 3.6734
Loss change: -0.0002 (10 steps)
✅ Training loop test PASSED!
```

### 当前限制

见 `phase2/IMPLEMENTATION_NOTES.md`：
- **1D params 跳过**：bias, layer_norm (LoRA不支持)
- **Embeddings 跳过**：embed_tokens, embed_positions (需要修改 vLLM 源码)

### 环境要求

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0  # 单进程模式（mock需要）
VLLM_ALLOW_INSECURE_SERIALIZATION=1
VLLM_BATCH_INVARIANT=1
CUDA_VISIBLE_DEVICES=5
```

## Phase 2 Milestone 4: 梯度对齐验证

### 测试脚本

| 文件 | 用途 | 关键对比 |
|------|------|----------|
| `test_gradient_alignment.py` | 单步验证 | perturbation + batch invariance + c |
| `test_gradient_alignment_detail.py` | 详细对齐测试 | U/V + loss + c |
| `test_gradient_alignment_vllm.py` | 使用baseline U/V | 读取json对比 |
| `test_multi_step_alignment.py` | 多步对齐 | 5步trajectory |
| `test_lozo_baseline_alignment.py` | 单步完整对比 | baseline vs vLLM |
| `test_real_lozo_baseline_side_by_side.py` | 并行运行 | subprocess调用LOZO |

### 当前进展（5步trajectory）

| Step | Seed | loss_base | c值差异 |
|------|------|-----------|---------|
| 0 | 534895718 | ✓ 对齐 | **不对齐** |
| 1 | 199900595 | ✓ 对齐 | **不对齐** |
| 2-4 | ... | ✓ 对齐 | **不对齐** |

**关键发现**：
- `loss_base` 对齐 ✓（weight sync正确）
- `c = (loss_plus - loss_minus) / (2*eps)` 不对齐 ❌

### Loss计算方式

**统一使用 avg（不是 sum）**：
- vLLM `compute_nll_from_prompt_logprobs`：返回 avg per token
- HF `compute_loss`：返回 avg per token (默认 `reduction="mean"`)
- LOZO baseline：调用 HF compute_loss，使用 avg

### 可能原因

1. **U/V采样对齐**：random seed相同但采样顺序/设备可能不同
2. **扰动方向**：plus/minus定义可能反向
3. **dtype精度**：CPU fp32 vs GPU fp16

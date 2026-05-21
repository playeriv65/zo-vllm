# AGENTS.md

## 项目概述

ZO-vLLM: 在vLLM上实现LOZO（零阶优化）的LoRA适配器验证框架

## 环境配置

- Python: 3.12 (`.venv`)
- GPU: 仅使用GPU 5
- 模型: `facebook/opt-2.7b` (fp16训练，不支持bf16)
- vLLM: vllm-ZO子模块 (`third_party/vllm/`)
- 包管理: uv

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

logs/
└── *.log                   # 运行日志

results/
└── *.json                  # 结果文件
```

## 下一步

- [ ] Phase 2: 实现内存中的临时LoRA slots（无文件I/O）
- [ ] Sample-level batch invariance 验证
- [ ] 记录正式结果表格用于论文

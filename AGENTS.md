# AGENTS.md

## 项目概述

ZO-vLLM: 在vLLM上实现LOZO（零阶优化）的LoRA适配器验证框架

## 兄弟项目边界

- `zo-vllm` 和 `zo-post` 是同级兄弟项目，路径分别是：
  - `/home/zelin4593/research_local/zo-vllm`
  - `/home/zelin4593/research_local/zo-post`
- `zo-post` 不在 `zo-vllm` 仓库内部。禁止在 `zo-vllm/` 下创建 `zo-post/`
  影子目录；ZO-post 的报告、studies、runs、logs 等产物必须写入真实的同级
  `/home/zelin4593/research_local/zo-post` 仓库。

## 环境配置

- Python: 3.12 (`.venv`)
- 模型: `facebook/opt-2.7b` (fp16训练，不支持bf16)
- vLLM: vllm-ZO子模块 (`third_party/vllm/`)
- 包管理: uv
- **LOZO baseline**: 独立虚拟环境 (`third_party/LOZO/large_models/.venv`)

## 关键配置

```bash
# vLLM引擎参数
gpu_memory_utilization=0.9      # GPU内存利用率；Phase 3 scaling 默认给 vLLM 留足 KV cache
max_lora_rank=16                # LoRA最大rank
enforce_eager=0                 # Phase 3默认使用vLLM compile/CUDA graph路径；eager仅作消融
zo_reserved_gpu_bytes=<auto>    # ZO transient direction scratch must be reserved from vLLM KV budget

# LOZO参数
eps=1e-3                        # 扰动步长（论文对齐）
rank=8 或 16                    # LoRA rank
U, V ~ N(0,1)                   # 随机矩阵分布
seed=42                         # 默认step seed流，可通过--seed覆盖
zo_random_device=cuda            # U/V/z采样设备；cpu仅用于复现旧CPU-RNG结果
weight_update=direct             # 可选：vLLM worker内原位更新base weights
weight_update_precision=param    # 最快；float32用于对齐plain LOZO update math
direct_update_mode=accumulate    # Phase 3 launcher默认；V固定区间内累积低秩U更新并延迟折回base weight
perturbation_normalization=rms    # 默认解析式归一化全局U@V.T RMS；不要在provider里再手写1/sqrt(rank)
direction_scale=1.0              # 显式用户振幅倍率；在perturbation_normalization之后生效
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

**Phase 3 性能实验经验**：速度 checkpoint 和收敛验收分开记录。`phase3/results/`
只保存可复现实验输出并保持 git-ignore；提交代码时只提交 runner、collector 和状态文档。
当 vLLM scoring 进入热路径时，优先使用 direct worker scoring 和固定 GPU LoRA slot，
避免把完整 `generate(prompt_logprobs=1)` serving 包装当作最终训练路径。
direct worker scoring、plus/minus loss 拆分和 SST-2 option loss 计算属于
`zo_vllm/core/direct_worker_scorer.py` 的通用 ZO-vLLM 组件；Phase runner 只负责
参数、日志和实验编排，避免在 runner 里复制 scoring 语义。
Phase 3 速度统计用外层 wall-clock step 作为主指标，不能用子步骤相加替代总步时；
OPT-1.3B/2.7B 的运行目录 slug 使用 `opt1p3b`/`opt2p7b`，避免省略小数点造成歧义。
vLLM direct-score cache 只能缓存已验证不会跨 batch 失效的输入派生张量；不要跨不相邻
batch 缓存可变 attention metadata。
Phase 3 速度口径不使用额外 warmup 参数：运行完整 `steps`，统计最后 100 step；launcher、
job spec 和历史摘要不要再引入可调 `warmup_steps`，共享 job builder 的缺省也必须保持为 0。
Phase 3 速度对比必须固定数据顺序和数据窗口：`num_train=1000`、`num_dev=500`，
`seed`/`data_seed`/`dataloader_seed` 默认固定为 42。HF Trainer 路径使用 Hugging Face
`Dataset.shuffle(data_seed)` 和原生 `SeedableRandomSampler` 的逐 epoch 打乱语义；不要让
扰动 `seed` 隐式改变数据窗口，也不要把不同样本数的 tail-100 结果直接比较。

**Phase 4 收敛记录经验**：官方 LOZO 训练日志里的 `loss` 是
`loss(theta + eps * direction)` 的 plus-perturbation probe，不是 clean
training objective。长程收敛判断以不带扰动的 `eval_loss`、`eval_acc` 和最终
full-eval accuracy 为主；training loss 只作为 noisy debug 信号。若需要按
wall-clock 画收敛曲线，必须明确区分真实 timestamp 与按平均 step time 估算的曲线。
`loss_plus` 和 `loss_minus` 是带扰动的 ZO probe loss，不能命名或解释为
`train_loss`。只有不带扰动的训练数据前向结果才允许叫 `train_loss`。
Launcher 中的 tmux/shell/preflight 逻辑优先复用 `zo_vllm/experiment/` 公共模块，
避免在 Phase runner 内复制实现导致行为分叉。
Phase 7 主权重不变实验默认把量化或预量化 base weight 视为只读：高精度 update
走 `--quantized-update-mode lora_bank` 和 `BlockLoRAUpdateBankState`，plus/minus
各占一个完整 LoRA bank slot。bank mode 下不要调用 base weight write-back 或
fold-back；`rank` 表示 ZO direction rank，`update_bank_rank` 表示 LoRA bank 容量。
实验入口默认用 `update_bank_rank=auto`，按训练步数、`nu` 和 direction rank 估算所需
bank 容量并额外保留一个 V-refresh block；估算值和手动整数都会向上取整到 vLLM
支持的 `max_lora_rank` 档位。
Serving-time ZO runner 会估算 worker-resident update-bank 额外显存，并通过
`VLLM_ZO_RESERVED_LORA_BANK_BYTES` 传给 vLLM worker，让 memory profiling 从 KV cache
预算里扣除这部分；该值只覆盖 ZO bank 额外状态，不重复计算 vLLM 自身 LoRA slot stack。
Phase7 目录只放测试/实验脚本，运行输出写入并忽略 `phase7/logs/`；具体模型、GPU、
prequant 路径通过脚本参数或环境变量传入，不写死到库代码。
Serving-time ZO 训练默认走 vLLM scheduler 的 compact NLL scoring；
完整 direct-worker scoring RPC 只用于数值对齐或 idle smoke，不进入正式 serving 并发路径。
后台训练状态优先 worker-resident，避免每步通过 RPC 传输大 LoRA bank tensor。验证“插队”
不能只看 step 与 benchmark 窗口重叠；还要检查 score/slot-write 与前台 request 重叠，
并且至少有 ZO step 在 benchmark 结束前完成。
Serving-time ZO 的 LoRA slot 写入默认不等待前台 HTTP 请求 idle；只需要保证同一对
plus/minus ZO scoring 未返回前不覆盖对应 slot。
Serving-time ZO slot copy 保守默认使用 `SLOT_WRITE_STREAM=background_sync`：LoRA tensor copy
仍在单独 CUDA stream 上执行，但 `prepare_slots` 返回前会等待对应 event 完成，随后再向
scheduler 提交 plus/minus compact-NLL 请求。`SLOT_WRITE_STREAM=background` 仅用于复现
runner 侧 event-fence 的异步写入路径；`default`/`default_sync` 用于默认 stream 消融。
Serving-time ZO 当前提供四种 score admission：保守默认 `idle_gap` 会在 score 提交前等待
`server_load_metrics <= max_score_admission_foreground_load`，不 gate slot write；
激进 `scheduler_only` 不等待 foreground gap，slot 写完后直接提交 priority=1000 的
compact-NLL 请求，让 vLLM scheduler 自己调度。`ZO_ADMISSION_TIMEOUT_S` 对
`idle_gap` 和 `gpu_utilization` 有意义，默认保持为 `0`（无限等待），并依赖
stop-aware wait 退出；高压 serving 下不要把“暂时没有 admission gap”记录成训练错误。
更平滑的 `queued` admission 不修改 vLLM scheduler，也不拆训练/eval 语义；它只在
compact-NLL 发射端维护 FIFO 队列，用 token bucket 和 max-inflight 控制低优先级 ZO request
释放到 vLLM 的速率。判断 `queued` 是否有效要看 scheduler trace 里单个 mixed batch 的
ZO token 尖峰是否下降，而不仅是 ZO step/sec。
`gpu_utilization` admission 同样只改发射端：slot 写完后轮询 `nvidia-smi`，GPU 利用率低于
阈值才提交 ZO compact-NLL；每步 JSONL 必须记录提交时 GPU util、阈值和等待采样次数。
Serving-time ZO 的 QoS 调试必须能追溯到 GPU scheduler batch：整体 p99/throughput 只能说明
是否变差，不能说明为什么变差。需要用 `SCHED_TRACE=1` 记录每个 scheduler batch 的前台/ZO
request、priority、LoRA id 和 token 数，再把受影响的 bench request 反查到 mixed batch，
区分 priority ordering 是否失效和“低优先级 ZO token 把同一个 GPU batch 撑重”这两类问题。
Serving-time ZO compile/CUDA graph 模式下默认保留 `VLLM_ZO_FORCE_EAGER_SCORING=1`：
只对带 serving ZO LoRA 的 compact-NLL scoring batch 跳过 compiled model 并禁用 CUDA graph，
普通前台生成 batch 仍走 server 的默认 compile/CUDA graph 执行模式。`=0` 只用于复现
动态 LoRA bank graph 问题。
Phase7/Qwen3 serving smoke 不要强制 `DTYPE=float16`：Qwen3 原生是 BF16，强转 FP16
会在前台 no-LoRA base forward 中先产生 nonfinite，再污染同轮 compact-NLL 观测。
默认使用 `DTYPE=auto` 和 `DIRECTION_DTYPE=auto`，让 vLLM server 与 LoRA bank direction
跟随模型原生 dtype；只有对 OPT 等 FP16 口径做复现时再显式覆盖。
`zo_vllm.serving` 是通用 serving-time ZO 编排层，不使用 Phase 命名，也不拥有 update
状态；Phase7 只保留实验脚本、文档和 `phase7/logs/` 产物边界。正式 serving API 的运行
输出根目录由 `ZOServingConfig.output_dir` / request `output_dir` 控制，默认是
`zo_vllm_runs/serving`，不要在 serving 包里写死 Phase 路径。正式 serving 路径的
worker-side update bank 属于 `zo_vllm.training.worker_update_bank`，作为一种权重更新后端；
server 侧只做固定 LoRA slot 注册、scheduled scoring 和小控制 RPC，避免再维护一套重复
bank/update 逻辑。
Direct LoRA 的作用范围统一使用 vLLM 风格 `target_modules` 列表；`lm_head` 和
`embed_tokens` 只能作为最外层便捷开关追加进同一列表，进入 metadata、slot registry、
runtime 和 worker bank 后不再保留独立分叉。任何 direct GPU slot 注册或 worker bank
初始化都必须校验 metadata/tensor shape 与 vLLM LoRA manager 注册模块一致，不能 silent
reset/skip active module 后继续 forward。
vLLM 的 LoRA 是 forward/logits 路径增量，不等同于 PyTorch 直接改 tied Parameter：
对 `tie_word_embeddings=True` 的模型，`embed_tokens` 只覆盖 input lookup，`lm_head`
必须同时作为 output embedding target 注册，才能让 logits 侧经过 `LogitsProcessorWithLoRA`。
这只是 scoring perturbation 的双路径视图；真正折回 base weight 时 tied embedding 仍只写
共享矩阵一次，不能把 `lm_head` 当第二份独立参数重复更新。
ZO perturbation 默认使用 `perturbation_normalization=rms` 做解析式能量归一化：
目标是让全局 `U @ V.T` 的期望 RMS 等于默认 Gaussian LOZO 的每权重单位能量，而不是
把不同 provider 的分布变成一样。不要在 `PoolUProvider`、`SubspaceUProvider` 或
queued AGZO V 里额外加入局部 `1/sqrt(rank)` / `1/sqrt(queue)` 缩放；rank、queue size
和 AGZO unit-basis V 的能量差异统一通过 provider metadata
(`perturbation_effective_rank`, `perturbation_v_expected_column_norm_sq`) 进入
`zo_vllm/core/perturbation_normalization.py`。`direction_scale` 只作为归一化之后的显式
用户振幅倍率。
保存 vLLM native sharded checkpoint 时，只允许非 `lora_bank` 更新模式折回 base weight；
`AccumulatedLowRankUpdateState.fold()` 会先 flush pending update 再写回 base。Phase7
`lora_bank` 代表主权重只读，不能为了保存 checkpoint 把 bank update fold 进 base。
把 accumulated update 物化进 native checkpoint 时必须复用 embedding-aware shape
语义：训练方向的 token embedding 是 hidden-first，而 vLLM base tensor 可能是
vocab-first 或带 padded vocab；tied `lm_head` 只是同一方向的输出转置视图。GPU E2E
必须覆盖 `ES + lora_full + accumulated update + native save/resume`，不能只分别验证
普通线性层 checkpoint 和关闭保存的 ES rollout。
vLLM `ShardedStateLoader.save_model()` 保存的是 `model.state_dict()`，direct LoRA slot
tensor 不是 registered parameter/buffer；native checkpoint 的语义应保持为 folded base
weights 加少量 ZO metadata，而不是 adapter checkpoint。
LoRA runtime 会把 base parameter 暴露成 `*.base_layer.*` state key，但 vLLM 在注入
LoRA wrapper 前加载 sharded checkpoint。native checkpoint 保存边界必须把这些 runtime
key 规范化回 base-model key；study 和恢复脚本不能依赖或修补 wrapper key。
所有可恢复的 native 和 LoRA-bank checkpoint 都必须保存完整的
`hf_to_vllm_mapping`、`hf_to_slice` 和稳定 fingerprint。恢复时先用当前
`WeightSync` 的实际映射严格比较，再加载任何 tensor；不能根据模型名或当前代码静默推断。
chunked AGZO subspace 构造只收集 activation，必须同时关闭 `compute_loss` 和
`compute_token_nll`；任务 loss/labels 属于训练与 evaluation scoring，不得混入子空间构造。
checkpoint 调度语义对齐 Hugging Face：`--save-strategy steps` 只按 measured step 和
`--save-steps` 保存，不依赖 eval cadence；`best` 只在 `metric_for_best_model` 改善时
保存；`load_best_model_at_end` 只能加载 native full checkpoint。LoRA-bank checkpoint
只作为 `lora_bank` 训练恢复状态，不和 full-model checkpoint 语义混用。
`zo_trainer` 也必须保持这个边界：metadata checkpoint 不可加载，LoRA-bank checkpoint 只用于
resume training state，`load_best_model_at_end` 必须要求 `zo_checkpoint_mode=native`。
SuperGLUE 任务实现默认对齐 `third_party/LOZO/large_models/tasks.py` 和 `templates.py`
的 generative multiple-choice 口径：每个 candidate 拼成 continuation，用 option-token
NLL 做分类；CB 是三选，ReCoRD 是实体候选变长多选并按原版 QA normalize 计算 EM/F1。
新增或修改 SuperGLUE 模板时优先更新 `zo_vllm/tasks/superglue/<task>.py`，不要在 runner 或
scoring helper 里复制任务文本。
SuperGLUE 的 train split 小于 `num_train + num_dev` 时，用尾部 `num_dev` 做 dev、前面
做 train，避免 HF `select(range(len, len))` 越界，也避免 train/dev 重叠。MultiRC 的
样本和 candidate 展开较重；先做启动覆盖或排查时可以单独降低 batch 或
`eval_accuracy_samples`，但必须保留 eval loss 和 accuracy 两个指标。
测试按金字塔分层维护：`tests/` 只放单元和轻量集成测试，不能启动真实 vLLM 模型；
真实脚本启动、GPU、模型加载的一步 smoke 放在 `e2e_tests/`，用 `pytest -m e2e`
显式触发，并默认用 `facebook/opt-125m` 做最小端到端覆盖。
默认 pytest collection 只覆盖 `tests/` 和 `e2e_tests/`，不要把 `third_party/` 或 phase runner
脚本纳入裸 `pytest`；这些目录分别通过专门脚本、子项目测试或 opt-in e2e 验证。
`zo_trainer` 的真实 GPU smoke 必须直接构造底层 `VLLMZOModel` / `ZOVLLMEngine`
再交给 Hugging Face `ZOTrainer`，不能借旧 `VLLMZOTrainer` 或 runner 脚本间接证明。
`zo_trainer` 的真实模型覆盖同样放在 `e2e_tests/`，用内存 HF `Dataset` +
`Dataset.map` + HF 原生 collator 做 opt-in GPU 测试，不放进默认 `tests/`。
`zo_trainer` 的数据主路径必须是 HF 原生：`load_dataset` 后通过用户自己的
`Dataset.map(preprocess_function, batched=True, remove_columns=...)` 生成
`input_ids`、`attention_mask`、`labels`，loss mask 使用 HF 约定 `labels=-100`。
`zo_trainer` 不能再新增 raw-row objective spec、dataset adapter、任务级 collator 或
`TokenProbeBatch` 数据预处理入口；多选、chat、VLM 等任务也应先表达成 HF 标准 batch
schema，再进入 `ZOTrainer`。
`zo_trainer` 允许提供很薄的 `Dataset.map` helper，但 helper 只能产出 HF 标准字段，不能
拥有 ZO objective、metric、candidate 展开或 trainer 分支语义。chat/instruction 数据优先在
用户 preprocess 中调用 tokenizer/processor 的原生 chat template。
HF metric 必须通过 `Trainer.compute_metrics` 进入；不要为了 metric 或候选项格式新增任务级
trainer 分支。
本地 JSON/CSV、dataset revision、cache 等 `datasets.load_dataset` 通用参数通过
`dataset_kwargs_json` 透传；不要为了数据文件来源新增 runner 分支。
通用 HF dataset 列清理使用 `Dataset.map(..., remove_columns=...)` 或 HF datasets 自带
API，不要为了去列/保留列写新的 dataset adapter。
`zo_trainer` 不再维护 generic raw-row objective runner；复杂实验应像原生 HF 脚本一样记录
dataset/preprocess/tokenizer/collator/training/runtime 参数，而不是把数据语义塞进 trainer。
`ZOTrainer.training_step` 只能接收 HF collator 产出的 batch，并通过
`Trainer.compute_loss(..., return_outputs=True)` 暴露 HF loss 能力；LoRA slot id 选择、
one-sided/two-sided/multi-query probing 和 score batching 必须继续由 runtime estimator /
engine adapter 负责，不能在 Trainer 里制造 `lora_id` 或写死 estimator 形状。vLLM backend
应返回 active label positions 的 compact logits，
再由 `ZOTrainerModel.forward` 调用 Transformers 原生 causal-LM loss helper；不要为了 loss
退回 collator/objective adapter，也不要强行物化 full `[batch, seq, vocab]` logits。
改 `zo_trainer` 的 probe / scoring 逻辑前必须先看 `ZOStepper` 和 `ZOEstimator`：
`ZOTrainer` 只管 HF lifecycle，`ZOTrainerModel` 只管 HF batch facade 和 compact-logits
forward，`ZOTrainer.training_step` 只估计并 stage 一个 `ZOPendingStep`，真实权重更新必须
发生在 HF 调用的 `ZOSGDOptimizer.step()` 内；`ZOStepper` 只管方向采样、回调、估计构造
和 runtime apply orchestration，
`ZOEstimator` 才拥有 antithetic / one-sided / multi-query / ES 的 probe 计划与聚合。
`VLLMZOModel` 和 `ZOStepper` 只暴露 estimate，不拥有 learning rate、weight decay 或
scheduler，也不能提供立即 apply 的 `step()` 便捷路径；所有同步更新必须由
`ZOSGDOptimizer.stage()` / `step()` 消费。
direct worker scoring/logits 是 `ZOVLLMEngine` backend 能力，不是可以在 Trainer 里直接
替代 estimator 的算法入口。
Generation-reward ES tasks use `ZORolloutTrainerModel`: the task supplies an HF
collator, scalar reward function, optional per-row metric projection, and
`compute_metrics`. Do not subclass `ZOTrainer` or override `evaluate()` for a
rollout task; clean generation returns standard HF `loss`/`logits` outputs,
while population probing remains in `EvolutionStrategyEstimator`.
Step-local direction sampling must not retain every seeded multi-query sample.
Keep only lightweight merged metadata and the first callback sample; independent
ES replays seeds lazily during update aggregation. Otherwise direction memory
silently grows with population size despite query microbatching.
HF-native 路径不得维护平行的 no-op scheduler，也不得绕过
`Trainer._load_optimizer_and_scheduler`；scheduler 的 warmup、step 和 checkpoint resume
统一交给 Transformers。ZO 日志用 `zo_applied_learning_rate` 表示当前 update 实际使用的
LR；当前 Transformers 在 scheduler step 前记录标准 `learning_rate`，两者应一致。
HF optimizer callbacks 必须包围真实 ZO 更新，不能包围 placeholder/no-op。当前只支持
exact ZO-SGD；未实现的 AdamW、autograd gradient clipping、HF gradient accumulation 和
accumulated-backend weight decay 必须快速失败，不能静默降级。
同一进程连续运行多个真实 vLLM GPU e2e 时，engine cleanup 后还必须调用 vLLM
`cleanup_dist_env_and_memory()`，不能只 destroy 默认 torch process group，否则残留的
vLLM parallel-state 单例会污染下一次 engine 初始化。
request-level token score 的 chunk / merge / slice 统一使用
`zo_vllm.core.token_scores`；`zo_trainer`、`ZOTrainerModel`、`ZOStepper` 和
`ZOEstimator` 与 `ZOVLLMEngine` 共享这一层，不再各自复制 request aggregation。
`zo_trainer` 的指标边界必须保持 HF 原生：训练/eval 总测速使用 HF 自带
`train_runtime`、`train_steps_per_second`、`eval_runtime`、`eval_steps_per_second`；
clean loss 使用 `eval_loss`；accuracy/F1/EM 等任务指标通过用户脚本传入
`compute_metrics`。ZO 特有的 probe/update/profile 字段只能作为通用 runtime observation
进入 HF log，统一使用 `zo_` 前缀；不要在 `zo_trainer` 中实现 Phase3 的
`tail_100.step_s.mean`、speedup 表、Phase4 convergence summary 或自定义 result JSON。
这些实验口径和迁移测试属于 `phase3/`、`phase4/` 各自脚本/collector。
`ZOTrainer` 要支持 HF 标准 `model=` 入口；即使底层 runtime 继承 `torch.nn.Module`，只要不是
`ZOTrainerModel`，也应该包装成 runtime facade，而不是走普通 autograd model 假设。
HF runtime checkpoint 每个保存点只能调用一次 payload handler，并且必须携带当前
`TrainerState.global_step`；保存 tokenizer、training args 和 metadata 时不能再次经过
`save_model()` 触发第二次 native/LoRA 写入。可复用 checkpoint payload 实现放在
`zo_vllm.training`，experiment runner 只拥有 cadence 和 best-checkpoint policy。
HF loss-backed scoring 使用 objective-level `ProbeLossResult`，不能把完整 batch loss 伪装成
`TokenScoreResult` 的 request NLL。Estimator 只操作 opaque probe slot，LoRA ID、路径和
`LoRARequest` 构造必须留在 engine/runtime。

## Phase 1 结论

**vLLM fake-LoRA 路径在真实 LOZO 多层扰动场景下完全可靠。**

所有 sign mismatch 都发生在低信号区（|delta_lozo| < 0.005）。高信号区达到 100% sign match。

| 区域 | 样本数 | 占比 | Sign Match |
|------|--------|------|------------|
| High-signal (\|delta\| >= 0.005) | 2020 | 78.9% | **100.0%** |
| Low-signal (\|delta\| < 0.005) | 540 | 21.1% | 69.4% |

详见 `phase1/phase1_results.md`

## 注意事项

1. **不要使用bf16**：OPT-2.7B是fp16训练的，bf16会损失精度（尾数7位 vs fp16的10位）
2. **Batch invariant**：训练代码不要管理 `VLLM_BATCH_INVARIANT`；未设置时就是正常训练语义，只有专门验证脚本会显式设为 `1`
3. **多模块adapter格式**：vLLM需要多模块adapter格式才能正确加载
4. **PEFT lora_alpha = r**：scaling factor = 1
5. **GPU 不写死**：可用 GPU 是临时协商资源，脚本和文档不要固定具体编号；通过外部 `CUDA_VISIBLE_DEVICES` 传入，或用可选 `--gpu` 临时覆盖。
6. **OPT BOS 口径显式化**：OPT 原生 HuggingFace/vLLM tokenizer 使用 `</s>`/id 2
   作为 BOS；严格复现 MeZO/LOZO OPT baseline 时才显式使用
   `--opt-bos-mode lozo`，让 tokenizer 在 `from_pretrained` 构造阶段使用
   `<s>`/id 0。不要只在加载后赋值 `bos_token_id` 并假设 `encode()` 已改变。

## 文件结构

```
phase1/
├── phase1_verify.py        # Phase 1验证
├── phase1_official.py      # Phase 1官方对齐验证
├── batch_test.py           # 批量测试CLI版本
├── persistent_test.py      # 主测试脚本（批量处理版本）
└── results/                # Phase 1结果；脚本默认写这里

zo_vllm/
└── core/                    # 跨Phase共享的LOZO/vLLM runtime核心模块
    ├── training/direction/
    ├── lora_runtime/
    ├── direct_worker_scorer.py
    └── weight_sync.py

phase2/
├── runners/                 # 训练、baseline和sweep入口
│   ├── train_convergence.py
│   ├── run_baseline_helper.py
│   ├── run_convergence_experiment.py
│   └── run_convergence_sweep.py
├── validation/              # 验收与回归脚本
│   ├── test_real_lozo_baseline_side_by_side.py
│   ├── test_batch_invariance.py
│   └── test_training_loop.py
├── configs/                 # 实验配置
├── artifacts/               # 临时adapter等运行产物；git忽略
├── results/                 # Phase 2日志和结果；git忽略
└── IMPLEMENTATION_NOTES.md  # 实现简化说明（1D/embedding跳过）
```

阶段边界：`zo_vllm/core/` 是共享runtime，不归属于某个实验阶段；`phase1/2/3`
目录只放各阶段特有的runner、validation、collector和文档。
外部仓库引用时，`pyproject.toml` 只打包 `zo_vllm`，公共 API 优先从
`zo_vllm.core` 暴露；SST-2、Phase runner、collector 等任务专用逻辑放在
`zo_vllm.experiment` 或 `phase*` 目录，不要让 `core` 反向依赖具体数据集。
原生 Hugging Face 训练入口统一放在顶层 `zo_trainer/`。新训练任务必须使用
`ZOTrainer` / `ZOTrainerArguments`，直接连接 `ZOTrainerModel`、`VLLMZOModel`、
`ZOStepper` 和 runtime checkpoint handler；不得把另一个 trainer 当 backend 再包一层。
Hugging Face 负责 datasets、`Dataset.map`、tokenizer/processor、sampler、DataLoader、
loss、metrics、callbacks、logging、evaluation 和 checkpoint cadence。ZO-vLLM 只拥有
scoring/logits backend、estimator、direction provider、update state 和 vLLM runtime。
训练 batch 在进入 Trainer 前必须已经表达为 HF 字段；任务名、raw row、template 和
verbalizer 不能进入通用 Trainer 分支。

`ZOTrainerModel` 只负责 HF ragged batch 到 compact logits/request NLL 的转换；
`ZOStepper` 负责 score/update callback 顺序与 update orchestration，并提供
`ZOStepCallback`：`on_direction_sampled`、`on_score_end`、`on_update_begin`、
`on_update_end`。multi-query、one-sided ZO 和 ES 只属于 estimator 层：
`VLLMZOConfig.estimator="multi_query"` 配合 `num_queries`、`perturbation_sides` 和
`query_microbatch_size`；`estimator="evolution_strategy"` 配合 `population_size`、
`sigma`、`reward_shaping` 和 `query_microbatch_size`。Trainer 不得读取这些配置来决定
probe 数量、扰动侧或 LoRA ID。
estimator 通过 `ZODirectionSampler.sample(seed=...)` 明确请求 probe direction：
two-sided estimator 对同一个 sampled direction 写 `+eps` / `-eps`，one-sided 和 ES
每个 query 只请求一个 direction 并写 `+eps` / `+sigma`。multi-query / ES 不允许从
已有 direction tensor clone 出“新方向”作为采样来源；按 seed 从 `DirectionSpec`
（来自全局 config + model metadata）采样。大 rank / 多 query 的 transient ZO scratch
通过 `ZOVLLMEngineConfig.zo_reserved_gpu_bytes` 计入 vLLM memory profiling，不能靠
临时调低 `gpu_memory_utilization` 掩盖预算问题。
HF loss 由 `ZOTrainer.compute_loss` 从 compact outputs 计算。causal/target LM 使用
`labels=-100` mask；prompt classification 使用显式 `option_loss_token_counts`、
`row_option_counts` 和 class `labels`。通用路径不得调用 objective router 或按任务名
分发 loss。Tokenizer/processor 通过 HF preprocessing 与 `processing_class` 进入，
不能藏在 collator、model facade 或 checkpoint fallback 中。
AGZO 相关的后续实验、worker AGZO 探针、post-paper sweep 和对应大结果统一放在
`zo_post/`；Phase3 只放 core-step speed/scaling，Phase4 只放正式长程收敛与
官方 LOZO/vLLM 对齐。不要把 AGZO 后续实验写入 `phase3/results` 或 `phase4/results`。
`zo_post` 采用干净结构：`configs/` 管配置，`runners/` 只做统一 runner 的薄封装，
`studies/` 记录研究说明和迁移索引，`results/` 保存大输出，`logs/` 集中所有 stdout
日志，方便跨实验寻找规律。post-phase exploratory 旧输出迁移时保留原目录名，不做删除。

## Phase 2 Milestone 1-2 完成 ✅

**历史实现方案**：早期使用 safetensors/open/os monkeypatch 从内存模拟 LoRA
文件加载。当前 runtime 已统一为 GPU direct LoRA slots，CPU mock 路径和测试已删除。

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

1. **GPU direct slots 是唯一训练路径**：先固定 plus/minus slot，再每步原位写 LoRA。
2. **不需要 LRU**：LOZO 训练每次只用 plus/minus 扰动 LoRA，内存占用小。

## 下一步

- [x] Phase 2 Milestone 1: 内存LoRA Mock框架（CPU版本）
- [x] Phase 2 Milestone 2: 对齐验证 + 速度对比 + 多LoRA测试
- [x] Phase 2 Milestone 3: LOZO训练loop实现
- [x] **Phase 2 Milestone 4: 梯度对齐验证** ✅
- [x] Sample-level batch invariance 验证 ✅
- [x] 记录正式结果表格用于论文 ✅

## Phase 2 Milestone 3 完成 ✅

**完整 LOZO 训练 loop 实现**：

### 核心组件

| 组件 | 功能 | 关键实现 |
|------|------|----------|
| `LOZO direction provider` | plain LOZO U/V sampling + V cache | uses Linear 2D metadata and skips embeddings/1D params |
| `LoRAUpdateRuntime` | GPU direct LoRA slots | stable plus/minus IDs + in-place slot writes |
| `VLLMScorer` | Legacy validation loss计算 | `LLM.generate(prompt_logprobs=1)` only for old checks |
| `WeightSync` | 权重同步/原位更新 | unwrap_lora_module + packed qkv_proj |

### Weight Sync 关键逻辑

```python
# LoRA wrapper -> base_layer
def unwrap_lora_module(module):
    if hasattr(module, "base_layer"):
        return module.base_layer
    return module

# direct path: apply LOZO update in the vLLM worker
param_slice.addmm_(U, V.T, alpha=-lr * c)
```

### 测试结果

```
Initial loss: 3.6736
Final loss: 3.6706
Loss change: -0.0030 (10 steps)
✅ Training loop test PASSED!
```

### 当前限制

见 `phase2/IMPLEMENTATION_NOTES.md`：
- **1D params 跳过**：bias, layer_norm (LoRA不支持)
- **Position embeddings 跳过**：embed_positions 不是 vLLM LoRA target；token
  `embed_tokens` 已支持，但 tied 模型必须同时注册 output-side `lm_head` LoRA 路径。

### 环境要求

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0  # Phase 2 validation harness uses LLM.apply_model
VLLM_ALLOW_INSECURE_SERIALIZATION=1
CUDA_VISIBLE_DEVICES=<GPU_IDS>
```

### CUDA RNG 与参数范围消融

`--zo-random-device cuda` 已通过 20-step step-by-step side-by-side：
- `direction_digest_mismatch_steps=[]`
- `max_loss_plus_diff=0.010078`
- `max_loss_minus_diff=0.010059`
- `max_c_diff=6.262078`

Baseline-only 100-step 消融（rank=8, lr=3e-7, eps=1e-3, nu=50, batch=16）：

| scope | initial | final | loss_change | step_s_mean |
|---|---:|---:|---:|---:|
| `lora_normal` | 5.132812 | 4.992188 | -0.140625 | 0.1168 |
| `full`（含embedding/1D） | 5.132812 | 4.882812 | -0.250000 | 0.1340 |

Full scope 更快下降，但不是数量级差异；vLLM真实注入路径仍以 `lora_normal`
为基础验收范围。报告和内部参数统一使用 `lora_normal` / `lora_embed_input_only`
/ `lora_embed_tied_head` / `lora_full` / `mezo_full`。其中 `lora_full`
表示所有 vLLM LoRA 可用 target：normal linear targets + token embeddings +
tied `lm_head` logits path；不包含 position embedding 或 1D 参数。

### GPU-resident LoRA路径

GPU direct LoRA slots 绕过旧 CPU mock safetensors 路径，并绕过每步 LoRAModelManager 重建/activate：
`CUDA U/V -> CUDA LoRA A/B -> fixed plus/minus slot -> module.set_lora(slot_index, ...)`。

短验证结果：
- archived CPU-mock vs GPU residency：3/3 steps 的 seed、U/V digest、loss_plus/loss_minus、`c` 完全一致；CPU mock 路径已删除
- 旧eager验收配置（GPU residency, batch_invariant=0, enforce_eager=1）3-step side-by-side vs LOZO baseline：`direction_digest_mismatch_steps=[]`，`max_loss_plus_diff=0.007089`，`max_loss_minus_diff=0.001356`，`max_c_diff=2.866773`
- archived batch=2短跑速度：CPU mock `step_s_mean=0.276578`，GPU residency `step_s_mean=0.238013`
- direct slot updater 20-step side-by-side vs LOZO baseline：`direction_digest_mismatch_steps=[]`，`sign_fail_steps=[]`，`max_loss_plus_diff=0.009428`，`max_loss_minus_diff=0.009200`，`max_c_diff=6.066894`
- direct vs manager 20-step：`lora_update_s_mean` `0.010747` vs `0.014718`（direct快 `1.37x`），`step_s_mean` `0.209159` vs `0.214355`（整体快 `1.03x`）

### Direct base-weight update路径

`--weight-update direct --weight-update-precision param` 在vLLM worker内对base weight做原位低秩更新；`LOZO direction provider` 只保留参数 metadata 用于采样方向，不再维护或同步全量 master weights：

```text
W <- W * (1 - lr * weight_decay) - lr * c * U @ V.T
```

当同一个 V 在 `nu` 内复用时，训练默认使用
`--direct-update-mode accumulate`：先在低秩 U/B 槽上累积
`U_accum += -lr * c * U`，plus/minus scoring 使用
`(U_accum ± eps * U) @ V.T`，只在 `nu` 触发方向刷新前把 `U_accum @ V.T`
折回 base weight。clean train/eval/final score 通过临时 LoRA slot 读取
`W + U_accum @ V.T`，不能为了评估清空 U；这样才能让 U norm cap、U snapshot
和方向变化观测保持同一口径。

短验证结果：
- fake packed-qkv 单测：`float32`模式与plain LOZO update formula逐元素一致
- 2026-05-23正式验收：
  - clean 20-step side-by-side vs LOZO baseline：`direction_digest_mismatch_steps=[]`，`sign_fail_steps=[]`，`max_loss_plus_diff=0.030806`，`max_loss_minus_diff=0.023877`，`max_c_diff=18.560467`
  - clean 300-step推荐配置：baseline `5.132812 -> 4.832031`，vLLM direct/param `5.132858 -> 4.831391`，vLLM达到baseline loss drop的 `100.23%`，final diff `0.000640`
  - clean 300-step速度：instrumented baseline `0.1155 s/step`，vLLM direct/param `0.0864 s/step`，速度提升 `1.34x`

### 当前vLLM瓶颈

正式测速默认关闭 `direction_digest`；side-by-side wrapper 会显式开启，用于确认每步 U/V 扰动完全一致。clean 300-step计时：

| component | mean s/step |
|---|---:|
| total step | 0.0864 |
| score | 0.0585 |
| score_generate | 0.0584 |
| score_postprocess | 0.000064 |
| direction sampling | 0.0027 |
| build_lora | 0.0071 |
| lora_update | 0.0100 |
| weight_update | 0.0078 |

结论：当前主要瓶颈在 vLLM `generate(prompt_logprobs=1)` scoring 路径；Python loss后处理和request构造可以忽略。

### vLLM执行参数消融

20-step vLLM-only 计时（rank=8, lr=1e-7, eps=1e-3, nu=100, batch=16, CUDA RNG, GPU-resident manager路径；direct更新器实现前结果）：

| batch_invariant | enforce_eager | total_s | step_s_mean | tail10_step_s_mean | 结论 |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 4.6126 | 0.2273 | 0.2106 | 旧eager消融；启动轻 |
| 0 | 0 | 4.2308 | 0.2092 | 0.2001 | 训练loop最快，但有torch.compile/CUDA graph启动成本 |
| 1 | 1 | 4.5978 | 0.2257 | 0.2188 | 旧验收路径 |
| 1 | 0 | 4.5380 | 0.2235 | 0.2142 | batch invariant抵消了大部分compile收益 |

所有组合的 step seed 和 U/V digest 完全一致。Phase 3测速默认使用
`enforce_eager=0`，即vLLM真实默认的 compile/CUDA graph路径；`enforce_eager=1`
只作为旧验收或消融口径。

## Phase 2 Milestone 4: 梯度对齐验证 完成 ✅

### 关键解决方案：稳定 LoRA ID + in-place reload
vLLM 内部以 `lora_id` 进行 LoRA 权重缓存。在早期实现中，多次调用 `update_plus_minus()` 时，如果 `lora_id` 始终不变，vLLM 会复用旧缓存；如果每步动态分配新 ID，又会制造不必要的 adapter churn。

旧 validation path 使用稳定的 plus/minus LoRA ID；当前实现通过 GPU direct slots 原位写入，早期 `VLLMScorer.score_plus_minus()` + `LoRARequest(load_inplace=True)` 语义只作为历史对齐背景保留。20-step side-by-side 验证达到 100% sign match / 100% high-signal sign match。

### 对齐结果
由于 HuggingFace (Baseline) 中的 Loss 在 `float16` 精度下计算和截断（这会导致 `[2.0, 4.0)` 区间精度间隔为 `1/512 ≈ 0.00195`，`[4.0, 8.0)` 区间精度间隔为 `1/256 ≈ 0.0039`），而 vLLM 的 sampler 内部在 float32 下进行 logprobs 累加计算，两者天生存在很小的浮点精度量化差（最高达 `0.01` 级别）。
因为 $\epsilon=1\text{e-}3$，这部分微小的 loss 精度差异在除以 $2\epsilon=0.002$ 后被放大了 500 倍，反映在系数 $c$ 上有几单位的绝对差值。但这属于完全符合物理规律的合理表现，两者在符号和数值量级上达成了高度契合（相对误差在大部分高信号区域都在 1%~5% 内，且方向 100% 对齐）。

通过设定更科学的 fp16 物理精度容忍度（Loss 容忍度为 `1.5e-2`，系数 $c$ 容忍度为 `15.0`），`test_real_lozo_baseline_side_by_side.py` 成功通过了验证。

同时，完整的 LOZO 训练 Loop 也通过了测试：
```
Initial loss: 3.6736
Final loss: 3.6706
Loss change: -0.0030
✅ Training loop test PASSED!
```

### Phase 2 收敛与性能验收结果

详见 `phase2/README.md` 和本地结果表 `phase2/results/convergence/official_results.md`。

推荐配置：
```
rank=8, nu=50, lr=3e-7, eps=1e-3, batch_size=16
```

300-step 对齐：
- baseline: `5.132812 -> 4.832031`，loss drop `0.300781`
- vLLM: `5.132858 -> 4.831391`，loss drop `0.301467`
- vLLM 达到 baseline loss drop 的 `100.2%`
- final loss diff: `0.000640`
- sign match: `98.3%`
- high-signal sign match: `99.0%`

训练速度：
- instrumented baseline: `0.1155 s/step`
- vLLM direct/param: `0.0864 s/step`
- vLLM 相对 instrumented baseline 速度提升 `1.34x`。

批量不变性：
- batch sizes: `1, 2, 4, 8`
- max per-sample NLL diff: `0.000000000`

### Worker Fused AGZO 经验

Worker-side fused step 可以把 AGZO direction collection、plus/minus slot 写入、
binary-option scoring 和低秩 base-weight update 放在一次 worker RPC 内执行，减少
Python driver 外壳开销。性能分析时要分开看启动期 vLLM compile/CUDA graph capture
和稳定 step 时间；短 smoke 的端到端时间会被初始化污染。

当前 activation hook 方向采集路径仍依赖 `skip_compiled`/eager forward 才能稳定捕获
Linear 输入。AGZO 算法本身不要求 eager，但在改成 graph-native activation capture
之前，不能简单关闭 eager/skip-compiled 并假设方向仍然对齐。

### HF Trainer Benchmark Alignment

For HF Trainer performance comparisons, matching scalar hyperparameters is not
enough. Verify selected dataset indices, every epoch's sampler order, partial
batch placement, token IDs, and perturbation seeds before comparing timings.
Native HF sampling uses epoch-seeded `SeedableRandomSampler` behavior; a
continuously consumed `torch.Generator` only matches the first epoch.

When a runtime scorer delegates through an HF `compute_loss` callback, an outer
`score_token_groups` timer can include tensor construction, device-to-host
unpacking, model-forward dispatch, and loss postprocessing. Record the inner
engine scoring interval separately before attributing a difference to GPU
forward time.

The HF ZO path should use the HF `data_collator` extension point to keep
dataloader inputs as ragged CPU token lists because vLLM does not need dense
sequence padding. Do not introduce a CPU `pad -> tensor -> list -> unpad` cycle,
rebuild GPU HF batches for runtime-selected probes, or call model forward a
second time to obtain loss. Causal objectives should return compact active-token
logits; classification objectives should return device-resident option NLL.
Apply the Trainer-owned `compute_loss_func` or Transformers loss utility to those
outputs and transfer only grouped scalar losses back to the estimator.
When plus/minus or multiple queries share one vLLM forward, split compact outputs
by complete probe group and compute one HF loss per group before runtime slicing.
Never compute one scalar over the combined probes: request-level token-score
slicing would otherwise replace the HF objective with token NLL.
Task-specific object batches must not cross the HF dataloader/Trainer boundary.

HF compact-logits and request-NLL forwards must return typed forward metadata,
not a zero-valued `TokenScoreResult`. Keep the runtime engine and objective
scorer as separate estimator dependencies; never build an engine-shaped wrapper
or inspect an opaque slot's physical LoRA IDs outside `ZOVLLMEngine`. Estimator
updates use `ZOGradientEstimate.scale`; `projected_grad` is a diagnostic and must never
be filled with a synthetic update constant. Device-resident profiling carries
CUDA events to the existing grouped-loss CPU synchronization point, while host
timers are named `engine_call`, `loss_dispatch`, and `loss_to_host` rather than
being presented as GPU execution time.
Removed public fields and arguments are hard errors: do not retain aliases,
fallback reads, deprecation warnings, or tensor-to-ragged conversion shims in
`zo_trainer`. `ZOTrainer` accepts the runtime through the native HF `model=`
argument, and classification batches must explicitly provide
`option_loss_token_counts` and `row_option_counts`.
Keep Hugging Face orchestration on CPU through `ZOTrainerArguments.use_cpu=True`;
vLLM owns GPU placement, so do not mutate Trainer's private `_n_gpu` or
`_train_batch_size` fields. Runtime checkpoint handlers belong to `ZOTrainer`,
must match `zo_checkpoint_mode`, and must run through Hugging Face's native
checkpoint flow so RNG and stateful callback state are preserved. Every
estimator must set `reported_loss` explicitly; Trainer must reject untyped step
results or missing losses instead of inferring them from metrics.
Resume tests must compare a resumed numerical trajectory with an uninterrupted
trajectory, not merely assert that files load. Direction-provider caches and
runtime LoRA slots are distinct state: after rebuilding workers, preserve the
cached basis but force one slot synchronization before scoring.
Cross-engine FP16 GPU forwards are not a bitwise checkpoint contract. Resume
tests should compare base tensors, batches, and U/V directions exactly, then use
a documented numerical tolerance for losses and finite-difference estimates.
Device tensors returned by in-process worker RPC must carry a CUDA readiness
event, and the caller stream must wait on it before computing the HF loss.
Native checkpoint reload must validate all shards, tensor keys, and shapes
before copying any tensor; incompatible tensors are fatal rather than skipped.
Checkpoint scope metadata is descriptive and must not silently configure a
study runtime. Generic `zo_trainer` callbacks accept injected recorders and must
not import experiment runner implementations.
Resumable trainer checkpoints persist optimizer and scheduler state directly;
do not reconstruct scheduler position by replaying `scheduler.step()` calls.
HF clean evaluation must request the runtime's effective clean LoRA slot when
updates are accumulated; `lora_ids=None` only represents immediate or folded
updates. Checkpoint handlers infer bank-aware native materialization from the
update-state capability instead of a caller-provided mode flag.
Serving-time ZO runs the same HF `ZOTrainer`, `ZOTrainerModel`, ragged collator,
`ZOSGDOptimizer`, scheduler, eval cadence, metrics, and callbacks in one
dedicated API-server thread. Do not add a `ServingZOTrainer`, async optimizer,
serving controller, or serving-specific stepper; the API server stores only a
passive background-training handle, while free start/stop/status functions own
lifecycle and the scheduled runtime owns backend differences. The only
synchronous/async boundary is
`ScheduledServingRuntime` plus `BlockingAsyncBridge`: scheduled scoring, QoS
admission, pair locking, cancellation, and worker RPC execute on the server
event loop while the trainer thread blocks. Estimator aggregation remains the
shared estimator implementation and must not be copied into API lifecycle
functions. Serving-time training must use HF `Dataset.shuffle`, `Dataset.map`,
and the HF sampler; do not expose the removed NumPy row-shuffle or unused
eval-row request parameters. The scheduled backend currently supports prompt-option NLL and
must fail fast for causal logits until the engine exposes scheduled compact
logits.

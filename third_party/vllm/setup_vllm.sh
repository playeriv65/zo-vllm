#!/bin/bash
# vLLM 编译安装脚本
# 使用方法: bash setup_vllm.sh

set -e

# 1. 加载模块
module load gcc/13.3.1-p20240614 cuda/12.4.0

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 安装构建依赖
uv pip install -r requirements/build/cuda.txt

# 4. 编译安装 vLLM (开发模式)
uv pip install -e . --torch-backend=auto

# 5. 安装 lint 工具
uv pip install -r requirements/lint.txt
pre-commit install

# 6. 验证
python -c "import vllm; print(f'vLLM {vllm.__version__} installed successfully')"

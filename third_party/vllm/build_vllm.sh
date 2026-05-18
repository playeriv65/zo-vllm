#!/bin/bash
set -e

GCC_DIR=/opt/packages/gcc/v13.3.1-p20240614/b2gpu
CUDA_DIR=/opt/packages/cuda/v13.0.2

export CC=$GCC_DIR/bin/gcc
export CXX=$GCC_DIR/bin/g++
export PATH=$GCC_DIR/bin:$CUDA_DIR/bin:$PATH
export LD_LIBRARY_PATH=$GCC_DIR/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDA_DIR

source .venv/bin/activate

echo "CXX=$CXX"
echo "g++ version: $($CXX --version | head -1)"

# 清理旧的cmake缓存
rm -rf build/
rm -rf .deps/

export CMAKE_ARGS="-DCMAKE_CXX_COMPILER=$CXX -DCMAKE_C_COMPILER=$CC"

# 用 stdbuf 禁用缓冲，确保日志实时输出
stdbuf -oL uv pip install -e . --torch-backend=auto --no-build-isolation

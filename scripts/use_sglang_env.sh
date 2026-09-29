#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/use_cuda_12_9.sh"

export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10
export NVCC_CCBIN=/usr/bin/g++-10

echo "SGLang environment:"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  CC=$CC"
echo "  CXX=$CXX"
echo "  NVCC_CCBIN=$NVCC_CCBIN"

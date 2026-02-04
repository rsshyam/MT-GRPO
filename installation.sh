#!/bin/bash
set -e  # stop on first error

########################
# CUDA / cuDNN settings
########################
# Optional: set these only on clusters / custom installs
# export CUDA_HOME=...
# export CUDNN_HOME=...
# export PATH=...
# export LD_LIBRARY_PATH=...


python3.11 -m venv .venv

nvcc --version

########################
# Activate virtual env
########################
source .venv/bin/activate

pip install --upgrade pip

pip install "pyairports @ git+https://github.com/ozeliger/pyairports.git"

pip install "outlines==0.0.46"

pip install "vllm==0.6.3"

pip install packaging ninja

pip install --upgrade wheel

MAX_JOBS=4 pip install flash-attn --no-build-isolation

pip install -r verl/requirements.txt

pip install -e ./verl

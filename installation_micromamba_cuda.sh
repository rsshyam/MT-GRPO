#!/usr/bin/env bash
set -euo pipefail

# This setup script installs a local CUDA 12.2 toolchain and is intended for machines with an NVIDIA GPU and driver.


# ---------- 0) micromamba install (idempotent) ----------
if ! command -v micromamba >/dev/null 2>&1; then
  echo "[setup] Installing micromamba..."
  tmpdir="$(mktemp -d)"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$tmpdir" bin/micromamba
  mkdir -p "$HOME/micromamba/bin"
  mv "$tmpdir/bin/micromamba" "$HOME/micromamba/bin/micromamba"
  rm -rf "$tmpdir"
  # add to PATH for this script run
  export PATH="$HOME/micromamba/bin:$PATH"
  # initialize shell hooks into ~/.bashrc (idempotent)
  # micromamba shell init -s bash -r "$HOME/micromamba" || true
else
  echo "[setup] micromamba found: $(command -v micromamba)"
fi

# Make micromamba shell functions available in *this* script
eval "$(micromamba shell hook -s bash)"

# ---------- 1) env create / activate ----------
ENV_NAME="mt-grpo"

if micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] Env '$ENV_NAME' already exists."
else
  echo "[setup] Creating env '$ENV_NAME' with Python 3.11..."
  micromamba create -y -n "$ENV_NAME" python=3.11 -c conda-forge
fi

micromamba activate "$ENV_NAME"

# ---------- 2) CUDA 12.2 toolchain & build tools ----------
echo "[setup] Installing CUDA 12.2 toolchain..."
micromamba install -y -c nvidia -c conda-forge \
  cuda-toolkit=12.2 cuda-nvcc=12.2 cuda-cudart-dev=12.2 \
  cuda-profiler-api=12.2 ninja setuptools wheel packaging

# ---------- 3) Python deps ----------
python -m pip install --upgrade pip
pip install --upgrade wheel

pip install "pyairports @ git+https://github.com/ozeliger/pyairports.git"
pip install "outlines==0.0.46"
pip install "vllm==0.6.3"
pip install packaging ninja

pip install debugpy==1.8.0

# FlashAttention (uses nvcc from the env). Idempotent: pip skips if satisfied.
MAX_JOBS=4 pip install --no-build-isolation flash-attn

# Project deps
if [ -f "verl/requirements.txt" ]; then
  pip install -r verl/requirements.txt
fi
pip install -e ./verl

# ---------- 4) Sanity checks ----------
echo "[check] nvcc: $(command -v nvcc)"
nvcc --version | sed -n '1,5p' || true

python - <<'PY'
import os, torch
print("[check] Torch available:", torch.cuda.is_available())
print("[check] CUDA runtime reported by torch:", torch.version.cuda)
print("[check] CUDA_HOME:", os.environ.get("CUDA_HOME"))
PY

echo "[done] Environment '$ENV_NAME' is ready."

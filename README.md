This repository contains the **anonymous code release** accompanying the work: *Multi-Task GRPO: Reliable LLM Reasoning Across Tasks*.

---

## Repository Structure

```text
verl/                 Core library (adapted for this work from verl)
scripts/exp-1/        Scripts to reproduce Experiment 1
scripts/exp-2/        Scripts to reproduce Experiment 2
installation.sh       Lightweight installation (assumes CUDA available)
installation_micromamba_cuda.sh  Full installation (local CUDA + micromamba)
```

---

## Installation

We provide two installation options.

### Option A — Full install (Recommended)

This option creates an isolated environment, installing micromamba, a local CUDA 12.2 toolkit, and all Python dependencies.

```bash
bash installation_micromamba_cuda.sh
```

> **Requirements:** A Linux machine with an NVIDIA GPU and driver.

### Option B — Lightweight install

If you already have a working CUDA and Python environment, you can install the dependencies directly:

```bash
bash installation.sh
```

---

## Running Experiments

All reproduction scripts are located in the `scripts/` directory.

Example usage:

```bash
bash scripts/exp-1/run_mt_grpo_0.2.sh
```

---

## Logging (Weights & Biases)

By default, the provided scripts may attempt to use Weights & Biases (wandb). 
To run experiments without external logging, set:

```bash
export WANDB_MODE=offline
```

If you prefer to log runs to your own account during the review process, set your API key and entity:

```bash
export WANDB_API_KEY=your_key_here
export WANDB_ENTITY=your_entity_here
```


## Notes

*   **Base Library:** This codebase builds upon the open-source [VeRL](https://github.com/volcengine/verl) framework. Files modified for this submission are included in the `verl/` directory.
*   **Model Downloads:** Pretrained models (e.g., Qwen2.5) are configured to download automatically from Hugging Face.

---

## License

This code is released for academic review purposes. A full license and attribution will be included in the de-anonymized release.
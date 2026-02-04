#!/bin/bash

# Note: This script enables Weights & Biases (wandb) logging by default.
# To disable online logging, set: export WANDB_MODE=disabled


if [ $# -gt 0 ]; then
  exp_name=$1
  shift 1
else
  STAMP=$(date +%F_%H-%M-%S)
  exp_name="SEC_GRPO_EXP_2_${STAMP}"
fi

other_args=$@


export WANDB_API_KEY="" 
export WANDB_ENTITY=""

export RAY_DEBUG=legacy


# --- Ray & temp dirs (set as required) ---
export RAY_DEBUGGER=0
: "${RAY_TMPDIR:=/tmp/ray}"
: "${TMPDIR:=/tmp}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

# --- Activate micromamba env non-interactively ---
# (needed inside scripts; don’t use `micromamba shell init` here)
export PATH="$HOME/micromamba/bin:$PATH"

eval "$(micromamba shell hook -s bash)"
micromamba activate mt-grpo

export N_GPUS=2

MODEL_PATH="Qwen/Qwen2.5-3B"
log_dir=logs/puzzle
mkdir -p $log_dir


export VLLM_ATTENTION_BACKEND=XFORMERS

export VRAM_RESERVE_GB=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="[data/combined/train_countdown.parquet,data/combined/train_zebra.parquet,data/combined/train_arc.parquet]" \
    data.val_files="[data/combined/test_countdown.parquet,data/combined/test_zebra.parquet,data/combined/test_arc.parquet]" \
    trainer.sec.bandit.type=joint \
    trainer.sec.bandit.feature="[difficulty,type]" \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=8 \
    actor_rollout_ref.rollout.swap_space=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='mt-grpo' \
    trainer.experiment_name=$exp_name \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    trainer.total_training_steps=720 \
    actor_rollout_ref.actor.log_task_wise_kl=True \
    trainer.sec.strategy=bandit \
    trainer.sec.enable=True \
    trainer.sec.bandit.lr=0.5 \
    trainer.sec.bandit.temperature=0.2 \
    trainer.rejection_sample=False \
    trainer.save_freq=180 \
    actor_rollout_ref.use_checkpoint_manager=True \
    $other_args 2>&1 | tee $log_dir/${exp_name}.log

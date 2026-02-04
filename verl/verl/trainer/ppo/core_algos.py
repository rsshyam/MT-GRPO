# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   normalize: bool = True,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if normalize:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = (scores[i] - id2mean[index[i]])
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask)

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores



def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio

def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss

# def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange, loss_vector=False, cliprange_high=None, agg_loss_mode="token-mean"):
#     """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

#     Args:
#         old_log_prob: `(torch.Tensor)`
#             shape: (bs, response_length)
#         log_prob: `(torch.Tensor)`
#             shape: (bs, response_length)
#         advantages: `(torch.Tensor)`
#             shape: (bs, response_length)
#         eos_mask: `(torch.Tensor)`
#             shape: (bs, response_length)
#         cliprange: (float)
#             The clip range used in PPO. See https://arxiv.org/abs/1707.06347

#     Returns:
#         pg_loss: `a scalar torch.Tensor`
#             policy gradient loss computed via PPO
#         pg_clipfrac: (float)
#             a float number indicating the fraction of policy gradient loss being clipped

#     """
#     if cliprange_high is None:
#         cliprange_high = cliprange
#     negative_approx_kl = log_prob - old_log_prob
#     ratio = torch.exp(negative_approx_kl)
#     ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

#     pg_losses = -advantages * ratio
#     pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange_high)

    

#     # pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
#     pg_loss = agg_loss(torch.max(pg_losses, pg_losses2), eos_mask, agg_loss_mode)

#     pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)

#     if loss_vector:
#         # calculate per sample loss
#         ppo_kl_vec = verl_F.masked_mean(-negative_approx_kl, eos_mask, axis=-1)
#         pg_loss_vec = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask, axis=-1)
#         pg_clipfrac_vec = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask, axis=-1)
#         return pg_loss, pg_clipfrac, ppo_kl, [pg_loss_vec, pg_clipfrac_vec, ppo_kl_vec]
#     return pg_loss, pg_clipfrac, ppo_kl, None


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    eos_mask,
    cliprange,
    loss_vector: bool = False,
    cliprange_high: float | None = None,
    agg_loss_mode: str = "token-mean",
    **kwargs,
):
    """Policy loss for PPO / GSPO / CISPO.

    loss_type:
        - "ppo"  (default): classic PPO clipping (original behavior)
        - "gspo": Group Sequence Policy Optimization
        - "cispo": Clipped IS-weight Policy Optimization
    """
    loss_type = kwargs.get("loss_type", "ppo")

    if cliprange_high is None:
        cliprange_high = cliprange

    # Shared quantity: negative approximate KL per token
    # (log π_new - log π_old)
    negative_approx_kl = log_prob - old_log_prob

    # =========================
    # 1) Standard PPO (old behavior)
    # =========================
    if loss_type == "ppo":
        print("Using PPO loss")
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange_high)

        pg_loss = agg_loss(torch.max(pg_losses, pg_losses2), eos_mask, agg_loss_mode)

        pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)

        if loss_vector:
            # calculate per sample loss
            ppo_kl_vec = verl_F.masked_mean(-negative_approx_kl, eos_mask, axis=-1)
            pg_loss_vec = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask, axis=-1)
            pg_clipfrac_vec = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask, axis=-1)
            return pg_loss, pg_clipfrac, ppo_kl, [pg_loss_vec, pg_clipfrac_vec, ppo_kl_vec]

        return pg_loss, pg_clipfrac, ppo_kl, None

    # =========================
    # 2) GSPO (Group Sequence Policy Optimization)
    # =========================
    if loss_type == "gspo":
        print("Using GSPO loss")
        # seq_lengths: number of valid tokens per sequence
        seq_lengths = torch.sum(eos_mask, dim=-1).clamp(min=1)

        # sequence-level negative approximate KL:
        # negative_approx_kl_seq[i] = (1 / |y_i|) * sum_t (log π_new - log π_old)
        negative_approx_kl_seq = torch.sum(negative_approx_kl * eos_mask, dim=-1) / seq_lengths

        # Combined ratio at token level (from GSPO paper / verl implementation):
        # log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
        log_seq_importance_ratio = (
            log_prob
            - log_prob.detach()
            + negative_approx_kl_seq.detach().unsqueeze(-1)
        )
        log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # numeric stability
        seq_importance_ratio = torch.exp(log_seq_importance_ratio)

        pg_losses1 = -advantages * seq_importance_ratio
        pg_losses2 = -advantages * torch.clamp(
            seq_importance_ratio,
            1.0 - cliprange,
            1.0 + cliprange_high,
        )
        pg_losses = torch.maximum(pg_losses1, pg_losses2)

        # For GSPO it’s recommended to aggregate as seq-mean over token-mean
        pg_loss = agg_loss(
            loss_mat=pg_losses,
            loss_mask=eos_mask,
            loss_agg_mode="seq-mean-token-mean",
        )

        pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

        if loss_vector:
            pg_loss_vec = verl_F.masked_mean(pg_losses, eos_mask, axis=-1)
            ppo_kl_vec = verl_F.masked_mean(-negative_approx_kl, eos_mask, axis=-1)
            pg_clipfrac_vec = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask, axis=-1)
            return pg_loss, pg_clipfrac, ppo_kl, [pg_loss_vec, pg_clipfrac_vec, ppo_kl_vec]

        return pg_loss, pg_clipfrac, ppo_kl, None

    # =========================
    # 3) CISPO (Clipped IS-weight Policy Optimization)
    # =========================
    if loss_type == "cispo":
        print("Using CISPO loss")
        # negative_approx_kl = log π_new - log π_old (already computed above)
        log_is_ratio = torch.clamp(negative_approx_kl, max=10.0)
        is_ratio = torch.exp(log_is_ratio)  # r_t

        # Interpret cliprange_high as epsilon_high for CISPO.
        # Paper / Swift suggest a relatively large value, e.g. 5.0
        epsilon_high = cliprange_high if cliprange_high is not None else 5.0

        # One-sided upper clipping (Swift-style CISPO)
        is_ratio_clipped = torch.clamp(is_ratio, max=epsilon_high)

        # Stop-gradient on IS weights
        is_weight = is_ratio_clipped.detach()

        # CISPO loss: - sg(r_hat) * A * log π_new
        pg_losses = -is_weight * advantages * log_prob

        pg_loss = agg_loss(
            loss_mat=pg_losses,
            loss_mask=eos_mask,
            loss_agg_mode="token-mean",  # usually "token-mean"
        )

        # Fraction of tokens where IS ratio was clipped
        clipped_mask = (is_ratio > epsilon_high).float()
        pg_clipfrac = verl_F.masked_mean(clipped_mask, eos_mask)

        # Approx KL for logging only (CISPO doesn't use a KL penalty)
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

        if loss_vector:
            pg_loss_vec = verl_F.masked_mean(pg_losses, eos_mask, axis=-1)
            ppo_kl_vec = verl_F.masked_mean(-negative_approx_kl, eos_mask, axis=-1)
            pg_clipfrac_vec = verl_F.masked_mean(clipped_mask, eos_mask, axis=-1)
            return pg_loss, pg_clipfrac, ppo_kl, [pg_loss_vec, pg_clipfrac_vec, ppo_kl_vec]

        return pg_loss, pg_clipfrac, ppo_kl, None


    # =========================
    # 4) Fallback if unknown loss_type
    # =========================
    raise ValueError(f"Unknown loss_type: {loss_type}")

def compute_is_token_reward(
    old_log_prob: torch.Tensor,          # [B, T]
    log_prob: torch.Tensor,              # [B, T]
    token_level_rewards: torch.Tensor,   # [B, T]
    eos_mask: torch.Tensor,              # [B, T]
    *,
    log_w_clip: float = 20.0,
):
    assert old_log_prob.shape == log_prob.shape == eos_mask.shape
    assert token_level_rewards.shape == log_prob.shape

    mask = eos_mask.to(log_prob.dtype)

    # per-sample scalar reward from old rollout
    denom = torch.sum(mask, dim=-1).clamp(min=1.0)
    r = torch.sum(token_level_rewards * mask, dim=-1) / denom  # [B]

    # per-sample log importance weight
    log_w = torch.sum((log_prob - old_log_prob) * mask, dim=-1)  # [B]
    log_w = torch.clamp(log_w, min=-log_w_clip, max=log_w_clip)

    return r, log_w



def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError

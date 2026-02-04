
from typing import Any
import numpy as np
from collections import defaultdict
import random
import torch
import copy
import verl.utils.torch_functional as verl_F


import numpy as np
from typing import Any, Dict, List
import torch.nn.functional as F

from ray.util import pdb 

from .decay_strategies import get_decay

from types import SimpleNamespace

def fixed_masked_softmax(logits: torch.Tensor,
                         mask: torch.Tensor,
                         fixed_p_each: float = 0.05,
                         dim: int = -1) -> torch.Tensor:
    """
    mask: True for active/trainable entries, False for masked/fixed entries.
    fixed_p_each: fixed probability assigned to each masked entry.
    """
    if logits.ndim != 1:
        raise ValueError("This helper assumes 1D logits (num_arms). Adapt as needed.")
    if mask.dtype != torch.bool:
        mask = mask.bool()

    num_masked = (~mask).sum().item()
    # if num_masked == 0:
    #     # no masked arms: standard softmax
    #     return torch.softmax(logits, dim=dim)

    fixed_mass = fixed_p_each * num_masked
    if not (0.0 <= fixed_mass < 1.0):
        raise ValueError("fixed_p_each * num_masked must be in [0,1).")

    # softmax only over active logits
    active_logits = logits[mask]
    # numerical stability: subtract max over active entries
    m = active_logits.max()
    active_probs = torch.softmax(active_logits - m, dim=0)

    # scale active probs to use the remaining mass
    active_probs = active_probs * (1.0 - fixed_mass)

    # stitch back together; masked probs are constants (no grad path)
    probs = torch.zeros_like(logits)
    probs[~mask] = logits.new_full((num_masked,), float(fixed_p_each))
    probs[mask] = active_probs

    # Optional exploration / per-arm floor
    if fixed_p_each > 0.0:
        n = probs.numel()
        # cap the floor so it's feasible (sum floors ≤ 1)
        floor = min(fixed_p_each, 1.0 / n - 1e-12)
        if floor > 0:
            probs = (1.0 - n * floor) * probs + floor
            probs = probs / probs.sum()  # re-normalize (numerical safety)

    return probs

def gated_softmax(
    logits: torch.Tensor,
    softmask: torch.Tensor,
    beta: float = 1.0,
    fixed_p_each: float = 0.0,
    eps: float = 1e-12,
    dim: int = -1
) -> torch.Tensor:
    if logits.ndim != 1:
        raise ValueError("This helper assumes 1D logits (num_arms). Adapt as needed.")
    if softmask.shape != logits.shape:
        raise ValueError("softmask must have same shape as logits.")

    # Product-of-experts: p ∝ exp(logits) * softmask^beta  →  logits + beta*log(softmask)
    log_gate = torch.log(torch.clamp(torch.from_numpy(softmask), min=eps))
    combined = logits + beta * log_gate

    # Stable softmax
    m = combined.max()
    p = torch.softmax(combined - m, dim=dim)

    # Optional exploration / per-arm floor
    if fixed_p_each > 0.0:
        n = p.numel()
        # cap the floor so it's feasible (sum floors ≤ 1)
        floor = min(fixed_p_each, 1.0 / n - 1e-12)
        if floor > 0:
            p = (1.0 - n * floor) * p + floor
            p = p / p.sum()  # re-normalize (numerical safety)

    return p

def _compute_masked_logit_y(logits: torch.Tensor, mask: torch.Tensor, fixed_p_each: float) -> torch.Tensor:
    """Scalar y so that each masked entry would have prob fixed_p_each if unmasked now."""
    if logits.ndim != 1:
        raise ValueError("Assumes 1D logits.")
    if mask.dtype is not torch.bool:
        mask = mask.bool()
    k = (~mask).sum()
    p = torch.tensor(float(fixed_p_each), dtype=logits.dtype, device=logits.device)
    if k.item() == 0:
        raise ValueError("No masked entries.")
    if (p * k) >= 1:
        raise ValueError("fixed_p_each * num_masked must be < 1.")
    if fixed_p_each == 0.0:
        return torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device)
    lse_active = torch.logsumexp(logits[mask], dim=0)  # log(sum(exp(active)))
    y = lse_active + torch.log(p) - torch.log(1.0 - k.to(logits.dtype) * p)
    return y

def _reset_opt_state_indices(optimizer, param: torch.Tensor, idx_bool: torch.Tensor, scale: float = 0.0):
    """
    Zero (scale=0) or damp (0<scale<1) optimizer state tensors for `param`
    at positions where idx_bool is True. Works for Adam/AdamW, SGD(momentum), RMSprop, etc.
    """
    st = optimizer.state.get(param, None)
    if not st:
        return
    with torch.no_grad():
        for k, v in st.items():
            if torch.is_tensor(v) and v.shape == param.shape:
                v[idx_bool] *= scale

def get_bandit_priority(*args, **kwargs):
    assert 'type' in kwargs, 'type is required for initializing bandit priority'
    type = kwargs['type']

    enable = kwargs.get('enable', False) ## corresponds to famo enable
    

    if type == 'joint':
        if enable:
            return JointFamoBandit(*args, **kwargs)
        else:
            return JointBandit(*args, **kwargs)
    elif type == 'independent':
        return IndependentBandit(*args, **kwargs)
    else:
        raise ValueError(f'Invalid type: {type}')

class BasePriority:

    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class DifficultyPriority(BasePriority):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, data: dict[str, Any]):
        return data['extra_info']['difficulty']


class ReverseDifficultyPriority(BasePriority):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, data: dict[str, Any]):
        return -data['extra_info']['difficulty']

class FamoPriorityDecay(BasePriority):
    """
    Priority / sampler that groups datapoints (e.g., by ['difficulty','type']),
    tracks per-group losses, and updates per-group sampling weights w using a
    FAMO-style loss-change signal:

        delta = log(prev_loss - min_loss + eps) - log(new_loss - min_loss + eps)

    Then w is updated via autograd on softmax(w) with grad_outputs=delta
    (same pattern as your _famo_weights_update).

      *,
        dataset: List[Dict[str, Any]],
        feature: List[str] | str = ("difficulty",),
        lr: float = 0.1,
        temperature: float = 1.0,       # kept for parity; not used when softmax(w) drives probs
        epsilon: float = 0.1,           # only used for 'greedy'
        method: str = "softmax",        # 'softmax' (recommended), 'greedy', or 'boltzmann' (legacy)
        device: str = "cpu",
        use_log_ratio_delta: bool = True,   # True => log-ratio delta (FAMO), False => linear delta
        eps: float = 1e-8,
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

       
        self.feature = kwargs.get('feature', ['difficulty'])
        self.learning_rate = kwargs.get('lr', 0.1)
        self.temperature = kwargs.get('temperature', 1.0)
        self.gamma = kwargs.get('gamma', 0.1)
        self.max_rate = kwargs.get('max_rate', False)
        self.fixed_p_each = kwargs.get('fixed_p_each', 0.05)
        self.decay_type = kwargs.get('decay_type', None)
        self.reward_threshold = kwargs.get('reward_threshold', 0.8)
        self.decay_rate = kwargs.get('decay_rate', 0.5)
        self.update_object = kwargs.get('update_object', 'task_wise_diff')

        self.average_delta = kwargs.get('average_delta', False)
        self.delta_threshold = kwargs.get('delta_threshold', None)
        self.use_reward_diff = kwargs.get('use_reward_diff', False)
        self.use_acc = kwargs.get('use_acc', False)

        self.change_decay_rate = kwargs.get('change_decay_rate', 0.5)
        self.change_decay_rate_2 = kwargs.get('change_decay_rate_2', 0.5)
        self.SNR_threshold = kwargs.get('SNR_threshold', 0.5)
        self.SNR_eps = kwargs.get('SNR_eps', 1e-8)

        self.SNR_update_step_threshold = kwargs.get('SNR_update_step_threshold', 50)

        self.sharpness = kwargs.get('sharpness', 1.0)

        self.index = kwargs.get('index', None)

        self.mean_diff_decay = kwargs.get('mean_diff_decay', False)

        self.lambda_bias = kwargs.get('lambda_bias', 0.1) 
        self.bias_threshold = kwargs.get('bias_threshold', 0.01)
        self.lambda_bias_type = kwargs.get('lambda_bias_type', 'add')
        self.conv_penalty = kwargs.get('conv_penalty', 0.1)


        self.lambda_bias_adaptive = kwargs.get('lambda_bias_adaptive', False)
        self.lambda_bias_start = kwargs.get('lambda_bias_start', 0.01)
        self.lambda_bias_end = kwargs.get('lambda_bias_end', 0.1)
        self.lambda_bias_ramp_frac = kwargs.get('lambda_bias_ramp_frac', 0.9)
        self.total_steps = kwargs.get('total_steps', 720)

        self.z_decay_rate = kwargs.get('z_decay_rate', 0.0)
        self.one_decay_rate = kwargs.get('one_decay_rate', 0.0)
        self.nznone_decay_rate = kwargs.get('nznone_decay_rate', 0.0)
        self.opt_decay_rate = kwargs.get('opt_decay_rate', 0.0)
        self.solve_none_decay_rate = kwargs.get('solve_none_decay_rate', 0.0)
        self.solve_all_decay_rate = kwargs.get('solve_all_decay_rate', 0.0)
        self.batch_acc_decay_rate = kwargs.get('batch_acc_decay_rate', 0.0)
        self.batch_acc_delta_decay_rate = kwargs.get('batch_acc_delta_decay_rate', 0.0)
        self.solve_half_decay_rate = kwargs.get('solve_half_decay_rate', 0.0)
        self.var_decay_rate = kwargs.get('var_decay_rate', 0.0)
        self.optdist_decay_rate = kwargs.get('optdist_decay_rate', 0.0)

        self.acc_decay_rate = kwargs.get('acc_decay_rate', 0.0)

        self.opt_threshold = kwargs.get('opt_threshold', 0.5)
        self.use_opt_ratio = kwargs.get('use_opt_ratio', False)
        self.bias_metric = kwargs.get('bias_metric', 'reward_gap')

        self.preset_probs = kwargs.get('preset_probs', None)
        if self.preset_probs is not None:
            self.preset_probs = torch.tensor(list(self.preset_probs))

        self.temp_probs = None


        if isinstance(self.feature, str):
            self.feature = [self.feature]

        cfg = kwargs.get('knapsack', {}) or {}

        self.knapsack = SimpleNamespace(
            enabled=cfg.get('enabled', False),
            rollouts_per_task=cfg.get('rollouts_per_task', 8),
            type=cfg.get('type', 'vanilla')
        )
        
        cfg_dynamic_rollouts = kwargs.get('dynamic_rollouts', {}) or {}
        self.dynamic_rollouts = SimpleNamespace(
            enabled=cfg_dynamic_rollouts.get('enabled', False),
            rollouts_per_task=cfg_dynamic_rollouts.get('rollouts_per_task', 8),
            max_rollouts_per_task=cfg_dynamic_rollouts.get('max_rollouts_per_task', 144),
            filter_threshold=cfg_dynamic_rollouts.get('filter_threshold', 0.1),
            filter_threshold_upper=cfg_dynamic_rollouts.get('filter_threshold_upper', 0.6),
            filtered_ratio_decay_rate=cfg_dynamic_rollouts.get('filtered_ratio_decay_rate', 0.9),
            filtered_ratio_update_step_threshold=cfg_dynamic_rollouts.get('filtered_ratio_update_step_threshold', 50),
            filtered_ratio_update_step_freq=cfg_dynamic_rollouts.get('filtered_ratio_update_step_freq', 25),
        )
        
        cfg_efficient_sampling = kwargs.get('efficient_sampling', {}) or {}
        self.efficient_sampling = SimpleNamespace(
            enabled=cfg_efficient_sampling.get('enabled', False),
            beta=cfg_efficient_sampling.get('beta', 1.0),
            eps=cfg_efficient_sampling.get('eps', 1e-2),
            mult_max=cfg_efficient_sampling.get('mult_max', 5.0),
            deterministic=cfg_efficient_sampling.get('deterministic', False),
        )


       

        assert 'dataset' in kwargs, 'dataset is required for initializing bandit priority'
        
        self.init_bandit(kwargs['dataset'])

        # ---- state: weights, prev & min losses (torch on device) ----
        self.w = torch.zeros(self.num_arms, requires_grad=True)
        self.opt = torch.optim.Adam([self.w], lr=self.learning_rate, weight_decay=self.gamma)
        # initialize probs cache

        self._probs_cache = self._softmax_probs().detach().cpu().numpy()

        resume_from = kwargs.get('resume_from', None)
        if resume_from is not None:
            self.load_checkpoint(resume_from)
    
    def save_checkpoint(self, local_dir: str, remote_dir: str | None = None):
        """
        Save bandit state (weights + optimizer + a bit of extra state) to a directory.
        """
        import os
        import torch

        os.makedirs(local_dir, exist_ok=True)

        state = {
            "w": self.w.detach().cpu(),             # weights
            "opt": self.opt.state_dict(),           # optimizer state
            "mask": self.mask,                      # basic mask
            "q_values": self.q_values,              # main stats driving mask/decay
            "w_update_step": getattr(self, "w_update_step", 0),
            "group_to_n": getattr(self, "group_to_n", None),
            "opt_ratio": getattr(self, "opt_ratio", None),
            "batch_acc": getattr(self, "batch_acc", None),
            "batch_acc_sq": getattr(self, "batch_acc_sq", None),
            "batch_acc_var": getattr(self, "batch_acc_var", None),
            "batch_acc_delta": getattr(self, "batch_acc_delta", None),
            "var": getattr(self, "var", None),
            "optdist": getattr(self, "optdist", None),
            "filtered_ratio": getattr(self, "filtered_ratio", None),
            "idx_to_group": getattr(self, "idx_to_group", None),
            "group_to_idx": getattr(self, "group_to_idx", None),
            "arm_to_group": getattr(self, "arm_to_group", None),
            "group_to_arm": getattr(self, "group_to_arm", None),
            "num_arms": getattr(self, "num_arms", None),
            "arm_to_idx": getattr(self, "arm_to_idx", None),
        }

        torch.save(state, os.path.join(local_dir, "priority_state.pt"))

        # If you really want to mirror actor/critic and copy to HDFS, you can do:
        if remote_dir is not None:
            from verl.utils import hdfs_io
            hdfs_io.makedirs(remote_dir, exist_ok=True)
            hdfs_io.copy(src=local_dir, dst=remote_dir)#
        
        print(f"Saved priority checkpoint to {local_dir}")

    def load_checkpoint(self, local_dir: str):
        """
        Load bandit state from a directory created by save_checkpoint.
        """
        import os
        import torch

        state_path = os.path.join(local_dir, "priority_state.pt")
        state = torch.load(state_path, map_location=self.w.device)

        # IMPORTANT: keep same tensor object, just overwrite its data
        self.w.data.copy_(state["w"])
        self.opt.load_state_dict(state["opt"])

        # restore simple buffers/arrays
        if "mask" in state:
            self.mask = state["mask"]
        if "q_values" in state:
            self.q_values = state["q_values"]
        if "w_update_step" in state:
            self.w_update_step = int(state["w_update_step"])
        if "group_to_n" in state and state["group_to_n"] is not None:
            self.group_to_n = state["group_to_n"]
        if "opt_ratio" in state and state["opt_ratio"] is not None:
            self.opt_ratio = state["opt_ratio"]
        if "batch_acc" in state and state["batch_acc"] is not None:
            self.batch_acc = state["batch_acc"]
        if "batch_acc_sq" in state and state["batch_acc_sq"] is not None:
            self.batch_acc_sq = state["batch_acc_sq"]
        if "batch_acc_var" in state and state["batch_acc_var"] is not None:
            self.batch_acc_var = state["batch_acc_var"]
        if "batch_acc_delta" in state and state["batch_acc_delta"] is not None:
            self.batch_acc_delta = state["batch_acc_delta"]
        if "var" in state and state["var"] is not None:
            self.var = state["var"]
        if "optdist" in state and state["optdist"] is not None:
            self.optdist = state["optdist"]
        if "filtered_ratio" in state and state["filtered_ratio"] is not None:
            self.filtered_ratio = state["filtered_ratio"]
        if "idx_to_group" in state and state["idx_to_group"] is not None:
            self.idx_to_group = state["idx_to_group"]
        if "group_to_idx" in state and state["group_to_idx"] is not None:
            self.group_to_idx = state["group_to_idx"]
        if "arm_to_group" in state and state["arm_to_group"] is not None:
            self.arm_to_group = state["arm_to_group"]
        if "group_to_arm" in state and state["group_to_arm"] is not None:
            self.group_to_arm = state["group_to_arm"]
        if "num_arms" in state and state["num_arms"] is not None:
            self.num_arms = state["num_arms"]
        if "arm_to_idx" in state and state["arm_to_idx"] is not None:
            self.arm_to_idx = state["arm_to_idx"]
    
        # refresh cached probabilities
        self._probs_cache = self._softmax_probs().detach().cpu().numpy()
        
        print(f"Loaded priority checkpoint from {local_dir}")



        
    def init_bandit(self, dataset):

        self.idx_to_group = {idx: self.data_to_group(item) for idx, item in enumerate(dataset)}
        self.group_to_idx = {group: [] for group in self.idx_to_group.values()}
        for idx, group in self.idx_to_group.items():
            self.group_to_idx[group].append(idx)
        self.arm_to_group = {i: key for i, key in enumerate(self.group_to_idx.keys())}
        print(self.arm_to_group)
        self.group_to_arm = {key: i for i, key in enumerate(self.group_to_idx.keys())}
        self.num_arms = len(self.group_to_idx)

        self.arm_to_idx = {i: [] for i in range(self.num_arms)}
        for group, arm in self.group_to_arm.items():
            self.arm_to_idx[arm].extend(self.group_to_idx[group])

        self.mask = torch.ones(self.num_arms, dtype=torch.bool)
        self.q_values = np.zeros(self.num_arms)
        if self.update_object == 'reward_gap':
            self.last_reward = np.zeros(self.num_arms)
        
        if self.update_object == 'task_wise_diff':
            self.task_delta = np.zeros(self.num_arms)
        
        if self.update_object == 'task_diff_bias':
            self.last_reward = np.zeros(self.num_arms)
            self.task_delta = np.zeros(self.num_arms)
            self.combined_delta =  np.zeros(self.num_arms)
            self.w_update_step = 0

            self.avg_task_wise_diff = np.zeros(self.num_arms)
            self.avg_task_wise_count = np.zeros(self.num_arms)
            self.avg_task_wise_ess_full = np.zeros(self.num_arms)
            self.avg_task_wise_ess_frac = np.zeros(self.num_arms)
            self.avg_task_wise_reward_diff = np.zeros(self.num_arms)
        
        if self.decay_type == 'q_conv_mask':
            self.change_qs = np.zeros(self.num_arms)
            self.change_qs_mom = np.zeros(self.num_arms)
            self.change_qs_mom_2 = np.zeros(self.num_arms)
            self.q_SNR = np.zeros(self.num_arms)
            self.q_update_step = np.zeros(self.num_arms)
        
        if self.decay_type == 'task_diff_q_conv_mask':
            self.q_mom_2 = np.zeros(self.num_arms)
            self.q_SNR = np.zeros(self.num_arms)
            self.q_update_step = np.zeros(self.num_arms)

        if self.decay_type == 'soft_task_diff_q_conv_mask':
            self.softmask = np.ones(self.num_arms)
            self.q_update_step = np.zeros(self.num_arms)

        if self.decay_type == 'var_conv_mask':
            self.q_mom_2 = np.zeros(self.num_arms)
            self.q_SNR = np.zeros(self.num_arms)
            self.q_update_step = np.zeros(self.num_arms)

        if self.decay_type == 'soft_var_conv_mask':
            self.softmask = np.ones(self.num_arms)
            self.q_update_step = np.zeros(self.num_arms)
        
        if self.decay_type == 'soft_q_conv_mask':
            self.softmask = np.ones(self.num_arms)
            self.q_update_step = np.zeros(self.num_arms)
            self.change_qs = np.zeros(self.num_arms)
            self.change_qs_mom = np.zeros(self.num_arms)

        if self.decay_type == 'soft_opt_conv_mask':
            self.softmask = np.ones(self.num_arms)
            self.q_update_step = np.zeros(self.num_arms)
            self.z_ratio = np.zeros(self.num_arms)
            self.one_ratio = np.zeros(self.num_arms)
            self.nznone_ratio = np.zeros(self.num_arms)
        
        if self.decay_type == 'coeff_var_conv_mask':
            self.q_update_step = np.zeros(self.num_arms)

        if self.decay_type == 'soft_coeff_var_conv_mask':
            self.softmask = np.ones(self.num_arms)
            self.q_update_step = np.zeros(self.num_arms)
        
        if self.decay_type == 'soft_half_opt_conv_mask':
            self.softmask = np.ones(self.num_arms)
            self.q_update_step = np.zeros(self.num_arms)
            self.z_ratio = np.zeros(self.num_arms)
            self.one_ratio = np.zeros(self.num_arms)
            self.nznone_ratio = np.zeros(self.num_arms)
            self.opt_ratio = np.zeros(self.num_arms)
            self.acc_ratio = defaultdict(lambda: defaultdict(float))
            self.solve_none_ratio = np.zeros(self.num_arms)
            self.solve_all_ratio = np.zeros(self.num_arms)
            self.batch_acc = np.zeros(self.num_arms)
            self.batch_acc_sq = np.zeros(self.num_arms)
            self.batch_acc_var = np.zeros(self.num_arms)
            self.batch_acc_delta = np.zeros(self.num_arms)
            self.var = np.zeros(self.num_arms)
            self.solve_half_ratio = np.zeros(self.num_arms)
            self.optdist = np.zeros(self.num_arms)
            self.filtered_ratio = np.zeros(self.num_arms)
        
        if self.knapsack.enabled:
            self.group_to_n = {self.arm_to_group[i]: self.knapsack.rollouts_per_task for i in range(self.num_arms)}
        if self.dynamic_rollouts.enabled:
            self.group_to_n = {self.arm_to_group[i]: self.dynamic_rollouts.rollouts_per_task for i in range(self.num_arms)}
        if self.efficient_sampling.enabled:
            self.original_group_ratios = {self.arm_to_group[i]: 1/self.num_arms for i in range(self.num_arms)}

    def extra_info_to_group(self, extra_info):
        return '-'.join([str(extra_info[feature]) for feature in self.feature])

    def data_to_group(self, data):
        if 'difficulty' in data['extra_info']:
            assert int(data['extra_info']['difficulty']) == float(data['extra_info']['difficulty']) # make sure difficulty is an integer
        return self.extra_info_to_group(data['extra_info'])

    def _softmax_probs(self) -> torch.Tensor:
        # keep the name but route to the masked-fixed version
        if self.temp_probs is not None:
            return self.temp_probs
        elif self.preset_probs is not None:
            return self.preset_probs
        elif isinstance(self.decay_type, str) and 'soft' in self.decay_type and self.update_object == 'task_wise_diff':
            return gated_softmax(self.w, self.softmask, fixed_p_each=self.fixed_p_each, dim=-1)
        return fixed_masked_softmax(self.w, self.mask, fixed_p_each=self.fixed_p_each, dim=-1)

    def sample_arms(self, num_samples: int) -> np.ndarray:
        probs = self._softmax_probs().detach().cpu().numpy()
        if self.knapsack.enabled:
            return self.knapsack_sample_arms(num_samples, probs)
        elif self.dynamic_rollouts.enabled:
            return self.sample_arms_dynamic_rollouts(num_samples, probs)
        elif self.efficient_sampling.enabled:
            return self.sample_arms_efficient_sampling(num_samples, probs, beta=self.efficient_sampling.beta, eps=self.efficient_sampling.eps, mult_max=self.efficient_sampling.mult_max)
        return np.random.choice(self.num_arms, size=num_samples, p=probs)
    
    def sample_arms_efficient_sampling(self, num_samples: int,
                    probs: np.ndarray,
                    beta: float = 1.0,
                    eps: float = 1e-2,
                    mult_max: float = 5.0) -> np.ndarray:

        # store original ratios (group == arm in your case)
        original = np.random.choice(self.num_arms, size=num_samples, p=probs)
        uniq, cnt = np.unique(original, return_counts=True)
        self.original_group_ratios = {self.arm_to_group[int(a)]: c / num_samples for a, c in zip(uniq, cnt)}

        fr = np.asarray(self.filtered_ratio, dtype=float)          # EMA of filtered fraction
        a = np.clip(1.0 - fr, eps, 1.0)                            # acceptance rate
        mult = np.minimum(a ** (-beta), mult_max)                  # temper + clip

        if self.efficient_sampling.deterministic:
          
            # empirical per-arm probs from the original sample (THIS is the key change)
            c = np.bincount(original, minlength=self.num_arms).astype(float)
            p_emp = c / max(c.sum(), 1.0)

  

            # IMPORTANT: inflate the empirical target (derived from original), not probs
            w = p_emp * mult
            s = w.sum()
   

            q = w / s
            print("acceptance-aware-inflation (conditioned on original)", q)

            # 3) deterministic apportionment (largest remainder)
            expected = q * num_samples
            counts = np.floor(expected).astype(int)
            remainder = int(num_samples - counts.sum())
            if remainder > 0:
                frac = expected - counts
                top = np.argsort(-frac)[:remainder]
                counts[top] += 1

            # 4) build deterministic arm list
            arms = np.repeat(np.arange(self.num_arms), counts)

            seed = int(getattr(self, "w_update_step", 0))
            rng = np.random.default_rng(seed)
            rng.shuffle(arms)

            return arms

        q = probs * mult
        q /= q.sum()

        print("acceptance-aware-inflation", q)

        return np.random.choice(self.num_arms, size=num_samples, p=q)

    
    def set_temp_probs(self, temp_probs: dict | None):
        # convert to torch tensore based on arm to group
        if temp_probs is None:
            self.temp_probs = None
            return
        temp_probs = torch.tensor([temp_probs.get(self.arm_to_group[i], 0) for i in range(self.num_arms)])
        # normalize
        temp_probs = temp_probs / temp_probs.sum()
        self.temp_probs = temp_probs
    
    def knapsack_sample_arms(self, num_samples: int, probs: np.ndarray) -> np.ndarray:
        # probs is all zeros and one 1, just sample from it
        if np.any(probs >= 1 - 1e-6):
            # find the index with prob >= 1 - 1e-6
            idx = np.where(probs >= 1 - 1e-6)[0][0]
            self.prompt_to_n = {i: self.knapsack.rollouts_per_task for i in range(num_samples)}
            self.group_to_n = {
                self.arm_to_group[i]: (self.knapsack.rollouts_per_task if i == idx else 0)
                for i in range(self.num_arms)
            }
            return np.full(num_samples, idx)
        sample_probs = np.zeros(self.num_arms)
        sample_probs[probs > 1e-6] = 1
        sample_probs = sample_probs / sample_probs.sum()
        samples = np.random.choice(self.num_arms, size=num_samples, p=sample_probs)

        wts = np.zeros(num_samples)

        # assign probs for samples based on arms
        for i in range(num_samples):
            wts[i] = probs[samples[i]]

        # convert weights to rollouts based on mathematics of apportionment
        S = self.knapsack.rollouts_per_task * num_samples   # total rollouts, e.g. 72
        total = sum(wts)
        if total <= 0:
            # set equal weights
            wts = np.ones(num_samples) / num_samples
        w = [p / total for p in wts]          # normalize
        quotas = [wi * S for wi in w]           # real quotas
        base = [int(q) for q in quotas]         # floors
        R = S - sum(base)                       # remaining rollouts to assign
        remainders = [(q - int(q), i) for i, q in enumerate(quotas)]
        remainders.sort(reverse=True)           # largest remainders first
        for k in range(R):
            base[remainders[k][1]] += 1
        self.prompt_to_n = {i: base[i] for i in range(num_samples)}


        ## compute average group n for self.group_to_n based on ids from samples
        self.group_to_n = {self.arm_to_group[i]: np.mean([base[j] for j in range(num_samples) if samples[j]==i]) for i in range(self.num_arms)}
        print(self.group_to_n, "group_to_n")
        
        return samples
        # if self.knapsack.type == 'vanilla':
        #     # return with uniform weights
        #     return np.random.choice(self.num_arms, size=num_samples, p=np.ones(self.num_arms) / self.num_arms)

    def sample_arms_dynamic_rollouts(self, num_samples: int, probs: np.ndarray) -> np.ndarray:

        # probs is all zeros and one 1, just sample from it
        if np.any(probs >= 1 - 1e-6):
            # find the index with prob >= 1 - 1e-6
            idx = np.where(probs >= 1 - 1e-6)[0][0]
            # get n for idx from group
            g = self.arm_to_group[idx]
            n = self.group_to_n[g]
            self.prompt_to_n = {i: n for i in range(num_samples)}
            return np.full(num_samples, idx)

        samples = np.random.choice(self.num_arms, size=num_samples, p=probs)
        out = []
        for a in samples:
            g = self.arm_to_group[a]
            gn = float(self.group_to_n[g])

            # factor f: <1 => thin, >1 => replicate
            f = self.dynamic_rollouts.rollouts_per_task / max(gn, 1e-12)

            if f >= 1.0:
                k = int(f)                       # floor
                out.extend([a] * k)
                if np.random.rand() < (f - k):   # fractional part
                    out.append(a)
            else:
                if np.random.rand() < f:         # keep with prob f
                    out.append(a)

        out = np.asarray(out, dtype=int)

        if len(out) == 0:
            out = samples

        self.prompt_to_n = {i: self.group_to_n[self.arm_to_group[out[i]]] for i in range(len(out))}
        return out

    def update_group_n(self):
        for g in self.group_to_n:
            base_n = self.group_to_n.get(g, 1)
            arm = self.group_to_arm[g]
            if 1 - self.filtered_ratio[arm] < self.dynamic_rollouts.filter_threshold:
                print("updating group n for group", g, "from", base_n, "to", base_n * 2)
                print("filtered ratio is lower than threshold", self.filtered_ratio[arm], "threshold", self.dynamic_rollouts.filter_threshold)
                mult = 2.0
            elif 1 - self.filtered_ratio[arm] > self.dynamic_rollouts.filter_threshold_upper:
                print("updating group n for group", g, "from", base_n, "to", base_n * 0.5)
                print("filtered ratio is higher than threshold", self.filtered_ratio[arm], "threshold", self.dynamic_rollouts.filter_threshold_upper)
                mult = 0.5
            else:
                mult = 1.0
            n = int(np.clip(base_n * mult, 1, self.dynamic_rollouts.max_rollouts_per_task))
            self.group_to_n[g] = n
        
        



    def get_metrics(self) -> Dict[str, float]:
        probs = self._softmax_probs().detach().cpu().numpy()
        out = {}
        for g, arm in self.group_to_arm.items():
            out[f"famo/{g}_prob"] = float(probs[arm])
            out[f"famo/{g}_w"] = float(self.w.detach().cpu().numpy()[arm])
            if self.decay_type == 'mask':
                out[f"famo/{g}_mask"] = float(self.mask[arm])
                out[f"famo/{g}_q"] = float(self.q_values[arm])
            if self.update_object == 'reward_gap':
                out[f"famo/{g}_reward_gap"] = float(self.last_reward[arm])
            if self.update_object == 'task_diff_bias':
                out[f"famo/{g}_reward_gap"] = float(self.last_reward[arm])
                out[f"famo/{g}_task_delta"] = float(self.task_delta[arm])
                out[f"famo/{g}_combined_delta"] = float(self.combined_delta[arm])

                out[f"famo/{g}_avg_task_wise_diff"] = float(self.avg_task_wise_diff[arm])
                out[f"famo/{g}_avg_task_wise_count"] = float(self.avg_task_wise_count[arm])
                out[f"famo/{g}_avg_task_wise_ess_full"] = float(self.avg_task_wise_ess_full[arm])
                out[f"famo/{g}_avg_task_wise_ess_frac"] = float(self.avg_task_wise_ess_frac[arm])
                out[f"famo/{g}_avg_task_wise_reward_diff"] = float(self.avg_task_wise_reward_diff[arm])

            if self.update_object == 'task_wise_diff':
                out[f"famo/{g}_task_delta"] = float(self.task_delta[arm])
            if self.update_object == 'reward_gap_bandit':
                out[f"famo/{g}_q"] = float(self.q_values[arm])
            if self.decay_type == 'q_conv_mask':
                out[f"famo/{g}_change_qs_mom"] = float(self.change_qs_mom[arm])
                out[f"famo/{g}_change_qs_mom_2"] = float(self.change_qs_mom_2[arm])
                out[f"famo/{g}_q_SNR"] = float(self.q_SNR[arm])
                out[f"famo/{g}_mask"] = float(self.mask[arm])
                out[f"famo/{g}_q"] = float(self.q_values[arm])
            if self.decay_type == 'task_diff_q_conv_mask':
                out[f"famo/{g}_q_mom_2"] = float(self.q_mom_2[arm])
                out[f"famo/{g}_q_SNR"] = float(self.q_SNR[arm])
                out[f"famo/{g}_mask"] = float(self.mask[arm])
                out[f"famo/{g}_q"] = float(self.q_values[arm])
            if self.decay_type == 'soft_task_diff_q_conv_mask':
                out[f"famo/{g}_softmask"] = float(self.softmask[arm])
                out[f"famo/{g}_q_values"] = float(self.q_values[arm])
            if self.decay_type == 'soft_q_conv_mask':
                out[f"famo/{g}_softmask"] = float(self.softmask[arm])
                out[f"famo/{g}_q_values"] = float(self.q_values[arm])
                out[f"famo/{g}_change_qs"] = float(self.change_qs[arm])
                out[f"famo/{g}_change_qs_mom"] = float(self.change_qs_mom[arm])
            if self.decay_type == 'var_conv_mask':
                out[f"famo/{g}_q_mom_2"] = float(self.q_mom_2[arm])
                out[f"famo/{g}_q_SNR"] = float(self.q_SNR[arm])
                out[f"famo/{g}_mask"] = float(self.mask[arm])
                out[f"famo/{g}_q"] = float(self.q_values[arm])
            if self.decay_type == 'soft_var_conv_mask':
                out[f"famo/{g}_softmask"] = float(self.softmask[arm])
                out[f"famo/{g}_q_values"] = float(self.q_values[arm])
            if self.decay_type == 'soft_opt_conv_mask':
                out[f"famo/{g}_softmask"] = float(self.softmask[arm])
                out[f"famo/{g}_q_values"] = float(self.q_values[arm])
                out[f"famo/{g}_zero_ratio"] = float(self.z_ratio[arm])
                out[f"famo/{g}_one_ratio"] = float(self.one_ratio[arm])
                out[f"famo/{g}_nznone_ratio"] = float(self.nznone_ratio[arm])
            if self.decay_type == 'coeff_var_conv_mask':
                out[f"famo/{g}_q_values"] = float(self.q_values[arm])
            if self.decay_type == 'soft_coeff_var_conv_mask':
                out[f"famo/{g}_softmask"] = float(self.softmask[arm])
                out[f"famo/{g}_q_values"] = float(self.q_values[arm])
            
            if self.decay_type == 'soft_half_opt_conv_mask':
                out[f"famo/{g}_softmask"] = float(self.softmask[arm])
                out[f"famo/{g}_q_values"] = float(self.q_values[arm])
                out[f"famo/{g}_optdist"] = float(self.optdist[arm])
                out[f"famo/{g}_zero_ratio"] = float(self.z_ratio[arm])
                out[f"famo/{g}_one_ratio"] = float(self.one_ratio[arm])
                out[f"famo/{g}_nznone_ratio"] = float(self.nznone_ratio[arm])
                out[f"famo/{g}_opt_ratio"] = float(self.opt_ratio[arm])
                for acc in self.acc_ratio[arm]:
                    out[f"famo/{g}_{acc}"] = float(self.acc_ratio[arm][acc])
                out[f"famo/{g}_solve_none_ratio"] = float(self.solve_none_ratio[arm])
                out[f"famo/{g}_solve_all_ratio"] = float(self.solve_all_ratio[arm])
                out[f"famo/{g}_batch_acc"] = float(self.batch_acc[arm])
                out[f"famo/{g}_batch_acc_sq"] = float(self.batch_acc_sq[arm])
                out[f"famo/{g}_batch_acc_var"] = float(self.batch_acc_var[arm])
                out[f"famo/{g}_batch_acc_delta"] = float(self.batch_acc_delta[arm])
                out[f"famo/{g}_var"] = float(self.var[arm])
                out[f"famo/{g}_solve_half_ratio"] = float(self.solve_half_ratio[arm])
                out[f"famo/{g}_filtered_ratio"] = float(self.filtered_ratio[arm])
            
            if self.knapsack.enabled:
                out[f"famo/{g}_rollouts"] = float(self.group_to_n[self.arm_to_group[arm]])
            
            if self.dynamic_rollouts.enabled:
                out[f"famo/{g}_rollouts"] = float(self.group_to_n[self.arm_to_group[arm]])
                
        return out
    
    def update_q_values(self, new_q_values: np.ndarray, arms_present: np.ndarray):
        self.q_values[arms_present] = self.decay_rate * self.q_values[arms_present] + (1-self.decay_rate) * new_q_values[arms_present]
    
    def update_optdist(self, new_optdist: np.ndarray, arms_present: np.ndarray):
        self.optdist[arms_present] = self.decay_rate * self.optdist[arms_present] + (1-self.decay_rate) * new_optdist[arms_present]
    
    def update_z_ratio(self, new_z_ratio: np.ndarray, arms_present: np.ndarray):
        self.z_ratio[arms_present] = self.z_decay_rate * self.z_ratio[arms_present] + (1-self.z_decay_rate) * new_z_ratio[arms_present]
    
    def update_one_ratio(self, new_one_ratio: np.ndarray, arms_present: np.ndarray):
        self.one_ratio[arms_present] = self.one_decay_rate * self.one_ratio[arms_present] + (1-self.one_decay_rate) * new_one_ratio[arms_present]

    def update_nznone_ratio(self, new_nznone_ratio: np.ndarray, arms_present: np.ndarray):
        self.nznone_ratio[arms_present] = self.nznone_decay_rate * self.nznone_ratio[arms_present] + (1-self.nznone_decay_rate) * new_nznone_ratio[arms_present]
    
    def update_opt_ratio(self, new_opt_ratio: np.ndarray, arms_present: np.ndarray):
        self.opt_ratio[arms_present] = self.opt_decay_rate * self.opt_ratio[arms_present] + (1-self.opt_decay_rate) * new_opt_ratio[arms_present]
    
    def update_acc_ratio(self, new_acc_ratio: defaultdict, arms_present: np.ndarray):
        for arm in np.where(arms_present)[0]:
            for acc in new_acc_ratio[arm]:
                self.acc_ratio[arm][acc] = self.acc_decay_rate * self.acc_ratio[arm][acc] + (1-self.acc_decay_rate) * new_acc_ratio[arm][acc]

    def update_solve_none(self, solve_none_ratio: defaultdict):
        for group in solve_none_ratio:
            arm = self.group_to_arm[group]
            self.solve_none_ratio[arm] = self.solve_none_decay_rate * self.solve_none_ratio[arm] + (1-self.solve_none_decay_rate) * solve_none_ratio[group]

    def update_solve_all(self, solve_all_ratio: defaultdict):
        for group in solve_all_ratio:
            arm = self.group_to_arm[group]
            self.solve_all_ratio[arm] = self.solve_all_decay_rate * self.solve_all_ratio[arm] + (1-self.solve_all_decay_rate) * solve_all_ratio[group]
    
    def update_batch_acc(self, batch_acc: defaultdict):
        a = self.batch_acc_decay_rate
        for group, acc in batch_acc.items():
            arm = self.group_to_arm[group]

            # previous EMAs
            prev_mean = self.batch_acc[arm]
            prev_sq   = self.batch_acc_sq[arm]

            # EMA of mean
            new_mean = a * prev_mean + (1 - a) * acc
            self.batch_acc[arm] = new_mean

            # EMA of squared value
            new_sq = a * prev_sq + (1 - a) * (acc * acc)
            self.batch_acc_sq[arm] = new_sq

            # moving variance of batch_acc over time
            batch_acc_var = new_sq - new_mean * new_mean
            self.batch_acc_var[arm] = max(batch_acc_var, 0.0)  # clamp for numerical safety
            #self.batch_acc[arm] = self.batch_acc_decay_rate * self.batch_acc[arm] + (1-self.batch_acc_decay_rate) * batch_acc[group]
    
    def update_batch_acc_delta(self, batch_acc_delta: defaultdict):
        for group, delta in batch_acc_delta.items():
            arm = self.group_to_arm[group]
            self.batch_acc_delta[arm] = self.batch_acc_delta_decay_rate * self.batch_acc_delta[arm] + (1-self.batch_acc_delta_decay_rate) * delta

    def update_var(self, var: defaultdict):
        for group in var:
            arm = self.group_to_arm[group]
            self.var[arm] = self.var_decay_rate * self.var[arm] + (1-self.var_decay_rate) * var[group]
    
    def update_solve_half(self, solve_half_ratio: defaultdict):
        for group in solve_half_ratio:
            arm = self.group_to_arm[group]
            self.solve_half_ratio[arm] = self.solve_half_decay_rate * self.solve_half_ratio[arm] + (1-self.solve_half_decay_rate) * solve_half_ratio[group]
    
    def update_filtered_ratio(self, filtered_ratio: defaultdict):
        for group in filtered_ratio:
            arm = self.group_to_arm[group]
            print("current batch filtered ratio for group", group, "is", filtered_ratio[group])
            old_filtered_ratio = self.filtered_ratio[arm]
            self.filtered_ratio[arm] = self.dynamic_rollouts.filtered_ratio_decay_rate * self.filtered_ratio[arm] + (1-self.dynamic_rollouts.filtered_ratio_decay_rate) * filtered_ratio[group]
            print("updating filtered ratio for group", group, "from", old_filtered_ratio, "to", self.filtered_ratio[arm])
    
    def get_inactive_tasks(self):
        inactive_tasks = set()
        for arm in range(self.num_arms):
            if not self.mask[arm]:
                inactive_tasks.add(self.arm_to_group[arm])
        return inactive_tasks
    
    def lambda_bias_adapt(self, step, total_steps, start=0.01, end=0.1, ramp_frac=0.9):
        T = max(1, int(total_steps * ramp_frac))
        t = min(step / T, 1.0)
        return start + (end - start) * t

    def combine(self, delta, lambda_bias, rewards, w_update_step):
        if w_update_step < self.SNR_update_step_threshold:
            return delta
            
        if self.lambda_bias_type == 'add':
            return delta + lambda_bias * rewards
        elif self.lambda_bias_type == 'threshold':
            for i in range(len(delta)):
                if rewards[i] > -1*self.bias_threshold:
                    delta[i] = self.conv_penalty
                else:
                    delta[i] = delta[i]
            return delta
        elif self.lambda_bias_type == 'bias_only':
            return lambda_bias * rewards            
        else:
            raise ValueError(f'Unknown combine_type: {self.combine_type}')
    
    def update_training_lr(self, lr):
        self.training_lr = lr

    # ---------- main update ----------
    @torch.no_grad()
    def update(self, *args, **kwargs):
        """
        group_to_new_loss: mapping {group_name: scalar_loss_value} for *this step*,
        where groups are the same keys as self.group_to_arm.

        This performs the FAMO-style weight update:
            delta = log(prev - min + eps) - log(new - min + eps)
        (or linear delta if use_log_ratio_delta=False),
        then autograd on softmax(w) with grad_outputs=delta to update w.
        """
        assert 'task_wise_diff' in kwargs, 'task_wise_diff is required for updating priority'
        assert 'task_wise_count' in kwargs, 'task_wise_count is required for updating decay'
        task_wise_diff = kwargs['task_wise_diff']
        task_wise_count = kwargs['task_wise_count']
        task_wise_ess_full = kwargs['task_wise_ess_full']
        task_wise_ess_frac = kwargs['task_wise_ess_frac']
        task_wise_reward_diff = kwargs['task_wise_reward_diff']


        # split keys based on batch id and group name
        #  for k, v in task_wise_diff.items():
        #               task_data[f'actor/task_wise_diff_{k}_{batch_idx}'] = v.detach().item()

        
        # group keys and values w.r.t. batch ids
        batch_to_task_wise_diff = defaultdict(dict)
        batch_to_task_wise_count = defaultdict(dict)
        batch_to_task_wise_ess_full = defaultdict(dict)
        batch_to_task_wise_ess_frac = defaultdict(dict)
        batch_to_task_wise_reward_diff = defaultdict(dict)
        for k, v in task_wise_diff.items():
            batch_id = k.split('_')[-1]
            group_name = k.split('_')[-2]
            batch_to_task_wise_diff[batch_id][group_name] = v
            k_count = f"actor/task_wise_count_{group_name}_{batch_id}"
            batch_to_task_wise_count[batch_id][group_name] = task_wise_count[k_count]
            k_ess_full = f"actor/task_wise_ess_full_{group_name}_{batch_id}"
            batch_to_task_wise_ess_full[batch_id][group_name] = task_wise_ess_full[k_ess_full]
            k_ess_frac = f"actor/task_wise_ess_frac_{group_name}_{batch_id}"
            batch_to_task_wise_ess_frac[batch_id][group_name] = task_wise_ess_frac[k_ess_frac]
            k_reward_diff = f"actor/task_wise_reward_diff_{group_name}_{batch_id}"
            batch_to_task_wise_reward_diff[batch_id][group_name] = task_wise_reward_diff[k_reward_diff]

        avg_task_wise_diff = defaultdict(float)
        avg_task_wise_count = defaultdict(float)
        avg_task_wise_ess_full = defaultdict(float)
        avg_task_wise_ess_frac = defaultdict(float)
        avg_task_wise_reward_diff = defaultdict(float)
        # compute and log averages across batch_id
        for batch_id in batch_to_task_wise_diff:
            for group_name in batch_to_task_wise_diff[batch_id]:
                avg_task_wise_diff[group_name] += batch_to_task_wise_diff[batch_id][group_name]
                avg_task_wise_count[group_name] += batch_to_task_wise_count[batch_id][group_name]
                avg_task_wise_ess_full[group_name] += batch_to_task_wise_ess_full[batch_id][group_name]
                avg_task_wise_ess_frac[group_name] += batch_to_task_wise_ess_frac[batch_id][group_name]
                avg_task_wise_reward_diff[group_name] += batch_to_task_wise_reward_diff[batch_id][group_name]
        
        for group_name in avg_task_wise_diff:
            avg_task_wise_diff[group_name] /= len(batch_to_task_wise_diff)
            avg_task_wise_count[group_name] /= len(batch_to_task_wise_count)
            avg_task_wise_ess_full[group_name] /= len(batch_to_task_wise_ess_full)
            avg_task_wise_ess_frac[group_name] /= len(batch_to_task_wise_ess_frac)
            avg_task_wise_reward_diff[group_name] /= len(batch_to_task_wise_reward_diff)
        
        # log averages
        for group_name in avg_task_wise_diff:
            self.avg_task_wise_diff[self.group_to_arm[group_name]] = avg_task_wise_diff[group_name]
            self.avg_task_wise_count[self.group_to_arm[group_name]] = avg_task_wise_count[group_name]
            self.avg_task_wise_ess_full[self.group_to_arm[group_name]] = avg_task_wise_ess_full[group_name]
            self.avg_task_wise_ess_frac[self.group_to_arm[group_name]] = avg_task_wise_ess_frac[group_name]
            self.avg_task_wise_reward_diff[self.group_to_arm[group_name]] = avg_task_wise_reward_diff[group_name]
            

        if self.use_reward_diff:
            batch_to_task_wise_diff = batch_to_task_wise_reward_diff


        # ------ mask logic that toggles based on q_values ------
        prev_mask = self.mask.clone()  # NEW: remember mask before we change it

        
        decay = get_decay(self.decay_type)
        decay.update(self, batch_to_task_wise_diff, batch_to_task_wise_count, kwargs)
        
        if self.update_object == 'reward_gap':

            assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
            assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'

            
            data_info = kwargs['data_info']
            
            
            obj = kwargs['rewards']
            index = kwargs['index']
            non_zero_mask = (obj != 0)
            obj = (obj * non_zero_mask).sum(-1)

            id_to_group = {}
            id2obj = defaultdict(list)

            id2nzmean = {}
            
            for data, obj_val,idx in zip(data_info, obj,index):
                group = self.extra_info_to_group(data)
                id_to_group[idx] = group
                id2obj[idx].append(obj_val.item())

            for idx in id2obj:
                # check if any element in id2obj[idx] is 1
                if len(id2obj[idx]) == 1:
                    if id2obj[idx][0] == 1:
                        id2nzmean[idx] = 1.0
                elif len(id2obj[idx]) > 1:
                    if any([obj_val >= 1 - 1e-6 for obj_val in id2obj[idx]]):
                        id2nzmean[idx] = np.mean([obj_val for obj_val in id2obj[idx]])
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            
            group_to_obj = defaultdict(list)
            for idx in id2nzmean.keys(): 
                group = id_to_group[idx]
                group_to_obj[group].append(id2nzmean[idx])
            
            for group, obj_vals in group_to_obj.items():
                group_to_obj[group] = np.mean(obj_vals)


            
            new_rs = np.zeros(self.num_arms)
            arms_present = np.array(self.num_arms*[False])
            for group, obj_vals in group_to_obj.items(): 
                new_rs[self.group_to_arm[group]] = obj_vals - 1 
                arms_present[self.group_to_arm[group]] = True

            rewards_by_arm = [new_rs[arm] for arm in range(self.num_arms)]

            self.last_reward = np.array(rewards_by_arm)

            delta = torch.tensor(rewards_by_arm)            

            # ------ handle mask transitions BEFORE computing grads ------
            # indices that just became active (masked False -> True)
            # recall: in your convention, True == active, False == masked
            unmask_idx = (self.mask & ~prev_mask)  # NEW
            if unmask_idx.any():
                _reset_opt_state_indices(self.opt, self.w, unmask_idx, scale=0.0)  # NEW: clear stale momentum/adam

            # update caches & trackers
            # autograd on softmax(w) with grad_outputs=delta
            self.opt.zero_grad(set_to_none=True)


            with torch.enable_grad():
                #probs = F.softmax(self.w, dim=-1)
                probs = fixed_masked_softmax(self.w, self.mask, fixed_p_each=self.fixed_p_each, dim=-1)
                # d softmax / d w dotted with delta: use autograd to get grad wrt w
                d = torch.autograd.grad(outputs=probs, inputs=self.w, grad_outputs=delta.detach())[0]
            # SGD step: w <- w - lr * d
            self.w.grad = d
            self.opt.step()

            # ------ re-impose calibrated value on currently masked entries (post-step) ------
            if (~self.mask).any():
                y = _compute_masked_logit_y(self.w, self.mask, self.fixed_p_each)  # scalar
                self.w[~self.mask] = y  # keeps them fixed and prevents AdamW drift

            print(self.w,"w")
            print(self.last_reward,"last_reward")




        elif self.update_object == 'task_wise_diff':

            self.task_delta = np.zeros(self.num_arms)

            # update priority for each batch
            for batch_id, task_wise_diff in batch_to_task_wise_diff.items():
                # get sorted keys 
                values_by_arm = [ task_wise_diff.get(self.arm_to_group[arm], 0) for arm in range(self.num_arms) ]

                delta = torch.tensor(values_by_arm)

                if not self.max_rate:
                    delta = -delta
                
                # clip delta to be in [-0.1,0.1]
                delta = torch.clamp(delta, -0.1, 0.1)
                
                # ------ handle mask transitions BEFORE computing grads ------
                # indices that just became active (masked False -> True)
                # recall: in your convention, True == active, False == masked
                unmask_idx = (self.mask & ~prev_mask)  # NEW
                if unmask_idx.any():
                    _reset_opt_state_indices(self.opt, self.w, unmask_idx, scale=0.0)  # NEW: clear stale momentum/adam

                # update caches & trackers
                # autograd on softmax(w) with grad_outputs=delta
                self.opt.zero_grad(set_to_none=True)


                with torch.enable_grad():
                    #probs = F.softmax(self.w, dim=-1)
                    # if 'soft' in self.decay_type:
                    #    probs = gated_softmax(self.w, self.softmask, fixed_p_each=self.fixed_p_each, drim=-1)
                    # else:
                    probs = fixed_masked_softmax(self.w, self.mask, fixed_p_each=self.fixed_p_each, dim=-1) # keep updates independent of softmask
                    # d softmax / d w dotted with delta: use autograd to get grad wrt w
                    d = torch.autograd.grad(outputs=probs, inputs=self.w, grad_outputs=delta.detach())[0]
                # SGD step: w <- w - lr * d
                self.w.grad = d
                self.opt.step()

                # ------ re-impose calibrated value on currently masked entries (post-step) ------
                if (~self.mask).any():
                    y = _compute_masked_logit_y(self.w, self.mask, self.fixed_p_each)  # scalar
                    self.w[~self.mask] = y  # keeps them fixed and prevents AdamW drift
                
                self.task_delta += np.asarray(delta.detach().cpu().numpy())
                
                print(self.w,"w")
                print(self.mask,"mask")
                if isinstance(self.decay_type, str) and 'soft' in self.decay_type:
                    print(np.log(self.softmask),"log-softmask")
            
            self.task_delta = self.task_delta/(int(batch_id)+1)

        elif self.update_object == 'task_diff_bias':

            self.w_update_step += 1

            assert self.decay_type == 'soft_opt_conv_mask' or self.decay_type == 'soft_half_opt_conv_mask', 'decay_type must be soft_opt_conv_mask or soft_half_opt_conv_mask'
            # if self.lambda_bias_type == 'add':
            # assert self.decay_rate == 0.0, 'decay_rate must be 0.0'

            if self.decay_type == 'soft_opt_conv_mask':
                rewards_by_arm = [self.q_values[arm]-1 for arm in range(self.num_arms)]
            elif self.decay_type == 'soft_half_opt_conv_mask':
                if self.bias_metric == 'opt_ratio':
                    rewards_by_arm = [-self.opt_ratio[arm] for arm in range(self.num_arms)]
                elif self.bias_metric == 'solve_none':
                    rewards_by_arm = [-self.solve_none_ratio[arm] for arm in range(self.num_arms)]
                elif self.bias_metric == 'solve_all':
                    rewards_by_arm = [self.solve_all_ratio[arm] for arm in range(self.num_arms)]
                elif self.bias_metric == 'reward_gap':
                    rewards_by_arm = [self.q_values[arm]-self.opt_threshold for arm in range(self.num_arms)]
                elif self.bias_metric == 'batch_acc':
                    rewards_by_arm = [self.batch_acc[arm] for arm in range(self.num_arms)]
                elif self.bias_metric == 'solve_half':
                    rewards_by_arm = [self.solve_half_ratio[arm] for arm in range(self.num_arms)]
                elif self.bias_metric == 'optdist':
                    rewards_by_arm = [self.optdist[arm] for arm in range(self.num_arms)]
                elif self.bias_metric == 'var':
                    rewards_by_arm = [-np.sqrt(self.var[arm]/0.25) for arm in range(self.num_arms)]
                elif self.bias_metric == 'batch_acc_var':
                    beta = self.batch_acc_decay_rate
                    N_eff = (1+beta)/(1-beta)
                    rewards_by_arm = [-np.sqrt(self.batch_acc_var[arm]/N_eff) for arm in range(self.num_arms)]
                else:
                    raise ValueError('bias_metric must be opt_ratio or solve_none or solve_all or reward_gap')

            self.last_reward = np.array(rewards_by_arm)

            self.softmask = np.ones(self.num_arms)

            self.task_delta = np.zeros(self.num_arms)
            self.combined_delta = np.zeros(self.num_arms)

            # update priority for each batch
            for batch_id, task_wise_diff in batch_to_task_wise_diff.items():
                # get sorted keys 
                values_by_arm = [ task_wise_diff.get(self.arm_to_group[arm], 0) for arm in range(self.num_arms) ]

                delta = torch.tensor(values_by_arm)

                delta = self.calculate_delta(delta,batch_to_task_wise_count[batch_id],threshold=self.delta_threshold)

                self.task_delta += np.asarray(delta.detach().cpu().numpy())

                if self.lambda_bias_adaptive:
                    lambda_bias = self.lambda_bias_adapt(self.w_update_step,self.total_steps,self.lambda_bias_start,self.lambda_bias_end,self.lambda_bias_ramp_frac)
                else:
                    lambda_bias = self.lambda_bias
                delta = self.combine(delta, lambda_bias, torch.tensor(rewards_by_arm), self.w_update_step)  

               
                
                # ------ handle mask transitions BEFORE computing grads ------
                # indices that just became active (masked False -> True)
                # recall: in your convention, True == active, False == masked
                unmask_idx = (self.mask & ~prev_mask)  # NEW
                if unmask_idx.any():
                    _reset_opt_state_indices(self.opt, self.w, unmask_idx, scale=0.0)  # NEW: clear stale momentum/adam

                # update caches & trackers
                # autograd on softmax(w) with grad_outputs=delta
                self.opt.zero_grad(set_to_none=True)


                with torch.enable_grad():
                    #probs = F.softmax(self.w, dim=-1)
                    # if 'soft' in self.decay_type:
                    #    probs = gated_softmax(self.w, self.softmask, fixed_p_each=self.fixed_p_each, dim=-1)
                    # else:
                    probs = fixed_masked_softmax(self.w, self.mask, fixed_p_each=self.fixed_p_each, dim=-1) # keep updates independent of softmask
                    # d softmax / d w dotted with delta: use autograd to get grad wrt w
                    d = torch.autograd.grad(outputs=probs, inputs=self.w, grad_outputs=delta.detach())[0]
                # SGD step: w <- w - lr * d
                self.w.grad = d
                self.opt.step()

                # ------ re-impose calibrated value on currently masked entries (post-step) ------
                if (~self.mask).any():
                    y = _compute_masked_logit_y(self.w, self.mask, self.fixed_p_each)  # scalar
                    self.w[~self.mask] = y  # keeps them fixed and prevents AdamW drift

                self.combined_delta += np.asarray(delta.detach().cpu().numpy())# convert torch tensor to numpy
                
                print(self.last_reward,"last_reward")
            
            self.combined_delta = self.combined_delta/(int(batch_id)+1)
            self.task_delta = self.task_delta/(int(batch_id)+1)

        
        elif self.update_object == 'reward_gap_bandit':

            assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
            assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'

            
            data_info = kwargs['data_info']
            
            
            obj = kwargs['rewards']
            index = kwargs['index']
            non_zero_mask = (obj != 0)
            obj = (obj * non_zero_mask).sum(-1)

            id_to_group = {}
            id2obj = defaultdict(list)

            id2nzmean = {}
            
            for data, obj_val,idx in zip(data_info, obj,index):
                group = self.extra_info_to_group(data)
                id_to_group[idx] = group
                id2obj[idx].append(obj_val.item())

            for idx in id2obj:
                # check if any element in id2obj[idx] is 1
                if len(id2obj[idx]) == 1:
                    if id2obj[idx][0] == 1:
                        id2nzmean[idx] = 1.0
                elif len(id2obj[idx]) > 1:
                    if any([obj_val >= 1 - 1e-6 for obj_val in id2obj[idx]]):
                        id2nzmean[idx] = np.mean([obj_val for obj_val in id2obj[idx]])
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            
            group_to_obj = defaultdict(list)
            for idx in id2nzmean.keys(): 
                group = id_to_group[idx]
                group_to_obj[group].append(id2nzmean[idx])
            
            for group, obj_vals in group_to_obj.items():
                group_to_obj[group] = np.mean(obj_vals)


            
            new_qs = np.zeros(self.num_arms)
            arms_present = np.array(self.num_arms*[False])
            for group, obj_vals in group_to_obj.items(): 
                new_qs[self.group_to_arm[group]] = obj_vals 
                arms_present[self.group_to_arm[group]] = True
            self.update_q_values(new_qs,arms_present)
                    
            print(np.std(self.q_values),"std before clip")

            q_std = self.sharpness #* np.clip(np.std(self.q_values),1e-6,1.0)
        

            self.w = torch.from_numpy((-self.q_values + 1) / q_std).to(self.w.dtype)
            

            print(self.q_values,"q_values")
            print(self.w,"w")
            print(q_std,"q_std")
            print(self.arm_to_group,"arm_to_group")
            print(self.group_to_arm,"group_to_arm")

        else:
            raise ValueError(f'Invalid update object: {self.update_object}')



        self._probs_cache = self._softmax_probs().detach().cpu().numpy()
    
    def calculate_delta(self,delta,batch_to_task_wise_count,threshold=None):
        if self.use_acc:
            delta = torch.tensor(self.batch_acc_delta)
        if not self.max_rate:
            delta = -delta
        
        if self.average_delta:
            print(delta,"delta before average")
            task_wise_count = batch_to_task_wise_count
            counts_by_arm = [task_wise_count.get(self.arm_to_group[arm], 0) for arm in range(self.num_arms)]
            counts_by_arm = torch.tensor(counts_by_arm)
            print(counts_by_arm,"counts_by_arm")
            # avoind 0/0
            counts_by_arm[counts_by_arm==0] = 1
            delta = delta/counts_by_arm
            print(delta,"delta after average")
        
        # set threshold based on situation
        if self.use_acc:
            threshold = 0.2
        elif self.average_delta:
            threshold = 0.001
        else:
            if threshold is None:
                threshold = 0.1
            else:
                threshold = threshold
        

        # clip delta to be in [-threshold,threshold]
        # make it a function of learning rate if it is 1e-6 set it to 0.1 for other take a factor of 1e-6 and multuplut to 0.1
        delta = torch.clamp(delta, -threshold * (self.training_lr/1e-6), threshold * (self.training_lr/1e-6))
    
        return delta
        

        
        


class JointFamoBanditDecay(FamoPriorityDecay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('JointFamoBanditDecay initialized')


    def extra_info_to_group(self, extra_info):
        return '-'.join([str(extra_info[feature]) for feature in self.feature])

    def data_to_group(self, data):
        if 'difficulty' in data['extra_info']:
            assert int(data['extra_info']['difficulty']) == float(data['extra_info']['difficulty']) # make sure difficulty is an integer
        return self.extra_info_to_group(data['extra_info'])
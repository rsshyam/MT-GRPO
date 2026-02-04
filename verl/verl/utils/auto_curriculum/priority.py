from typing import Any
import numpy as np
from collections import defaultdict
import random
import torch
import copy
import verl.utils.torch_functional as verl_F

from verl.utils.auto_curriculum.priority_decay import JointFamoBanditDecay


import numpy as np
from typing import Any, Dict, List
import torch.nn.functional as F

from ray.util import pdb 

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
    if num_masked == 0:
        # no masked arms: standard softmax
        return torch.softmax(logits, dim=dim)

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
        if enable == 'decay':
            return JointFamoBanditDecay(*args, **kwargs)
        elif enable:
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

class BanditPriority(BasePriority):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.groups_to_train = kwargs.get('groups_to_train', None)

        self.temperature = kwargs.get('temperature', 1.0)
        self.learning_rate = kwargs.get('lr', 0.1)
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.method = kwargs.get('method', 'boltzmann')
        self.objective = kwargs.get('objective', 'adv')
        self.feature = kwargs.get('feature', ['difficulty'])

        if type(self.feature) == str:
            self.feature = [self.feature]


        assert 'dataset' in kwargs, 'dataset is required for initializing bandit priority'
        
        self.init_bandit(kwargs['dataset'])

        self.q_values = np.zeros(self.num_arms)
    
    def __call__(self, data: dict[str, Any]):
        return random.random()



    def get_metrics(self):
        q_vals = {f'bandit/{group}_q_value': self.q_values[self.group_to_arm[group]] for group in self.group_to_arm}
        probs = {f'bandit/{group}_prob': self.get_probs(self.method)[self.group_to_arm[group]] for group in self.group_to_arm}
        return {**q_vals, **probs}



    def init_bandit(self, dataset):
        if self.groups_to_train is not None:
            dataset = [item for item in dataset if self.data_to_group(item) in self.groups_to_train]
            print(len(dataset))

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

        if self.objective == 'progress':
            self.group_to_last_reward = {group: 0 for group in self.group_to_idx.keys()}

    def get_probs(self, method: str):
        if method == 'boltzmann':
            return np.exp(self.q_values / self.temperature) / np.sum(np.exp(self.q_values / self.temperature))
        elif method == 'normalized_q':
            assert all(self.q_values >= 0), 'q_values should be non-negative for normalized_q method'
            clip_q_values = np.maximum(self.q_values, 0.01)
            probs = clip_q_values / np.sum(clip_q_values)
            return probs
        elif method == 'greedy':
            return np.exp(self.q_values / self.temperature) / np.sum(np.exp(self.q_values / self.temperature))
        else:
            raise ValueError(f'Invalid sample method: {method}')


    def sample_arms(self, num_samples: int):
        if self.method == 'boltzmann':
            probs = self.get_probs('boltzmann')
            return np.random.choice(self.num_arms, size=num_samples, p=probs)
        elif self.method == 'normalized_q':
            probs = self.get_probs('normalized_q')
            return np.random.choice(self.num_arms, size=num_samples, p=probs)
        elif self.method == 'greedy':
            arms = []
            for i in range(num_samples):
                if random.random() < self.epsilon:
                    max_arm = np.argmax(self.q_values)
                    arm_to_sample = random.randint(0, self.num_arms - 1)
                    while arm_to_sample == max_arm:
                        arm_to_sample = random.randint(0, self.num_arms - 1)
                    arms.append(arm_to_sample)
                else:
                    arms.append(np.argmax(self.q_values))
            return np.array(arms)
        else:
            raise ValueError(f'Invalid sample method: {self.method}')

    def update(self, *args, **kwargs):
        assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
        assert 'adv' in kwargs, 'adv is required for updating bandit priority'

        response_mask = kwargs['response_mask']


        data_info = kwargs['data_info']
        if self.objective in ['adv', 'unnormalized_adv']:
            if self.objective == 'adv':
                obj = kwargs['adv']
            else:
                obj = kwargs['unnormalized_adv']

            obj = verl_F.masked_mean(obj.abs(), response_mask, axis=-1)

            group_to_obj = defaultdict(list)
            for data, obj_val in zip(data_info, obj):
                group = self.extra_info_to_group(data)
                group_to_obj[group].append(obj_val.item())
            for group, obj_vals in group_to_obj.items():
                group_to_obj[group] = np.mean(np.abs(obj_vals)) # New Q-value is the mean of the absolute advantages


        elif self.objective == 'progress':
            assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'
            rewards = kwargs['rewards']
            sequence_reward = rewards.sum(-1)
            group_to_reward = defaultdict(list)
            for data, reward in zip(data_info, sequence_reward):
                group = self.extra_info_to_group(data)
                group_to_reward[group].append(reward.item())

            
            group_to_obj = {}
            for group, rewards in self.group_to_last_reward.items():
                if group in group_to_reward:
                    group_to_reward[group] = np.mean(group_to_reward[group])
                else:
                    group_to_reward[group] = rewards
                group_to_obj[group] = group_to_reward[group] - self.group_to_last_reward[group]
                self.group_to_last_reward[group] = group_to_reward[group]
            
        else:
            raise ValueError(f'Invalid objective: {self.objective}')
        
        new_qs = np.zeros(self.num_arms)
        for group, obj_vals in group_to_obj.items(): 
            new_qs[self.group_to_arm[group]] = obj_vals 
        self.update_q_values(new_qs)

    def update_q_values(self, new_q_values: np.ndarray):
        self.q_values = (1 - self.learning_rate) * self.q_values + self.learning_rate * new_q_values


class FamoPriorityNew(BasePriority):
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

        self.change_decay_rate = kwargs.get('change_decay_rate', 0.5)
        self.change_decay_rate_2 = kwargs.get('change_decay_rate_2', 0.5)
        self.SNR_threshold = kwargs.get('SNR_threshold', 0.5)
        self.SNR_eps = kwargs.get('SNR_eps', 1e-8)

        self.SNR_update_step_threshold = kwargs.get('SNR_update_step_threshold', 50)

        self.sharpness = kwargs.get('sharpness', 1.0)

        self.index = kwargs.get('index', None)

        self.mean_diff_decay = kwargs.get('mean_diff_decay', False)


        if isinstance(self.feature, str):
            self.feature = [self.feature]

       

        assert 'dataset' in kwargs, 'dataset is required for initializing bandit priority'
        
        self.init_bandit(kwargs['dataset'])

        # ---- state: weights, prev & min losses (torch on device) ----
        self.w = torch.zeros(self.num_arms, requires_grad=True)
        self.opt = torch.optim.Adam([self.w], lr=self.learning_rate, weight_decay=self.gamma)
        # initialize probs cache

        self._probs_cache = self._softmax_probs().detach().cpu().numpy()
        
    def init_bandit(self, dataset):

        self.idx_to_group = {idx: self.data_to_group(item) for idx, item in enumerate(dataset)}
        self.group_to_idx = {group: [] for group in self.idx_to_group.values()}
        for idx, group in self.idx_to_group.items():
            self.group_to_idx[group].append(idx)
        self.arm_to_group = {i: key for i, key in enumerate(self.group_to_idx.keys())}
        self.group_to_arm = {key: i for i, key in enumerate(self.group_to_idx.keys())}
        self.num_arms = len(self.group_to_idx)

        self.arm_to_idx = {i: [] for i in range(self.num_arms)}
        for group, arm in self.group_to_arm.items():
            self.arm_to_idx[arm].extend(self.group_to_idx[group])

        self.mask = torch.ones(self.num_arms, dtype=torch.bool)
        self.q_values = np.zeros(self.num_arms)
        if self.update_object == 'reward':
            self.last_reward = np.zeros(self.num_arms)
        
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
            

    def extra_info_to_group(self, extra_info):
        return '-'.join([str(extra_info[feature]) for feature in self.feature])

    def data_to_group(self, data):
        if 'difficulty' in data['extra_info']:
            assert int(data['extra_info']['difficulty']) == float(data['extra_info']['difficulty']) # make sure difficulty is an integer
        return self.extra_info_to_group(data['extra_info'])

    def _softmax_probs(self) -> torch.Tensor:
        # keep the name but route to the masked-fixed version
        if 'soft' in self.decay_type:
            return gated_softmax(self.w, self.softmask, fixed_p_each=self.fixed_p_each, dim=-1)
        return fixed_masked_softmax(self.w, self.mask, fixed_p_each=self.fixed_p_each, dim=-1)

    def sample_arms(self, num_samples: int) -> np.ndarray:
        probs = self._softmax_probs().detach().cpu().numpy()
        return np.random.choice(self.num_arms, size=num_samples, p=probs)

    def get_metrics(self) -> Dict[str, float]:
        probs = self._softmax_probs().detach().cpu().numpy()
        out = {}
        for g, arm in self.group_to_arm.items():
            out[f"famo/{g}_prob"] = float(probs[arm])
            out[f"famo/{g}_w"] = float(self.w.detach().cpu().numpy()[arm])
            if self.decay_type == 'mask':
                out[f"famo/{g}_mask"] = float(self.mask[arm])
                out[f"famo/{g}_q"] = float(self.q_values[arm])
            if self.update_object == 'reward':
                out[f"famo/{g}_reward"] = float(self.last_reward[arm])
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
        return out
    
    def update_q_values(self, new_q_values: np.ndarray, arms_present: np.ndarray):
        self.q_values[arms_present] = self.decay_rate * self.q_values[arms_present] + (1-self.decay_rate) * new_q_values[arms_present]
    
    def get_inactive_tasks(self):
        inactive_tasks = set()
        for arm in range(self.num_arms):
            if not self.mask[arm]:
                inactive_tasks.add(self.arm_to_group[arm])
        return inactive_tasks

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


        # split keys based on batch id and group name
        #  for k, v in task_wise_diff.items():
        #               task_data[f'actor/task_wise_diff_{k}_{batch_idx}'] = v.detach().item()

        
        # group keys and values w.r.t. batch ids
        batch_to_task_wise_diff = defaultdict(dict)
        batch_to_task_wise_count = defaultdict(dict)
        for k, v in task_wise_diff.items():
            batch_id = k.split('_')[-1]
            group_name = k.split('_')[-2]
            batch_to_task_wise_diff[batch_id][group_name] = v
            k_count = f"actor/task_wise_count_{group_name}_{batch_id}"
            batch_to_task_wise_count[batch_id][group_name] = task_wise_count[k_count]

        # ------ mask logic that toggles based on q_values ------
        prev_mask = self.mask.clone()  # NEW: remember mask before we change it

        if self.decay_type == 'mask':
            assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
            assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'

            response_mask = kwargs['response_mask']

            data_info = kwargs['data_info']
          
            obj = kwargs['rewards'] #B*T
            non_zero_mask = (obj != 0)
            obj = (obj * non_zero_mask).sum(-1) #B

            group_to_obj = defaultdict(list)
            for data, obj_val in zip(data_info, obj):
                group = self.extra_info_to_group(data)
                group_to_obj[group].append(obj_val.item())
            for group, obj_vals in group_to_obj.items():
                group_to_obj[group] = np.mean(np.abs(obj_vals)) # New Q-value is the mean of the absolute advantages

            
            new_qs = np.zeros(self.num_arms)
            arms_present = np.array(self.num_arms*[False])
            for group, obj_vals in group_to_obj.items(): 
                new_qs[self.group_to_arm[group]] = obj_vals 
                arms_present[self.group_to_arm[group]] = True
            self.update_q_values(new_qs,arms_present)
            # update only if group is present in group_to_obj


            # set self.mask to zero if self.q_values is greater than threshold 0.8
            for arm, q_val in enumerate(self.q_values):
                if q_val > self.reward_threshold:
                    self.mask[arm] = False
                else:
                    self.mask[arm] = True
        elif self.decay_type == 'q_conv_mask':
            assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
            assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'

            response_mask = kwargs['response_mask']

            data_info = kwargs['data_info']
          
            obj = kwargs['rewards']
            non_zero_mask = (obj != 0)
            obj = (obj * non_zero_mask).sum(-1)

            group_to_obj = defaultdict(list)
            for data, obj_val in zip(data_info, obj):
                group = self.extra_info_to_group(data)
                group_to_obj[group].append(obj_val.item())
            for group, obj_vals in group_to_obj.items():
                group_to_obj[group] = np.mean(np.abs(obj_vals)) # New Q-value is the mean of the absolute advantages

            
            new_qs = np.zeros(self.num_arms)
            arms_present = np.array(self.num_arms*[False])
            for group, obj_vals in group_to_obj.items(): 
                new_qs[self.group_to_arm[group]] = obj_vals 
                arms_present[self.group_to_arm[group]] = True
            old_qs = self.q_values.copy()
            self.update_q_values(new_qs,arms_present)
            # update only if group is present in group_to_obj
            new_qs = self.q_values.copy()
            
            self.change_qs[arms_present] = new_qs[arms_present] - old_qs[arms_present]
            
            self.change_qs_mom[arms_present] = self.change_qs[arms_present] * (1-self.change_decay_rate) + self.change_decay_rate * self.change_qs_mom[arms_present]
            self.change_qs_mom_2[arms_present] = (self.change_qs[arms_present]) * (self.change_qs[arms_present]) * (1-self.change_decay_rate_2) + self.change_decay_rate_2 * self.change_qs_mom_2[arms_present]
            self.q_update_step[arms_present] += 1

            change_qs_mom_hat = np.zeros(self.num_arms)
            change_qs_mom_hat[arms_present] = self.change_qs_mom[arms_present] / (1-self.change_decay_rate**self.q_update_step[arms_present])
            change_qs_mom_2_hat = np.zeros(self.num_arms)
            change_qs_mom_2_hat[arms_present] = self.change_qs_mom_2[arms_present] / (1-self.change_decay_rate_2**self.q_update_step[arms_present])

            var_qs_hat = np.maximum(change_qs_mom_2_hat - change_qs_mom_hat * change_qs_mom_hat, 0)
            
            self.q_SNR[arms_present] = change_qs_mom_hat[arms_present] / (np.sqrt(var_qs_hat[arms_present]) + self.SNR_eps)

            for arm, q_SNR_val in enumerate(self.q_SNR):
                if q_SNR_val < self.SNR_threshold and self.q_update_step[arm] > self.SNR_update_step_threshold:
                    self.mask[arm] = False    
                else:
                    self.mask[arm] = True

        elif self.decay_type == 'soft_q_conv_mask':
            assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
            assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'

            response_mask = kwargs['response_mask']

            data_info = kwargs['data_info']
          
            obj = kwargs['rewards']
            non_zero_mask = (obj != 0)
            obj = (obj * non_zero_mask).sum(-1)

            group_to_obj = defaultdict(list)
            for data, obj_val in zip(data_info, obj):
                group = self.extra_info_to_group(data)
                group_to_obj[group].append(obj_val.item())
            for group, obj_vals in group_to_obj.items():
                group_to_obj[group] = np.mean(np.abs(obj_vals)) # New Q-value is the mean of the absolute advantages

            
            new_qs = np.zeros(self.num_arms)
            arms_present = np.array(self.num_arms*[False])
            for group, obj_vals in group_to_obj.items(): 
                new_qs[self.group_to_arm[group]] = obj_vals 
                arms_present[self.group_to_arm[group]] = True
            old_qs = self.q_values.copy()
            self.update_q_values(new_qs,arms_present)
            # update only if group is present in group_to_obj
            new_qs = self.q_values.copy()
            
            self.change_qs[arms_present] = new_qs[arms_present] - old_qs[arms_present]
            
            self.change_qs_mom[arms_present] = self.change_qs[arms_present] * (1-self.change_decay_rate) + self.change_decay_rate * self.change_qs_mom[arms_present]
            self.q_update_step[arms_present] += 1

            change_q_std = self.sharpness * np.clip(np.std(self.change_qs_mom),1e-6,1e-2)
            
            # softmask: before threshold = 1, after threshold = sigmoid(q/τ)
            step_mask = self.q_update_step > self.SNR_update_step_threshold
            self.softmask[step_mask] = 1 / (1 + np.exp(-self.change_qs_mom[step_mask] / change_q_std))

            print(self.q_values,"q_values")
            print(self.change_qs,"change_qs")
            print(self.change_qs_mom,"change_qs_mom")
            print(self.softmask,"softmask")
            print(change_q_std,"change_q_std")
            print(self.arm_to_group,"arm_to_group")
            print(self.group_to_arm,"group_to_arm")

        elif self.decay_type == 'var_conv_mask':
            assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
            assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'

            response_mask = kwargs['response_mask']

            data_info = kwargs['data_info']
          
            
            obj = kwargs['rewards']
            index = kwargs['index']
            non_zero_mask = (obj != 0)
            obj = (obj * non_zero_mask).sum(-1)

            group_to_obj = defaultdict(list)
            id2obj = defaultdict(list)

            id2std = {}
            
            for data, obj_val,idx in zip(data_info, obj,index):
                group = self.extra_info_to_group(data)
                group_to_obj[group].append(idx)
                id2obj[idx].append(obj_val.item())
            for idx in id2obj:
                if len(id2obj[idx]) == 1:
                    id2std[idx] = 1.0
                elif len(id2obj[idx]) > 1:
                    id2std[idx] = np.std(id2obj[idx])
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            for group, idxs in group_to_obj.items():
                group_to_obj[group] = np.mean(np.abs([id2std[idx] for idx in idxs])) # New Q-value is the mean of the variances

            
            new_qs = np.zeros(self.num_arms)
            arms_present = np.array(self.num_arms*[False])
            for group, obj_vals in group_to_obj.items(): 
                new_qs[self.group_to_arm[group]] = obj_vals 
                arms_present[self.group_to_arm[group]] = True
            old_qs = self.q_values.copy()
            self.update_q_values(new_qs,arms_present)
                       
            self.q_mom_2[arms_present] = (new_qs[arms_present]) * (new_qs[arms_present]) * (1-self.change_decay_rate_2) + self.change_decay_rate_2 * self.q_mom_2[arms_present]
            self.q_update_step[arms_present] += 1
            
            var_qs_hat = np.maximum(self.q_mom_2 - self.q_values * self.q_values, 0)
            
            self.q_SNR[arms_present] = self.q_values[arms_present] / (np.sqrt(var_qs_hat[arms_present]) + self.SNR_eps)

            for arm, q_SNR_val in enumerate(self.q_SNR):
                if q_SNR_val < self.SNR_threshold and self.q_update_step[arm] > self.SNR_update_step_threshold:
                    self.mask[arm] = False    
                else:
                    self.mask[arm] = True

            print(self.q_values,"q_values")
            print(self.mask,"mask")
            print(self.q_SNR,"q_SNR")
            print(self.q_mom_2,"q_mom_2")
            print(var_qs_hat,"var_qs_hat")
            print(self.arm_to_group,"arm_to_group")
            print(self.group_to_arm,"group_to_arm")
        
        elif self.decay_type == 'soft_var_conv_mask':
            assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
            assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'

            response_mask = kwargs['response_mask']

            data_info = kwargs['data_info']
          
            
            obj = kwargs['rewards']
            index = kwargs['index']
            non_zero_mask = (obj != 0)
            obj = (obj * non_zero_mask).sum(-1)

            group_to_obj = defaultdict(list)
            id2obj = defaultdict(list)

            id2std = {}
            
            for data, obj_val,idx in zip(data_info, obj,index):
                group = self.extra_info_to_group(data)
                group_to_obj[group].append(idx)
                id2obj[idx].append(obj_val.item())
            for idx in id2obj:
                if len(id2obj[idx]) == 1:
                    id2std[idx] = 1.0
                elif len(id2obj[idx]) > 1:
                    id2std[idx] = np.std(id2obj[idx])
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            for group, idxs in group_to_obj.items():
                group_to_obj[group] = np.mean(np.abs([id2std[idx] for idx in idxs])) # New Q-value is the mean of the variances

            
            new_qs = np.zeros(self.num_arms)
            arms_present = np.array(self.num_arms*[False])
            for group, obj_vals in group_to_obj.items(): 
                new_qs[self.group_to_arm[group]] = obj_vals 
                arms_present[self.group_to_arm[group]] = True
            old_qs = self.q_values.copy()
            self.update_q_values(new_qs,arms_present)
                       
            self.q_update_step[arms_present] += 1
            print(np.std(self.q_values),"std before clip")
            q_std = self.sharpness * np.clip(np.std(self.q_values),1e-4,1)
            

            # softmask: before threshold = 1, after threshold = sigmoid(q/τ)
            step_mask = self.q_update_step > self.SNR_update_step_threshold
            self.softmask[step_mask] = 2.0*(1 / (1 + np.exp(-self.q_values[step_mask] / q_std)))-1

            print(self.q_values,"q_values")
            print(self.softmask,"softmask")
            print(q_std,"q_std")
            print(self.arm_to_group,"arm_to_group")
            print(self.group_to_arm,"group_to_arm")
        
        elif self.decay_type == 'task_diff_q_conv_mask':
            
            group_to_obj = defaultdict(list)
            # update priority for each batch
            for batch_id, task_wise_diff in batch_to_task_wise_diff.items():
                # get sorted keys 
                values_by_arm = [ task_wise_diff.get(self.arm_to_group[arm], 0) for arm in range(self.num_arms) ]
                for arm in range(self.num_arms):
                    group_to_obj[self.arm_to_group[arm]]= group_to_obj.get(self.arm_to_group[arm],0) + values_by_arm[arm]

            
            new_qs = np.zeros(self.num_arms)
            arms_present = np.array(self.num_arms*[False])
            for group, obj_vals in group_to_obj.items(): 
                # clip obj_vals to be in [-0.1,0.1]
                obj_vals = np.clip(obj_vals, -0.1, 0.1)
                new_qs[self.group_to_arm[group]] = obj_vals if self.max_rate else -obj_vals 
                arms_present[self.group_to_arm[group]] = True
            self.update_q_values(new_qs,arms_present)
            
            self.q_mom_2[arms_present] = (new_qs[arms_present]) * (new_qs[arms_present]) * (1-self.change_decay_rate_2) + self.change_decay_rate_2 * self.q_mom_2[arms_present]
            self.q_update_step[arms_present] += 1
            
            var_qs_hat = np.maximum(self.q_mom_2 - self.q_values * self.q_values, 0)
            
            self.q_SNR[arms_present] = self.q_values[arms_present] / (np.sqrt(var_qs_hat[arms_present]) + self.SNR_eps)

            for arm, q_SNR_val in enumerate(self.q_SNR):
                if q_SNR_val < self.SNR_threshold and self.q_update_step[arm] > self.SNR_update_step_threshold:
                    self.mask[arm] = False    
                else:
                    self.mask[arm] = True

            print(self.q_values,"q_values")
            print(self.mask,"mask")
            print(self.q_SNR,"q_SNR")
            print(self.q_mom_2,"q_mom_2")
            print(var_qs_hat,"var_qs_hat")
            print(self.arm_to_group,"arm_to_group")
            print(self.group_to_arm,"group_to_arm")
        
        elif self.decay_type == 'soft_task_diff_q_conv_mask':
            
            group_to_obj = defaultdict(list)
            group_to_count = defaultdict(list)
            # update priority for each batch
            for batch_id, task_wise_diff in batch_to_task_wise_diff.items():
                # get sorted keys 
                values_by_arm = [ task_wise_diff.get(self.arm_to_group[arm], 0) for arm in range(self.num_arms) ]
                
                for arm in range(self.num_arms):
                    group_to_obj[self.arm_to_group[arm]]= group_to_obj.get(self.arm_to_group[arm],0) + values_by_arm[arm]
            
            for batch_id, task_wise_count in batch_to_task_wise_count.items():
                counts_by_arm = [ task_wise_count.get(self.arm_to_group[arm], 0) for arm in range(self.num_arms) ]
                for arm in range(self.num_arms):
                    group_to_count[self.arm_to_group[arm]]= group_to_count.get(self.arm_to_group[arm],0) + counts_by_arm[arm]

            if self.mean_diff_decay:
                print(group_to_obj,"group_to_obj")
                print(group_to_count,"group_to_count")
                for group, obj_vals in group_to_obj.items():
                    group_to_obj[group] = obj_vals / group_to_count[group]
                
                print(group_to_obj,"group_to_obj after mean")


            
            new_qs = np.zeros(self.num_arms)
            arms_present = np.array(self.num_arms*[False])
            for group, obj_vals in group_to_obj.items(): 
                # clip obj_vals to be in [-0.1,0.1]
                obj_vals = np.clip(obj_vals, -0.1, 0.1)
                new_qs[self.group_to_arm[group]] = obj_vals if self.max_rate else -obj_vals 
                arms_present[self.group_to_arm[group]] = True
            self.update_q_values(new_qs,arms_present)

            self.q_update_step[arms_present] += 1

            q_std = self.sharpness * np.clip(np.std(self.q_values),1e-4,1e-2)
            

            # softmask: before threshold = 1, after threshold = sigmoid(q/τ)
            step_mask = self.q_update_step > self.SNR_update_step_threshold
            self.softmask[step_mask] = 1 / (1 + np.exp(-self.q_values[step_mask] / q_std))

            print(self.q_values,"q_values")
            print(self.softmask,"softmask")
            print(q_std,"q_std")
            print(self.arm_to_group,"arm_to_group")
            print(self.group_to_arm,"group_to_arm")
        
        if self.update_object == 'reward':

            assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
            assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'

            response_mask = kwargs['response_mask']


            data_info = kwargs['data_info']
          
            obj = kwargs['rewards']
            non_zero_mask = (obj != 0)
            obj = (obj * non_zero_mask).sum(-1)

            group_to_obj = defaultdict(list)
            for data, obj_val in zip(data_info, obj):
                group = self.extra_info_to_group(data)
                group_to_obj[group].append(obj_val.item())
            for group, obj_vals in group_to_obj.items():
                group_to_obj[group] = np.mean(np.abs(obj_vals)) # New Q-value is the mean of the absolute advantages

            
            new_rs = np.zeros(self.num_arms)
            for group, obj_vals in group_to_obj.items(): 
                new_rs[self.group_to_arm[group]] = obj_vals 

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



        elif self.update_object == 'task_wise_diff':

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
                    #     probs = gated_softmax(self.w, self.softmask, fixed_p_each=self.fixed_p_each, dim=-1)
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
        

        else:
            raise ValueError(f'Invalid update object: {self.update_object}')



        self._probs_cache = self._softmax_probs().detach().cpu().numpy()


class JointBandit(BanditPriority):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def extra_info_to_group(self, extra_info):
        return '-'.join([str(extra_info[feature]) for feature in self.feature])

    def data_to_group(self, data):
        if 'difficulty' in data['extra_info']:
            assert int(data['extra_info']['difficulty']) == float(data['extra_info']['difficulty']) # make sure difficulty is an integer
        return self.extra_info_to_group(data['extra_info'])
        
class JointFamoBandit(FamoPriorityNew):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('JointFamoBandit initialized')


    def extra_info_to_group(self, extra_info):
        return '-'.join([str(extra_info[feature]) for feature in self.feature])

    def data_to_group(self, data):
        if 'difficulty' in data['extra_info']:
            assert int(data['extra_info']['difficulty']) == float(data['extra_info']['difficulty']) # make sure difficulty is an integer
        return self.extra_info_to_group(data['extra_info'])
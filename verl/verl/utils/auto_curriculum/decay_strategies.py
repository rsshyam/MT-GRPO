from __future__ import annotations
from typing import Any, Counter, Dict, Callable
import numpy as np

from collections import defaultdict

# Simple registry for strategies
_REGISTRY: Dict[str, "DecayStrategy"] = {}

def register_decay(name: str) -> Callable[["DecayStrategy"], "DecayStrategy"]:
    def _wrap(cls: "DecayStrategy") -> "DecayStrategy":
        _REGISTRY[name] = cls()
        return cls
    return _wrap

def get_decay(name: str | None) -> "DecayStrategy":
    if name is None:
        return _REGISTRY["noop"]
    if name not in _REGISTRY:
        raise ValueError(f"Unknown decay_type '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]

class DecayStrategy:
    # Each strategy mutates fields on famo (e.g., q_values, mask/softmask)
    def update(
        self,
        famo: "FamoPriorityDecay",
        batch_to_task_wise_diff: Dict[str, Dict[str, float]],
        batch_to_task_wise_count: Dict[str, Dict[str, float]],
        kwargs: Dict[str, Any],
    ) -> None:
        raise NotImplementedError

@register_decay("noop")
class NoOpDecay(DecayStrategy):
    def update(self, famo, batch_to_task_wise_diff, batch_to_task_wise_count, kwargs):
        # Do nothing if no decay is desired
        return

@register_decay("mask")
class MaskDecay(DecayStrategy):
    # adapted to use `famo` instead of `self`
    def update(self, famo, batch_to_task_wise_diff, batch_to_task_wise_count, kwargs):

        assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
        assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'

        data_info = kwargs["data_info"]
        rewards = kwargs["rewards"]

        non_zero = (rewards != 0)
        obj = (rewards * non_zero).sum(-1)

        
        group_to_obj = defaultdict(list)
        for data, obj_val in zip(data_info, obj):
            group = famo.extra_info_to_group(data)
            group_to_obj[group].append(obj_val.item())

        new_qs = np.zeros(famo.num_arms)
        arms_present = np.zeros(famo.num_arms, dtype=bool)
        for group, vals in group_to_obj.items():
            mean_abs = float(np.mean(np.abs(vals)))
            arm = famo.group_to_arm[group]
            new_qs[arm] = mean_abs
            arms_present[arm] = True

        famo.update_q_values(new_qs, arms_present)

        # threshold → mask True/False
        for arm, q in enumerate(famo.q_values):
            famo.mask[arm] = not (q > famo.reward_threshold)


@register_decay("q_conv_mask")
class QConvMaskDecay(DecayStrategy):
    def update(self, famo, batch_to_task_wise_diff, batch_to_task_wise_count, kwargs):
        assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
        assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'

        data_info = kwargs['data_info']
        
        obj = kwargs['rewards']
        non_zero_mask = (obj != 0)
        obj = (obj * non_zero_mask).sum(-1)

        group_to_obj = defaultdict(list)
        for data, obj_val in zip(data_info, obj):
            group = famo.extra_info_to_group(data)
            group_to_obj[group].append(obj_val.item())
        for group, obj_vals in group_to_obj.items():
            group_to_obj[group] = np.mean(np.abs(obj_vals)) # New Q-value is the mean of the absolute advantages

        
        new_qs = np.zeros(famo.num_arms)
        arms_present = np.array(famo.num_arms*[False])
        for group, obj_vals in group_to_obj.items(): 
            new_qs[famo.group_to_arm[group]] = obj_vals 
            arms_present[famo.group_to_arm[group]] = True
        old_qs = famo.q_values.copy()
        famo.update_q_values(new_qs,arms_present)
        # update only if group is present in group_to_obj
        new_qs = famo.q_values.copy()
        
        famo.change_qs[arms_present] = new_qs[arms_present] - old_qs[arms_present]
        
        famo.change_qs_mom[arms_present] = famo.change_qs[arms_present] * (1-famo.change_decay_rate) + famo.change_decay_rate * famo.change_qs_mom[arms_present]
        famo.change_qs_mom_2[arms_present] = (famo.change_qs[arms_present]) * (famo.change_qs[arms_present]) * (1-famo.change_decay_rate_2) + famo.change_decay_rate_2 * famo.change_qs_mom_2[arms_present]
        famo.q_update_step[arms_present] += 1

        change_qs_mom_hat = np.zeros(famo.num_arms)
        change_qs_mom_hat[arms_present] = famo.change_qs_mom[arms_present] / (1-famo.change_decay_rate**famo.q_update_step[arms_present])
        change_qs_mom_2_hat = np.zeros(famo.num_arms)
        change_qs_mom_2_hat[arms_present] = famo.change_qs_mom_2[arms_present] / (1-famo.change_decay_rate_2**famo.q_update_step[arms_present])

        var_qs_hat = np.maximum(change_qs_mom_2_hat - change_qs_mom_hat * change_qs_mom_hat, 0)
        
        famo.q_SNR[arms_present] = change_qs_mom_hat[arms_present] / (np.sqrt(var_qs_hat[arms_present]) + famo.SNR_eps)

        for arm, q_SNR_val in enumerate(famo.q_SNR):
            if q_SNR_val < famo.SNR_threshold and famo.q_update_step[arm] > famo.SNR_update_step_threshold:
                famo.mask[arm] = False    
            else:
                famo.mask[arm] = True

@register_decay("soft_q_conv_mask")
class SoftQConvMaskDecay(DecayStrategy):
    def update(self, famo, batch_to_task_wise_diff, batch_to_task_wise_count, kwargs):
        assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
        assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'

        data_info = kwargs['data_info']
        
        obj = kwargs['rewards']
        non_zero_mask = (obj != 0)
        obj = (obj * non_zero_mask).sum(-1)

        group_to_obj = defaultdict(list)
        for data, obj_val in zip(data_info, obj):
            group = famo.extra_info_to_group(data)
            group_to_obj[group].append(obj_val.item())
        for group, obj_vals in group_to_obj.items():
            group_to_obj[group] = np.mean(np.abs(obj_vals)) # New Q-value is the mean of the absolute advantages

        
        new_qs = np.zeros(famo.num_arms)
        arms_present = np.array(famo.num_arms*[False])
        for group, obj_vals in group_to_obj.items(): 
            new_qs[famo.group_to_arm[group]] = obj_vals 
            arms_present[famo.group_to_arm[group]] = True
        old_qs = famo.q_values.copy()
        famo.update_q_values(new_qs,arms_present)
        # update only if group is present in group_to_obj
        new_qs = famo.q_values.copy()
        
        famo.change_qs[arms_present] = new_qs[arms_present] - old_qs[arms_present]
        
        famo.change_qs_mom[arms_present] = famo.change_qs[arms_present] * (1-famo.change_decay_rate) + famo.change_decay_rate * famo.change_qs_mom[arms_present]
        famo.q_update_step[arms_present] += 1

        change_q_std = famo.sharpness * np.clip(np.std(famo.change_qs_mom),1e-6,1e-2)
        
        # softmask: before threshold = 1, after threshold = sigmoid(q/τ)
        step_mask = famo.q_update_step > famo.SNR_update_step_threshold
        famo.softmask[step_mask] = 1 / (1 + np.exp(-famo.change_qs_mom[step_mask] / change_q_std))

        print(famo.q_values,"q_values")
        print(famo.change_qs,"change_qs")
        print(famo.change_qs_mom,"change_qs_mom")
        print(famo.softmask,"softmask")
        print(change_q_std,"change_q_std")
        print(famo.arm_to_group,"arm_to_group")
        print(famo.group_to_arm,"group_to_arm")
        

@register_decay("var_conv_mask")
class VarConvMaskDecay(DecayStrategy):
    def update(self, famo, batch_to_task_wise_diff, batch_to_task_wise_count, kwargs):
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
            group = famo.extra_info_to_group(data)
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

        
        new_qs = np.zeros(famo.num_arms)
        arms_present = np.array(famo.num_arms*[False])
        for group, obj_vals in group_to_obj.items(): 
            new_qs[famo.group_to_arm[group]] = obj_vals 
            arms_present[famo.group_to_arm[group]] = True
        old_qs = famo.q_values.copy()
        famo.update_q_values(new_qs,arms_present)
                    
        famo.q_mom_2[arms_present] = (new_qs[arms_present]) * (new_qs[arms_present]) * (1-famo.change_decay_rate_2) + famo.change_decay_rate_2 * famo.q_mom_2[arms_present]
        famo.q_update_step[arms_present] += 1
        
        var_qs_hat = np.maximum(famo.q_mom_2 - famo.q_values * famo.q_values, 0)
        
        famo.q_SNR[arms_present] = famo.q_values[arms_present] / (np.sqrt(var_qs_hat[arms_present]) + famo.SNR_eps)

        for arm, q_SNR_val in enumerate(famo.q_SNR):
            if q_SNR_val < famo.SNR_threshold and famo.q_update_step[arm] > famo.SNR_update_step_threshold:
                famo.mask[arm] = False    
            else:
                famo.mask[arm] = True

        print(famo.q_values,"q_values")
        print(famo.mask,"mask")
        print(famo.q_SNR,"q_SNR")
        print(famo.q_mom_2,"q_mom_2")
        print(var_qs_hat,"var_qs_hat")
        print(famo.arm_to_group,"arm_to_group")
        print(famo.group_to_arm,"group_to_arm")

@register_decay("soft_var_conv_mask")
class SoftVarConvMaskDecay(DecayStrategy):
    def update(self, famo, batch_to_task_wise_diff, batch_to_task_wise_count, kwargs):
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
            group = famo.extra_info_to_group(data)
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

        
        new_qs = np.zeros(famo.num_arms)
        arms_present = np.array(famo.num_arms*[False])
        for group, obj_vals in group_to_obj.items(): 
            new_qs[famo.group_to_arm[group]] = obj_vals 
            arms_present[famo.group_to_arm[group]] = True
        old_qs = famo.q_values.copy()
        famo.update_q_values(new_qs,arms_present)
                    
        famo.q_update_step[arms_present] += 1
        print(np.std(famo.q_values),"std before clip")
        q_std = famo.sharpness * np.clip(np.std(famo.q_values),1e-4,1)
        

        # softmask: before threshold = 1, after threshold = sigmoid(q/τ)
        step_mask = famo.q_update_step > famo.SNR_update_step_threshold
        famo.softmask[step_mask] = 2.0*(1 / (1 + np.exp(-famo.q_values[step_mask] / q_std)))-1

        print(famo.q_values,"q_values")
        print(famo.softmask,"softmask")
        print(q_std,"q_std")
        print(famo.arm_to_group,"arm_to_group")
        print(famo.group_to_arm,"group_to_arm")

@register_decay("task_diff_q_conv_mask")
class TaskDiffQConvMaskDecay(DecayStrategy):
    def update(self, famo, batch_to_task_wise_diff, batch_to_task_wise_count, kwargs):
          
        group_to_obj = defaultdict(list)
        # update priority for each batch
        for batch_id, task_wise_diff in batch_to_task_wise_diff.items():
            # get sorted keys 
            values_by_arm = [ task_wise_diff.get(famo.arm_to_group[arm], 0) for arm in range(famo.num_arms) ]
            for arm in range(famo.num_arms):
                group_to_obj[famo.arm_to_group[arm]]= group_to_obj.get(famo.arm_to_group[arm],0) + values_by_arm[arm]

        
        new_qs = np.zeros(famo.num_arms)
        arms_present = np.array(famo.num_arms*[False])
        for group, obj_vals in group_to_obj.items(): 
            # clip obj_vals to be in [-0.1,0.1]
            obj_vals = np.clip(obj_vals, -0.1, 0.1)
            new_qs[famo.group_to_arm[group]] = obj_vals if famo.max_rate else -obj_vals 
            arms_present[famo.group_to_arm[group]] = True
        famo.update_q_values(new_qs,arms_present)
        
        famo.q_mom_2[arms_present] = (new_qs[arms_present]) * (new_qs[arms_present]) * (1-famo.change_decay_rate_2) + famo.change_decay_rate_2 * famo.q_mom_2[arms_present]
        famo.q_update_step[arms_present] += 1
        
        var_qs_hat = np.maximum(famo.q_mom_2 - famo.q_values * famo.q_values, 0)
        
        famo.q_SNR[arms_present] = famo.q_values[arms_present] / (np.sqrt(var_qs_hat[arms_present]) + famo.SNR_eps)

        for arm, q_SNR_val in enumerate(famo.q_SNR):
            if q_SNR_val < famo.SNR_threshold and famo.q_update_step[arm] > famo.SNR_update_step_threshold:
                famo.mask[arm] = False    
            else:
                famo.mask[arm] = True

        print(famo.q_values,"q_values")
        print(famo.mask,"mask")
        print(famo.q_SNR,"q_SNR")
        print(famo.q_mom_2,"q_mom_2")
        print(var_qs_hat,"var_qs_hat")
        print(famo.arm_to_group,"arm_to_group")
        print(famo.group_to_arm,"group_to_arm")
    
@register_decay("soft_task_diff_q_conv_mask")
class SoftTaskDiffQConvMaskDecay(DecayStrategy):
    def update(self, famo, batch_to_task_wise_diff, batch_to_task_wise_count, kwargs):
        group_to_obj = defaultdict(list)
        group_to_count = defaultdict(list)
        # update priority for each batch
        for batch_id, task_wise_diff in batch_to_task_wise_diff.items():
            # get sorted keys 
            values_by_arm = [ task_wise_diff.get(famo.arm_to_group[arm], 0) for arm in range(famo.num_arms) ]
            
            for arm in range(famo.num_arms):
                group_to_obj[famo.arm_to_group[arm]]= group_to_obj.get(famo.arm_to_group[arm],0) + values_by_arm[arm]
        
        for batch_id, task_wise_count in batch_to_task_wise_count.items():
            counts_by_arm = [ task_wise_count.get(famo.arm_to_group[arm], 0) for arm in range(famo.num_arms) ]
            for arm in range(famo.num_arms):
                group_to_count[famo.arm_to_group[arm]]= group_to_count.get(famo.arm_to_group[arm],0) + counts_by_arm[arm]

        if famo.mean_diff_decay:
            print(group_to_obj,"group_to_obj")
            print(group_to_count,"group_to_count")
            for group, obj_vals in group_to_obj.items():
                group_to_obj[group] = obj_vals / group_to_count[group]
            
            print(group_to_obj,"group_to_obj after mean")


        
        new_qs = np.zeros(famo.num_arms)
        arms_present = np.array(famo.num_arms*[False])
        for group, obj_vals in group_to_obj.items(): 
            # clip obj_vals to be in [-0.1,0.1]
            obj_vals = np.clip(obj_vals, -0.1, 0.1)
            new_qs[famo.group_to_arm[group]] = obj_vals if famo.max_rate else -obj_vals 
            arms_present[famo.group_to_arm[group]] = True
        famo.update_q_values(new_qs,arms_present)

        famo.q_update_step[arms_present] += 1

        q_std = famo.sharpness * np.clip(np.std(famo.q_values),1e-4,1e-2)
        

        # softmask: before threshold = 1, after threshold = sigmoid(q/τ)
        step_mask = famo.q_update_step > famo.SNR_update_step_threshold
        famo.softmask[step_mask] = 1 / (1 + np.exp(-famo.q_values[step_mask] / q_std))

        print(famo.q_values,"q_values")
        print(famo.softmask,"softmask")
        print(q_std,"q_std")
        print(famo.arm_to_group,"arm_to_group")
        print(famo.group_to_arm,"group_to_arm")


@register_decay("soft_opt_conv_mask")
class SoftOptConvMaskDecay(DecayStrategy):
    def update(self, famo, batch_to_task_wise_diff, batch_to_task_wise_count, kwargs):

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
        id2onemean = {}

        
        for data, obj_val,idx in zip(data_info, obj,index):
            group = famo.extra_info_to_group(data)
            id_to_group[idx] = group
            id2obj[idx].append(obj_val.item())

        for idx in id2obj:
            # check if any element in id2obj[idx] is 1
            if len(id2obj[idx]) == 1:
                if id2obj[idx][0] == 1:
                    id2nzmean[idx] = 1.0
                print('question with one response found')
            elif len(id2obj[idx]) > 1:
                if any([obj_val >= 1 - 1e-6 for obj_val in id2obj[idx]]):
                    id2nzmean[idx] = np.mean([obj_val for obj_val in id2obj[idx]])
                    if all([obj_val >= 1 - 1e-6 for obj_val in id2obj[idx]]):
                        id2onemean[idx] = 1.0
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        group_to_obj = defaultdict(list)
        group_to_nzcount = defaultdict(int)
        group_to_zcount =  defaultdict(int)
        group_to_onecount = defaultdict(int)

        for idx in id2obj:
            group = id_to_group[idx]
            if idx in id2nzmean.keys(): 
                group_to_obj[group].append(id2nzmean[idx])
                group_to_nzcount[group]+=1
            else:
                group_to_zcount[group]+=1
            if idx in id2onemean.keys():
                group_to_onecount[group]+=1

        group_to_zratio = {}
        group_to_oneratio = {}
        group_to_nznoneratio = {}

        for group, obj_vals in group_to_obj.items():
            group_to_obj[group] = np.mean(obj_vals)
            group_to_zratio[group] = group_to_zcount[group]/(group_to_nzcount[group]+group_to_zcount[group])
            group_to_oneratio[group] = group_to_onecount[group]/(group_to_nzcount[group]+group_to_zcount[group])
            group_to_nznoneratio[group] = (group_to_nzcount[group]-group_to_onecount[group])/(group_to_nzcount[group]+group_to_zcount[group])

        new_qs = np.zeros(famo.num_arms)
        new_zratio = np.zeros(famo.num_arms)
        new_oneratio = np.zeros(famo.num_arms)
        new_nznoneratio = np.zeros(famo.num_arms)
        arms_present = np.array(famo.num_arms*[False])
        for group, obj_vals in group_to_obj.items(): 
            new_qs[famo.group_to_arm[group]] = obj_vals 
            new_zratio[famo.group_to_arm[group]] = group_to_zratio[group]
            new_oneratio[famo.group_to_arm[group]] = group_to_oneratio[group]
            new_nznoneratio[famo.group_to_arm[group]] = group_to_nznoneratio[group]
            arms_present[famo.group_to_arm[group]] = True

        famo.update_q_values(new_qs,arms_present)
        famo.update_z_ratio(new_zratio,arms_present)
        famo.update_one_ratio(new_oneratio,arms_present)
        famo.update_nznone_ratio(new_nznoneratio,arms_present)                    
        famo.q_update_step[arms_present] += 1

        q_std = famo.sharpness * np.clip(np.std(famo.q_values),1e-6,1.0)
        

        # softmask: before threshold = 1, after threshold = sigmoid(q/τ)
        step_mask = famo.q_update_step > famo.SNR_update_step_threshold
        famo.softmask[step_mask] = np.clip(1.0-2.0*(1 / (1 + np.exp((-famo.q_values[step_mask]+1) / q_std))),0,1)

        print(famo.q_values,"q_values")
      

@register_decay("soft_half_opt_conv_mask")
class SoftHalfOptConvMaskDecay(DecayStrategy):
    def update(self, famo, batch_to_task_wise_diff, batch_to_task_wise_count, kwargs):

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
        id2optmean = {}
        id2onemean = {}
        id2optdist = {}

        acc2idx = defaultdict(list)

        
        for data, obj_val,idx in zip(data_info, obj,index):
            group = famo.extra_info_to_group(data)
            id_to_group[idx] = group
            id2obj[idx].append(obj_val.item())

        for idx in id2obj:
            # check if any element in id2obj[idx] is 1
            if len(id2obj[idx]) == 1:
                if id2obj[idx][0] >= 1 - 1e-6:
                    id2nzmean[idx] = 1.0
                    id2onemean[idx] = 1.0
                print('question with one response found')
            elif len(id2obj[idx]) > 1:
                if any([obj_val >= 1 - 1e-6 for obj_val in id2obj[idx]]):
                    id2nzmean[idx] = np.mean([obj_val for obj_val in id2obj[idx]])
                    if all([obj_val >= 1 - 1e-6 for obj_val in id2obj[idx]]):
                        id2onemean[idx] = 1.0

                    # create bins based on number of obj_val greater than 1- 1e-6 within id2obj[idx]
                    # count number of obj_val greater than 1- 1e-6 within id2obj[idx]
                succ = sum(obj_val >= 1 - 1e-6 for obj_val in id2obj[idx])
                acc2idx[f"{succ}_{len(id2obj[idx])}"].append(idx)

                if (succ/len(id2obj[idx])) <= famo.opt_threshold and any([obj_val >= 1 - 1e-6 for obj_val in id2obj[idx]]):
                    id2optmean[idx] = np.mean([obj_val for obj_val in id2obj[idx]])
                    id2optdist[idx] = (np.mean([obj_val for obj_val in id2obj[idx]])-0.5)**2

            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        group_to_obj = defaultdict(list)
        group_to_optdist = defaultdict(list)
        group_to_nzcount = defaultdict(int)
        group_to_zcount =  defaultdict(int)
        group_to_onecount = defaultdict(int)
        group_to_optcount = defaultdict(int)

        group_acc_count = defaultdict(dict)

        for idx in id2obj:
            group = id_to_group[idx]
            if idx in id2nzmean.keys(): 
                group_to_nzcount[group]+=1
            else:
                group_to_zcount[group]+=1
            if idx in id2onemean.keys():
                group_to_onecount[group]+=1
            if idx in id2optmean.keys():
                group_to_obj[group].append(id2optmean[idx])
                group_to_optcount[group]+=1
            if idx in id2optdist.keys():
                group_to_optdist[group].append(id2optdist[idx])
            
        for acc, idxs in acc2idx.items():
            group_acc = [id_to_group[idx] for idx in idxs]
            for group in set(group_acc):
                group_acc_count[group][acc] = group_acc.count(group)

        group_to_zratio = {}
        group_to_oneratio = {}
        group_to_nznoneratio = {}
        group_to_optratio = {}
        group_acc_ratio = defaultdict(dict)

        for group, obj_vals in group_to_obj.items():
            group_to_obj[group] = np.mean(obj_vals)
            group_to_optdist[group] = np.mean(group_to_optdist[group])
            group_to_zratio[group] = group_to_zcount[group]/(group_to_nzcount[group]+group_to_zcount[group])
            group_to_oneratio[group] = group_to_onecount[group]/(group_to_nzcount[group]+group_to_zcount[group])
            group_to_nznoneratio[group] = (group_to_nzcount[group]-group_to_onecount[group])/(group_to_nzcount[group]+group_to_zcount[group])
            group_to_optratio[group] = group_to_optcount[group]/(group_to_nzcount[group]+group_to_zcount[group])

            for acc, count in group_acc_count[group].items():
                group_acc_ratio[group][acc] = count/(group_to_nzcount[group]+group_to_zcount[group])

        new_qs = np.zeros(famo.num_arms)
        new_optdist = np.zeros(famo.num_arms)
        new_zratio = np.zeros(famo.num_arms)
        new_oneratio = np.zeros(famo.num_arms)
        new_nznoneratio = np.zeros(famo.num_arms)
        new_optratio = np.zeros(famo.num_arms)
        new_optdist = np.zeros(famo.num_arms)
        new_accratio = defaultdict(dict)
        arms_present = np.array(famo.num_arms*[False])
        for group, obj_vals in group_to_obj.items(): 
            new_qs[famo.group_to_arm[group]] = obj_vals 
            new_zratio[famo.group_to_arm[group]] = group_to_zratio[group]
            new_oneratio[famo.group_to_arm[group]] = group_to_oneratio[group]
            new_nznoneratio[famo.group_to_arm[group]] = group_to_nznoneratio[group]
            new_optratio[famo.group_to_arm[group]] = group_to_optratio[group]
            new_optdist[famo.group_to_arm[group]] = group_to_optdist[group]
            arms_present[famo.group_to_arm[group]] = True


            for acc, ratio in group_acc_ratio[group].items():
                new_accratio[famo.group_to_arm[group]][acc] = ratio

        famo.update_q_values(new_qs,arms_present)
        famo.update_z_ratio(new_zratio,arms_present)
        famo.update_one_ratio(new_oneratio,arms_present)
        famo.update_nznone_ratio(new_nznoneratio,arms_present)
        famo.update_opt_ratio(new_optratio,arms_present)
        famo.update_optdist(new_optdist,arms_present)
        famo.update_acc_ratio(new_accratio,arms_present)                    
        famo.q_update_step[arms_present] += 1

        q_std = famo.sharpness * np.clip(np.std(famo.q_values),1e-6,1.0)
        

        # softmask: before threshold = 1, after threshold = sigmoid(q/τ)
        step_mask = famo.q_update_step > famo.SNR_update_step_threshold
        famo.softmask[step_mask] = np.clip(1.0-2.0*(1 / (1 + np.exp((-famo.q_values[step_mask]+famo.opt_threshold) / q_std))),0,1)

        print(famo.q_values,"q_values")

@register_decay("coeff_var_conv_mask")
class CoeffVarConvMaskDecay(DecayStrategy): # cannot solve hard vs converged tasks distinction fully
    def update(self, famo, batch_to_task_wise_diff, batch_to_task_wise_count, kwargs):
        assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
        assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'

        response_mask = kwargs['response_mask']

        data_info = kwargs['data_info']
        
        
        obj = kwargs['rewards']
        index = kwargs['index']
        non_zero_mask = (obj != 0)
        obj = (obj * non_zero_mask).sum(-1)

        group_to_id = defaultdict(list)
        id2obj = defaultdict(list)

        id2std = {}
        id2mean = {}
        
        for data, obj_val,idx in zip(data_info, obj,index):
            group = famo.extra_info_to_group(data)
            group_to_id[group].append(idx)
            id2obj[idx].append(obj_val.item())
        for idx in id2obj:
            if len(id2obj[idx]) == 1:
                id2std[idx] = np.nan
            elif len(id2obj[idx]) > 1:
                id2std[idx] = np.std(id2obj[idx],ddof=1)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
            id2mean[idx] = np.mean(id2obj[idx])

        group_to_coeff = {}
        for group, idxs in group_to_id.items():
            std_list = [id2std[idx] for idx in idxs if not np.isnan(id2std[idx])]
            if len(std_list) > 0:
                group_to_coeff[group] = np.sqrt(np.mean(np.square(std_list)))/(max(np.mean([id2mean[idx] for idx in idxs]),1e-6))

        
        new_qs = np.zeros(famo.num_arms)
        arms_present = np.array(famo.num_arms*[False])
        for group, obj_vals in group_to_coeff.items(): 
            new_qs[famo.group_to_arm[group]] = obj_vals 
            arms_present[famo.group_to_arm[group]] = True
        old_qs = famo.q_values.copy()
        famo.update_q_values(new_qs,arms_present)
                    
        famo.q_update_step[arms_present] += 1

        for arm, q_val in enumerate(famo.q_values):
            if q_val < famo.SNR_threshold and famo.q_update_step[arm] > famo.SNR_update_step_threshold:
                famo.mask[arm] = False    
            else:
                famo.mask[arm] = True

        print(famo.q_values,"q_values")
        print(famo.mask,"mask")
        print(famo.q_update_step,"q_update_step")
        print(famo.arm_to_group,"arm_to_group")
        print(famo.group_to_arm,"group_to_arm")


@register_decay("soft_coeff_var_conv_mask")
class SoftCoeffVarConvMaskDecay(DecayStrategy): # cannot solve hard vs converged tasks distinction fully
    def update(self, famo, batch_to_task_wise_diff, batch_to_task_wise_count, kwargs):
        assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
        assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'

        response_mask = kwargs['response_mask']

        data_info = kwargs['data_info']
        
        
        obj = kwargs['rewards']
        index = kwargs['index']
        non_zero_mask = (obj != 0)
        obj = (obj * non_zero_mask).sum(-1)

        group_to_id = defaultdict(list)
        id2obj = defaultdict(list)

        id2std = {}
        id2mean = {}
        
        for data, obj_val,idx in zip(data_info, obj,index):
            group = famo.extra_info_to_group(data)
            group_to_id[group].append(idx)
            id2obj[idx].append(obj_val.item())
        for idx in id2obj:
            if len(id2obj[idx]) == 1:
                id2std[idx] = np.nan
            elif len(id2obj[idx]) > 1:
                id2std[idx] = np.std(id2obj[idx],ddof=1)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
            id2mean[idx] = np.mean(id2obj[idx])

        group_to_coeff = {}
        for group, idxs in group_to_id.items():
            std_list = [id2std[idx] for idx in idxs if not np.isnan(id2std[idx])]
            if len(std_list) > 0:
                group_to_coeff[group] = np.sqrt(np.mean(np.square(std_list)))/(max(np.mean([id2mean[idx] for idx in idxs]),1e-6))

        
        new_qs = np.zeros(famo.num_arms)
        arms_present = np.array(famo.num_arms*[False])
        for group, obj_vals in group_to_coeff.items(): 
            new_qs[famo.group_to_arm[group]] = obj_vals 
            arms_present[famo.group_to_arm[group]] = True
        old_qs = famo.q_values.copy()
        famo.update_q_values(new_qs,arms_present)
                    
        famo.q_update_step[arms_present] += 1
        print(np.std(famo.q_values),"std before clip")
        q_std = famo.sharpness * np.clip(np.std(famo.q_values),1e-4,10)
        

        # softmask: before threshold = 1, after threshold = sigmoid(q/τ)
        step_mask = famo.q_update_step > famo.SNR_update_step_threshold
        famo.softmask[step_mask] = 2.0*(1 / (1 + np.exp(-famo.q_values[step_mask] / q_std)))-1

        print(famo.q_values,"q_values")
        print(famo.softmask,"softmask")
        print(q_std,"q_std")
        print(famo.arm_to_group,"arm_to_group")
        print(famo.group_to_arm,"group_to_arm")
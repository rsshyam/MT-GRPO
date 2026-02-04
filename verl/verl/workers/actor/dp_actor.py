# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple, Counter

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

from verl.trainer.ppo import core_algos

from ray.util import pdb 


__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
        self.groups = self.config.groups
        
        if self.config.kl_ctrl.type == 'adaptive':
            assert self.config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {self.config.kl_ctrl.horizon}'
            self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=self.config.kl_ctrl.kl_coef,
                                                           target_kl=self.config.kl_ctrl.target_kl,
                                                           horizon=self.config.kl_ctrl.horizon)
            print(f'Adaptive Actor kl_ctrl: {self.kl_ctrl}')

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs
    
    def extra_info_to_group(self, extra_info, features):
        return '-'.join([str(extra_info[feature]) for feature in features])

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        batch_size = data.batch.batch_size[0]

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'token_level_rewards']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        if self.config.famo.enable:
            if hasattr(data, "non_tensor_batch") and "extra_info" in data.non_tensor_batch:
                extra_info_list = data.non_tensor_batch["extra_info"]  # this is a list of dicts
                bs = self.config.ppo_mini_batch_size
                # manually chunk into mini-batches
                extra_info_chunks = [
                    extra_info_list[i:i+bs] for i in range(0, len(extra_info_list), bs)
                ]
                task_data = {}
        
        if self.config.log_task_wise_kl:
            if hasattr(data, "non_tensor_batch") and "extra_info" in data.non_tensor_batch:
                extra_info_list = data.non_tensor_batch["extra_info"]  # this is a list of dicts
                bs = self.config.ppo_mini_batch_size
                # manually chunk into mini-batches
                extra_info_chunks = [
                    extra_info_list[i:i+bs] for i in range(0, len(extra_info_list), bs)
                ]
        

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, micro_batch_idx = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)
                    micro_batch_idx = []
                    offset = 0
                    for mb in micro_batches:
                        sz = len(mb)
                        micro_batch_idx.append(list(range(offset, offset+sz)))
                        offset += sz

                self.actor_optimizer.zero_grad()

                if self.config.famo.enable:
                    # ----- NEW: container for grouped stats this mini-batch -----
                    task_wise_obj = {}
                    # -----------------------------------------------------------
                    extra_info_mini_batch = extra_info_chunks[batch_idx]

                    for group in self.groups:
                        task_wise_obj[group] = []
                
                if self.config.log_task_wise_kl:
                    task_wise_kl = {}
                    extra_info_mini_batch = extra_info_chunks[batch_idx]
                    for group in self.groups:
                        task_wise_kl[group] = []

                for data_idx, data in enumerate(micro_batches):
                    print("MICROBATCH STEP")
                    data = data.cuda()  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    entropy_coeff = self.config.entropy_coeff

                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                    pg_loss, pg_clipfrac, ppo_kl, loss_vector = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                log_prob=log_prob,
                                                                                advantages=advantages,
                                                                                eos_mask=response_mask,
                                                                                cliprange=clip_ratio,
                                                                                cliprange_high=self.config.clip_ratio_high,
                                                                                agg_loss_mode=self.config.agg_loss_mode,
                                                                                loss_vector=self.config.famo.enable,
                                                                                loss_type=self.config.loss_type)
                    # compute entropy loss from entropy
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                        if self.config.log_task_wise_kl:
                            kl_loss_vec = masked_mean(kld, response_mask, axis=-1)

                            cur_micro_batch_idx = micro_batch_idx[data_idx]

                            for i, sample_kl_loss in zip(cur_micro_batch_idx, kl_loss_vec):
                                ei = extra_info_mini_batch[i]
                                group = self.extra_info_to_group(ei, features=getattr(self.config, 'famo_features', ('difficulty','type')))
                                task_wise_kl[group].append(sample_kl_loss.detach()) 

                    if self.config.kl_ctrl.type == 'adaptive':
                        print(f'Used latest kl coefficient {self.config.kl_loss_coef}')
                        self.kl_ctrl.update(current_kl=kl_loss.detach().item(), n_steps=batch_size)
                        print(f'Updating kl_ctrl: {self.kl_ctrl.value}')
                    loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, data)

                    if self.config.famo.enable:
                        # Map idxs_in_minibatch -> global indices into the full batch
                        # recalculate sample wise loss
                        pg_loss_vec, pg_clipfrac_vec, ppo_kl_vec = loss_vector
                        entropy_vec = verl_F.masked_mean(entropy, response_mask, axis=-1)
                        policy_loss_vec = pg_loss_vec - entropy_vec * self.config.entropy_coeff
                        
                        if self.config.use_kl_loss:
                            kl_loss_vec = masked_mean(kld, response_mask, axis=-1)
                            policy_loss_vec = policy_loss_vec + kl_loss_vec * self.config.kl_loss_coef

                        cur_micro_batch_idx = micro_batch_idx[data_idx]

                        for i, sample_loss in zip(cur_micro_batch_idx, policy_loss_vec):
                            ei = extra_info_mini_batch[i]
                            group = self.extra_info_to_group(ei, features=getattr(self.config, 'famo_features', ('difficulty','type')))
                            task_wise_obj[group].append(sample_loss.detach()) 

                        # ------------------------------------------------
                    
                

                # Store params before optimizer step to check for changes
                # if self.config.famo.enable:
                #     params_before = [p.clone().detach() for p in self.actor_module.parameters()]

                grad_norm = self._optimizer_step()
                print(self.actor_optimizer.state_dict()['param_groups'][0]['lr'])

                # # Check if parameters have changed
                # if self.config.famo.enable:
                #     params_after = [p.clone().detach() for p in self.actor_module.parameters()]
                #     params_changed = False
                #     for p_before, p_after in zip(params_before, params_after):
                #         if not torch.equal(p_before, p_after):
                #             params_changed = True
                #             break
                #     print(f"\nModel parameters changed after optimizer_step: {params_changed}\n")

                # recalculate loss after every update

                if self.config.famo.enable:

                    task_wise_new_obj = {}
                    task_wise_reward = {}
                    task_wise_logw = {}
                    
                    for group in self.groups:
                        task_wise_new_obj[group] = []
                        task_wise_logw[group] = []
                        task_wise_reward[group] = []
                    for data_idx, data in enumerate(micro_batches):
                        print("MICROBATCH STEP")
                        data = data.cuda()  # actor device is cpu when using offload
                        responses = data['responses']
                        response_length = responses.size(1)
                        attention_mask = data['attention_mask']
                        response_mask = attention_mask[:, -response_length:]
                        old_log_prob = data['old_log_probs']
                        advantages = data['advantages'] 

                        clip_ratio = self.config.clip_ratio
                        entropy_coeff = self.config.entropy_coeff

                        
                        with torch.no_grad():
                            new_entropy, new_log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                            new_pg_loss, new_pg_clipfrac, new_ppo_kl, new_loss_vector = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                        log_prob=new_log_prob,
                                                                                        advantages=advantages,
                                                                                        eos_mask=response_mask,
                                                                                        cliprange=clip_ratio,
                                                                                        cliprange_high=self.config.clip_ratio_high,
                                                                                        agg_loss_mode=self.config.agg_loss_mode,
                                                                                        loss_vector=self.config.famo.enable,
                                                                                        loss_type=self.config.loss_type)
                            
                            reward_vec, logw_vec  = core_algos.compute_is_token_reward(old_log_prob=old_log_prob,
                                                log_prob=new_log_prob,
                                                token_level_rewards=data['token_level_rewards'],
                                                eos_mask=response_mask)

                        new_pg_loss_vec, new_pg_clipfrac_vec, new_ppo_kl_vec = new_loss_vector
                        new_entropy_vec = verl_F.masked_mean(new_entropy, response_mask, axis=-1)
                        new_policy_loss_vec = new_pg_loss_vec - new_entropy_vec * self.config.entropy_coeff
                        
                        if self.config.use_kl_loss:
                            ref_log_prob = data['ref_log_prob']
                            # compute kl loss
                            new_kld = core_algos.kl_penalty(logprob=new_log_prob,
                                                        ref_logprob=ref_log_prob,
                                                        kl_penalty=self.config.kl_loss_type)
                            new_kl_loss_vec = masked_mean(new_kld, response_mask, axis=-1)
                            new_policy_loss_vec = new_policy_loss_vec + new_kl_loss_vec * self.config.kl_loss_coef

                        cur_micro_batch_idx = micro_batch_idx[data_idx]

                        for row, i in enumerate(cur_micro_batch_idx):
                            ei = extra_info_mini_batch[i]   # <-- IMPORTANT: use i, not row
                            group = self.extra_info_to_group(
                                ei,
                                features=getattr(self.config, 'famo_features', ('difficulty','type'))
                            )

                            task_wise_new_obj[group].append(new_policy_loss_vec[row].detach())
                            task_wise_logw[group].append(logw_vec[row].detach())
                            task_wise_reward[group].append(reward_vec[row].detach())

                        # ------------------------------------------------
                            
                    # compute task-wise total and mean loss where each key has list of tensors
                    task_wise_total_loss = {
                        group: torch.sum(torch.stack(losses)) if len(losses) > 0 else torch.tensor(0.0, device='cuda:0')
                        for group, losses in task_wise_obj.items()
                    }
                    # compute task-wise total and mean loss where each key has list of tensors
                    task_wise_new_total_loss = {group: torch.sum(torch.stack(losses)) if len(losses) > 0 else torch.tensor(0.0, device='cuda:0') for group, losses in task_wise_new_obj.items()}
                    task_wise_count = {group: torch.tensor(len(losses), device='cuda:0') if len(losses) > 0 else torch.tensor(0, device='cuda:0') for group, losses in task_wise_obj.items()}
                    #collect across gpus
                    for group in task_wise_new_total_loss:
                        torch.distributed.all_reduce(task_wise_new_total_loss[group], op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(task_wise_total_loss[group], op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(task_wise_count[group], op=torch.distributed.ReduceOp.SUM)
            
                    task_wise_diff = {group: task_wise_new_total_loss[group] - task_wise_total_loss[group] for group in task_wise_new_total_loss}
                 
                    
                    for k, v in task_wise_diff.items():
                        task_data[f'actor/task_wise_diff_{k}_{batch_idx}'] = v.detach().item()
                        task_data[f'actor/task_wise_count_{k}_{batch_idx}'] = task_wise_count[k].detach().item()

                    task_wise_total_reward = {}
                
                    # 1) old sums / counts
                    for g in self.groups:
                        if len(task_wise_reward[g]) > 0:
                            r = torch.stack(task_wise_reward[g])                # [Ng]
                            task_wise_total_reward[g] = r.sum()
                        else:
                            task_wise_total_reward[g] = torch.tensor(0.0, device='cuda:0')

                    for g in self.groups:
                        torch.distributed.all_reduce(task_wise_total_reward[g], op=torch.distributed.ReduceOp.SUM)


                    # 2) WIS per-group: need group-wise max(logw) across ranks, then sum weights/numerators
                    task_wise_max_logw = {}
                    for g in self.groups:
                        if len(task_wise_logw[g]) > 0:
                            lw = torch.stack(task_wise_logw[g])                 # [Ng]
                            task_wise_max_logw[g] = lw.max()
                        else:
                            task_wise_max_logw[g] = torch.tensor(-1e9, device='cuda:0')

                    # all-reduce MAX to get global group max
                    for g in self.groups:
                        torch.distributed.all_reduce(task_wise_max_logw[g], op=torch.distributed.ReduceOp.MAX)

                    task_wise_total_w = {}
                    task_wise_total_wr = {}
                    task_wise_total_w2 = {}

                    for g in self.groups:
                        if len(task_wise_reward[g]) > 0:
                            r  = torch.stack(task_wise_reward[g])              # [Ng]
                            lw = torch.stack(task_wise_logw[g])                # [Ng]
                            w  = torch.exp(lw - task_wise_max_logw[g])         # stable weights

                            task_wise_total_w[g]  = w.sum()
                            task_wise_total_wr[g] = (w * r).sum()
                            task_wise_total_w2[g] = (w * w).sum()
                        else:
                            task_wise_total_w[g]  = torch.tensor(0.0, device='cuda:0')
                            task_wise_total_wr[g] = torch.tensor(0.0, device='cuda:0')
                            task_wise_total_w2[g] = torch.tensor(0.0, device='cuda:0')

                    for g in self.groups:
                        torch.distributed.all_reduce(task_wise_total_w[g],  op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(task_wise_total_wr[g], op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(task_wise_total_w2[g], op=torch.distributed.ReduceOp.SUM)

                    task_wise_new_wis = {}
                    task_wise_ess = {}
                    for g in self.groups:
                        task_wise_new_wis[g] = task_wise_total_wr[g] / (task_wise_total_w[g] + 1e-8)
                        task_wise_ess[g] = (task_wise_total_w[g] * task_wise_total_w[g]) / (task_wise_total_w2[g] + 1e-8)

                    task_wise_reward_diff = {g: task_wise_new_wis[g] - task_wise_total_reward[g]/ task_wise_count[g].clamp(min=1).to(task_wise_total_reward[g].dtype) for g in self.groups}

                    for g, v in task_wise_reward_diff.items():
                        task_data[f'actor/task_wise_reward_diff_{g}_{batch_idx}'] = v.detach().item()
                        # optional but highly useful:
                        task_data[f'actor/task_wise_ess_full_{g}_{batch_idx}'] = task_wise_ess[g].detach().item()
                        ess_frac = task_wise_ess[g] / task_wise_count[g].clamp(min=1).to(task_wise_ess[g].dtype)
                        task_data[f'actor/task_wise_ess_frac_{g}_{batch_idx}'] = ess_frac.detach().item()


                    append_to_dict(metrics, task_data)

                    
                
                if self.config.log_task_wise_kl:
                    task_wise_count = {group: torch.tensor(len(losses), device='cuda:0') if len(losses) > 0 else torch.tensor(0, device='cuda:0') for group, losses in task_wise_kl.items()}
                    task_wise_total_kl_loss = {group: torch.sum(torch.stack(losses)) if len(losses) > 0 else torch.tensor(0.0, device='cuda:0') for group, losses in task_wise_kl.items()}

                    for group in task_wise_total_kl_loss:
                        torch.distributed.all_reduce(task_wise_total_kl_loss[group], op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(task_wise_count[group], op=torch.distributed.ReduceOp.SUM)
                        #task_wise_total_kl_loss[group] = task_wise_total_kl_loss[group] / task_wise_count[group]
                    
                    task_wise_kl_data = {}
                    for k, v in task_wise_total_kl_loss.items():
                        task_wise_kl_data[f'actor/task_wise_kl_loss_{k}_{batch_idx}'] = v.detach().item()
                        task_wise_kl_data[f'actor/task_wise_kl_count_{k}_{batch_idx}'] = task_wise_count[k].detach().item()

                data = {'actor/grad_norm': grad_norm.detach().item()}
                append_to_dict(metrics, data)
            if self.config.log_task_wise_kl:
                # group keys and values w.r.t. batch ids
                batch_to_task_wise_kl = Counter()
                batch_to_task_wise_count = Counter()
                for k, v in task_wise_kl_data.items():
                    if 'kl_count' in k:
                        group_name = k.split('_')[-2]
                        batch_to_task_wise_count[group_name] += v
                    elif 'kl_loss' in k:
                        group_name = k.split('_')[-2]
                        batch_to_task_wise_kl[group_name] += v
                    else: 
                        raise ValueError(f'Invalid key: {k}')
                kl_loss_task_wise = {}
                for k, v in batch_to_task_wise_kl.items():
                    if batch_to_task_wise_count[k] == 0:
                        kl_loss_task_wise['kl_loss_' + k] = 0.0
                    else:
                        kl_loss_task_wise['kl_loss_' + k] = v / batch_to_task_wise_count[k]
                
                append_to_dict(metrics, kl_loss_task_wise)
        self.actor_optimizer.zero_grad()
        if self.config.kl_ctrl.type == 'adaptive':
            self.config.kl_loss_coef = float(self.kl_ctrl.value)
            print(f'Adapted Actor kl_loss_coef={self.config.kl_loss_coef}')
        return metrics

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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem, collate_fn
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
import verl.utils.torch_functional as verl_F

from ray.util import pdb 

from collections import defaultdict, deque, Counter
import random
import torch
import gc



WorkerType = Type[Worker]



def dataprotoitem_to_dataproto(item: DataProtoItem) -> DataProto:
    """Convert a DataProtoItem to a DataProto object"""
    return DataProto.from_dict(
        tensors=item.batch,  # TensorDict is already in correct format
        non_tensors=item.non_tensor_batch,  # Dict is already in correct format 
        meta_info=item.meta_info
    )

def _infer_step_from_resume_path(path: str) -> int:
    base = os.path.basename(path.rstrip("/"))
    if base.startswith("global_step_"):
        return int(base.split("_")[-1])
    return 0

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        normalize=True)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns

        unnormalized_advantages, _ = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        normalize=False)
        data.batch['unnormalized_advantages'] = unnormalized_advantages
    elif adv_estimator == 'rloo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        response_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'drgrpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        normalize=False)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError(f'Advantage estimator {adv_estimator} not implemented')
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }

# put this near the top of fit(), before the loop
def _wg_union(batch, dpw, fn):
    padded, pad = pad_dataproto_to_divisor(batch, dpw)
    out = fn(padded)
    if pad:
        out = unpad_dataproto(out, pad)  # may return a DataProtoItem slice
        # normalize type to DataProto (optional but tidy)
        if isinstance(out, DataProtoItem):
            out = DataProto(batch=out.batch, non_tensor_batch=out.non_tensor_batch, meta_info=out.meta_info)
    return batch.union(out)

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)
        
        if self.config.replay.enabled:
            # Ring buffers: each task -> deque(maxlen=100) of per-sample tensors
            self.replay = defaultdict(lambda: deque(maxlen=self.config.replay.max_buffer_size))

            # Tasks you consider inactive; treat as a set of task keys (e.g., "easy:math" or whatever your grouping produces)
            self.inactive_tasks = set()
            self.replay_prob = getattr(self.config.replay, "prob", 0.05)
            self.replay_per_group = getattr(self.config.replay, "max_prompts_per_group", 8)

        self._create_dataloader()

    def _create_dataloader(self):
        # from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')

        # filter train dataset
        if self.config.trainer.sec.bandit.groups_to_train is not None:
            self.train_dataset = [item for item in self.train_dataset if self.data_to_group(item) in self.config.trainer.sec.bandit.groups_to_train]
            print(len(self.train_dataset))

        train_batch_size = self.config.data.train_batch_size
        if self.config.trainer.rejection_sample:
            train_batch_size *= self.config.trainer.rejection_sample_multiplier
            train_batch_size = int(train_batch_size)

        from verl.utils.auto_curriculum.dataloader import RandomDataloader, PriorityDataloader, BanditDataloader
        from verl.utils.auto_curriculum.priority import DifficultyPriority, BanditPriority, ReverseDifficultyPriority, get_bandit_priority
        
        if not self.config.trainer.sec.enable:
            DataLoader = RandomDataloader
            priority_func = None
        elif self.config.trainer.sec.strategy == 'random':
            DataLoader = RandomDataloader
            priority_func = None
        elif self.config.trainer.sec.strategy == 'difficulty':
            DataLoader = PriorityDataloader
            priority_func = DifficultyPriority()
        elif self.config.trainer.sec.strategy == 'reverse_difficulty':
            DataLoader = PriorityDataloader
            priority_func = ReverseDifficultyPriority()
        elif self.config.trainer.sec.strategy == 'bandit':
            DataLoader = BanditDataloader
            print("using bandit priority")
            famo_args = getattr(self.config.trainer, "famo", {}) 
            priority_func = get_bandit_priority(dataset=self.train_dataset, **self.config.trainer.sec.bandit, **famo_args)
        else:
            raise ValueError(f'Invalid strategy: {self.config.trainer.sec.strategy}')

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=train_batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           priority_func=priority_func,
                                           max_steps=self.config.trainer.total_training_steps,
                                           sample_with_replacement=self.config.trainer.sec.sample_with_replacement) # if total_training_steps is not None, then we let the dataloader run forever and stop when the total_training_steps is reached instead of counting epochs
        if self.config.trainer.famo.enable:
            self.train_dataloader.priority_func.update_training_lr(self.config.actor_rollout_ref.actor.optim.lr)
        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')

        self.val_dataloader = RandomDataloader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=False,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            n_val_samples = self.config.actor_rollout_ref.rollout.n_val
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_gen_batch_padded.meta_info['val_temperature'] = self.config.actor_rollout_ref.rollout.val_temperature
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('Validation: Generation end.')


            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        if getattr(self.train_dataloader, 'priority_func', None) is not None:
            self.config.actor_rollout_ref.actor.groups = list(self.train_dataloader.priority_func.group_to_idx.keys())
            self.config.actor_rollout_ref.ref.groups = list(self.train_dataloader.priority_func.group_to_idx.keys())
        else:
            self.idx_to_group = {idx: self.data_to_group(item) for idx, item in enumerate(self.train_dataset)}
            self.group_to_idx = {group: [] for group in self.idx_to_group.values()}
            for idx, group in self.idx_to_group.items():
                self.group_to_idx[group].append(idx)
            self.config.actor_rollout_ref.actor.groups = list(self.group_to_idx.keys())
            self.config.actor_rollout_ref.ref.groups = list(self.group_to_idx.keys())

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in ['grpo', 'drgrpo', 'rloo']:
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        resume_actor = self.config.actor_rollout_ref.get("resume_from", None)
        if resume_actor is not None:
            self.global_steps_add = _infer_step_from_resume_path(resume_actor)
            print(f"[trainer] Factoring in {self.global_steps_add} as we are Resuming from step {self.global_steps_add}")
        else:
            self.global_steps_add = 0

        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps + self.global_steps_add}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps + self.global_steps_add}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)
        
        # if famo is enabled save priority's optimizer state
        if self.config.trainer.famo.enable:
            priority_local_path = os.path.join(self.config.trainer.default_local_dir, 'priority',
                                            f'global_step_{self.global_steps + self.global_steps_add}')
            priority_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'priority')
            self.train_dataloader.priority_func.save_checkpoint(priority_local_path, priority_remote_path)


    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

       
        dpw = self.actor_rollout_wg.world_size

        # --- NEW: Initialize state variables for accumulation ---
        accumulated_batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        num_filtered_samples = 0
        initial_group_ratios = None
        self.init_rollouts_per_task = self.config.trainer.sec.bandit.dynamic_rollouts.rollouts_per_task
        self.old_batch_acc = Counter()
        self.ensure_famo_ratio = self.config.trainer.sec.bandit.ensure_famo_ratio
        self.try_famo_ratio = self.config.trainer.sec.bandit.try_famo_ratio
        
        # --- END NEW ---

         # always set ensure_famo_ratio initially to false if ensure_famo_ratio_from is greater than 0
        if self.global_steps < self.config.trainer.sec.bandit.ensure_famo_ratio_from:
            self.ensure_famo_ratio = False
        
        # always set try_famo_ratio initially to false if try_famo_ratio_from is greater than 0
        if self.global_steps < self.config.trainer.sec.bandit.try_famo_ratio_from:
            self.try_famo_ratio = False




        for _ in range(self.config.trainer.total_epochs):
            
            for batch_dict in self.train_dataloader:
                if self.config.replay.enabled:
                    self._store_batch_in_replay(batch_dict)

                    # 2) optionally append replay samples
                    batch_dict = self._append_replay_tokenized(batch_dict)

                # 3) proceed as usual
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                metrics = {}
                timing_raw = {}

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                # This code matches a prompt ID with its N responses.
                batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                        dtype=object)

                if self.config.trainer.sec.bandit.knapsack.enabled or self.config.trainer.sec.bandit.dynamic_rollouts.enabled:
                    n_list = [int(self.train_dataloader.priority_func.prompt_to_n[i]) for i in range(len(batch))]
                    extras = batch.non_tensor_batch['extra_info']

                    # Optional: validate difficulty is an int for all rows where it's present
                    for ex in extras:
                        if 'difficulty' in ex:
                            assert int(ex['difficulty']) == float(ex['difficulty']), \
                                f"difficulty must be integer-like, got {ex['difficulty']}"

                    # Build group ids using your helper
                    groups = [self.extra_info_to_group(ex) for ex in extras] # list of strings

                    print(set(n_list), 'unique n')
                
                if self.config.trainer.sec.bandit.dynamic_rollouts.enabled:
                    group_n = self.train_dataloader.priority_func.group_to_n
                    print(group_n, 'group n based on current filtered ratio')



            


                with _timer('step', timing_raw):
                    # generate a batch
                    if self.config.trainer.sec.bandit.knapsack.enabled or self.config.trainer.sec.bandit.dynamic_rollouts.enabled:
                        merged_parts = []

                        with _timer('gen', timing_raw):
                            for n_value in sorted(set(n_list)):
                                if n_value == 0:
                                    continue
                                rows = [i for i in range(len(n_list)) if n_list[i] == n_value]
                                if len(rows) == 0:
                                    continue

                                do_sample = gen_batch.meta_info.get('do_sample', True)
                                # force greedy to 1
                                n_for_this = 1 if not do_sample else int(n_value)

                                # ---- Build sub-batches from existing APIs (no new helpers)
                                # tensors for generation
                                gen_items  = [gen_batch[i] for i in rows]        # DataProtoItem list
                                gen_sub    = collate_fn(gen_items)               # -> DataProto
                                gen_sub.meta_info = gen_batch.meta_info          # preserve meta_info (1 line)

                                # prompt-side metadata (tensors + non-tensors)
                                meta_items = [batch[i] for i in rows]            # DataProtoItem list
                                batch_sub  = collate_fn(meta_items)              # -> DataProto

                                # Force-attach the uids from the parent batch for these rows
                                uids_sub = batch.non_tensor_batch['uid'][rows]
                                batch_sub.non_tensor_batch['uid'] = uids_sub


                                # ---- Call generate_sequences exactly as before, but per-group
                                # Make its internal repeat_interleave use n_for_this:
                                gen_sub.meta_info.setdefault("sampling_overrides", {})["n"] = int(n_for_this)

                                # before calling generate_sequences
                                dp_world = self.actor_rollout_wg.world_size  # or worker_group.world_size
                                bs = len(gen_sub)
                                rem = bs % dp_world
                                pad = (dp_world - rem) % dp_world

                                if pad:
                                    # make a 1-row DataProto from the last row
                                    last = DataProto(
                                        batch=gen_sub.batch[-1:] if gen_sub.batch is not None else None,
                                        non_tensor_batch={k: v[-1:] for k, v in gen_sub.non_tensor_batch.items()},
                                        meta_info=gen_sub.meta_info,
                                    )
                                    last_repeat = last.repeat(repeat_times=pad, interleave=False)  # instance method
                                    gen_sub_padded = DataProto.concat([gen_sub, last_repeat])      # no axis kwarg
                                else:
                                    gen_sub_padded = gen_sub

                                gen_out_sub = self.actor_rollout_wg.generate_sequences(gen_sub_padded)

                                # If we padded, drop the extra rows from BOTH the input-side bookkeeping and the outputs
                                if pad:
                                    trim = pad * n_for_this   # remove all rows created by padded prompts
                                    trimmed_batch = gen_out_sub.batch[:-trim] if gen_out_sub.batch is not None else None
                                    trimmed_ntb   = {k: v[:-trim] for k, v in gen_out_sub.non_tensor_batch.items()}
                                    gen_out_sub = DataProto(batch=trimmed_batch,
                                                            non_tensor_batch=trimmed_ntb,
                                                            meta_info=gen_out_sub.meta_info)

                                bs = len(batch_sub)                      # unpadded prompts in this group
                                expected = bs * n_for_this
                                actual   = len(gen_out_sub)
                                assert actual == expected, f"n mismatch: asked {n_for_this} => {expected}, got {actual}"
                                # ---- Repeat ONLY metadata to match B_group * n_for_this
                                
                                batch_part = batch_sub.repeat(repeat_times=int(n_for_this), interleave=True)
                                # Align & merge
                                assert len(batch_part) == len(gen_out_sub), "row count mismatch in group merge"
                                merged_parts.append(batch_part.union(gen_out_sub))
                            
                        # stitch all groups back together
                        batch = DataProto.concat(merged_parts)
                    else:
                        with _timer('gen', timing_raw):
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                    # TODO: remove this
                    # # print(f'{batch.meta_info=}')
                    # # print(f'{batch.non_tensor_batch["extra_info"]=}')
                    # print(f'{type(batch.non_tensor_batch["extra_info"][0]["embedding"])=}')
                    # print(f'{batch.non_tensor_batch["extra_info"][0]["embedding"].shape=}')

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            # if self.config.trainer.sec.bandit.knapsack.enabled:
                            #     batch = _wg_union(batch, dpw, self.critic_wg.compute_values)
                            # else:
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            # if self.config.trainer.sec.bandit.knapsack.enabled:
                            #     batch = _wg_union(batch, dpw, self.rm_wg.compute_rm_score)
                            # else:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        extras = batch.non_tensor_batch['extra_info']

                        # Optional: validate difficulty is an int for all rows where it's present
                        for ex in extras:
                            if 'difficulty' in ex:
                                assert int(ex['difficulty']) == float(ex['difficulty']), \
                                f"difficulty must be integer-like, got {ex['difficulty']}"

                        # Build group ids using your helper
                        groups = [self.extra_info_to_group(ex) for ex in extras] # list of strings

                        # get group corresponding to each uid
                        uid_to_group = {}
                        for uid, group in zip(uids, groups):
                            if uid not in uid_to_group:
                                uid_to_group[uid] = group
                            else:
                                assert uid_to_group[uid] == group
                        
                        # --- NEW: compute initial group ratios once, based on unique_uids ---
                        if initial_group_ratios is None:
                            if not self.config.trainer.sec.bandit.efficient_sampling.enabled:
                                uid_groups_initial = defaultdict(set)
                                for uid in unique_uids:
                                    g = uid_to_group[uid]
                                    uid_groups_initial[g].add(uid)
                                total_unique_initial = sum(len(s) for s in uid_groups_initial.values())
                                initial_group_ratios = {
                                    g: len(s) / total_unique_initial
                                    for g, s in uid_groups_initial.items()
                                }
                            else:
                                initial_group_ratios = self.train_dataloader.priority_func.original_group_ratios
                        print(f'Initial group ratios: {initial_group_ratios}')
                        # --- END NEW --
                        

                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        solve_none_group_wise = defaultdict(int)
                        solve_all_group_wise = defaultdict(int)
                        solve_half_group_wise = defaultdict(int)
                        acc_group_wise = defaultdict(int)
                        var_group_wise = defaultdict(float)
                        total_group_wise = defaultdict(int)
                        filtered_prompts_group_wise = defaultdict(int)
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                            total_group_wise[uid_to_group[uid]] += 1
                            acc_group_wise[uid_to_group[uid]] += (uid_rewards >= 1-1e-6).sum()/len(uid_rewards)
                           
                            # calculate vatiance of (uid_rewards >= 1-1e-6).float()
                            var_group_wise[uid_to_group[uid]] += (uid_rewards >= 1-1e-6).float().var()

                            
                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards == 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                                solve_none_group_wise[uid_to_group[uid]] += 1
                                filtered_prompts_group_wise[uid_to_group[uid]] += 1
                            elif (uid_rewards == 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1
                                solve_all_group_wise[uid_to_group[uid]] += 1
                                filtered_prompts_group_wise[uid_to_group[uid]] += 1
                            elif (uid_rewards <= 1-1e-6).all() and self.config.trainer.pure_acc_filter: # avoid all 0.1s
                                valid_mask[uid_mask] = False
                                solve_none += 1
                                solve_none_group_wise[uid_to_group[uid]] += 1
                                filtered_prompts_group_wise[uid_to_group[uid]] += 1
                            elif (uid_rewards >= 1-1e-6).sum() >= len(uid_rewards)/2:
                                solve_half_group_wise[uid_to_group[uid]] += 1
                        
                        # Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_all'] = solve_all
                        metrics['batch/num_zero_grad'] = solve_none + solve_all
                        for group in solve_none_group_wise:
                            metrics[f'batch/solve_none_{group}'] = solve_none_group_wise[group]
                        for group in solve_all_group_wise:
                            metrics[f'batch/solve_all_{group}'] = solve_all_group_wise[group]
                        for group in total_group_wise:
                            metrics[f'batch/filtered_prompts_{group}'] = filtered_prompts_group_wise[group]
                            metrics[f'batch/filtered_ratio_{group}'] = (
                                filtered_prompts_group_wise[group] / max(total_group_wise[group], 1)
                            )
                        for group in total_group_wise:
                            metrics[f'batch/total_{group}'] = total_group_wise[group]
                        print("total_group_wise", total_group_wise)
                        for group in acc_group_wise:
                            metrics[f'batch/acc_{group}'] = acc_group_wise[group]/total_group_wise[group]
                            metrics[f'batch/acc_delta_{group}'] = acc_group_wise[group]/total_group_wise[group] - self.old_batch_acc[group]
                            self.old_batch_acc[group] = acc_group_wise[group]/total_group_wise[group]
                        for group in var_group_wise:
                            metrics[f'batch/var_{group}'] = var_group_wise[group]/total_group_wise[group]
                        for group in solve_half_group_wise:
                            metrics[f'batch/solve_half_{group}'] = solve_half_group_wise[group]/total_group_wise[group]
                        if self.config.trainer.famo.enable:
                            self.train_dataloader.priority_func.update_solve_none({group: solve_none_group_wise[group]/total_group_wise[group] for group in solve_none_group_wise})
                            self.train_dataloader.priority_func.update_solve_all({group: solve_all_group_wise[group]/total_group_wise[group] for group in solve_all_group_wise})    
                            self.train_dataloader.priority_func.update_batch_acc({group: acc_group_wise[group]/total_group_wise[group] for group in acc_group_wise})
                            self.train_dataloader.priority_func.update_batch_acc_delta({group: metrics[f'batch/acc_delta_{group}'] for group in acc_group_wise})
                            self.train_dataloader.priority_func.update_solve_half({group: solve_half_group_wise[group]/total_group_wise[group] for group in solve_half_group_wise})
                            self.train_dataloader.priority_func.update_var({group: var_group_wise[group]/total_group_wise[group] for group in var_group_wise})
                            self.train_dataloader.priority_func.update_filtered_ratio({
                                group: filtered_prompts_group_wise[group] / max(total_group_wise[group], 1)
                                for group in filtered_prompts_group_wise
                            })



                        if self.config.trainer.rejection_sample:
                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                num_gen_batches += 1
                                print("Current generation batch was fully invalid. Skipping to fetch more data...")
                                continue # Skips to the next batch_dict

                            # Filter batch to keep only valid samples
                            current_filtered_batch = batch[valid_mask]
                            current_filtered_batch = dataprotoitem_to_dataproto(current_filtered_batch)


                            # 3. Accumulate the valid samples
                            if accumulated_batch is None:
                                accumulated_batch = current_filtered_batch
                            else:
                                accumulated_batch = DataProto.concat([accumulated_batch, current_filtered_batch])

                            # 4. Update the total count of unique prompts we've gathered
                            num_prompt_in_batch = len(set(accumulated_batch.non_tensor_batch['uid']))
                            num_gen_batches += 1

                            # --- NEW: Regeneration Control Flow ---
                            prompt_bsz = self.config.data.train_batch_size # Define your target prompt batch size

                            if (self.ensure_famo_ratio or self.try_famo_ratio) and initial_group_ratios is not None : # remove ensure famo here

                                # (ii) current unique_uids per group in accumulated_batch
                                acc_uids = accumulated_batch.non_tensor_batch['uid']
                                acc_extras = accumulated_batch.non_tensor_batch['extra_info']
                                acc_groups = [self.extra_info_to_group(ex) for ex in acc_extras]

                                acc_group_to_uids = defaultdict(set)
                                for uid, g in zip(acc_uids, acc_groups):
                                    acc_group_to_uids[g].add(uid)

                                if not self.config.trainer.sec.bandit.dynamic_rollouts.enabled:

                                    # (iii) desired unique_uids per group = initial_ratio * prompt_bsz
                                    desired_counts = {
                                        g: int(round(initial_group_ratios.get(g, 0.0) * prompt_bsz))
                                        for g in initial_group_ratios.keys()
                                    }
                                else:

                                    # We want datapoints_g ≈ ratio_g * total_datapoints.
                                    # datapoints_g = (#prompts_g) * n_g.
                                    # => #prompts_g ∝ ratio_g / n_g.
                                    print("adjusting weights based on group_n", group_n)

                                    weights = {}
                                    for g, ratio in initial_group_ratios.items():
                                        n_g = group_n.get(g, 1)  # fallback to 1 if somehow missing
                                        weights[g] = (ratio * self.init_rollouts_per_task) / max(n_g, 1e-8) 
                                        #upscale low rollout tasks to have more questions

                                    print("weights", weights)
                                    print("initial_group_ratios", initial_group_ratios)

                                    desired_counts = {
                                        g: int(round(prompt_bsz * (weights[g])))
                                        for g in weights
                                    }
                                    print("desired_counts", desired_counts)


                                # (iv) how many more uids we still need for each group
                                required_samples = {}
                                for g, target in desired_counts.items():
                                    current = len(acc_group_to_uids.get(g, set()))
                                    deficit = target - current
                                    if deficit > 0:
                                        required_samples[g] = deficit
                                    else:
                                        required_samples[g] = 0

                            if num_prompt_in_batch < prompt_bsz:
                                max_num_gen_batches = self.config.algorithm.filter_groups.get("max_num_gen_batches", 10) # Get from config
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    print(f"Accumulated {num_prompt_in_batch}/{prompt_bsz} prompts. Keep generating...")
                                    num_filtered_samples += len(valid_mask) - valid_mask.sum()
                                        # --- END NEW ---
                                    
                                    continue # This is the key: skips to the next batch_dict to get more data
                                else:
                                    # raise ValueError(f"Generated {num_gen_batches} times but failed to fill the batch. Stopping.")
                                    # continue with accumulated batch and log length of accumulated batch
                                    print(f"Generated {num_gen_batches} times but failed to fill the batch. Continuing with accumulated batch.")
                                    metrics['batch/accumulated_batch_size'] = len(accumulated_batch)
                                # --- END NEW ---
                            elif self.ensure_famo_ratio and initial_group_ratios is not None and any(required_samples.values()):
                                # (v) if some groups are underfilled, bias next sampling to those groups only
                                print("Required samples is positive for some groups. This should not happen. regenerate")
                                print(required_samples)
                                ## keep it as dict and pass the dict to priority_func
                                #self.train_dataloader.priority_func.set_temp_probs(required_samples)
                                if callable(getattr(getattr(self.train_dataloader, "priority_func", None), "set_temp_probs", None)):
                                    self.train_dataloader.priority_func.set_temp_probs(required_samples)

                                max_num_gen_batches = self.config.algorithm.filter_groups.get("max_num_gen_batches", 10) # Get from config
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    print(f"Accumulated {num_prompt_in_batch}/{prompt_bsz} prompts but some groups are underfilled. Keep generating...")
                                    num_filtered_samples += len(valid_mask) - valid_mask.sum()
                                        # --- END NEW ---
                                    
                                    continue # This is the key: skips to the next batch_dict to get more data
                                else:
                                    # raise ValueError(f"Generated {num_gen_batches} times but failed to fill the batch. Stopping.")
                                    # continue with accumulated batch and log length of accumulated batch
                                    print(f"Generated {num_gen_batches} times but failed to fill the batch. Continuing with accumulated batch.")
                                    metrics['batch/accumulated_batch_size'] = len(accumulated_batch)

                            # log number of regeneration batches
                            metrics['batch/num_regen_batches'] = num_gen_batches
                            metrics['batch/avg_num_filtered_samples'] = num_filtered_samples / num_gen_batches

                            batch = accumulated_batch

                            # --- NEW: ALIGN THE BATCH CORRECTLY FOR VARIABLE ROLLOUTS ---
                            if num_prompt_in_batch > prompt_bsz and not (self.ensure_famo_ratio or self.try_famo_ratio):
                                # 1. Get all unique prompt IDs from the oversized batch
                                unique_uids_in_batch = np.unique(batch.non_tensor_batch['uid'])
                                
                                # 2. Select the first `prompt_bsz` unique IDs to keep
                                uids_to_keep = set(unique_uids_in_batch[:prompt_bsz])
                                
                                # 3. Create a boolean mask to filter the batch
                                filter_mask = torch.tensor([uid in uids_to_keep for uid in batch.non_tensor_batch['uid']], dtype=torch.bool)
                                
                                # 4. Apply the mask to get the correctly sized batch
                                batch = batch[filter_mask]
                                batch = dataprotoitem_to_dataproto(batch)
                            # --- END NEW ALIGNMENT LOGIC ---
                            elif num_prompt_in_batch > prompt_bsz and (self.ensure_famo_ratio or self.try_famo_ratio):
                                # --- NEW: GROUP-RATIO AWARE ALIGNMENT FOR VARIABLE ROLLOUTS ---
                                

                                batch_uids = np.array(batch.non_tensor_batch['uid'])
                                batch_extras = batch.non_tensor_batch['extra_info']
                                batch_groups = [self.extra_info_to_group(ex) for ex in batch_extras]

                                # (ii) unique_uids with valid mask True per group in the accumulated batch
                                group_to_uids = defaultdict(set)
                                for uid, g in zip(batch_uids, batch_groups):
                                    group_to_uids[g].add(uid)

                                rng = np.random.default_rng()
                                chosen_uids = []
                                leftover_uids = []

                                print("desired_counts", desired_counts)


                                # First, try to satisfy per-group targets
                                for g, uids_set in group_to_uids.items():
                                    available = list(uids_set)
                                    target = desired_counts.get(g, 0)
                                    
                                    if target <= 0:
                                        continue

                                    if len(available) <= target:
                                        # Not enough: take all; we'll fill the rest from other groups
                                        chosen_uids.extend(available)
                                    else:
                                        # Enough: randomly choose `target` uids, keep the rest as leftovers
                                        chosen = rng.choice(available, size=target, replace=False)
                                        chosen_uids.extend(chosen.tolist())
                                        leftovers = [u for u in available if u not in chosen]
                                        leftover_uids.extend(leftovers)
                                    
                                    print(len(available), target, len(chosen), "available, target, chosen for group", g)

                                # (v) If we still have fewer than prompt_bsz unique uids, fill from leftovers
                                chosen_uids = list(set(chosen_uids))  # ensure uniqueness
                                if len(chosen_uids) < prompt_bsz:
                                    remaining = prompt_bsz - len(chosen_uids)
                                    # Fill from leftovers or from any available uids if leftovers too small
                                    pool = list(set(leftover_uids) | set(u for s in group_to_uids.values() for u in s))
                                    pool = [u for u in pool if u not in chosen_uids]
                                    if pool:
                                        extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
                                        chosen_uids.extend(extra.tolist())

                                uids_to_keep = set(chosen_uids)

                                # Build mask over the accumulated batch
                                filter_mask = torch.tensor(
                                    [uid in uids_to_keep for uid in batch_uids],
                                    dtype=torch.bool,
                                )

                                batch = batch[filter_mask]
                                batch = dataprotoitem_to_dataproto(batch)

                                # Update num_prompt_in_batch to the number of unique uids we actually kept
                                num_prompt_in_batch = len(uids_to_keep)
                                metrics['batch/num_prompt_in_batch'] = num_prompt_in_batch
                            if (self.ensure_famo_ratio or self.try_famo_ratio):
                                if callable(getattr(getattr(self.train_dataloader, "priority_func", None), "set_temp_probs", None)):
                                    self.train_dataloader.priority_func.set_temp_probs(None)
                                initial_group_ratios = None
                                print("Reset initial_group_ratios, temp_probs")
                            # --- END GROUP-RATIO AWARE ALIGNMENT ---
                                

                            # Round down the final, accumulated batch to the nearest multiple of world size
                            num_trainer_replicas = self.actor_rollout_wg.world_size 
                            max_batch_size = (len(batch) // num_trainer_replicas) * num_trainer_replicas

                            if not max_batch_size:
                                # This can happen if the accumulated batch is still smaller than the world size
                                # after the loop exits (e.g., due to a max_num_gen_batches limit).
                                print(f"Warning: Accumulated batch size ({len(batch)}) is less than world size ({num_trainer_replicas}). Skipping this training step.")
                                
                                # CRITICAL: Reset state and continue to the next data fetch
                                accumulated_batch = None
                                num_prompt_in_batch = 0
                                num_gen_batches = 0
                                initial_group_ratios = None
                                continue

                            size_mask = torch.zeros(batch.batch['input_ids'].shape[0], dtype=torch.bool)
                            size_mask[:max_batch_size] = True
                            batch = batch[size_mask]
                            batch = dataprotoitem_to_dataproto(batch)

                            # # Round down to the nearest multiple of world size
                            # num_trainer_replicas = self.actor_rollout_wg.world_size 
                            # max_batch_size = (current_filtered_batch.batch['input_ids'].shape[0] // num_trainer_replicas) * num_trainer_replicas
                            # if not max_batch_size:
                            #     # give up, you got everything either all wrong or right.
                            #     continue

                            # size_mask = torch.zeros(batch.batch['input_ids'].shape[0], dtype=torch.bool)
                            # size_mask[:max_batch_size] = True
                            # batch = batch[size_mask]
                            # batch = dataprotoitem_to_dataproto(batch)

                        if initial_group_ratios is not None:
                            initial_group_ratios = None

                        # log composition of each batch

                        extras = batch.non_tensor_batch['extra_info']
                        uids = batch.non_tensor_batch['uid']

                        groups = {}

                        for ex, uid in zip(extras, uids):
                            group = self.extra_info_to_group(ex)
                            if group not in groups:
                                groups[group] = set()
                            groups[group].add(uid)   # no need to check if it's already there

                        # count uids in each group
                        for group, uids_in_group in groups.items():
                            metrics[f"batch/group_uid_counts/{group}"] = len(uids_in_group)


                        # recompute old_log_probs
                        with _timer('old_log_prob', timing_raw):
                            # if self.config.trainer.sec.bandit.knapsack.enabled:
                            #     batch = _wg_union(batch, dpw, self.actor_rollout_wg.compute_log_prob)
                            # else:
                            
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                # if self.config.trainer.sec.bandit.knapsack.enabled:
                                #     batch = _wg_union(batch, dpw, self.ref_policy_wg.compute_ref_log_prob)
                                # else:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss
                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                        kl_ctrl=self.kl_ctrl,
                        #                                        kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']


                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)



                    
                    

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # update train_dataloader state

                    responses = batch.batch['responses']
                    response_length = responses.size(-1)
                    attention_mask = batch.batch['attention_mask']
                    response_mask = attention_mask[:, -response_length:]

                    # print(f'{batch.batch["advantages"].shape=}')
                    # print(f'{batch.batch["advantages"]=}')
                    # print(f'{response_mask.shape=}')
                    # print(f'{response_mask=}')
                    # mean_adv = verl_F.masked_mean(batch.batch['advantages'].abs(), response_mask, axis=-1)
                    # print(f'{mean_adv=}')
                    # print(f'{mean_adv.shape=}')

                    metrics.update(self.train_dataloader.get_metrics())
                    metrics['loader/average_difficulty'] = sum([data['difficulty'] for data in batch.non_tensor_batch['extra_info']]) / len(batch.non_tensor_batch['extra_info'])
                    if not self.config.trainer.famo.enable:
                        self.train_dataloader.update(data_info=batch.non_tensor_batch['extra_info'], 
                                                    adv=batch.batch['advantages'], 
                                                    rewards=batch.batch['token_level_rewards'],
                                                    response_mask=response_mask)
                        

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    if self.config.trainer.famo.enable:
                        task_wise_diff = {k: actor_output.meta_info['metrics'][k] for k in actor_output.meta_info['metrics'].keys() if 'task_wise_diff' in k}
                        task_wise_count = {k: actor_output.meta_info['metrics'][k] for k in actor_output.meta_info['metrics'].keys() if 'task_wise_count' in k}
                        task_wise_ess_full = {k: actor_output.meta_info['metrics'][k] for k in actor_output.meta_info['metrics'].keys() if 'task_wise_ess_full' in k}
                        task_wise_ess_frac = {k: actor_output.meta_info['metrics'][k] for k in actor_output.meta_info['metrics'].keys() if 'task_wise_ess_frac' in k}
                        task_wise_reward_diff = {k: actor_output.meta_info['metrics'][k] for k in actor_output.meta_info['metrics'].keys() if 'task_wise_reward_diff' in k}
                        with _timer('update_famo', timing_raw):
                            self.train_dataloader.update(data_info=batch.non_tensor_batch['extra_info'], 
                                                 task_wise_diff=task_wise_diff,
                                                 task_wise_count=task_wise_count,
                                                 task_wise_ess_full=task_wise_ess_full,
                                                 task_wise_ess_frac=task_wise_ess_frac,
                                                 task_wise_reward_diff=task_wise_reward_diff,
                                                 rewards=batch.batch['token_level_rewards'],
                                                 index=batch.non_tensor_batch['uid'],
                                                 response_mask=response_mask)
                            # get replay masks from famo
                            if self.config.replay.enabled:
                                self.inactive_tasks = self.train_dataloader.priority_func.get_inactive_tasks()
                                print(f'Inactive tasks: {self.inactive_tasks}')

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        print(f"Validating at step {self.global_steps}")
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)
                    
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps) % self.config.trainer.save_freq == 0:
                        print(f"Saving checkpoint at step {self.global_steps}")
                        print("clearing cache")
                        #gc.collect()
                        #torch.cuda.empty_cache()
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if self.config.trainer.rejection_sample:
                    accumulated_batch = None
                    num_prompt_in_batch = 0
                    num_gen_batches = 0

                self.global_steps += 1

                if self.config.trainer.famo.enable:
                    if self.train_dataloader.priority_func.dynamic_rollouts.enabled:
                        # update group n after increase in step
                        if self.global_steps > self.train_dataloader.priority_func.dynamic_rollouts.filtered_ratio_update_step_threshold and self.global_steps % self.train_dataloader.priority_func.dynamic_rollouts.filtered_ratio_update_step_freq == 0:
                            # update group n based on filtered ratio
                            self.train_dataloader.priority_func.update_group_n()
                
                if self.global_steps >= self.config.trainer.sec.bandit.ensure_famo_ratio_from:
                    self.ensure_famo_ratio = self.config.trainer.sec.bandit.ensure_famo_ratio
                    # set to true value after initial steps
                
                if self.global_steps >= self.config.trainer.sec.bandit.try_famo_ratio_from:
                    self.try_famo_ratio = self.config.trainer.sec.bandit.try_famo_ratio
                    # set to true value after initial steps

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    
                    # save final checkpoint
                    print(f"Saving checkpoint at step {self.global_steps}")
                    with _timer('save_checkpoint', timing_raw):
                        self._save_checkpoint()
                    return
    
    def extra_info_to_group(self, extra_info):
        return '-'.join([str(extra_info[feature]) for feature in self.config.trainer.sec.bandit.feature])

    def data_to_group(self, data):
        if 'difficulty' in data['extra_info']:
            assert int(data['extra_info']['difficulty']) == float(data['extra_info']['difficulty']) # make sure difficulty is an integer
        return self.extra_info_to_group(data['extra_info'])

    def _task_key(self, extra_info: dict) -> str:
        return self.extra_info_to_group(extra_info)

    def _store_batch_in_replay(self, batch_dict: dict):
        ids  = batch_dict['input_ids']        # (B, L) torch
        attn = batch_dict['attention_mask']   # (B, L)
        pos  = batch_dict['position_ids']     # (B, L)
        extra_list = batch_dict['extra_info']
        data_list = batch_dict.get('data_source', None)
        ability_list = batch_dict.get('ability', None)
        reward_model_list = batch_dict.get('reward_model', None)
        index_list = batch_dict.get('index', None)


        # save detached CPU copies (no graph, no VRAM growth)
        for i in range(ids.size(0)):
            ei = extra_list[i]
            if ei is None:
                continue
            g = self._task_key(ei)
            self.replay[g].append({
                "input_ids":      ids[i].clone().detach(),
                "attention_mask": attn[i].clone().detach(),
                "position_ids":   pos[i].clone().detach(),
                "extra_info":     ei,
                "data_source":    data_list[i],
                "ability":        ability_list[i],
                "reward_model":   reward_model_list[i],
                "index":          index_list[i],
            })

    def _append_replay_tokenized(self, batch_dict: dict):
        """Sample tokenized items from inactive tasks & append to current batch (no re-tokenization)."""
        if not self.replay_prob:
            return batch_dict

        targets = self.inactive_tasks
        picks = []
        for g in sorted(targets):
            if random.random() < self.replay_prob and len(self.replay[g]) > 0:
                k = min(self.replay_per_group, len(self.replay[g]))
                picks.extend(random.sample(list(self.replay[g]), k))

        if not picks:
            return batch_dict

        print(f"Replay pick size: {len(picks)}")

        # stack replay tensors
        rep_ids  = torch.stack([p["input_ids"]      for p in picks], dim=0)  # (K,L)
        rep_attn = torch.stack([p["attention_mask"] for p in picks], dim=0)  # (K,L)
        rep_pos  = torch.stack([p["position_ids"]   for p in picks], dim=0)  # (K,L)

        rep_extra_info = np.array([p["extra_info"] for p in picks],dtype=object)
        rep_data_source = np.array([p["data_source"] for p in picks],dtype=object)
        rep_ability = np.array([p["ability"] for p in picks],dtype=object)
        rep_reward_model = np.array([p["reward_model"] for p in picks],dtype=object)
        rep_index = np.array([p["index"] for p in picks],dtype=object)


        # assert length compatibility (should always match if same tokenizer/max_length)
        L_cur = batch_dict['input_ids'].size(1)
        L_rep = rep_ids.size(1)
        if L_cur != L_rep:
            # If this ever happens, the configs changed; safest is to skip these picks
            # (or implement trivial pad/truncate here).
            return batch_dict

        # concat along batch dimension
        batch_dict['input_ids']      = torch.cat([batch_dict['input_ids'],      rep_ids],  dim=0)
        batch_dict['attention_mask'] = torch.cat([batch_dict['attention_mask'], rep_attn], dim=0)
        batch_dict['position_ids']   = torch.cat([batch_dict['position_ids'],   rep_pos],  dim=0)

        batch_dict['extra_info'] = np.concatenate([batch_dict['extra_info'], rep_extra_info], axis=0)
        batch_dict['data_source'] = np.concatenate([batch_dict['data_source'], rep_data_source], axis=0)
        batch_dict['ability'] = np.concatenate([batch_dict['ability'], rep_ability], axis=0)
        batch_dict['reward_model'] = np.concatenate([batch_dict['reward_model'], rep_reward_model], axis=0)
        batch_dict['index'] = np.concatenate([batch_dict['index'], rep_index], axis=0)

        return batch_dict


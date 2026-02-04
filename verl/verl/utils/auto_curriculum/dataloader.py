from torch.utils.data import DataLoader
import random
from dataclasses import dataclass, field
import heapq
from verl.utils.auto_curriculum.priority import BasePriority, BanditPriority
import copy
from collections import defaultdict
import math
@dataclass(order=True)
class ACLData:
    data_idx: int = field(compare=False)
    priority: tuple = field(compare=True)





class BaseACLDataloader:
    def __init__(self, dataset, batch_size=1, 
                 shuffle=True, drop_last=True, 
                 collate_fn=None, seed=42, 
                 max_steps=None,
                 sample_with_replacement=False,
                 **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.seed = seed
        self.max_steps = max_steps
        self.perpetual = max_steps is not None
        self.sample_with_replacement = sample_with_replacement

        if self.perpetual:
            print('Using perpetual dataloader')

    def _fetch_data(self, idx_list):
        return self.collate_fn([self.dataset[i] for i in idx_list])
    
    def update(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_metrics(self):
        return {}
    
    def __len__(self):
        num_batches = len(self.dataset) // self.batch_size
        if self.drop_last:
            return num_batches
        else:
            if len(self.dataset) % self.batch_size == 0:
                return num_batches
            else:
                return num_batches + 1

    def __next__(self):
        raise NotImplementedError
        
    def __iter__(self):
        self.reset()
        return self



class RandomDataloader(BaseACLDataloader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, *args, **kwargs):
        pass

    def reset(self):
        self.idx_to_sample = list(range(len(self.dataset)))
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.idx_to_sample)

    def __next__(self):
        if self.perpetual and len(self.idx_to_sample) < self.batch_size:
            # refill idx_to_sample if the dataloader is perpetual
            remaining_idx = copy.deepcopy(self.idx_to_sample)
            self.reset()
            self.idx_to_sample = remaining_idx + self.idx_to_sample
            assert len(self.idx_to_sample) >= self.batch_size

        if len(self.idx_to_sample) == 0:
            assert not self.perpetual, 'Dataloader is perpetual but no remaining data'
            raise StopIteration
        if len(self.idx_to_sample) < self.batch_size:
            assert not self.perpetual, 'Dataloader is perpetual but no remaining data'
            if self.drop_last:
                raise StopIteration
            else:
                data = self._fetch_data(self.idx_to_sample)
                self.idx_to_sample = []
                return data
        else:
            data = self._fetch_data(self.idx_to_sample[:self.batch_size])
            

            ### if self.sample_with_replacement is True, dont remove the data from idx_to_sample and reshuflle
            if self.sample_with_replacement:
                random.shuffle(self.idx_to_sample)
            else:
                self.idx_to_sample = self.idx_to_sample[self.batch_size:]
                
            
            return data

class PriorityDataloader(BaseACLDataloader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if 'priority_func' not in kwargs:
            raise ValueError('priority_func must be provided')

        self.priority_func: BasePriority = kwargs['priority_func']
        

    def update(self, *args, **kwargs):
        pass

    def reset(self):

        if self.perpetual:
            total_data_len = len(self.dataset)
            growth_factor = self.max_steps / (total_data_len / self.batch_size)
            

        
        if self.shuffle:
            random.seed(self.seed)
            self.priority_list = [ACLData(data_idx = i, 
                                 priority = (self.priority_func(self.dataset[i]), random.random())
                                 ) 
                                 for i in range(len(self.dataset))]
        else:
            self.priority_list = [ACLData(data_idx = i, 
                                 priority = (self.priority_func(self.dataset[i]), i)
                                 ) 
                                 for i in range(len(self.dataset))]
        
        if self.perpetual and growth_factor > 1:
            diff_to_idx = defaultdict(list)
            for i in range(len(self.priority_list)):
                diff_to_idx[self.priority_list[i].priority[0]].append(i)

            data_to_add = []
            for diff, idx_list in diff_to_idx.items():
                random.shuffle(idx_list)
                idx_list_extended = idx_list * math.ceil(growth_factor)
                num_to_add = math.ceil(len(idx_list) * growth_factor) - len(idx_list)
                assert len(idx_list_extended) >= num_to_add
                data_to_add.extend([self.priority_list[i] for i in idx_list_extended[:num_to_add]])
                print(f'Add {num_to_add} data with diff {diff}')
            self.priority_list = self.priority_list + data_to_add
            print(f'{len(self.priority_list)=}')
            print(f'{len(data_to_add)=}')
            print(self.max_steps * self.batch_size)

            assert len(self.priority_list) >= self.max_steps * self.batch_size

        self.priority_list.sort()

    def __next__(self):
        if self.perpetual and len(self.priority_list) < self.batch_size:
            # refill priority_list if the dataloader is perpetual
            remaining_idx = copy.deepcopy(self.priority_list)
            self.reset()
            self.priority_list = remaining_idx + self.priority_list
            assert len(self.priority_list) >= self.batch_size

        if len(self.priority_list) == 0:
            assert not self.perpetual, 'Dataloader is perpetual but no remaining data'
            raise StopIteration
        if len(self.priority_list) < self.batch_size:
            assert not self.perpetual, 'Dataloader is perpetual but no remaining data'
            if self.drop_last:
                raise StopIteration
            else:
                data = self._fetch_data([i.data_idx for i in self.priority_list])
                self.priority_list = []
                return data
        else:
            data = self._fetch_data([i.data_idx for i in self.priority_list[:self.batch_size]])
            self.priority_list = self.priority_list[self.batch_size:]
            return data

class BanditDataloader(BaseACLDataloader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if 'priority_func' not in kwargs:
            raise ValueError('priority_func must be provided')

        self.priority_func: BanditPriority = kwargs['priority_func']
        self.arm_to_remaining_idx = copy.deepcopy(self.priority_func.arm_to_idx)

        for idx_list in self.arm_to_remaining_idx.values():
            random.shuffle(idx_list)

    def get_metrics(self):
        bandit_metrics = self.priority_func.get_metrics()
        group_to_num_remaining = {f'loader/{group}_num_left': len(self.arm_to_remaining_idx[self.priority_func.group_to_arm[group]]) for group in self.priority_func.group_to_arm}
        return {**bandit_metrics, **group_to_num_remaining}


    @property
    def num_remaining_data(self):
        return sum(len(v) for v in self.arm_to_remaining_idx.values())

    def update(self, *args, **kwargs):
        self.priority_func.update(*args, **kwargs)

    def reset(self):
        self.arm_to_remaining_idx = copy.deepcopy(self.priority_func.arm_to_idx)
        num_data_per_arm = {arm: len(idx) for arm, idx in self.arm_to_remaining_idx.items()}
        assert all(num_data_per_arm[arm] >= self.batch_size for arm in self.arm_to_remaining_idx), f'We should have at least one batch of data for each arm. Got {num_data_per_arm}'
        for idx_list in self.arm_to_remaining_idx.values():
            random.shuffle(idx_list)

    def _sample_data(self):
        arms_to_sample = self.priority_func.sample_arms(self.batch_size)
        data_idx = []
        for arm in arms_to_sample:
            if len(self.arm_to_remaining_idx[arm]) > 0:
                data_idx.append(self.arm_to_remaining_idx[arm].pop(0))
            else:
                assert not self.perpetual, 'Dataloader is perpetual but no remaining data'
                arms_to_sample = random.sample(self._get_arm_has_remaining_data(), 1)
                data_idx.append(self.arm_to_remaining_idx[arms_to_sample[0]].pop(0))
        return data_idx
    
    def _get_arm_has_remaining_data(self):
        return [arm for arm in self.arm_to_remaining_idx if len(self.arm_to_remaining_idx[arm]) > 0]

    def __next__(self):
        
        if self.perpetual and any(len(remaining_idx) < self.batch_size for remaining_idx in self.arm_to_remaining_idx.values()):
            # refill arm_to_remaining_idx if the dataloader is perpetual
            remaining_idx = copy.deepcopy(self.arm_to_remaining_idx)
            self.reset()
            for arm, idx in remaining_idx.items():
                self.arm_to_remaining_idx[arm] = idx + self.arm_to_remaining_idx[arm]
            assert all(len(remaining_idx) >= self.batch_size for remaining_idx in self.arm_to_remaining_idx.values())

        if self.num_remaining_data == 0:
            assert not self.perpetual, 'Dataloader is perpetual but no remaining data'
            raise StopIteration
        if self.num_remaining_data < self.batch_size:
            assert not self.perpetual, 'Dataloader is perpetual but no remaining data'
            if self.drop_last:
                raise StopIteration
            else:
                data_idx = []
                for arm, idx in self.arm_to_remaining_idx.items():
                    data_idx.extend(idx)
                self.arm_to_remaining_idx = {arm: [] for arm in self.arm_to_remaining_idx}
                return self._fetch_data(data_idx)
        else:
            data_idx = self._sample_data()
            return self._fetch_data(data_idx)



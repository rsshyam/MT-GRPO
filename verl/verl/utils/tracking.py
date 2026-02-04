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
A unified tracking interface that supports logging data to different backend
"""
import dataclasses
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Union, Dict, Any

import json

import os
import wandb

import shutil


def copy_hydra_config(source_run_dir: str, checkpoint_root: str):
    src = Path(source_run_dir) / ".hydra"
    dst = Path(checkpoint_root) / ".hydra"

    if not src.exists():
        print(f"[hydra] Source .hydra not found at {src}")
        return

    if dst.exists():
        print(f"[hydra] .hydra already exists at {dst}")
        return

    shutil.copytree(src, dst)
    print(f"[hydra] Copied .hydra to {dst}")



class Tracking(object):
    supported_backend = ['wandb', 'mlflow', 'console']

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = 'console', config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == 'tracking':
                import warnings
                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning)
            else:
                assert backend in self.supported_backend, f'{backend} is not supported'

        self.logger = {}

        if 'tracking' in default_backend or 'wandb' in default_backend:
            
            # resume_path = config.get("actor_rollout_ref", {}).get("resume_from", None)

            # if resume_path:
            #     exp_dir = Path(resume_path).parents[1]  # because you save: .../<exp_dir>/actor/global_step_X
            #     meta = load_wandb_meta(exp_dir)

            #     if meta:
            #         run = wandb.init(
            #             project=meta["project"],
            #             name=meta["name"],
            #             id=meta["run_id"],
            #             resume="allow",
            #             config=config,
            #         )
            #         print(f"[wandb] Resuming run {meta['run_id']} at {exp_dir}")

                   

            #     else:
            #         print(f"[wandb] No wandb_meta.json found at {exp_dir}, starting new run.")

            #         # fresh run
            #         run = wandb.init(project=project_name, name=experiment_name, config=config)
            # else:
            #     print(f"[wandb] No resume path found, starting new run.")
            run = wandb.init(project=project_name, name=experiment_name, config=config)

            # save metadata in the new experiment directory
            exp_dir = config.get("trainer", {}).get("default_local_dir")
            
            source_hydra_dir = config.get("hydra_run_dir")
            copy_hydra_config(source_hydra_dir, exp_dir)
            save_wandb_meta(exp_dir, run)
            self.logger['wandb'] = wandb

        if 'mlflow' in default_backend:
            import mlflow
            mlflow.start_run(run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger['mlflow'] = _MlflowLoggingAdapter()

        if 'console' in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger['console'] = self.console_logger

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def __del__(self):
        if 'wandb' in self.logger:
            print('finish wandb')
            self.logger['wandb'].finish()


class _MlflowLoggingAdapter:

    def log(self, data, step):
        import mlflow
        mlflow.log_metrics(metrics=data, step=step)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep='/')


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {'list_len': len(x)} | {f'{i}': _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd
    ans = pd.json_normalize(raw, sep=sep).to_dict(orient='records')[0]
    assert isinstance(ans, dict)
    return ans

def save_wandb_meta(exp_dir: str, run):
    meta = {
        "run_id": run.id,
        "project": run.project,
        "name": run.name,
    }
    meta_path = Path(exp_dir) / "wandb_meta.json"

    # FIX: Create the directory
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "w") as f:
        json.dump(meta, f)


def load_wandb_meta(exp_dir: str):
    meta_path = Path(exp_dir) / "wandb_meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return None
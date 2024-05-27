import os
import random
from pathlib import Path

import torch
import numpy as np
import wandb


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def maybe_setup_wandb(logdir, args=None, run_name_suffix=None, **init_kwargs):

    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT")

    if wandb_entity is None or wandb_project is None:
        print(f"{wandb_entity=}", f"{wandb_project=}")
        print("Not initializing WANDB")
        return

    origin_run_name = Path(logdir).name

    api = wandb.Api()

    name_runs = list(api.runs(f'{wandb_entity}/{wandb_project}', filters={'display_name': origin_run_name}))
    group_runs = list(api.runs(f'{wandb_entity}/{wandb_project}', filters={'group': origin_run_name}))

    print(f'Retrieved {len(name_runs)} for run_name: {origin_run_name}')

    assert len(name_runs) <= 1, f'retrieved_runs: {len(name_runs)}'

    new_run_name = origin_run_name if len(name_runs) == 0 else f"{origin_run_name}_{len(group_runs)}"

    if run_name_suffix is not None:
        new_run_name = f"{new_run_name}_{run_name_suffix}"

    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        config=args,
        name=new_run_name,
        dir=logdir,
        resume="never",
        group=origin_run_name,
        sync_tensorboard=True,
        **init_kwargs
    )

    print("WANDB run", wandb.run.id, new_run_name, origin_run_name)
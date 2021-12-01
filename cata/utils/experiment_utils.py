import datetime
import os
import time
from typing import Optional


def set_random_seeds(seed: int) -> None:
    # import packages with non-deterministic behaviour
    import random
    import torch
    import numpy as np

    # set random seeds for these packages
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_experiment_timestamp() -> str:
    """Get a timestamp in YY-MM-DD-HH-MM-SS format."""
    raw_datetime = datetime.datetime.fromtimestamp(time.time())
    exp_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    return exp_timestamp


def get_checkpoint_path(
    folder: str,
    timestamp: str,
    experiment_name: str,
    subfolder_name: Optional[str] = "",
) -> str:
    """Get full checkpoint path for experiment logs etc."""
    checkpoint_path = os.path.join(folder, timestamp, experiment_name, subfolder_name)
    return checkpoint_path

from typing import Dict, List, Union

from .base_teachers import _BaseTeachers
from utils import Parameters

import torch

class PureMNISTTeachers(_BaseTeachers):

    """Dummy teachers class for pure mnist teachers (just returns back label)"""

    def __init__(self, config: Parameters):
        _BaseTeachers.__init__(self, config)

    def test_set_forward(self, teacher_index, batch) -> torch.Tensor:
        all_task_labels: List[torch.Tensor] = batch['y']
        labels = all_task_labels[teacher_index]
        return labels

    def forward(self, teacher_index: int, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch['y']
        return labels

    def _setup_teachers(self, config: Parameters):
        pass

    def signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def signal_step_boundary_to_teacher(self, step: int, current_task: int):
        pass
    
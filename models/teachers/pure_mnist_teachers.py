from typing import Dict

from .base_teachers import _BaseTeachers

import torch

class PureMNISTTeachers(_BaseTeachers):

    """Dummy teachers class for pure mnist teachers (just returns back label)"""

    def __init__(self, config: Dict):
        _BaseTeachers.__init__(self, config)

    def test_set_forward(self, teacher_index: int, batch: Dict) -> torch.Tensor:
        labels = batch['y'][teacher_index]
        return labels

    def forward(self, teacher_index: int, batch: Dict):
        labels = batch['y']
        return labels

    def _setup_teachers(self, config: Dict):
        pass

    def signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def signal_step_boundary_to_teacher(self, step: int, current_task: int):
        pass
    
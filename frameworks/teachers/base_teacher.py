from abc import ABC, abstractmethod

from typing import Dict

class _BaseTeacher(ABC):

    def __init__(self, config: Dict):

        self.num_teachers = config.get(["task", "num_teachers"])
        self.device = config.get("device")

        self._setup_teachers(config=config)

    @abstractmethod
    def _setup_teachers(self, config):
        raise NotImplementedError("Base class method")

    def forward(self, teacher_index: int, x):
        output = self._teachers[teacher_index](x)
        return output

    def forward_all(self, x):
        outputs = [teacher(x) for teacher in self._teachers]
        return outputs

    def get_teacher_networks(self):
        networks = self._teachers 
        return networks

    @abstractmethod
    def signal_task_boundary_to_teacher(self, new_task: int):
        raise NotImplementedError("Base class method")

    @abstractmethod
    def signal_step_boundary_to_teacher(self, step: int, current_task: int):
        raise NotImplementedError("Base class method")



from models import ContinualStudent

from .base_learner import _BaseLearner

from typing import Dict

class ContinualLearner(_BaseLearner):

    """Orchestrates students in continual learning setting (one head per task)"""

    def __init__(self, config):
        _BaseLearner.__init__(self, config)

    def _setup_student(self, config: Dict) -> None:
        """Instantiate student"""
        self._student_network = ContinualStudent(config=config).to(self._device)

    def signal_task_boundary_to_learner(self, new_task: int) -> None:
        self._student_network.set_task(new_task)

    def signal_step_boundary_to_learner(self, step: int, current_task: int):
        pass

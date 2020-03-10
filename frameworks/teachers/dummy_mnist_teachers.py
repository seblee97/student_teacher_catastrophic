from typing import Dict

from .base_teacher import _BaseTeacher

class DummyMNISTTeachers(_BaseTeacher):

    """Dummy teachers class for pure mnist teachers"""

    def __init__(self, config: Dict):
        _BaseTeacher.__init__(self, config)

    def _setup_teachers(self, config: Dict):
        pass

    def _signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def _signal_step_boundary_to_teacher(self, step: int, current_task: int):
        pass
    
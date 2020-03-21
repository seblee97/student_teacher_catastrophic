from typing import Dict

from .base_teachers import _BaseTeachers

class DummyMNISTTeachers(_BaseTeachers):

    """Dummy teachers class for pure mnist teachers"""

    def __init__(self, config: Dict):
        _BaseTeachers.__init__(self, config)

    def _setup_teachers(self, config: Dict):
        pass

    def signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def signal_step_boundary_to_teacher(self, step: int, current_task: int):
        pass
    
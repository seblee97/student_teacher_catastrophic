from .base_teachers import _BaseTeachers
from utils import Parameters

from typing import Dict

class IndependentTeachers(_BaseTeachers):

    def __init__(self, config: Parameters):
        _BaseTeachers.__init__(self, config)

    def _setup_teachers(self, config: Parameters) -> None:
        """Instantiate all teachers"""
        # initialise teacher networks, freeze
        teacher_noise = config.get(["task", "teacher_noise"])
        if type(teacher_noise) is int:
            teacher_noises = [teacher_noise for _ in range(self._num_teachers)]
        elif type(teacher_noise) is list:
            assert len(teacher_noise) == self._num_teachers, \
            "Provide one noise for each teacher. {} noises given, {} teachers specified".format(len(teacher_noise), self._num_teachers)
            teacher_noises = teacher_noise

        self._teachers = []
        for t in range(self._num_teachers):
            teacher = self._init_teacher(config=config, index=t)
            teacher.freeze_weights()
            if teacher_noises[t] != 0:
                teacher_output_std = teacher.get_output_statistics()
                teacher.set_noise_distribution(mean=0, std=teacher_noises[t] * teacher_output_std)
            self._teachers.append(teacher)

    def _signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def _signal_step_boundary_to_teacher(self, step: int, current_task: int):
        pass
    
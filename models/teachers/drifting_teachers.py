from models.networks.teachers import DriftingTeacher
from utils import Parameters
from .base_teachers import _BaseTeachers


class DriftingTeachers(_BaseTeachers):

    def __init__(self, config: Parameters):
        _BaseTeachers.__init__(self, config)
        self._drift_frequency = config.get(["task", "drift_frequency"])

    def _setup_teachers(self, config: Parameters):
        """Instantiate Teachers"""
        # initialise teacher networks, freeze
        teacher_noise = config.get(["task", "teacher_noise"])

        if type(teacher_noise) is int:
            teacher_noises = [teacher_noise for _ in range(self._num_teachers)]

        elif type(teacher_noise) is list:
            assert len(
                teacher_noise
            ) == self._num_teachers, f"Provide one noise for each teacher. {len(teacher_noise)} noises given, {self._num_teachers} teachers specified"
            teacher_noises = teacher_noise

        self.teachers = []
        for t in range(self._num_teachers):
            teacher = DriftingTeacher(config=config, index=t).to(self._device)
            teacher.freeze_weights()
            if teacher_noises[t] != 0:
                teacher_output_std = teacher.get_output_statistics()
                teacher.set_noise_distribution(mean=0, std=teacher_noises[t] * teacher_output_std)
            self.teachers.append(teacher)

    def _signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def _signal_step_boundary_to_teacher(self, step: int, current_task: int):
        self._teachers[current_task].forward_count += 1
        if step % self._drift_frequency == 0:
            self._teachers[current_task].rotate_weights()

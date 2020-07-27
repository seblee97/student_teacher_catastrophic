import copy

from utils import Parameters
from .base_teachers import _BaseTeachers


class NoisyTeachers(_BaseTeachers):

    def __init__(self, config: Parameters):
        _BaseTeachers.__init__(self, config)

    def _setup_teachers(self, config: Parameters):
        """
        Instantiate all teachers

        Start with one 'original' teacher. Copy this teacher given amount
        of times. Noise distributions set separately for each teacher.
        This represents tasks in which all share an 'underlying' structure
        but have slight differences in their environments.
        """
        # initialise teacher networks, freeze
        teacher_noises = config.get(["task", "teacher_noises"])

        assert len(teacher_noises) == self.num_teachers, \
            f"Provide one noise for each teacher. {len(teacher_noises)} noises given, {self.num_teachers} teachers specified"

        self.teachers = []
        base_teacher = self._init_teacher(config=config, index=0)
        base_teacher.freeze_weights()
        base_teacher_output_std = base_teacher.get_output_statistics()

        for t in range(self.num_teachers):
            teacher = copy.deepcopy(base_teacher)
            if teacher_noises[t] != 0:
                teacher.set_noise_distribution(
                    mean=0, std=teacher_noises[t] * base_teacher_output_std)
            self.teachers.append(teacher)

    def _signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def _signal_step_boundary_to_teacher(self, step: int, current_task: int):
        pass

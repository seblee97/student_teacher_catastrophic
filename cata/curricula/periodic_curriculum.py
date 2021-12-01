from cata.curricula import base_curriculum
from cata.run import student_teacher_config


class PeriodicCurriculum(base_curriculum.BaseCurriculum):
    """Curriculum defined by fixed period of switching."""

    def __init__(self, config: student_teacher_config.StudentTeacherConfig) -> None:
        self._curriculum_period = config.fixed_period
        super().__init__(config=config)

    def to_switch(self, task_step: int, error: float) -> bool:
        """Establish whether condition for switching has been met.

        Here, see if task_step is equal to switch period.

        Args:
            task_step: number of steps completed for current task
            being trained (not overall step count).
            error: generalisation error associated with current teacher.

        Returns:
            bool indicating whether or not to switch.
        """
        return task_step == self._curriculum_period

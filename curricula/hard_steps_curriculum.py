import itertools

import numpy as np
from curricula import base_curriculum
from run import student_teacher_config
from utils import custom_functions


class HardStepsCurriculum(base_curriculum.BaseCurriculum):
    """Curriculum defined by hard-coded steps at which to switch."""

    def __init__(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> None:
        self._curriculum_switch_steps = iter(config.switch_steps)

        if config.interleave_period is not None:
            assert (
                len(config.switch_steps) < 2
            ), "interleaving currently only implemented for single switch."
            offset = config.switch_steps[0] + config.interleave_period
            increments = [
                config.interleave_duration,
                config.interleave_period,
            ]
            interleave_iterator = custom_functions.create_iterator(
                offset=offset, increments=increments
            )
            self._curriculum_switch_steps = itertools.chain(
                self._curriculum_switch_steps, interleave_iterator
            )

        self._next_switch_step = next(self._curriculum_switch_steps)
        super().__init__(config=config)

    def to_switch(self, task_step: int, error: float) -> bool:
        """Establish whether condition for switching has been met.

        Here, see if step equals next switch step.

        Args:
            task_step: number of steps completed for current task
            being trained (not overall step count).
            error: generalisation error associated with current teacher.

        Returns:
            bool indicating whether or not to switch.
        """
        if task_step == self._next_switch_step:
            try:
                self._next_switch_step = next(self._curriculum_switch_steps)
            except StopIteration:
                self._next_switch_step = np.inf
            return True
        else:
            return False

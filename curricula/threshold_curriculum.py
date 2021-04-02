from curricula import base_curriculum
from run import student_teacher_config


class ThresholdCurriculum(base_curriculum.BaseCurriculum):
    """Curriculum defined by switching after loss threshold."""

    def __init__(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> None:
        loss_threshold_sequence = config.loss_thresholds
        # make n copies of threshold sequence
        # e.g. [thresh1, thresh2] -> [thresh1, thresh1, thresh2, thresh2]
        self._curriculum_loss_thresholds = iter(
            [
                threshold
                for threshold in loss_threshold_sequence
                for _ in range(config.num_teachers)
            ]
        )
        self._current_loss_threshold = next(self._curriculum_loss_thresholds)
        super().__init__(config=config)

    def to_switch(self, task_step: int, error: float) -> bool:
        """Establish whether condition for switching has been met.

        Here, see if error is below current threshold for switching.

        Args:
            task_step: number of steps completed for current task
            being trained (not overall step count).
            error: generalisation error associated with current teacher.

        Returns:
            bool indicating whether or not to switch.
        """
        if error < self._current_loss_threshold:
            try:
                self._current_loss_threshold = next(self._curriculum_loss_thresholds)
                return True
            except StopIteration:
                print("Sequence of thresholds exhausted...")
                self._current_loss_threshold = 0
                return None
        else:
            return False

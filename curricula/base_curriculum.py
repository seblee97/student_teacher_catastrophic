import abc
import itertools

from run import student_teacher_config


class BaseCurriculum(abc.ABC):
    """Base class for curriculum objects."""

    def __init__(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> None:
        self._curriculum = itertools.cycle(list(range(config.num_teachers)))

    def __next__(self):
        """Get next index from curriculum."""
        return next(self._curriculum)

    @abc.abstractmethod
    def to_switch(self, task_step: int, error: float) -> bool:
        """Establish whether condition for switching has been met.

        Args:
            task_step: number of steps completed for current task
            being trained (not overall step count).
            error: generalisation error associated with current teacher.

        Returns:
            bool indicating whether or not to switch.
        """
        pass

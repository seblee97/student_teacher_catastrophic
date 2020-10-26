import abc


class BaseCurriculum(abc.ABC):
    """Base class for curriculum objects."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def to_switch(self, step: int, error: float):
        """Establish whether condition for switching has been met."""
        pass

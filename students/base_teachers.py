import abc
import constants

import torch

from typing import Dict
from typing import List

from students import classification_teacher
from students import regression_teacher


class BaseTeachers(abc.ABC):
    """Base class for sets/ensembles of teachers
    (as opposed to single teacher network)."""

    def __init__(self, loss_type: str):
        self._loss_type = loss_type

    @abc.abstractmethod
    def _setup_teachers(self, ARGS) -> None:
        """instantiate teacher network(s)"""
        pass

    @abc.abstractmethod
    def test_set_forward(self, batch) -> List[torch.Tensor]:
        pass

    @abc.abstractmethod
    def forward(
        self, teacher_index: int, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Call to current teacher forward."""
        pass

    def _init_teacher(self):
        if self._loss_type == constants.Constants.CLASSIFICATION:
            return classification_teacher.ClassificationTeacher(
                config=config, index=index
            )
        elif self._loss_type == constants.Constants.REGRESSION:
            return regression_teacher.RegressionTeacher(config=config, index=index).to(
                self._device
            )

    def save_weights(self, teacher_index: int, save_path: str):
        """Save weights associated with given teacher index"""
        torch.save(self._teachers[teacher_index].state_dict(), save_path)

    def forward_all(self, batch: Dict) -> List[torch.Tensor]:
        """Call to forward of all teachers (used primarily for evaluation)"""
        outputs = [self.forward(t, batch) for t in range(self._num_teachers)]
        return outputs

    @property
    def teachers(self) -> List:
        """Getter method for teacher networks."""
        return self._teachers

import numpy as np

from abc import ABC, abstractmethod

from typing import List, Tuple


class Overlap(ABC):

    def __init__(self, initial_values: np.ndarray, final: bool):
        self._overlap_values = initial_values.astype(float)
        self._timestep = 0
        self._final = final

    @abstractmethod
    def __getitem__(self, indices: List[Tuple[str, int]]):
        pass

    @property
    def timestep(self) -> int:
        return self._timestep

    @property
    def values(self) -> np.ndarray:
        return self._overlap_values

    def step(self, value_change: np.ndarray) -> None:

        if self._final:
            raise TypeError("Cannot step on a final Overlap.")

        self._timestep += 1
        self._overlap_values += value_change

    @property 
    def shape(self):
        return self._overlap_values.shape

class SelfOverlap(Overlap):

    def __init__(self, initial_values: np.ndarray, final: bool):
        super().__init__(initial_values, final)

    def __getitem__(self, indices: List[Tuple[str, int]]):
        # symmetric
        return self._overlap_values[indices[0][1]][indices[1][1]]

class CrossOverlap(Overlap):

    def __init__(self, initial_values: np.ndarray, final: bool):
        super().__init__(initial_values, final)

    def __getitem__(self, indices: List[Tuple[str, int]]):
        # student index first
        ordered_indices = sorted(indices)
        return self._overlap_values[ordered_indices[0][1]][ordered_indices[1][1]]

from typing import List
from typing import Optional

import numpy as np


class NetworkConfiguration:
    """object to store configuration of student/teacher networks in unified way."""

    def __init__(
        self,
        student_head_weights: List[np.ndarray],
        teacher_head_weights: List[np.ndarray],
        student_self_overlap: np.ndarray,
        teacher_self_overlaps: List[np.ndarray],
        teacher_cross_overlaps: List[np.ndarray],
        student_teacher_overlaps: List[np.ndarray],
        old_student_self_overlap: Optional[np.ndarray] = None,
        student_old_student_overlap: Optional[np.ndarray] = None,
        teacher_old_student_overlaps: Optional[List[np.ndarray]] = [],
    ):
        self._student_head_weights = student_head_weights
        self._teacher_head_weights = teacher_head_weights
        self._student_self_overlap = student_self_overlap
        self._teacher_self_overlaps = teacher_self_overlaps
        self._teacher_cross_overlaps = teacher_cross_overlaps
        self._student_teacher_overlaps = student_teacher_overlaps

        self._old_student_self_overlap = old_student_self_overlap
        self._student_old_student_overlap = student_old_student_overlap
        self._teacher_old_student_overlaps = teacher_old_student_overlaps

    @property
    def student_head_weights(self) -> List[np.ndarray]:
        return self._student_head_weights

    @property
    def teacher_head_weights(self) -> List[np.ndarray]:
        return self._teacher_head_weights

    @property
    def student_self_overlap(self) -> np.ndarray:
        return self._student_self_overlap

    @property
    def teacher_self_overlaps(self) -> List[np.ndarray]:
        return self._teacher_self_overlaps

    @property
    def teacher_cross_overlaps(self) -> List[np.ndarray]:
        return self._teacher_cross_overlaps

    @property
    def student_teacher_overlaps(self) -> List[np.ndarray]:
        return self._student_teacher_overlaps

    @property
    def old_student_self_overlap(self):
        return self._old_student_self_overlap

    @property
    def student_old_student_overlap(self):
        return self._student_old_student_overlap

    @property
    def teacher_old_student_overlaps(self):
        return self._teacher_old_student_overlaps

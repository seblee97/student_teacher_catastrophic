from models import Teacher, MetaStudent, ContinualStudent, MNISTContinualStudent, MNISTMetaStudent

from .base_learner import BaseLearner

import copy

import torch

class ContinualLearner(BaseLearner):

    def __init__(self, config):
        # super(ContinualLearner, self).__init__(config)
        BaseLearner.__init__(self, config)

    def _setup_student(self, config):
        """Instantiate student"""
        # initialise student network
        self._student_network = ContinualStudent(config=config).to(self.device)

    def signal_task_boundary_to_learner(self, new_task: int):
        self._student_network.set_task(new_task)

    def signal_step_boundary_to_learner(self, step: int, current_task: int):
        pass

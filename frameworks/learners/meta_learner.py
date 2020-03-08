from models import Teacher, MetaStudent, ContinualStudent, MNISTContinualStudent, MNISTMetaStudent

from frameworks import StudentTeacher, MNIST

import copy

import torch

class MetaLearner(StudentTeacher):

    def __init__(self, config):
        StudentTeacher.__init__(self, config)

    def _setup_student(self, config):
        """Instantiate student"""
        # initialise student network
        self.student_network = MetaStudent(config=config).to(self.device)

    def _signal_task_boundary_to_learner(self, new_task: int):
        self.student_network.set_task(new_task)

    def _signal_step_boundary_to_learner(self, step: int, current_task: int):
        pass

    def _compute_generalisation_errors(self, teacher_index=None):
        with torch.no_grad():
            student_outputs = self.student_network.test_all_tasks(self.test_input_data)
            generalisation_errors = [float(self._compute_loss(student_outputs, teacher_output)) for teacher_output in self.test_teacher_outputs]
            return {'generalisation_error': generalisation_errors}

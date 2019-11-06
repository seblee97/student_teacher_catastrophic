from models import Teacher, MetaStudent, ContinualStudent

from frameworks import StudentTeacher

import copy

import torch

class ContinualLearner(StudentTeacher):

    def __init__(self, config):
        StudentTeacher.__init__(self, config)

    def _setup_student(self, config):
        """Instantiate student"""
        # initialise student network
        self.student_network = ContinualStudent(config=config).to(self.device)

    def _signal_task_boundary_to_learner(self, new_task: int):
        self.student_network.set_task(new_task)

    def _signal_step_boundary_to_learner(self, step: int, current_task: int):
        pass

    def _compute_generalisation_errors(self):
        with torch.no_grad():
            student_outputs = self.student_network.test_all_tasks(self.test_input_data)
            generalisation_errors = [float(self._compute_loss(student_output, teacher_output)) for student_output, teacher_output in zip(student_outputs, self.test_teacher_outputs)]
            return generalisation_errors
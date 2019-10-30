from models import Teacher, MetaStudent, ContinualStudent

from frameworks import StudentTeacher

import copy

import torch

class MetaLearner(StudentTeacher):

    def __init__(self, config):
        StudentTeacher.__init__(self, config)

    def _setup_student(self, config):
        # initialise student network
        self.student_network = MetaStudent(config=config).to(self.device)

    def _signal_task_boundary(self, new_task: int):
        self.student_network.set_task(new_task)

    def _compute_generalisation_errors(self):
        with torch.no_grad():
            student_outputs = self.student_network.test_all_tasks(self.test_input_data)
            generalisation_errors = [float(self._compute_loss(student_outputs, teacher_output)) for teacher_output in self.test_teacher_outputs]
            return generalisation_errors

from models import Teacher, MetaStudent, ContinualStudent, MNISTContinualStudent, MNISTMetaStudent

from frameworks import StudentTeacher, MNIST

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

    def _compute_generalisation_errors(self, teacher_index=None):
        with torch.no_grad():
            student_outputs = self.student_network.test_all_tasks(self.test_input_data)
            generalisation_errors = [float(self._compute_loss(student_output, teacher_output)) for student_output, teacher_output in zip(student_outputs, self.test_teacher_outputs)]

            return_dict = {'generalisation_error': generalisation_errors}

            if self.student_network.classification_output:
                accuracies = [float(self._compute_classification_acc(student_output, teacher_output)) for student_output, teacher_output in zip(student_outputs, self.test_teacher_outputs)]
                return_dict['accuracy'] = accuracies

            return return_dict

class MNISTContinualLearner(MNIST):

    def __init__(self, config):
        MNIST.__init__(self, config)

    def _setup_student(self, config):
        """Instantiate student"""
        # initialise student network
        self.student_network = ContinualStudent(config=config).to(self.device)

    def _signal_task_boundary_to_learner(self, new_task: int):
        self.student_network.set_task(new_task)

    def _signal_step_boundary_to_learner(self, step: int, current_task: int):
        pass

    def _compute_generalisation_errors(self, teacher_index=None):
        with torch.no_grad():
            test_input_images = self.test_data_x[teacher_index]
            student_outputs = self.student_network.test_all_tasks(self.test_data_x[teacher_index])
            generalisation_errors = [float(self._compute_loss(student_output, self.test_data_y[teacher_index])) for student_output in student_outputs]
            
            accuracies = [float(self._compute_classification_acc(student_output, self.test_data_y[teacher_index])) for student_output in student_outputs]

            return {'generalisation_error': generalisation_errors, 'accuracy': accuracies}


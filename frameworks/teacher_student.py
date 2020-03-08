import numpy as np 
import copy
import itertools
import pandas as pd
import time
import logging

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

from typing import List, Tuple, Generator, Dict

from utils import visualise_matrix, get_binary_classification_datasets, load_mnist_data, load_mnist_data_as_dataloader

from abc import ABC, abstractmethod

from .learners import ContinualLearner
from .teachers import OverlappingTeachers
from .loggers import StudentMNISTLogger, StudentTeacherLogger
from .data_modules import MNISTData, IIDData
from .loss_modules import RegressionLoss, ClassificationLoss

class StudentTeacherRunner:

    def __init__(self, config: Dict) -> None:
        """
        Class for orchestrating student teacher framework including training and test loops

        :param config: dictionary containing parameters to specify training etc.
        """
        # extract relevant parameters from config
        self._extract_parameters(config)

        # initialise student, teachers, logger_module, data_module and loss_module
        self._setup_learner(config=config)
        self._setup_teachers(config=config)
        self._setup_logger(config=config)
        self._setup_data(config=config)
        self._setup_loss(config=config)

        # extract curriculum from config
        self._set_curriculum(config)

        # initialise optimiser with trainable parameters of student
        trainable_parameters = self.learner.get_trainable_parameters()
        self.optimiser = optim.SGD(trainable_parameters, lr=self.learning_rate)
        
        # initialise objects containing metrics of interest
        self._initialise_metrics()

    def _extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        """
        self.verbose = config.get("verbose")
        self.verbose_tb = config.get("verbose_tb")
        self.checkpoint_path = config.get("checkpoint_path")
        self.device = config.get("device")
        self.logfile_path = config.get("logfile_path")
        self.checkpoint_frequency = config.get("checkpoint_frequency")
        self.log_to_df = config.get("log_to_df")

        self.total_training_steps = config.get(["training", "total_training_steps"])

        self.input_dimension = config.get(["model", "input_dimension"])
        self.output_dimension = config.get(["model", "output_dimension"])
        self.soft_committee = config.get(["model", "soft_committee"])
        self.student_hidden = config.get(["model", "student_hidden_layers"])

        self.train_batch_size = config.get(["training", "train_batch_size"])
        self.test_batch_size = config.get(["training", "test_batch_size"])
        self.learning_rate = config.get(["training", "learning_rate"])
        self.scale_output_backward = config.get(["training", "scale_output_backward"])

        self.test_all_teachers = config.get(["testing", "test_all_teachers"])
        self.test_frequency = config.get(["testing", "test_frequency"])
        self.overlap_frequency = config.get(["testing", "overlap_frequency"])

        self.num_teachers = config.get(["task", "num_teachers"])
        self.label_task_bounaries = config.get(["task", "label_task_boundaries"])
        self.learner_configuration = config.get(["task", "learner_configuration"])
        self.teacher_configuration = config.get(["task", "teacher_configuration"])
        self.loss_type = config.get(["task", "loss_type"])

        self.input_source = config.get(["training", "input_source"])
        self.pca_input = config.get(["training", "pca_input"])

        if self.pca_input > 0:
            if self.pca_input != self.input_dimension:
                raise ValueError("Please ensure that if PCA is applied that the number of principal components\
                    matches the input dimension for the network.")

    def _setup_learner(self, config: Dict):
        if self.learner_configuration == "continual":
            self.learner = ContinualLearner(config=config)
        else:
            raise ValueError("Learner configuration {} not recognised".format(self.learner_configuration))
        # elif self.learner_configuration == "meta":
        #     self.learner = MetaLearner(config=config)

    def _setup_teachers(self, config: Dict):
        if self.teacher_configuration == "overlapping":
            self.teachers = OverlappingTeachers(config=config)
        elif self.teacher_configuration == "mnist":
            self.teachers = MNISTTeachers(config=config)
        else:
            raise NotImplementedError("Teacher configuration {} not recognised".format(self.teacher_configuration))

    def _setup_data(self, config: Dict):
        if self.input_source == 'iid_gaussian':
            self.data_module = IIDData(config)
        elif self.input_source == 'mnist':
            self.data_module = MNISTData(config)
        else:
            raise ValueError("Input source type {} not recognised. Please use either iid_gaussian or mnist".format(self.input_source))
        
        self.test_input_data, self.test_data_labels = self.data_module.get_test_set() # labels will be None if using student-teacher networks
        self.test_teacher_outputs = self.teachers.forward_all(self.test_input_data)
        
    def _setup_logger(self, config: Dict):
        if self.teacher_configuration == "mnist":
            self.logger = StudentMNISTLogger(config=config)
        else:
            self.logger = StudentTeacherLogger(config=config)

    def _setup_loss(self, config: Dict):
        if self.loss_type == "regression":
            self.loss_module = RegressionLoss(config=config)
        elif self.loss_type == "classification":
            self.loss_module = ClassificationLoss(config=config)
        else:
            raise NotImplementedError("Loss type {} not recognised".format(self.loss_type))

    def _initialise_metrics(self) -> None:
        """
        Initialise objects that will keep metrix of training e.g. generalisation errors
        Useful to make them attributes of class so that they can be accessed by all methods in class
        """
        self.generalisation_errors = {i: [] for i in range(self.num_teachers)}

    def _set_curriculum(self, config: Dict) -> None:
        """
        Establish and assign curriculum from configuration file. This is used to determine how to 
        proceed with training (when to switch teacher, how to decide subsequent teacher etc.)
        """
        curriculum_type = config.get(["curriculum", "type"])
        self.curriculum_stopping_condition = config.get(["curriculum", "stopping_condition"])
        self.curriculum_period = config.get(["curriculum", "fixed_period"])
        self.curriculum_loss_threshold = config.get(["curriculum", "loss_threshold"])

        if curriculum_type == "custom":
            self.curriculum = itertools.cycle((config.get(["curriculum", "custom"])))
        elif curriculum_type == "standard":
            self.curriculum_selection_type = config.get(["curriculum", "selection_type"])
            if self.curriculum_selection_type == "cyclical":
                self.curriculum = itertools.cycle(list(range(self.num_teachers)))
            elif self.curriculum_selection_type == "random":
                raise NotImplementedError
            else:
                raise ValueError("You have specified a standard curriculum (as opposed to custom)," 
                                 "but the selection type is not recognised. Please choose between"
                                 "'cyclical' and 'random'")
        else:
            raise ValueError("Curriculum type {} not recognised".format(curriculum_type))

    def train(self) -> None:
        """
        Training loop
        """
        training_losses = []
        total_step_count = 0
        steps_per_task = []

        iter_time = time.time()

        # explicitly set model to training mode
        self.learner.set_to_train()

        while total_step_count < self.total_training_steps:
            
            if self.log_to_df:
                self.logger.add_row(row_label=total_step_count)
            
            teacher_index = next(self.curriculum)
            task_step_count = 0
            latest_task_generalisation_error = np.inf
            
            # alert learner/teacher(s) of task change e.g. change output head of student to relevant task (if in continual setting)
            self.learner.signal_task_boundary_to_learner(new_task=teacher_index)
            self.teachers.signal_task_boundary_to_teacher(new_task=teacher_index)

            while total_step_count < self.total_training_steps:

                batch_input = self.data_module.get_batch()

                teacher_output = self.teachers.forward(teacher_index, batch_input)
                student_output = self.learner.forward(batch_input)

                if self.verbose and task_step_count % 1000 == 0:
                    self.logger.log("Training step {}".format(task_step_count))

                # training iteration
                self.optimiser.zero_grad()
                loss = self.loss_module.compute_loss(student_output, teacher_output)
                loss.backward()
                self.optimiser.step()
                training_losses.append(float(loss))

                # log training loss
                self.logger.write_scalar_tb('training_loss', float(loss), total_step_count)
                if self.log_to_df:
                    self.logger.write_scalar_df('training_loss', float(loss), total_step_count)
                
                # test
                if total_step_count % self.test_frequency == 0 and total_step_count != 0:

                    # explicitly set model to evaluation mode
                    self.learner.set_to_eval()

                    latest_task_generalisation_error = self._perform_test_loop(teacher_index, task_step_count, total_step_count)

                    # explicitly set model back to training mode
                    self.learner.set_to_train()

                # overlap matrices
                if total_step_count % self.overlap_frequency == 0 and total_step_count != 0:
                    self.logger._compute_overlap_matrices(
                        student_network=self.learner.get_student_network(), 
                        teacher_networks=self.teachers.get_teacher_networks(), 
                        step_count=total_step_count
                        )

                total_step_count += 1
                task_step_count += 1

                # output layer weights
                self.logger._log_output_weights(step_count=total_step_count, student_network=self.learner.get_student_network())

                # alert learner/teacher(s) of step e.g. to drift teacher
                self.learner.signal_step_boundary_to_learner(step=task_step_count, current_task=teacher_index)
                self.teachers.signal_step_boundary_to_teacher(step=task_step_count, current_task=teacher_index)

                if total_step_count % 500 == 0:
                    current_time = time.time()
                    self.logger.log("Time taken for last {} steps: {}".format(500, current_time - iter_time))
                    iter_time = current_time

                # checkpoint dataframe
                if self.logfile_path and total_step_count % self.checkpoint_frequency == 0 and self.log_to_df:
                    self.logger.checkpoint_df(step=total_step_count)

                if self._switch_task(step=task_step_count, generalisation_error=latest_task_generalisation_error):
                    self.logger.write_scalar_tb('steps_per_task', task_step_count, total_step_count)
                    steps_per_task.append(task_step_count)
                    break

        # self._consolidate_dfs()

    def _switch_task(self, step: int, generalisation_error: float) -> bool: 
        """
        Method to determine whether to switch task 
        (i.e. whether condition set out by curriculum for switching has been met)

        :param step: number of steps completed for current task being trained (not overall step count)
        :param generalisation_error: generalisation error associated with current teacher 

        :return True / False: whether or not to switch task
        """
        if self.curriculum_stopping_condition == "fixed_period":
            if step == self.curriculum_period:
                return True
            else:
                return False
        elif self.curriculum_stopping_condition == "threshold":
            if generalisation_error < self.curriculum_loss_threshold:
                return True
            else:
                return False
        else:
            raise ValueError("curriculum stopping condition {} unknown. Please use 'fixed_period' or 'threshold'.".format(self.curriculum_stopping_condition))

    def _perform_test_loop(self, teacher_index: int, task_step_count: int, total_step_count: int) -> float:
        """
        Test loop. Evaluated generalisation error of student wrt teacher(s)

        :param teacher_index: index of current teacher student is being trained on
        :param task_step_count: number of steps completed under current teacher
        :param total_step_count: total number of steps student has beein training for

        :return generalisation_error_wrt_current_teacher: generalisation_error for current teacher 
        """
        with torch.no_grad():
            
            generalisation_metrics = self._compute_generalisation_errors(teacher_index=teacher_index)
            
            generalisation_error_per_teacher = generalisation_metrics.get('generalisation_error')
            classification_accuracy = generalisation_metrics.get('accuracy')

            # log average generalisation losses over teachers
            mean_generalisation_error_per_teacher = np.mean(generalisation_error_per_teacher)
            self.logger.write_scalar_tb(
                'mean_generalisation_error/linear', mean_generalisation_error_per_teacher, total_step_count
                )
            self.logger.write_scalar_tb(
                'mean_generalisation_error/log', np.log10(mean_generalisation_error_per_teacher), total_step_count
                )
            if self.log_to_df:
                self.logger.write_scalar_tb(
                    'mean_generalisation_error/linear', mean_generalisation_error_per_teacher, total_step_count
                    )
                self.logger.write_scalar_tb(
                    'mean_generalisation_error/log', np.log10(mean_generalisation_error_per_teacher), total_step_count
                    )

            for i, error in enumerate(generalisation_error_per_teacher):
                self.generalisation_errors[i].append(error)

                # log generalisation loss per teacher
                self.logger.write_scalar_tb('generalisation_error/linear', error, total_step_count, teacher_index=i)
                self.logger.write_scalar_tb('generalisation_error/log', np.log10(error), total_step_count, teacher_index=i)
                if self.log_to_df:
                    self.logger.write_scalar_df('teacher_{}_generalisation_error/log'.format(i), np.log10(error), total_step_count)
                    self.logger.write_scalar_df('teacher_{}_generalisation_error/linear'.format(i), error, total_step_count)

                if classification_accuracy:
                    self.logger.write_scalar_tb('classification_accuracy', classification_accuracy[i], total_step_count, teacher_index=i)
                    if self.log_to_df:
                        self.logger.write_scalar_df('teacher_{}_classification_accuracy'.format(i), classification_accuracy[i], total_step_count)

                if len(self.generalisation_errors[i]) > 1:
                    last_error = copy.deepcopy(self.generalisation_errors[i][-2])
                    error_delta = error - last_error
                    if error_delta != 0.:
                        # log generalisation loss delta per teacher
                        self.logger.write_scalar_tb('error_change/linear', error_delta, total_step_count, teacher_index=i)
                        self.logger.write_scalar_tb('error_change/log', np.sign(error_delta) * np.log10(abs(error_delta)), total_step_count, teacher_index=i)
                        if self.log_to_df:
                            self.logger.write_scalar_df('teacher_{}_error_change/linear'.format(i), error_delta, total_step_count)
                            self.logger.write_scalar_df('teacher_{}_error_change/log'.format(i), np.sign(error_delta) * np.log10(abs(error_delta)), total_step_count)

            if self.verbose and task_step_count % 500 == 0:
                self.logger.log("Generalisation errors @ step {} ({}'th step training on teacher {}):".format(total_step_count, task_step_count, teacher_index))
                for i, error in enumerate(generalisation_error_per_teacher):
                    self.logger.log(
                        "{}Teacher {}: {}\n".format("".rjust(10), i, error)
                    )
                if classification_accuracy:
                    self.logger.log("Classification accuracies @ step {} ({}'th step training on teacher {}):".format(total_step_count, task_step_count, teacher_index))
                    for i, acc in enumerate(classification_accuracy):
                        self.logger.log(
                            "{}Teacher {}: {}\n".format("".rjust(10), i, acc),
                        )

            generalisation_error_wrt_current_teacher = generalisation_error_per_teacher[teacher_index]

            return generalisation_error_wrt_current_teacher

    def _compute_classification_acc(self, prediction: torch.Tensor, target: torch.Tensor):
        class_predictions = torch.argmax(prediction, axis=1)
        accuracy = int(torch.sum(class_predictions == target)) / len(target)
        return accuracy

    def _compute_generalisation_errors(self, teacher_index=None):
        with torch.no_grad():
            student_outputs = self.learner.forward_all(self.test_input_data)
            generalisation_errors = [float(self.loss_module.compute_loss(student_output, teacher_output)) for student_output, teacher_output in zip(student_outputs, self.test_teacher_outputs)]

            return_dict = {'generalisation_error': generalisation_errors}

            if self.loss_type == "classification":
                accuracies = [float(self._compute_classification_acc(student_output, teacher_output)) for student_output, teacher_output in zip(student_outputs, self.test_teacher_outputs)]
                return_dict['accuracy'] = accuracies

            return return_dict


# class MNIST(Framework):

#     def __init__(self, config: Dict) -> None:
#         Framework.__init__(self, config)

#     def train(self) -> None:

#         training_losses = []
#         total_step_count = 0
#         steps_per_task = []

#         iter_time = time.time()

#         while total_step_count < self.total_training_steps:
            
#             teacher_index = next(self.curriculum)
#             task_step_count = 0
#             latest_task_generalisation_error = np.inf
            
#             # alert learner/teacher(s) of task change e.g. change output head of student to relevant task (if in continual setting)
#             self._signal_task_boundary_to_learner(new_task=teacher_index)
#             self._signal_task_boundary_to_teacher(new_task=teacher_index)

#             while True:

#                 training_batch = []
                
#                 for _ in range(self.train_batch_size):
#                     if len(self.teachers[teacher_index]) == 0:
#                         task_data = get_binary_classification_datasets(self.mnist_train_x, self.mnist_train_y, self.mnist_teacher_classes[teacher_index], rotations=self.rotations[teacher_index])
#                         self.teachers[teacher_index] = task_data
#                     training_data_point = self.teachers[teacher_index].pop()
#                     training_batch.append(training_data_point)

#                 image_input = torch.stack([item[0] for item in training_batch]).to(self.device)

#                 teacher_output = torch.flatten(torch.stack([item[1] for item in training_batch]).to(self.device))
#                 student_output = self.student_network(image_input)  

#                 if self.verbose and task_step_count % 1000 == 0:
#                     self.logger.info("Training step {}".format(task_step_count))

#                 import pdb; pdb.set_trace()

#                 # training iteration
#                 self.optimiser.zero_grad()
#                 loss = self._compute_loss(student_output, teacher_output)
#                 loss.backward()
#                 self.optimiser.step()
#                 training_losses.append(float(loss))

#                 # log training loss
#                 self.writer.add_scalar('training_loss', float(loss), total_step_count)
#                 if self.log_to_df:
#                     self.logger_df.at[total_step_count, 'training_loss'] = float(loss)

#                 # test
#                 if total_step_count % self.test_frequency == 0 and total_step_count != 0:
#                     latest_task_generalisation_error = self._perform_test_loop(teacher_index, task_step_count, total_step_count)

#                 total_step_count += 1
#                 task_step_count += 1

#                 # overlap matrices
#                 if total_step_count % self.overlap_frequency == 0 and total_step_count != 0:
#                     self._compute_overlap_matrices(step_count=total_step_count)

#                 total_step_count += 1
#                 task_step_count += 1

#                 # output layer weights
#                 self._log_output_weights(step_count=total_step_count)

#                 # alert learner/teacher(s) of step e.g. to drift teacher
#                 self._signal_step_boundary_to_learner(step=task_step_count, current_task=teacher_index)
#                 self._signal_step_boundary_to_teacher(step=task_step_count, current_task=teacher_index)

#                 if total_step_count % 500 == 0:
#                     current_time = time.time()
#                     self.logger.info("Time taken for last {} steps: {}".format(500, current_time - iter_time))
#                     iter_time = current_time

#                 # checkpoint dataframe
#                 if self.logfile_path and total_step_count % self.checkpoint_frequency == 0:
#                     self._checkpoint_df(step=total_step_count)

#                 if self._switch_task(step=task_step_count, generalisation_error=latest_task_generalisation_error):
#                     self.writer.add_scalar('steps_per_task', task_step_count, total_step_count)
#                     if self.log_to_df:
#                         self.logger_df.at[total_step_count, 'steps_per_task'] = task_step_count
#                     steps_per_task.append(task_step_count)
#                     break

#     def extract_parameters(self, config: Dict) -> None:
#         self.rotations = config.get(["training", "rotations"])

   
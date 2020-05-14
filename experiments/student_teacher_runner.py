import numpy as np 
import copy
import time
import logging
import itertools
import sys

import torch 
import torch.optim as optim

from typing import List, Tuple, Generator, Dict, Union

from components import data_modules, data_modules, loss_modules, loggers
from models import teachers, learners
from utils import Parameters

class StudentTeacherRunner:

    def __init__(self, config: Parameters) -> None:
        """
        Class for orchestrating student teacher framework including training and test loops

        :param config: dictionary containing parameters to specify training etc.
        """
        # extract relevant parameters from config
        self._extract_parameters(config)

        # initialise student, teachers, logger_module, data_module and loss_module
        t0 = time.time()
        self.learner: learners._BaseLearner = self._setup_learner(config=config)
        t1 = time.time()
        print("Learners setup in {}s".format(round(t1 - t0, 5)))
        self.teachers: teachers._BaseTeachers = self._setup_teachers(config=config)
        t2 = time.time()
        print("Teachers setup in {}s".format(round(t2 - t1, 5)))
        self.logger: loggers._BaseLogger = self._setup_logger(config=config)
        t3 = time.time()
        print("Logger setup in {}s".format(round(t3 - t2, 5)))
        self.data_module: data_modules._BaseData = self._setup_data(config=config)
        t4 = time.time()
        print("Data module setup in {}s".format(round(t4 - t3, 5)))
        self.loss_module: loss_modules._BaseLoss = self._setup_loss(config=config)
        t5 = time.time()
        print("Loss module setup in {}s".format(round(t5 - t4, 5)))

        # compute test outputs for teachers
        self.test_data: Union[List[Dict[str, torch.Tensor]], Dict[str, Union[torch.Tensor, List[torch.Tensor]]]] = self.data_module.get_test_data()
        self.test_teacher_outputs: List[torch.Tensor] = self._get_test_teacher_outputs()

        # extract curriculum from config
        self._set_curriculum(config)

        # initialise optimiser with trainable parameters of student
        trainable_parameters = self.learner.get_trainable_parameters()
        self.optimiser = optim.SGD(trainable_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # initialise objects containing metrics of interest
        self._initialise_metrics()

    def _extract_parameters(self, config: Parameters) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        """
        self.verbose = config.get(["logging", "verbose"])
        self.logfile_path = config.get("logfile_path")
        self.checkpoint_frequency = config.get(["logging", "checkpoint_frequency"])
        self.log_to_df = config.get(["logging", "log_to_df"])

        self.total_training_steps = config.get(["training", "total_training_steps"])
        if self.total_training_steps is None:
            self.total_training_steps = np.inf
        self.learning_rate = config.get(["training", "learning_rate"])
        self.weight_decay = config.get(["training", "weight_decay"])

        self.num_teachers = config.get(["task", "num_teachers"])
        self.learner_configuration = config.get(["task", "learner_configuration"])
        self.teacher_configuration = config.get(["task", "teacher_configuration"])
        self.loss_type = config.get(["task", "loss_type"])

        self.test_frequency = config.get(["testing", "test_frequency"])
        self.overlap_frequency = config.get(["testing", "overlap_frequency"])

        self.input_dimension = config.get(["model", "input_dimension"])
        self.input_source = config.get(["data", "input_source"])
        self.same_input_distribution = config.get(["data", "same_input_distribution"])

    def _setup_learner(self, config: Parameters) -> learners._BaseLearner :
        if self.learner_configuration == "continual":
            return learners.ContinualLearner(config=config)
        elif self.learner_configuration == "meta":
            return learners.MetaLearner(config=config)
        else:
            raise ValueError("Learner configuration {} not recognised".format(self.learner_configuration))

    def _setup_teachers(self, config: Parameters) -> teachers._BaseTeachers:  
        self.teacher_is_network: bool
        if self.teacher_configuration == "overlapping":
            self.teacher_is_network = True
            return teachers.OverlappingTeachers(config=config)
        elif self.teacher_configuration == "pure_mnist":
            self.teacher_is_network = False
            return teachers.PureMNISTTeachers(config=config) # 'teachers' are MNIST images with labels - provided by data module
        elif self.teacher_configuration == "trained_mnist":
            self.teacher_is_network = True
            return teachers.TrainedMNISTTeachers(config=config)
        else:
            raise NotImplementedError("Teacher configuration {} not recognised".format(self.teacher_configuration))
        
    def _setup_logger(self, config: Parameters) -> loggers._BaseLogger:
        if self.teacher_configuration == "mnist":
            return loggers.StudentMNISTLogger(config=config)
        else:
            return loggers.StudentTeacherLogger(config=config)

    def _setup_data(self, config: Parameters) -> data_modules._BaseData:
        if self.input_source == 'iid_gaussian':
            return data_modules.IIDData(config)
        elif self.input_source == 'mnist_stream':
            return data_modules.MNISTStreamData(config)
        elif self.input_source == 'mnist_digits':
            return data_modules.MNISTDigitsData(config)
        elif self.input_source == 'even_greater':
            return data_modules.MNISTEvenGreaterData(config)
        else:    
            raise ValueError("Input source type {} not recognised. Please use either iid_gaussian or mnist".format(self.input_source))

    def _get_test_teacher_outputs(self) -> List[torch.Tensor]:
        if self.same_input_distribution:
            return [self.teachers.test_set_forward(t, self.test_data) for t in range(self.num_teachers)]
        else:
            return [self.teachers.forward(t, test_data) for t, test_data in enumerate(self.test_data)]

    def _setup_loss(self, config: Parameters) -> loss_modules._BaseLoss:
        if self.loss_type == "regression":
            return loss_modules.RegressionLoss(config=config)
        elif self.loss_type == "classification":
            return loss_modules.ClassificationLoss(config=config)
        else:
            raise NotImplementedError("Loss type {} not recognised".format(self.loss_type))

    def _initialise_metrics(self) -> None:
        """
        Initialise objects that will keep metrix of training e.g. generalisation errors
        Useful to make them attributes of class so that they can be accessed by all methods in class
        """
        self.generalisation_errors: Dict[int, List] = {i: [] for i in range(self.num_teachers)}

    def _set_curriculum(self, config: Parameters) -> None:
        """
        Establish and assign curriculum from configuration file. This is used to determine how to 
        proceed with training (when to switch teacher, how to decide subsequent teacher etc.)
        """
        curriculum_type = config.get(["curriculum", "type"])
        self.curriculum_stopping_condition = config.get(["curriculum", "stopping_condition"])
        if self.curriculum_stopping_condition == "fixed_period":
            self.curriculum_period = config.get(["curriculum", "fixed_period"])
        elif self.curriculum_stopping_condition == "single_threshold":
            self.current_loss_threshold = config.get(["curriculum", "loss_threshold"])
        elif self.curriculum_stopping_condition == "threshold_sequence":
            loss_threshold_sequence = config.get(["curriculum", "loss_threshold"])
            # make n copies of threshold sequence e.g. [thresh1, thresh2] -> [thresh1, thresh1, thresh2, thresh2]
            self.curriculum_loss_threshold = iter([threshold for threshold in loss_threshold_sequence for _ in range(self.num_teachers)])
            self.current_loss_threshold = next(self.curriculum_loss_threshold) 

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
            self.data_module.signal_task_boundary_to_data_generator(new_task=teacher_index)

            while total_step_count < self.total_training_steps:

                batch: Dict[str, torch.Tensor] = self.data_module.get_batch()
                batch_input: torch.Tensor = batch['x']
                            
                # feed full batch into teachers (i.e. potentially with label since come teacher configurations will simply return label)
                teacher_output = self.teachers.forward(teacher_index, batch)

                # student on the other hand only gets input (avoids accidentally accessing label)
                student_output = self.learner.forward(batch_input)

                assert teacher_output.shape == student_output.shape, \
                    "Shape of student and teacher outputs are different. To ensure correctness please fix"

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
                if total_step_count % self.test_frequency == 0:

                    # explicitly set model to evaluation mode
                    self.learner.set_to_eval()

                    latest_task_generalisation_error = self._perform_test_loop(teacher_index, task_step_count, total_step_count)

                    # explicitly set model back to training mode
                    self.learner.set_to_train()

                # overlap matrices
                if total_step_count % self.overlap_frequency == 0 and self.teacher_is_network:
                    self.logger._compute_overlap_matrices(
                        student_network=self.learner, 
                        teacher_networks=self.teachers.get_teacher_networks(), 
                        step_count=total_step_count
                        )

                total_step_count += 1
                task_step_count += 1

                # output layer weights
                self.logger._log_output_weights(step_count=total_step_count, student_network=self.learner)

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
        
        self.logger._consolidate_dfs()

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
        elif self.curriculum_stopping_condition == "single_threshold":
            if generalisation_error < self.current_loss_threshold:
                return True
            else:
                return False
        elif self.curriculum_stopping_condition == "threshold_sequence":
            if generalisation_error < self.current_loss_threshold:
                try:
                    self.current_loss_threshold = next(self.curriculum_loss_threshold)
                except:
                    self.logger.log("Sequence of thresholds exhausted. Run complete...")
                    self.logger._consolidate_dfs()
                    sys.exit(0)
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
        class_predictions = (prediction > 0.5).long().squeeze()
        accuracy = float((class_predictions == target).sum()) / len(target)
        return accuracy

    def _compute_generalisation_errors(self, teacher_index=None):
        with torch.no_grad():
            if self.same_input_distribution:
                student_outputs = self.learner.forward_all(self.test_data['x'])
            else:
                student_outputs = self.learner.forward_batch_per_task(self.test_data)

            # import pdb; pdb.set_trace()
            generalisation_errors = [float(self.loss_module.compute_loss(student_output, teacher_output)) for student_output, teacher_output in zip(student_outputs, self.test_teacher_outputs)]

            return_dict = {'generalisation_error': generalisation_errors}

            if self.loss_type == "classification":
                accuracies = [float(self._compute_classification_acc(student_output, teacher_output)) for student_output, teacher_output in zip(student_outputs, self.test_teacher_outputs)]
                return_dict['accuracy'] = accuracies
            return return_dict
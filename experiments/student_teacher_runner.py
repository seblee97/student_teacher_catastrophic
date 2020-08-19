import copy
import itertools
import os
import time
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import torch
import torch.distributions as tdist
import torch.optim as optim

from components import data_modules
from components import loggers
from components import loss_modules
from constants import Constants
from models import learners
from models import teachers
from ode import configuration
from ode import dynamics
from utils import Parameters


class StudentTeacherRunner:

    def __init__(self, config: Parameters) -> None:
        """
        Class for orchestrating student teacher framework including training
        and test loops

        :param config: dictionary containing parameters to specify
                       training etc.
        """
        # extract relevant parameters from config
        self._extract_parameters(config)

        # initialise student, teachers, logger_module,
        # data_module and loss_module
        t0 = time.time()
        self.learner: learners._BaseLearner = \
            self._setup_learner(config=config)
        t1 = time.time()
        print("Learners setup in {}s".format(round(t1 - t0, 5)))
        self.teachers: teachers._BaseTeachers = \
            self._setup_teachers(config=config)
        t2 = time.time()
        print("Teachers setup in {}s".format(round(t2 - t1, 5)))
        self.logger: loggers._BaseLogger = self._setup_logger(config=config)
        t3 = time.time()
        print("Logger setup in {}s".format(round(t3 - t2, 5)))
        self.data_module: data_modules._BaseData = \
            self._setup_data(config=config)
        t4 = time.time()
        print("Data module setup in {}s".format(round(t4 - t3, 5)))
        self.loss_module: loss_modules._BaseLoss = \
            self._setup_loss(config=config)
        t5 = time.time()
        print("Loss module setup in {}s".format(round(t5 - t4, 5)))

        if self._save_initial_weights:
            self._save_network_initial_weights()

        # compute test outputs for teachers
        self.test_data: Constants.TEST_DATA_TYPES = \
            self.data_module.get_test_data()
        self.test_teacher_outputs: List[torch.Tensor] = \
            self.teachers.test_set_forward(self.test_data)

        # extract curriculum from config
        self._set_curriculum(config)

        # initialise optimiser with trainable parameters of student
        trainable_parameters = self.learner.get_trainable_parameters()
        self.optimiser = optim.SGD(
            trainable_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)

        # initialise objects containing metrics of interest
        self._initialise_metrics()

    def _extract_parameters(self, config: Parameters) -> None:
        """
        Method to extract relevant parameters from config and
        make them attributes of this class
        """
        self.verbose = config.get(["logging", "verbose"])
        self.logfile_path = config.get("logfile_path")
        self.experiment_path = config.get("checkpoint_path")
        self.checkpoint_frequency = \
            config.get(["logging", "checkpoint_frequency"])
        self.log_to_df = config.get(["logging", "log_to_df"])
        self._save_weights_at_switch = \
            config.get(["logging", "save_weights_at_switch"])
        self._save_initial_weights = \
            config.get(["logging", "save_initial_weights"])
        self._weight_save_path = \
            os.path.join(self.experiment_path, "learner_weights")
        os.makedirs(self._weight_save_path, exist_ok=False)

        self.total_training_steps = \
            config.get(["training", "total_training_steps"])
        if self.total_training_steps is None:
            self.total_training_steps = np.inf
        self.learning_rate = config.get(["training", "learning_rate"])
        self.weight_decay = config.get(["training", "weight_decay"])

        self.num_teachers = config.get(["task", "num_teachers"])
        self.learner_configuration = \
            config.get(["task", "learner_configuration"])
        self.teacher_configuration = \
            config.get(["task", "teacher_configuration"])
        self.loss_type = config.get(["task", "loss_type"])

        self.test_batch_size = config.get(["testing", "test_batch_size"])
        self.test_frequency = config.get(["testing", "test_frequency"])
        self.overlap_frequency = config.get(["testing", "overlap_frequency"])

        self.input_dimension = config.get(["model", "input_dimension"])
        self.student_hidden_layers = config.get(["model", "student_hidden_layers"])
        self.teacher_hidden_layers = config.get(["model", "teacher_hidden_layers"])
        self.student_initialisation_std = config.get(["model", "student_initialisation_std"])
        self.teacher_initialisation_std = config.get(["model", "teacher_initialisation_std"])
        self.initialise_student_outputs = config.get(["model", "initialise_student_outputs"])
        self.nonlinearity = config.get(["model", "student_nonlinearity"])
        self.normalise_teachers = config.get(["model", "normalise_teachers"])
        self.symmetric_student_initialisation = config.get(
            ["model", "symmetric_student_initialisation"])
        self.soft_committee = config.get(["model", "soft_committee"])
        self.input_source = config.get(["data", "input_source"])
        self.same_input_distribution = \
            config.get(["data", "same_input_distribution"])

        self._noise = config.get(["data", "noise"])
        self._noise_to_teacher = config.get(["data", "noise_to_teacher"])

        if self._noise is not None:
            self._noise_distribution = tdist.Normal(0, self._noise)

        self.ode_timestep_scaling = config.get(["training", "ode_timestep_scaling"])

    def _save_network_initial_weights(self):
        # save initial student weights
        learner_save_path = os.path.join(self._weight_save_path, "switch_0_step_0_saved_weights.pt")
        self.logger.log("Saving initial weights")
        self.learner.save_weights(learner_save_path)
        # save teacher weights
        for t in range(self.num_teachers):
            teacher_save_path = os.path.join(self._weight_save_path,
                                             "teacher_{}_weights.pt".format(t))
            self.teachers.save_weights(t, teacher_save_path)

    def _setup_learner(self, config: Parameters) -> learners._BaseLearner:
        if self.learner_configuration == "continual":
            return learners.ContinualLearner(config=config)
        elif self.learner_configuration == "meta":
            return learners.MetaLearner(config=config)
        else:
            raise ValueError("Learner configuration {} not \
                    recognised".format(self.learner_configuration))

    def _setup_teachers(self, config: Parameters) -> teachers._BaseTeachers:
        self.teacher_is_network: bool
        if self.teacher_configuration == "overlapping":
            self.teacher_is_network = True
            return teachers.OverlappingTeachers(config=config)
        elif self.teacher_configuration == "pure_mnist":
            self.teacher_is_network = False
            return teachers.PureMNISTTeachers(config=config)
        elif self.teacher_configuration == "trained_mnist":
            self.teacher_is_network = True
            return teachers.TrainedMNISTTeachers(config=config)
        else:
            raise NotImplementedError("Teacher configuration {} not \
                    recognised".format(self.teacher_configuration))

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
            raise ValueError("Input source type {} not \
                    recognised".format(self.input_source))

    def _setup_loss(self, config: Parameters) -> loss_modules._BaseLoss:
        if self.loss_type == "regression":
            return loss_modules.RegressionLoss(config=config)
        elif self.loss_type == "classification":
            return loss_modules.ClassificationLoss(config=config)
        else:
            raise NotImplementedError("Loss type {} not recognised".format(self.loss_type))

    def _initialise_metrics(self) -> None:
        """
        Initialise objects that will keep metrix of training
        e.g. generalisation errors
        Useful to make them attributes of class so that they
        can be accessed by all methods in class
        """
        self.generalisation_errors: Dict[int, List] = {i: [] for i in range(self.num_teachers)}

    def _set_curriculum(self, config: Parameters) -> None:
        """
        Establish and assign curriculum from configuration file.
        This is used to determine how to proceed with training
        (when to switch teacher, how to decide subsequent teacher etc.)
        """
        curriculum_type = config.get(["curriculum", "type"])
        self.curriculum_stopping_condition = \
            config.get(["curriculum", "stopping_condition"])
        if self.curriculum_stopping_condition == "fixed_period":
            self.curriculum_period = config.get(["curriculum", "fixed_period"])
        elif self.curriculum_stopping_condition == "single_threshold":
            self.current_loss_threshold = \
                config.get(["curriculum", "loss_threshold"])
        elif self.curriculum_stopping_condition == "threshold_sequence":
            loss_threshold_sequence = \
                config.get(["curriculum", "loss_threshold"])
            # make n copies of threshold sequence
            # e.g. [thresh1, thresh2] -> [thresh1, thresh1, thresh2, thresh2]
            self.curriculum_loss_threshold = iter([
                threshold for threshold in loss_threshold_sequence for _ in range(self.num_teachers)
            ])
            self.current_loss_threshold = next(self.curriculum_loss_threshold)

        if curriculum_type == "custom":
            self.curriculum = \
                itertools.cycle((config.get(["curriculum", "custom"])))
        elif curriculum_type == "standard":
            self.curriculum_selection_type = \
                config.get(["curriculum", "selection_type"])
            if self.curriculum_selection_type == "cyclical":
                self.curriculum = \
                    itertools.cycle(list(range(self.num_teachers)))
            elif self.curriculum_selection_type == "random":
                raise NotImplementedError
            else:
                raise ValueError("You have specified a standard curriculum \
                        (as opposed to custom), but the selection \
                        type is not recognised. Please choose between \
                        'cyclical' and 'random'")
        else:
            raise ValueError("Curriculum type {} not recognised".format(curriculum_type))

    def _log_to_df(self, tag: str, scalar: float, step: int) -> None:
        """Makes call to logger method, avoids constant checks for
        boolean log_to_df in training loop etc.
        """
        if self.log_to_df:
            self.logger.write_scalar_df(tag=tag, scalar=scalar, step=step)

    def run_ode(self) -> None:
        """Use configuration established in initialisation to run ODE equations.
        """
        self.logger.log("Starting ODE run...")

        student_weight_vector_1 = self.learner.state_dict()["layers.0.weight"].numpy()[0]
        student_weight_vector_2 = self.learner.state_dict()["layers.0.weight"].numpy()[1]

        student_weight_vectors = [student_weight_vector_1, student_weight_vector_2]

        teacher_1_weight_vector = self.teachers._teachers[0].state_dict()["layers.0.weight"].numpy()
        teacher_2_weight_vector = self.teachers._teachers[1].state_dict()["layers.0.weight"].numpy()

        Q = configuration.RandomStudentTwoTeacherConfiguration.weight_overlap_matrix(
            student_weight_vectors, student_weight_vectors, N=self.input_dimension)

        R = configuration.RandomStudentTwoTeacherConfiguration.weight_overlap_matrix(
            student_weight_vectors, teacher_1_weight_vector, N=self.input_dimension)

        U = configuration.RandomStudentTwoTeacherConfiguration.weight_overlap_matrix(
            student_weight_vectors, teacher_2_weight_vector, N=self.input_dimension)

        T = configuration.RandomStudentTwoTeacherConfiguration.weight_overlap_matrix(
            teacher_1_weight_vector, teacher_1_weight_vector, N=self.input_dimension)

        S = configuration.RandomStudentTwoTeacherConfiguration.weight_overlap_matrix(
            teacher_2_weight_vector, teacher_2_weight_vector, N=self.input_dimension)

        V = configuration.RandomStudentTwoTeacherConfiguration.weight_overlap_matrix(
            teacher_1_weight_vector, teacher_2_weight_vector, N=self.input_dimension)

        h1 = self.learner.state_dict()["heads.0.weight"].numpy().flatten()
        h2 = self.learner.state_dict()["heads.1.weight"].numpy().flatten()

        th1 = self.teachers._teachers[0].state_dict()["output_layer.weight"].numpy().flatten()
        th2 = self.teachers._teachers[1].state_dict()["output_layer.weight"].numpy().flatten()

        ode_configuration = \
            configuration.StudentTwoTeacherConfiguration(
                Q=Q,
                R=R,
                U=U,
                T=T,
                S=S,
                V=V,
                h1=h1,
                h2=h2,
                th1=th1,
                th2=th2
            )

        dt = self.ode_timestep_scaling

        curriculum = self._get_ode_curriculum(
            total_steps=self.total_training_steps,
            period=self.curriculum_period,
            scaling=self.input_dimension)

        ode = dynamics.StudentTeacherODE(
            overlap_configuration=ode_configuration,
            nonlinearity=self.nonlinearity,
            w_learning_rate=self.learning_rate,
            h_learning_rate=self.learning_rate,
            dt=dt,
            curriculum=curriculum,
            soft_committee=self.soft_committee)

        ode.step(self.total_training_steps / self.input_dimension)

        ode.save_to_csv(save_path=self.experiment_path)
        ode.make_plot(save_path=self.experiment_path, total_time=self.total_training_steps)

    def _get_ode_curriculum(self, total_steps: int, period: int, scaling: int = 1) -> List[int]:
        return np.arange(0, total_steps, period)[1:] / scaling

    def train(self) -> None:
        """Training loop
        """
        self.logger.log("Starting training...")

        training_losses = []
        total_step_count = 0
        steps_per_task = []
        num_switches = 0

        iter_time = time.time()

        # explicitly set model to training mode
        self.learner.set_to_train()

        while total_step_count < self.total_training_steps:

            task_step_count = 0

            if self.log_to_df:
                self.logger.add_row(row_label=total_step_count)

            teacher_index = next(self.curriculum)
            latest_task_generalisation_error = np.inf

            # alert learner/teacher(s) of task change
            # e.g. change output head of student to
            # relevant task (if in continual setting)
            self.learner.signal_task_boundary_to_learner(new_task=teacher_index)
            self.teachers.signal_task_boundary_to_teacher(new_task=teacher_index)
            self.data_module.signal_task_boundary_to_data_generator(new_task=teacher_index)

            while total_step_count < self.total_training_steps:  # train on given teacher

                pre_step_learner_weights = \
                    copy.deepcopy(self.learner.state_dict())

                total_step_count += 1
                task_step_count += 1

                batch: Dict[str, torch.Tensor] = self.data_module.get_batch()
                batch_input: torch.Tensor = batch['x']

                if self._noise is not None:
                    batch_input_noise = \
                        self._noise_distribution.sample(batch_input.shape)
                    batch_input_student = batch_input + batch_input_noise
                else:
                    batch_input_student = batch_input
                if self._noise_to_teacher:
                    batch['x'] = batch_input_student

                # feed full batch into teachers (i.e. potentially with label
                # since come teacher configurations will simply return label)
                teacher_output = self.teachers.forward(teacher_index, batch)

                # student on the other hand only gets input
                # (avoids accidentally accessing label)
                student_output = self.learner.forward(batch_input_student)

                assert teacher_output.shape == student_output.shape, \
                    "Shape of student and teacher outputs are different. \
                        To ensure correctness please fix"

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

                self._log_to_df('training_loss', float(loss), total_step_count)

                # test
                if total_step_count % self.test_frequency == 0:

                    # explicitly set model to evaluation mode
                    self.learner.set_to_eval()

                    latest_task_generalisation_error = self._perform_test_loop(
                        teacher_index, task_step_count, total_step_count)

                    # explicitly set model back to training mode
                    self.learner.set_to_train()

                # overlap matrices
                if (total_step_count % self.overlap_frequency == 0 and self.teacher_is_network):
                    self.logger.compute_overlap_matrices(
                        student_network=self.learner,
                        teacher_networks=self.teachers.get_teacher_networks(),
                        step_count=total_step_count)

                # weight changes
                self.logger._log_weight_diff(
                    step_count=total_step_count,
                    old_weights=pre_step_learner_weights,
                    student_network=self.learner)

                # output layer weights
                self.logger._log_output_weights(
                    step_count=total_step_count, student_network=self.learner)

                # alert learner/teacher(s) of step e.g. to drift teacher
                self.learner.signal_step_boundary_to_learner(
                    step=task_step_count, current_task=teacher_index)
                self.teachers.signal_step_boundary_to_teacher(
                    step=task_step_count, current_task=teacher_index)

                if total_step_count % 500 == 0:
                    current_time = time.time()
                    self.logger.log("Time taken for last {} steps: {}".format(
                        500, current_time - iter_time))
                    iter_time = current_time

                # checkpoint dataframe
                if (self.logfile_path and total_step_count % self.checkpoint_frequency == 0 and
                        self.log_to_df):
                    self.logger.checkpoint_df(step=total_step_count)

                to_switch = self._switch_task(
                    step=task_step_count, generalisation_error=latest_task_generalisation_error)

                if to_switch is True:
                    num_switches += 1
                    self.logger.write_scalar_tb('steps_per_task', task_step_count, total_step_count)
                    steps_per_task.append(task_step_count)
                    if self._save_weights_at_switch:
                        save_path = os.path.join(
                            self._weight_save_path, "switch_{}_step_{}_saved_weights.pt".format(
                                num_switches, total_step_count))
                        self.logger.log(
                            "Saving weights after {}th task switch".format(num_switches))
                        self.learner.save_weights(save_path)
                    break
                elif to_switch is None:
                    # checkpoint outstanding data
                    self.logger.log("TRAINING COMPLETE")
                    if self.log_to_df:
                        self.logger.checkpoint_df(step=total_step_count)
                    # set step to infinity to break out of loop
                    total_step_count = np.inf
                elif to_switch is False:
                    continue
                else:
                    raise ValueError("Output of switch _task call must be bool or None")

        # checkpoint outstanding data
        self.logger.log("TRAINING COMPLETE")
        if self.log_to_df:
            self.logger.checkpoint_df(step=total_step_count)

    def consolidate_run(self) -> None:
        self.logger._consolidate_dfs()

    def _switch_task(self, step: int, generalisation_error: float) -> Union[bool, None]:
        """
        Method to determine whether to switch task
        (i.e. whether condition set out by curriculum
        for switching has been met).

        Args:
            step: number of steps completed for current task
            being trained (not overall step count).

            generalisation_error: generalisation error associated
            with current teacher.

        Returns:
            True - if switch condition has been met.
            False - if switch condition has not been met.
            None - if switch condition ha been met and all tasks
            in cycle have been exhausted (end of training).
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
                    self.current_loss_threshold = \
                        next(self.curriculum_loss_threshold)
                    return True
                except StopIteration:
                    self.logger.log("Sequence of thresholds exhausted. Run complete...")
                    # checkpoint outstanding data
                    if self.log_to_df:
                        self.logger.checkpoint_df(step=step)
                    return None
            else:
                return False
        else:
            raise ValueError("curriculum stopping condition {} unknown. \
                    Please use 'fixed_period' or \
                    'threshold'.".format(self.curriculum_stopping_condition))

    def _perform_test_loop(self, teacher_index: int, task_step_count: int,
                           total_step_count: int) -> float:
        """
        Test loop. Evaluated generalisation error of student wrt teacher(s)

        :param teacher_index: index of current teacher student is being
                              trained on
        :param task_step_count: number of steps completed under current teacher
        :param total_step_count: total number of steps student has been
                                 training for

        :return generalisation_error_wrt_current_teacher: generalisation_error
                                                          for current teacher
        """
        with torch.no_grad():

            generalisation_metrics = self._compute_generalisation_errors(
                teacher_index=teacher_index)

            generalisation_error_per_teacher = \
                generalisation_metrics.get('generalisation_error')
            classification_accuracy = generalisation_metrics.get('accuracy')

            # log average generalisation losses over teachers
            mean_generalisation_error_per_teacher = \
                np.mean(generalisation_error_per_teacher)
            self.logger.write_scalar_tb('mean_generalisation_error/linear',
                                        mean_generalisation_error_per_teacher, total_step_count)
            self.logger.write_scalar_tb('mean_generalisation_error/log',
                                        np.log10(mean_generalisation_error_per_teacher),
                                        total_step_count)
            self._log_to_df('mean_generalisation_error/linear',
                            mean_generalisation_error_per_teacher, total_step_count)
            self._log_to_df('mean_generalisation_error/log',
                            np.log10(mean_generalisation_error_per_teacher), total_step_count)

            for i, error in enumerate(generalisation_error_per_teacher):
                self.generalisation_errors[i].append(error)

                # log generalisation loss per teacher
                self.logger.write_scalar_tb(
                    'generalisation_error/linear', error, total_step_count, teacher_index=i)
                self.logger.write_scalar_tb(
                    'generalisation_error/log', np.log10(error), total_step_count, teacher_index=i)
                self._log_to_df('teacher_{}_generalisation_error/log'.format(i), np.log10(error),
                                total_step_count)
                self._log_to_df('teacher_{}_generalisation_error/linear'.format(i), error,
                                total_step_count)

                if classification_accuracy:
                    self.logger.write_scalar_tb(
                        'classification_accuracy',
                        classification_accuracy[i],
                        total_step_count,
                        teacher_index=i)
                    self._log_to_df('teacher_{}_classification_accuracy'.format(i),
                                    classification_accuracy[i], total_step_count)

                if len(self.generalisation_errors[i]) > 1:
                    last_error = copy.deepcopy(self.generalisation_errors[i][-2])
                    error_delta = error - last_error
                    # log generalisation loss delta per teacher
                    if error_delta != 0.:
                        log_error_change = \
                            np.sign(error_delta) * np.log10(abs(error_delta))
                    else:
                        log_error_change = \
                            0.

                    self.logger.write_scalar_tb(
                        'error_change/linear', error_delta, total_step_count, teacher_index=i)
                    self.logger.write_scalar_tb(
                        'error_change/log', log_error_change, total_step_count, teacher_index=i)
                    self._log_to_df('teacher_{}_error_change/linear'.format(i), error_delta,
                                    total_step_count)
                    self._log_to_df('teacher_{}_error_change/log'.format(i), log_error_change,
                                    total_step_count)

            if self.verbose and task_step_count % 500 == 0:
                self.logger.log("Generalisation errors @ step {} ({}'th step training on \
                        teacher {}):".format(total_step_count, task_step_count, teacher_index))
                for i, error in enumerate(generalisation_error_per_teacher):
                    self.logger.log("{}Teacher {}: {}\n".format("".rjust(10), i, error))
                if classification_accuracy:
                    self.logger.log("Classification accuracies @ step \
                        {} ({}'th step training on teacher {}):".format(
                        total_step_count, task_step_count, teacher_index))
                    for i, acc in enumerate(classification_accuracy):
                        self.logger.log("{}Teacher {}: {}\n".format("".rjust(10), i, acc),)

            generalisation_error_wrt_current_teacher = \
                generalisation_error_per_teacher[teacher_index]

        return generalisation_error_wrt_current_teacher

    def _compute_classification_acc(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        class_predictions = (prediction > 0.5).long().squeeze()
        accuracy = float((class_predictions == target).sum()) / len(target)
        return accuracy

    def _compute_generalisation_errors(self, teacher_index=None) -> Dict[str, float]:
        with torch.no_grad():
            if self.same_input_distribution:
                student_outputs = self.learner.forward_all(self.test_data['x'])
            else:
                student_outputs = \
                    self.learner.forward_batch_per_task(self.test_data)

            zipped_outputs = zip(student_outputs, self.test_teacher_outputs)

            generalisation_errors = []
            accuracies = []
            for student_output, teacher_output in zipped_outputs:
                loss = self.loss_module.compute_loss(student_output, teacher_output)
                generalisation_errors.append(float(loss))
                if self.loss_type == "classification":
                    accuracy = self._compute_classification_acc(student_output, teacher_output)
                    accuracies.append(float(accuracy))

            return_dict = {'generalisation_error': generalisation_errors}
            if self.loss_type == "classification":
                return_dict['accuracy'] = accuracies

            return return_dict

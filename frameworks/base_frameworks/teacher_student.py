import numpy as np 
import copy
import itertools
import pandas as pd

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

from tensorboardX import SummaryWriter

from typing import List, Tuple, Generator, Dict

from utils import visualise_matrix, get_binary_classification_datasets, load_mnist_data, load_mnist_data_as_dataloader

from abc import ABC, abstractmethod

class Framework(ABC):

    def __init__(self, config: Dict) -> None:
        """
        Class for orchestrating student teacher framework including training and test loops

        :param config: dictionary containing parameters to specify training etc.
        """
        # extract relevant parameters from config
        self._extract_parameters(config)

        # setup teacher(s) and students according to config
        self._setup_teachers(config=config)
        self._setup_student(config=config)

        # extract curriculum from config
        self._set_curriculum(config)
        
        trainable_parameters = self.student_network.parameters()
        # trainable_parameters = filter(lambda param: param.requires_grad, self.student_network.parameters())

        # initialise optimiser with trainable parameters of student        
        self.optimiser = optim.SGD(trainable_parameters, lr=self.learning_rate)
        
        # initialise loss function
        if config.get(["training", "loss_function"]) == 'mse':
            self.loss_function = nn.MSELoss()
        elif config.get(["training", "loss_function"]) == 'cross_entropy':
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("{} is not currently supported, please use mse loss or cross_entropy for mnist".format(config.get("loss")))

        # initialise general tensorboard writer
        self.writer = SummaryWriter(self.checkpoint_path)
        # initialise separate tb writers for each teacher
        self.teacher_writers = [SummaryWriter("{}/teacher_{}".format(self.checkpoint_path, t)) for t in range(self.num_teachers)]

        # initialise dataframe to log metrics
        self.logger_df = pd.DataFrame()

        # initialise objects containing metrics of interest
        self._initialise_metrics()

    def _extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        """

        self.verbose = config.get("verbose")
        self.checkpoint_path = config.get("checkpoint_path")
        self.device = config.get("device")
        self.logfile_path = config.get("logfile_path")
        self.checkpoint_frequency = config.get("checkpoint_frequency")

        self.total_training_steps = config.get(["training", "total_training_steps"])

        self.input_dimension = config.get(["model", "input_dimension"])
        self.output_dimension = config.get(["model", "output_dimension"])
        # self.nonlinearity = config.get(["model", "nonlinearity"])
        self.soft_committee = config.get(["model", "soft_committee"])
        self.student_hidden = config.get(["model", "student_hidden_layers"])

        self.train_batch_size = config.get(["training", "train_batch_size"])
        self.test_batch_size = config.get(["training", "test_batch_size"])
        self.learning_rate = config.get(["training", "learning_rate"])

        self.test_all_teachers = config.get(["testing", "test_all_teachers"])
        self.test_frequency = config.get(["testing", "test_frequency"])
        self.overlap_frequency = config.get(["testing", "overlap_frequency"])

        self.num_teachers = config.get(["task", "num_teachers"])
        self.label_task_bounaries = config.get(["task", "label_task_boundaries"])
        self.learner_configuration = config.get(["task", "learner_configuration"])
        self.teacher_configuration = config.get(["task", "teacher_configuration"])

        self.input_source = config.get(["training", "input_source"])
        self.pca_input = config.get(["training", "pca_input"])

        if self.pca_input > 0:
            if self.pca_input != self.input_dimension:
                raise ValueError("Please ensure that if PCA is applied that the number of principal components\
                    matches the input dimension for the network.")

    @abstractmethod
    def _setup_teachers(self, config: Dict):
        """Instantiate all teachers"""
        raise NotImplementedError("Base class method")

    @abstractmethod
    def _setup_student(self, config: Dict):
        """Instantiate student"""
        raise NotImplementedError("Base class method")

    def _initialise_metrics(self) -> None:
        """
        Initialise objects that will keep metrix of training e.g. generalisation errors
        Useful to make them attributes of class so that they can be accessed by all methods in class
        """
        self.generalisation_errors = {i: [] for i in range(len(self.teachers))}

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

    @abstractmethod
    def train(self) -> None:
        """
        Training loop
        """
        raise NotImplementedError("Base class method")

    @abstractmethod
    def _signal_task_boundary_to_learner(self):
        raise NotImplementedError("Base class method")

    @abstractmethod
    def _signal_task_boundary_to_teacher(self):
        raise NotImplementedError("Base class method")

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

            generalisation_error_per_teacher = self._compute_generalisation_errors(teacher_index=teacher_index).get('generalisation_error')
            classification_accuracy = self._compute_generalisation_errors(teacher_index=teacher_index).get('accuracy')

            # log average generalisation losses over teachers
            mean_generalisation_error_per_teacher = np.mean(generalisation_error_per_teacher)
            self.writer.add_scalar(
                'mean_generalisation_error/linear', mean_generalisation_error_per_teacher, total_step_count
                )
            self.writer.add_scalar(
                'mean_generalisation_error/log', np.log10(mean_generalisation_error_per_teacher), total_step_count
                )
            self.logger_df.at[total_step_count, 'mean_generalisation_error/linear'] = mean_generalisation_error_per_teacher
            self.logger_df.at[total_step_count, 'mean_generalisation_error/log'] = np.log10(mean_generalisation_error_per_teacher)


            for i, error in enumerate(generalisation_error_per_teacher):
                self.generalisation_errors[i].append(error)

                # log generalisation loss per teacher
                self.teacher_writers[i].add_scalar('generalisation_error/linear', error, total_step_count)
                self.teacher_writers[i].add_scalar('generalisation_error/log', np.log10(error), total_step_count)
                self.logger_df.at[total_step_count, 'teacher_{}_generalisation_error/linear'.format(i)] = error
                self.logger_df.at[total_step_count, 'teacher_{}_generalisation_error/log'.format(i)] = np.log10(error)

                if classification_accuracy:
                    self.teacher_writers[i].add_scalar('classification_accuracy', classification_accuracy[i], total_step_count)
                    self.logger_df.at[total_step_count, 'teacher_{}_classification_accuracy'.format(i)] = classification_accuracy[i]

                if len(self.generalisation_errors[i]) > 1:
                    last_error = copy.deepcopy(self.generalisation_errors[i][-2])
                    error_delta = error - last_error
                    if error_delta != 0.:
                        # log generalisation loss delta per teacher
                        self.teacher_writers[i].add_scalar('error_change/linear', error_delta, total_step_count)
                        self.teacher_writers[i].add_scalar('error_change/log', np.sign(error_delta) * np.log10(abs(error_delta)), total_step_count)
                        self.logger_df.at[total_step_count, 'teacher_{}_error_change/linear'.format(i)] = error_delta
                        self.logger_df.at[total_step_count, 'teacher_{}_error_change/log'.format(i)] = np.sign(error_delta) * np.log10(abs(error_delta))

            if self.verbose and task_step_count % 500 == 0:
                print("Generalisation errors @ step {} ({}'th step training on teacher {}):".format(total_step_count, task_step_count, teacher_index))
                for i, error in enumerate(generalisation_error_per_teacher):
                    print(
                        "{}Teacher {}: {}".format("".rjust(10), i, error),
                        sep="\n"
                    )
                if classification_accuracy:
                    print("Classification accuracies @ step {} ({}'th step training on teacher {}):".format(total_step_count, task_step_count, teacher_index))
                    for i, acc in enumerate(classification_accuracy):
                        print(
                            "{}Teacher {}: {}".format("".rjust(10), i, acc),
                            sep="\n"
                        )

            generalisation_error_wrt_current_teacher = generalisation_error_per_teacher[teacher_index]

            return generalisation_error_wrt_current_teacher

    def _log_output_weights(self, step_count: int) -> None:
        """
        extract and log output weights of student.
        """
        # extract output layer weights for student
        student_output_layer_weights = self.student_network._get_head_weights()
        for h, head in enumerate(student_output_layer_weights):
            flattened_weights = torch.flatten(head)
            for w, weight in enumerate(flattened_weights):
                self.writer.add_scalar('student_head_{}_weight_{}'.format(h, w), float(weight), step_count)
                self.logger_df.at[step_count, 'student_head_{}_weight_{}'.format(h, w)] = float(weight)

    @abstractmethod
    def _compute_overlap_matrices(self, step_count: int) -> None:
        """
        calculated overlap matrices (order parameters) of student wrt itself, teacher wrt itself and student wrt teacher
        produces figures and logs to tb

        :param step_count: number of steps (overall) taken by student in training so far
        """
        raise NotImplementedError("Base class method")

    @abstractmethod
    def _compute_generalisation_errors(self, teacher_index: int=None) -> Dict[str, List[float]]:
        """
        calculated generalisation errors wrt fixed test dataset of student against teacher indices given

        :param teacher_indices: teachers against which to test student 
        :return generalisation_errors: list of generalisation errors of student against all teachers specified
        """
        raise NotImplementedError("Base class method")
    
    @abstractmethod
    def _compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates loss of prediction of student vs. target from teacher
        Loss function determined by configuration

        :param prediction: prediction made by student network on given input
        :param target: target - teacher output on same input

        :return loss: loss between target (from teacher) and prediction (from student)
        """
        raise NotImplementedError("Base class method")


class StudentTeacher(Framework):

    def __init__(self, config: Dict) -> None:
        Framework.__init__(self, config)

        # generate fixed test data
        if self.input_source == 'iid_gaussian':
            self.test_input_data = torch.randn(self.test_batch_size, self.input_dimension).to(self.device)
        elif self.input_source == 'mnist':
            # load mnist test data
            self.data_path = config.get("data_path")
            test_input_data = iter(load_mnist_data_as_dataloader(
                data_path=self.data_path, train=False, batch_size=self.test_batch_size, pca=self.pca_input)
                ).next()[0]

            self.data_mu = torch.mean(test_input_data, axis=0)
            self.data_sigma = torch.std(test_input_data, axis=0)

            # standardise test inputs
            self.test_input_data = torch.stack([(d - self.data_mu) / self.data_sigma for d in test_input_data])

            # load mnist training data
            self.mnist_dataloader = load_mnist_data_as_dataloader(data_path=self.data_path, batch_size=self.train_batch_size, pca=self.pca_input)
            # self.mnist_train_x = self.mnist_train_x.type(torch.FloatTensor)
            self.training_data_iterator = iter(self.mnist_dataloader)

        else:
            raise ValueError("Input source type {} not recognised. Please use either iid_gaussian or mnist".format(self.input_source))

        self.test_teacher_outputs = [teacher(self.test_input_data) for teacher in self.teachers]

    def train(self) -> None:

        training_losses = []
        total_step_count = 0
        steps_per_task = []

        while total_step_count < self.total_training_steps:
            
            teacher_index = next(self.curriculum)
            task_step_count = 0
            latest_task_generalisation_error = np.inf
            
            # alert learner/teacher(s) of task change e.g. change output head of student to relevant task (if in continual setting)
            self._signal_task_boundary_to_learner(new_task=teacher_index)
            self._signal_task_boundary_to_teacher(new_task=teacher_index)

            while True:
                
                if self.input_source == 'iid_gaussian':
                    batch_input = torch.randn(self.train_batch_size, self.input_dimension).to(self.device)
                elif self.input_source == 'mnist':
                    
                    try:
                        batch_input = next(self.training_data_iterator)[0].to(self.device)
                    except:
                        self.training_data_iterator = iter(self.mnist_dataloader)
                        batch_input = next(self.training_data_iterator)[0].to(self.device)

                    # normalise input
                    batch_input = (batch_input - self.data_mu) / self.data_sigma

                else:
                    raise ValueError("Input source type {} not recognised. Please use either iid_gaussian or mnist".format(self.input_source))

                teacher_output = self.teachers[teacher_index](batch_input)
                student_output = self.student_network(batch_input)

                if self.verbose and task_step_count % 1000 == 0:
                    print("Training step {}".format(task_step_count))

                # training iteration
                self.optimiser.zero_grad()
                loss = self._compute_loss(student_output, teacher_output)
                loss.backward()
                self.optimiser.step()
                training_losses.append(float(loss))

                # log training loss
                self.writer.add_scalar('training_loss', float(loss), total_step_count)
                self.logger_df.at[total_step_count, 'training_loss'] = float(loss)
                

                # test
                if total_step_count % self.test_frequency == 0 and total_step_count != 0:
                    latest_task_generalisation_error = self._perform_test_loop(teacher_index, task_step_count, total_step_count)

                # overlap matrices
                if total_step_count % self.overlap_frequency == 0 and total_step_count != 0:
                    self._compute_overlap_matrices(step_count=total_step_count)

                total_step_count += 1
                task_step_count += 1

                # output layer weights
                self._log_output_weights(step_count=total_step_count)

                # alert learner/teacher(s) of step e.g. to drift teacher
                self._signal_step_boundary_to_learner(step=task_step_count, current_task=teacher_index)
                self._signal_step_boundary_to_teacher(step=task_step_count, current_task=teacher_index)

                if self._switch_task(step=task_step_count, generalisation_error=latest_task_generalisation_error):
                    self.writer.add_scalar('steps_per_task', task_step_count, total_step_count)
                    self.logger_df.at[total_step_count, "steps_per_task"] = task_step_count
                    steps_per_task.append(task_step_count)
                    break

                # checkpoint dataframe
                if self.logfile_path and total_step_count % self.checkpoint_frequency == 0:
                    self.logger_df.to_csv(self.logfile_path)

    def _compute_overlap_matrices(self, step_count: int) -> None:

        for layer in range(len(self.student_hidden)):
            self._compute_layer_overlaps(layer=str(layer), step_count=step_count, head=None)

        self._compute_layer_overlaps(layer="output", step_count=step_count, head=0)
        self._compute_layer_overlaps(layer="output", step_count=step_count, head=1)

    def _compute_layer_overlaps(self, layer: str, step_count: int, head: int):

        # extract layer weights
        if head is None:
            student_layer = self.student_network.state_dict()['layers.{}.weight'.format(layer)].data
            teacher_layers = [teacher.state_dict()['layers.{}.weight'.format(layer)].data for teacher in self.teachers]
        else:
            student_layer = self.student_network.state_dict()['heads.{}.weight'.format(str(head))].data
            teacher_layers = [teacher.state_dict()['output_layer.weight'].data for teacher in self.teachers]
            layer = layer + "_head_{}".format(str(head))

        # compute overlap matrices
        student_self_overlap = (student_layer.mm(student_layer.t()) / self.input_dimension).cpu().numpy()
        if head is None:
            student_teacher_overlaps = [(student_layer.mm(teacher_layer.t()) / self.input_dimension).cpu().numpy() for teacher_layer in teacher_layers]
        else:
            student_teacher_overlaps = [(student_layer.t().mm(teacher_layer) / self.input_dimension).cpu().numpy() for teacher_layer in teacher_layers]
        teacher_self_overlaps = [(teacher_layer.mm(teacher_layer.t()) / self.input_dimension).cpu().numpy() for teacher_layer in teacher_layers]
        teacher_pairs = list(itertools.combinations(range(len(teacher_layers)), 2))
        teacher_teacher_overlaps = {(i, j): (teacher_layers[i].mm(teacher_layers[j].t()) / self.input_dimension).cpu().numpy() for (i, j) in teacher_pairs}

        # log overlap values (scalars vs image graphs below)
        def log_matrix_values(log_name: str, matrix):
            matrix_shape = matrix.shape
            for i in range(matrix_shape[0]):
                for j in range(matrix_shape[1]):
                    self.writer.add_scalar("layer_{}_{}/values_{}_{}".format(layer, log_name, i, j), matrix[i][j], step_count)
                    self.logger_df.at[step_count, "layer_{}_{}/values_{}_{}".format(layer, log_name, i, j)] = matrix[i][j]
        
        log_matrix_values("student_self_overlap", student_self_overlap)
        for s, student_teacher_overlap in enumerate(student_teacher_overlaps):
            log_matrix_values("student_teacher_overlaps/{}".format(s), student_teacher_overlap)
        for s, teacher_self_overlap in enumerate(teacher_self_overlaps):
            log_matrix_values("teacher_self_overlaps/{}".format(s), teacher_self_overlap)
        for (i, j) in teacher_teacher_overlaps:
            log_matrix_values("teacher_teacher_overlaps/{}_{}".format(i, j), teacher_teacher_overlaps[(i, j)])

        # generate visualisations
        student_self_fig = visualise_matrix(student_self_overlap, fig_title=r"$Q_{ik}^\mu$")
        teacher_cross_figs = {(i, j):
            visualise_matrix(teacher_teacher_overlaps[(i, j)], fig_title=r"$T_{nm}$") \
            for (i, j) in teacher_teacher_overlaps
        }
        teacher_self_figs = [
            visualise_matrix(teacher_self_overlap, fig_title=r"$T_{nm}$") \
            for teacher_self_overlap in teacher_self_overlaps
        ]
        student_teacher_figs = [
            visualise_matrix(student_teacher_overlap, fig_title=r"$R_{in}^\mu$") \
            for t, student_teacher_overlap in enumerate(student_teacher_overlaps)
        ]

        # log visualisations
        self.writer.add_figure("layer_{}_student_self_overlap".format(str(layer)), student_self_fig, step_count)
        for t, student_teacher_fig in enumerate(student_teacher_figs):
            self.writer.add_figure("layer_{}_student_teacher_overlaps/teacher_{}".format(layer, t), student_teacher_fig, step_count)
        for t, teacher_self_fig in enumerate(teacher_self_figs):
            self.writer.add_figure("layer_{}_teacher_self_overlaps/teacher_{}".format(layer, t), teacher_self_fig, step_count)
        for (i, j), teacher_cross_fig in list(teacher_cross_figs.items()):
            self.writer.add_figure("layer_{}_teacher_cross_overlaps/teacher{}x{}".format(layer, i, j), teacher_cross_fig, step_count)
        
    def _compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.5 * self.loss_function(prediction, target)
        return loss


class MNIST(Framework):

    def __init__(self, config: Dict) -> None:
        Framework.__init__(self, config)

    def train(self) -> None:

        training_losses = []
        total_step_count = 0
        steps_per_task = []

        while total_step_count < self.total_training_steps:
            
            teacher_index = next(self.curriculum)
            task_step_count = 0
            latest_task_generalisation_error = np.inf
            
            # alert learner/teacher(s) of task change e.g. change output head of student to relevant task (if in continual setting)
            self._signal_task_boundary_to_learner(new_task=teacher_index)
            self._signal_task_boundary_to_teacher(new_task=teacher_index)

            while True:

                training_batch = []
                
                for _ in range(self.train_batch_size):
                    if len(self.teachers[teacher_index]) == 0:
                        task_data = get_binary_classification_datasets(self.mnist_x_data, self.mnist_y_data, self.mnist_teacher_classes[teacher_index])
                        self.teachers[teacher_index] = task_data
                    training_data_point = self.teachers[teacher_index].pop()
                    training_batch.append(training_data_point)

                image_input = torch.stack([item[0] for item in training_batch]).to(self.device)

                teacher_output = torch.flatten(torch.stack([item[1] for item in training_batch]).to(self.device))
                student_output = self.student_network(image_input)  

                if self.verbose and task_step_count % 1000 == 0:
                    print("Training step {}".format(task_step_count))

                # training iteration
                self.optimiser.zero_grad()
                loss = self._compute_loss(student_output, teacher_output)
                loss.backward()
                self.optimiser.step()
                training_losses.append(float(loss))

                # log training loss
                self.writer.add_scalar('training_loss', float(loss), total_step_count)
                self.logger_df.at[total_step_count, 'training_loss'] = float(loss)

                # test
                if total_step_count % self.test_frequency == 0 and total_step_count != 0:
                    latest_task_generalisation_error = self._perform_test_loop(teacher_index, task_step_count, total_step_count)

                total_step_count += 1
                task_step_count += 1

                # alert learner/teacher(s) of step e.g. to drift teacher
                self._signal_step_boundary_to_learner(step=task_step_count, current_task=teacher_index)
                self._signal_step_boundary_to_teacher(step=task_step_count, current_task=teacher_index)

                if self._switch_task(step=task_step_count, generalisation_error=latest_task_generalisation_error):
                    self.writer.add_scalar('steps_per_task', task_step_count, total_step_count)
                    self.logger_df.at[total_step_count, 'steps_per_task'] = task_step_count
                    steps_per_task.append(task_step_count)
                    break

    def _compute_overlap_matrices(self, step_count: int) -> None:
        pass

    def _compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss
        """
        loss = self.loss_function(prediction, target)
        return loss

    def _compute_classification_acc(self, prediction: torch.Tensor, target: torch.Tensor):
        class_predictions = torch.argmax(prediction, axis=1)
        accuracy = int(sum(class_predictions == target)) / len(target)
        return accuracy

        

        

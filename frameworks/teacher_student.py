import numpy as np 
import copy
import itertools

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

from tensorboardX import SummaryWriter

from typing import List, Tuple, Generator, Dict

from models import Teacher, MetaStudent, ContinualStudent
from utils import visualise_matrix

class StudentTeacher:

    def __init__(self, config: Dict) -> None:
        """
        Class for orchestrating student teacher framework including training and test loops

        :param config: dictionary containing parameters to specify training etc.
        """
        # extract relevant parameters from config
        self._extract_parameters(config)

        # setup teacher(s) and students according to config
        self._setup_teacher_student_framework(config=config)

        # extract curriculum from config
        self._set_curriculum(config)
        
        trainable_parameters = filter(lambda param: param.requires_grad, self.student_network.parameters())

        # initialise optimiser with trainable parameters of student        
        self.optimiser = optim.SGD(trainable_parameters, lr=self.learning_rate)
        
        # initialise loss function
        if config.get(["training", "loss_function"]) == 'mse':
            self.loss_function = nn.MSELoss()
        else:
            raise NotImplementedError("{} is not currently supported, please use mse loss".format(config.get("loss")))

        # generate fixed test data
        self.test_input_data = torch.randn(self.test_batch_size, self.input_dimension).to(self.device)
        self.test_teacher_outputs = [teacher(self.test_input_data) for teacher in self.teachers]

        # initialise general tensorboard writer
        self.writer = SummaryWriter(self.checkpoint_path)
        # initialise separate tb writers for each teacher
        self.teacher_writers = [SummaryWriter("{}/teacher_{}".format(self.checkpoint_path, t)) for t in range(self.num_teachers)]

        # initialise objects containing metrics of interest
        self._initialise_metrics()

    def _extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        """

        self.verbose = config.get("verbose")
        self.checkpoint_path = config.get("checkpoint_path")
        self.device = config.get("device")

        self.total_training_steps = config.get(["training", "total_training_steps"])

        self.input_dimension = config.get(["model", "input_dimension"])
        self.output_dimension = config.get(["model", "output_dimension"])
        self.nonlinearity = config.get(["model", "nonlinearity"])
        self.soft_committee = config.get(["model", "soft_committee"])

        self.train_batch_size = config.get(["training", "train_batch_size"])
        self.test_batch_size = config.get(["training", "test_batch_size"])
        self.learning_rate = config.get(["training", "learning_rate"])

        self.test_all_teachers = config.get(["testing", "test_all_teachers"])
        self.test_frequency = config.get(["testing", "test_frequency"])
        self.overlap_frequency = config.get(["testing", "overlap_frequency"])

        self.num_teachers = config.get(["task", "num_teachers"])
        self.label_task_bounaries = config.get(["task", "label_task_boundaries"])
        self.task_setting = config.get(["task", "task_setting"])
        self.teacher_initialisation = config.get(["task", "teacher_initialisation"])

    def _setup_teacher_student_framework(self, config: Dict):
        # initialise student network
        self.student_network = MetaStudent(config=config).to(self.device)

        # initialise teacher networks, freeze
        self.teachers = []
        if self.teacher_initialisation == 'independent':
            for _ in range(self.num_teachers):
                teacher = Teacher(config=config).to(self.device)
                teacher.freeze_weights()
                teacher.set_noise_distribution(None, None)
                self.teachers.append(teacher)
        elif self.teacher_initialisation == 'underlying_task':
            base_teacher = Teacher(config=config).to(self.device)
            base_teacher.freeze_weights()
            for _ in range(self.num_teachers):
                teacher = copy.deepcopy(base_teacher)
                teacher.set_noise_distribution(mean=0, std=1)
                self.teachers.append(teacher)

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

    def train(self) -> None:
        """
        Training loop
        """
        training_losses = []
        total_step_count = 0
        steps_per_task = []

        while total_step_count < self.total_training_steps:
            
            teacher_index = next(self.curriculum)
            task_step_count = 0
            latest_task_generalisation_error = np.inf

            # change output head of student to relevant task
            self.student_network.set_task(teacher_index)

            while True:

                random_input = torch.randn(self.train_batch_size, self.input_dimension).to(self.device)

                # if self.label_tasks:
                #     import pdb; pdb.set_trace()
                #     task_label = torch.zeros(self.train_batch_size, len(self.teachers))
                #     task_label[:, teacher_index] = 1.
                #     random_input = torch.concat(random_input, task_label)
    
                teacher_output = self.teachers[teacher_index](random_input)
                student_output = self.student_network(random_input)

                if self.verbose and task_step_count % 1000 == 0:
                    print("Training step {}".format(task_step_count))

                # training iteration
                self.optimiser.zero_grad()
                loss = self._compute_loss(student_output, teacher_output)
                training_losses.append(float(loss))
                loss.backward()
                self.optimiser.step()

                # log training loss
                self.writer.add_scalar('training_loss', float(loss), total_step_count)

                # test
                if total_step_count % self.test_frequency == 0 and total_step_count != 0:
                    latest_task_generalisation_error = self._perform_test_loop(teacher_index, task_step_count, total_step_count)

                # overlap matrices
                if total_step_count % self.overlap_frequency == 0 and total_step_count != 0:
                    self._compute_overlap_matrices(step_count=total_step_count)

                total_step_count += 1
                task_step_count += 1

                if self._switch_task(step=task_step_count, generalisation_error=latest_task_generalisation_error):
                    self.writer.add_scalar('steps_per_task', task_step_count, total_step_count)
                    steps_per_task.append(task_step_count)
                    break

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
            raise ValueError("curriculum stopping condition {} unknown".format(self.curriculum_stopping_condition))

    def _perform_test_loop(self, teacher_index: int, task_step_count: int, total_step_count: int) -> float:
        """
        Test loop. Evaluated generalisation error of student wrt teacher(s)

        :param teacher_index: index of current teacher student is being trained on
        :param task_step_count: number of steps completed under current teacher
        :param total_step_count: total number of steps student has beein training for

        :return generalisation_error_wrt_current_teacher: generalisation_error for current teacher 
        """
        with torch.no_grad():

            if self.test_all_teachers:
                teachers_to_test = list(range(len(self.teachers)))
            else:
                teachers_to_test = [teacher_index]

            generalisation_error_per_teacher = self._compute_generalisation_errors(teachers_to_test)

            # log average generalisation losses over teachers
            self.writer.add_scalar(
                'mean_generalisation_error/linear', np.mean(generalisation_error_per_teacher), total_step_count
                )
            self.writer.add_scalar(
                'mean_generalisation_error/log', np.log10(np.mean(generalisation_error_per_teacher)), total_step_count
                )

            for i, error in enumerate(generalisation_error_per_teacher):
                self.generalisation_errors[i].append(error)
                # log generalisation loss per teacher
                self.teacher_writers[i].add_scalar('generalisation_error/linear', error, total_step_count)
                self.teacher_writers[i].add_scalar('generalisation_error/log', np.log10(error), total_step_count)

            if self.verbose and task_step_count % 500 == 0:
                print("Generalisation errors @ step {} ({}'th step training on teacher {}):".format(total_step_count, task_step_count, teacher_index))
                for i, error in enumerate(generalisation_error_per_teacher):
                    print(
                        "{}Teacher {}: {}".format("".rjust(10), i, error),
                        sep="\n"
                    )

            generalisation_error_wrt_current_teacher = generalisation_error_per_teacher[teacher_index]

            return generalisation_error_wrt_current_teacher

    def _compute_overlap_matrices(self, step_count: int) -> None:
        """
        calculated overlap matrices (order parameters) of student wrt itself, teacher wrt itself and student wrt teacher
        produces figures and logs to tb

        :param step_count: number of steps (overall) taken by student in training so far
        """
        # extract layer weights
        student_layer = self.student_network.state_dict()['layers.0.weight'].data
        teacher_layers = [teacher.state_dict()['layers.0.weight'].data for teacher in self.teachers]

        # compute overlap matrices
        student_self_overlap = student_layer.mm(student_layer.t()) / self.input_dimension
        student_teacher_overlaps = [student_layer.mm(teacher_layer.t()) / self.input_dimension for teacher_layer in teacher_layers]
        teacher_self_overlaps = [teacher_layer.mm(teacher_layer.t()) / self.input_dimension for teacher_layer in teacher_layers]

        # generate visualisations
        student_self_fig = visualise_matrix(student_self_overlap.cpu().numpy(), fig_title=r"$Q_{ik}^\mu$")
        student_teacher_figs = [
            visualise_matrix(student_teacher_overlap.cpu().numpy(), fig_title=r"$R_{in}^\mu$") \
            for t, student_teacher_overlap in enumerate(student_teacher_overlaps)
        ]

        # log visualisations
        self.writer.add_figure("student_self_overlap", student_self_fig, step_count)
        for t, student_teacher_fig in enumerate(student_teacher_figs):
            self.writer.add_figure("student_teacher_overlaps/teacher_{}".format(t), student_teacher_fig, step_count)


    def _compute_generalisation_errors(self, teacher_indices: List[int]) -> List[float]:
        """
        calculated generalisation errors wrt fixed test dataset of student against teacher indices given

        :param teacher_indices: teachers against which to test student 
        :return generalisation_errors: list of generalisation errors of student against all teachers specified
        """
        with torch.no_grad():
            student_outputs = self.student_network.test_all_tasks(self.test_input_data)
            if self.test_all_teachers:
                if self.task_setting == 'continual':
                    generalisation_errors = [float(self._compute_loss(student_output, teacher_output)) for student_output, teacher_output in zip(student_outputs, self.test_teacher_outputs)]
                elif self.task_setting == 'meta':
                    generalisation_errors = [float(self._compute_loss(student_outputs, teacher_output)) for teacher_output in self.test_teacher_outputs]
            else:
                if self.task_setting == 'continual':
                    generalisation_errors = [float(self._compute_loss(student_outputs[teacher_index], self.test_teacher_outputs[teacher_index])) for teacher_index in teacher_indices]
                elif self.task_setting == 'meta':
                    generalisation_errors = [float(self._compute_loss(student_outputs, self.test_teacher_outputs[teacher_index])) for teacher_index in teacher_indices]
            return generalisation_errors

    def _compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates loss of prediction of student vs. target from teacher
        Loss function determined by configuration

        :param prediction: prediction made by student network on given input
        :param target: target - teacher output on same input

        :return loss: loss between target (from teacher) and prediction (from student)
        """
        loss = 0.5 * self.loss_function(prediction, target)
        return loss


        

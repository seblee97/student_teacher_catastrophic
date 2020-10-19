import itertools

import constants
from run import student_teacher_config
from students import base_student
from students import base_teachers
from students import continual_student
from students import meta_student
from students import overlapping_teachers
from utils import custom_functions


class NetworkRunner:
    """Runner for network simulations.

    Class for orchestrating student teacher framework including training
    and test loops.
    """

    def __init__(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> None:
        """
        Class constructor.

        Args:
            config: configuration object containing parameters to specify run.
        """
        # initialise student, teachers, logger_module,
        # data_module and loss_module
        self._student = self._setup_student(config=config)
        self._teachers = self._setup_teachers(config=config)
        self._logger = self._setup_logger(config=config)
        self._data_module = self._setup_data(config=config)
        self._loss_module = self._setup_loss(config=config)

        self._set_curriculum(config=config)
        self._setup_optimiser()

    def get_network_configuration(self):
        pass

    @custom_functions.timer
    def _setup_student(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> base_student.BaseStudent:
        if config.learner_configuration == constants.Constants.CONTINUAL:
            student_class = continual_student.ContinualStudent
        elif config.learner_configuration == constants.Constants.META:
            student_class = meta_student.MetaStudent
        else:
            raise ValueError(
                f"Learner type '{config.learning_configuration}' not recognised"
            )

        return student_class(
            input_dimension=config.input_dimension,
            hidden_dimensions=config.student_hidden_layers,
            output_dimension=config.output_dimension,
            bias=config.student_bias_parameters,
            soft_committee=config.soft_committee,
            num_teachers=config.num_teachers,
            loss_type=config.loss_type,
            learning_rate=config.learning_rate,
            scale_head_lr=config.scale_head_lr,
            scale_hidden_lr=config.scale_hidden_lr,
            nonlinearity=config.student_nonlinearity,
            initialise_outputs=config.initialise_student_outputs,
            symmetric_initialisation=config.symmetric_student_initialisation,
            initialisation_std=config.student_initialisation_std,
        )

    @custom_functions.timer
    def _setup_teachers(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> base_teachers.BaseTeachers:
        if config.teacher_configuration == constants.Constants.OVERLAPPING:
            teachers_class = overlapping_teachers.OverlappingTeachers
        else:
            raise ValueError(
                f"Teacher configuration '{config.teacher_configuration}' not recognised."
            )
        return teachers_class

    @custom_functions.timer
    def _setup_logger(self, config: student_teacher_config.StudentTeacherConfiguration):
        pass

    @custom_functions.timer
    def _setup_data(self, config: student_teacher_config.StudentTeacherConfiguration):
        pass

    @custom_functions.timer
    def _setup_loss(self, config: student_teacher_config.StudentTeacherConfiguration):
        pass

    def _set_curriculum(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ):
        """Establish and setup curriculum (when to switch teacher,
        how to decide subsequent teacher etc.) according to configuration."""

        # order of teachers
        self._curriculum = itertools.cycle(list(range(config.num_teachers)))

        if config.stopping_condition == constants.Constants.FIXED_PERIOD:
            self._curriculum_period = config.fixed_period
        elif config.stopping_condition == constants.Constants.THRESHOLD:
            loss_threshold_sequence = config.loss_thresholds
            # make n copies of threshold sequence
            # e.g. [thresh1, thresh2] -> [thresh1, thresh1, thresh2, thresh2]
            self._curriculum_loss_threshold = iter(
                [
                    threshold
                    for threshold in loss_threshold_sequence
                    for _ in range(config.num_teachers)
                ]
            )
            self._current_loss_threshold = next(self._curriculum_loss_threshold)

    def _setup_optimiser(self):
        raise NotImplementedError

    def train(self):
        print("TRAINING")

import torch

from constants import Constants
from curricula import base_curriculum
from curricula import periodic_curriculum
from curricula import threshold_curriculum
from run import student_teacher_config
from students import base_student
from students import continual_student
from students import meta_student
from teachers.ensembles import base_teacher_ensemble
from teachers.ensembles import feature_rotation_ensemble
from teachers.ensembles import readout_rotation_ensemble
from utils import decorators


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
        # data_module, loss_module, torch optimiser, and curriculum object
        self._student = self._setup_student(config=config)
        self._teachers = self._setup_teachers(config=config)
        self._logger = self._setup_logger(config=config)
        self._data_module = self._setup_data(config=config)
        self._loss_module = self._setup_loss(config=config)
        self._optimiser = self._setup_optimiser(config=config)
        self._curriculum = self._setup_curriculum(config=config)

    def get_network_configuration(self):
        """Get macroscopic configuration of networks in terms of order parameters.

        Used for both logging purposes and as input to ODE runner.
        """
        pass

    @decorators.timer
    def _setup_student(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> base_student.BaseStudent:
        """Initialise object containing student network."""
        if config.learner_configuration == Constants.CONTINUAL:
            student_class = continual_student.ContinualStudent
        elif config.learner_configuration == Constants.META:
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

    @decorators.timer
    def _setup_teachers(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> base_teacher_ensemble.BaseTeacherEnsemble:
        """Initialise teacher object containing teacher networks."""
        base_arguments = {
            Constants.INPUT_DIMENSION: config.input_dimension,
            Constants.HIDDEN_DIMENSIONS: config.teacher_hidden_layers,
            Constants.OUTPUT_DIMENSION: config.output_dimension,
            Constants.BIAS: config.teacher_bias_parameters,
            Constants.NUM_TEACHERS: config.num_teachers,
            Constants.LOSS_TYPE: config.loss_type,
            Constants.NONLINEARITY: config.student_nonlinearity,
            Constants.UNIT_NORM_TEACHER_HEAD: config.unit_norm_teacher_head,
            Constants.INITIALISATION_STD: config.teacher_initialisation_std,
        }
        if config.teacher_configuration == Constants.FEATURE_ROTATION:
            teachers_class = feature_rotation_ensemble.FeatureRotationTeacherEnsemble
            additional_arguments = {
                Constants.ROTATION_MAGNITUDE: config.feature_rotation_magnitude
            }
        elif config.teacher_configuration == Constants.READOUT_ROTATION:
            teachers_class = readout_rotation_ensemble.ReadoutRotationTeacherEnsemble
            additional_arguments = {
                Constants.ROTATION_MAGNITUDE: config.readout_rotation_magnitude,
                Constants.FEATURE_COPY_PERCENTAGE: config.feature_copy_percentage,
            }
        else:
            raise ValueError(
                f"Teacher configuration '{config.teacher_configuration}' not recognised."
            )
        return teachers_class(**base_arguments, **additional_arguments)

    @decorators.timer
    def _setup_logger(self, config: student_teacher_config.StudentTeacherConfiguration):
        pass

    @decorators.timer
    def _setup_data(self, config: student_teacher_config.StudentTeacherConfiguration):
        pass

    @decorators.timer
    def _setup_loss(self, config: student_teacher_config.StudentTeacherConfiguration):
        pass

    @decorators.timer
    def _setup_curriculum(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> base_curriculum.BaseCurriculum:
        """Initialise curriculum object (when to switch teacher,
        how to decide subsequent teacher etc.)

        Raises:
            ValueError: if stopping condition is not recognised.
        """
        if config.stopping_condition == Constants.FIXED_PERIOD:
            curriculum = periodic_curriculum.PeriodicCurriculum(config=config)
        elif config.stopping_condition == Constants.LOSS_THRESHOLDS:
            curriculum = threshold_curriculum.ThresholdCurriculum(config=config)
        else:
            raise ValueError(
                f"Stopping condition {config.stopping_condition} not recognised."
            )
        return curriculum

    @decorators.timer
    def _setup_optimiser(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> torch.optim.SGD:
        """Initialise optimiser with trainable parameters of student."""
        trainable_parameters = self._student.get_trainable_parameters()
        return torch.optim.SGD(trainable_parameters, lr=config.learning_rate)

    def train(self):
        print("TRAINING")

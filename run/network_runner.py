import constants
from run import student_teacher_config
from utils import custom_functions
from students import continual_student
from students import meta_student


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
        self.student = self._setup_student(config=config)
        self.teachers = self._setup_teachers(config=config)
        self.logger = self._setup_logger(config=config)
        self.data_module = self._setup_data(config=config)
        self.loss_module = self._setup_loss(config=config)

        # self._set_curriculum(config=config)
        # self._setup_optimiser()

    def get_network_configuration(self):
        pass

    @custom_functions.timer
    def _setup_student(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> None:
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
            initialise_student_outputs=config.initialise_student_outputs,
            symmetric_initialisation=config.symmetric_student_initialisation,
            initialisation_std=config.student_initialisation_std,
        )

    @custom_functions.timer
    def _setup_teachers(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ):
        pass

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
        raise NotImplementedError

    def _setup_optimiser(self):
        raise NotImplementedError

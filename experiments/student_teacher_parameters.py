from typing import Dict

from utils import Parameters
from utils import _Template


class StudentTeacherParameters(Parameters):

    def __init__(self, params: Dict, root_config_template: _Template,
                 iid_data_config_template: _Template, mnist_data_config_template: _Template,
                 pure_mnist_config_template: _Template, trained_mnist_config_template: _Template):
        Parameters.__init__(self, params)

        self.root_config_template = root_config_template
        self.iid_data_config_template = iid_data_config_template
        self.mnist_data_config_template = mnist_data_config_template
        self.pure_mnist_config_template = pure_mnist_config_template
        self.trained_mnist_config_template = trained_mnist_config_template

        self._check_configs()
        self._ensure_consistent_config()

    def _check_configs(self):
        self.check_template(self.root_config_template)

        if self.get(["task", "teacher_configuration"]) == "trained_mnist":
            self.check_template(self.trained_mnist_config_template)

        input_source = self.get(["data", "input_source"])

        if input_source == "iid_gaussian":
            self.check_template(self.iid_data_config_template)
        elif input_source == "mnist_stream":
            self.check_template(self.mnist_data_config_template)
        elif input_source == "even_greater" or input_source == "mnist_digits":
            self.check_template(self.pure_mnist_config_template)

    def _ensure_consistent_config(self):

        self._consistent_loss()
        self._consistent_num_teachers()
        self._consistent_layers()
        self._consistent_input_dimension()
        self._consistent_same_input()
        self._consistent_curriculum()
        self._consistent_lr_scaling()

    def _consistent_loss(self):
        # loss function
        loss_type = self.get(["task", "loss_type"])
        loss_function = self.get(["training", "loss_function"])
        assert (loss_type == "regression" and loss_function in ["mse"] or
                loss_type == "classification" and
                loss_function in ["bce"]), "loss function {} is not compatible with loss \
            type {}".format(loss_function, loss_type)

    def _consistent_num_teachers(self):

        # number of teachers
        num_teachers = self.get(["task", "num_teachers"])
        teacher_nonlinearities = self.get(["model", "teacher_nonlinearities"])
        teacher_noises = self.get(["teachers", "teacher_noise"])
        assert len(
            teacher_nonlinearities
        ) == num_teachers, f"number of teacher nonlinearities provided ({len(teacher_nonlinearities)}) does not match num_teachers specification ({num_teachers})"
        assert len(
            teacher_noises
        ) == num_teachers, f"number of teacher noises provided ({len(teacher_noises)}) does not match num_teachers specification ({num_teachers})"

        curriculum_type = self.get(["curriculum", "type"])
        if curriculum_type == "custom":
            custom_curriculum = self.get(["curriculum", "custom"])
            assert len(
                custom_curriculum
            ) == num_teachers, "Custom curriculum specified is not compatible with number of teachers."

    def _consistent_layers(self):
        teacher_overlaps = self.get(["teachers", "overlap_percentages"])
        teacher_layers = self.get(["model", "teacher_hidden_layers"])
        assert len(teacher_overlaps) == len(
            teacher_layers
        ) + 1, f"number of teacher overlaps provided ({len(teacher_overlaps)}) does not match layers specified for teachers ({teacher_layers})"

    def _consistent_input_dimension(self):
        # mnist input dimension
        if self.get(["data", "input_source"]) == "mnist":
            input_specified = self.get(["model", "input_dimension"])
            pca_output = self.get(["mnist_data", "pca_input"])
            assert (
                input_specified == 784
                or (pca_output > 0 and pca_output == input_specified)), \
                "Input dimension for an MNIST task is not consistent \
                    with image size"

        # trained_mnist_task
        if self.get(["task", "teacher_configuration"]) == "trained_mnist":
            assert self.get(["training", "input_source"]) == "mnist", \
                "Task chosen is trained MNIST but input specified \
                    are not MNIST"

    def _consistent_same_input(self):
        # same input distribution
        same_input_distribution = self.get(["data", "same_input_distribution"])
        teacher_configuration = self.get(["task", "teacher_configuration"])
        input_source = self.get(["data", "input_source"])
        if (teacher_configuration == "pure_mnist" and input_source == "even_greater"):
            assert same_input_distribution, \
                "For even_greater, same_input_distribution should be True"
        elif (teacher_configuration == "pure_mnist" and input_source == "mnist_digits"):
            assert not same_input_distribution, \
                "For mnist_digits, same input_distribution should be False"
        elif teacher_configuration == "trained_mnist":
            assert same_input_distribution
        elif teacher_configuration == "overlapping":
            assert same_input_distribution

    def _consistent_curriculum(self):
        stopping_condition = self.get(["curriculum", "stopping_condition"])
        loss_thresholds = self.get(["curriculum", "loss_threshold"])
        if stopping_condition == "threshold_sequence":
            assert isinstance(loss_thresholds, list)
        elif stopping_condition == "single_threshold":
            assert isinstance(loss_thresholds, float)

    def _consistent_lr_scaling(self):
        scale_hidden_lr_forward = self.get(["training", "scale_hidden_lr_forward"])
        scale_hidden_lr_backward = self.get(["training", "scale_hidden_lr_backward"])
        assert not (
            scale_hidden_lr_forward and scale_hidden_lr_backward
        ), "Hidden layer learning rate should only be scaled on forward or backward pass, not both."

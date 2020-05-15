from utils import Parameters, _Template

from typing import Dict


class StudentTeacherParameters(Parameters):

    def __init__(
            self, params: Dict, root_config_template: _Template,
            mnist_data_config_template: _Template, 
            pure_mnist_config_template: _Template,
            trained_mnist_config_template: _Template
            ):
        Parameters.__init__(self, params)

        self.root_config_template = root_config_template
        self.mnist_data_config_template = mnist_data_config_template
        self.pure_mnist_config_template = pure_mnist_config_template
        self.trained_mnist_config_template = trained_mnist_config_template

        self._check_configs()
        self._ensure_consistent_config()

    def _check_configs(self):
        self.check_template(self.root_config_template)

        if self.get(["task", "teacher_configuration"]) == "trained_mnist":
            self.check_template(self.trained_mnist_config_template)

        if self.get(["data", "input_source"]) == "mnist":
            self.check_template(self.mnist_data_config_template)

    def _ensure_consistent_config(self):

        # loss function
        loss_type = self.get(["task", "loss_type"])
        loss_function = self.get(["training", "loss_function"])
        assert (
            loss_type == "regression" and loss_function in ["mse"] or
            loss_type == "classification" and loss_function in ["bce"]
        ), "loss function {} is not compatible with loss \
            type {}".format(loss_function, loss_type)

        # number of teachers
        num_teachers = self.get(["task", "num_teachers"])
        teacher_nonlinearities = self.get(["model", "teacher_nonlinearities"])
        teacher_overlaps = self.get(["teachers", "overlap_percentages"])
        teacher_noises = self.get(["teachers", "teacher_noise"])
        assert len(teacher_nonlinearities) == num_teachers, \
            "number of teacher nonlinearities provided ({}) does not match \
                num_teachers specification ({})".format(
                    len(teacher_nonlinearities), num_teachers
                    )
        assert len(teacher_overlaps) == num_teachers, \
            "number of teacher overlaps provided ({}) does not match \
                num_teachers specification ({})".format(
                    len(teacher_overlaps), num_teachers
                    )
        assert len(teacher_noises) == num_teachers, \
            "number of teacher noises provided ({}) does not match \
                num_teachers specification ({})".format(
                    len(teacher_noises), num_teachers
                    )

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

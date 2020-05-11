from context import utils

class StudentTeacherParameters(utils.Parameters):

    def __init__(self, params, root_config_template, mnist_config_template, trained_mnist_config_template):
        utils.Parameters.__init__(self, params)

        self.root_config_template = root_config_template
        self.mnist_config_template = mnist_config_template
        self.trained_mnist_config_template = trained_mnist_config_template

        self._check_configs()
        self._ensure_consistent_config()

    def _check_configs(self):
        self.check_template(self.root_config_template)

        if self._config["task"]["teacher_configuration"] == "trained_mnist":
            self.check_template(self.trained_mnist_config_template)

        if self._config["training"]["input_source"] == "mnist":
            self.check_template(self.mnist_config_template)

    def _ensure_consistent_config(self):
        
        # loss function
        loss_type = self._config["task"]["loss_type"]
        loss_function = self._config["training"]["loss_function"]
        assert (
            loss_type == "regression" and loss_function in ["mse"] or 
            loss_type == "classification" and loss_function in ["bce"]
        ), "loss function {} is not compatible with loss type {}".format(loss_function, loss_type)

        # number of teachers
        num_teachers = self._config["task"]["num_teachers"]
        teacher_nonlinearities = self._config["model"]["teacher_nonlinearities"]
        teacher_overlaps = self._config["teachers"]["overlap_percentages"]
        teacher_noises = self._config["teachers"]["teacher_noise"]
        assert len(teacher_nonlinearities) == num_teachers, \
        "number of teacher nonlinearities provided ({}) does not match num_teachers specification ({})".format(len(teacher_nonlinearities), num_teachers)
        assert len(teacher_overlaps) == num_teachers, \
        "number of teacher overlaps provided ({}) does not match num_teachers specification ({})".format(len(teacher_overlaps), num_teachers)
        assert len(teacher_noises) == num_teachers, \
        "number of teacher noises provided ({}) does not match num_teachers specification ({})".format(len(teacher_noises), num_teachers)
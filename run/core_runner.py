import os

import constants
from run import network_runner
from run import ode_runner
from run import student_teacher_config
from utils import plotter


class CoreRunner:
    """Core student-multi-teacher runner."""

    def __init__(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> None:
        """Class for orchestrating student teacher framework run.

        Initialise specific ODE simulation and network simulation runners.

        Args:
            config: configuration object specifying experiment setup.
        """
        self._config = config
        self._config.save_configuration(folder_path=self._config.checkpoint_path)

        self._setup_runners()

    def _setup_runners(self):
        # always generate initial configuration from networks,
        # even if only ODE simulations are performed.
        # This is a. easier and b. ensures correctness
        # e.g. in terms of positive semi-definiteness of covariance matrices.
        self._network_simulation_runner = network_runner.NetworkRunner(
            config=self._config
        )
        network_configuration = (
            self._network_simulation_runner.get_network_configuration()
        )
        if self._config.ode_simulation:
            self._ode_simulation_runner = ode_runner.ODERunner(
                config=self._config, network_configuration=network_configuration
            )

    def run(self):
        if self._config.network_simulation:
            self._network_simulation_runner.train()
        if self._config.ode_simulation:
            self._ode_simulation_runner.run()

    def post_process(self):
        """Method to make plots from data collected by loggers."""
        if self._config.ode_simulation:
            ode_log_path = os.path.join(
                self._config.checkpoint_path, constants.Constants.ODE_CSV
            )
        else:
            ode_log_path = None
        if self._config.network_simulation:
            network_log_path = os.path.join(
                self._config.checkpoint_path, constants.Constants.NETWORK_CSV
            )
        else:
            network_log_path = None

        plotter.Plotter(
            ode_logger_path=ode_log_path,
            network_logger_path=network_log_path,
            save_folder=self._config.checkpoint_path,
            num_steps=self._config.total_training_steps,
        ).make_plots()
        # if self._config.network_simulation and self._config.ode_simulation:
        #     self._overlay_plot()

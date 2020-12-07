import os

import constants
from plotters import split_plotter
from plotters import unified_plotter
from run import network_runner
from run import ode_runner
from run import student_teacher_config


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
        if self._config.ode_simulation:
            network_configuration = (
                self._network_simulation_runner.get_network_configuration()
            )
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
        if self._config.split_logging:
            plotter_class = split_plotter.SplitPlotter
        else:
            plotter_class = unified_plotter.UnifiedPlotter

        plotter = plotter_class(
            save_folder=self._config.checkpoint_path,
            num_steps=self._config.total_training_steps,
            log_overlaps=self._config.log_overlaps,
            log_ode=self._config.ode_simulation,
            log_network=self._config.network_simulation,
        )
        plotter.make_plots()
        # if self._config.network_simulation and self._config.ode_simulation:
        #     self._overlay_plot()

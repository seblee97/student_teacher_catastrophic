from run import student_teacher_config

from run import network_runner
from run import ode_runner


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
            self._network_simulation_runner.post_process()
        if self._config.ode_simulation:
            self._ode_simulation_runner.run()
            self._ode_simulation_runner.post_process()

    def overlay_plot(self):
        if self._config.network_simulation and self._config.ode_simulation:
            self._overlay_plot()

    def _perform_ode_simulations(self):
        """Initialise and start training of ODE runner."""
        pass

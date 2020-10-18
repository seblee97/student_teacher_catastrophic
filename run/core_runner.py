from run import student_teacher_config

from run import network_runner
from run import ode_runner


class CoreRunner:
    """Core student-multi-teacher runner."""

    def __init__(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> None:
        """Class for orchestrating student teacher framework run.

        Specific ODE simulation and/or network simulation runners are initialised
        by this runner depending on configuration provided.

        Args:
            config: configuration object specifying experiment setup.
        """
        self._config = config

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
        self._ode_simulation_runner = ode_runner.ODERunner(config=self._config)

        if self._config.network_simulation:
            self._perform_network_simulations()
        if self._config.ode_simulation:
            self._perform_ode_simulations()

    def _perform_ode_simulations(self):
        """Initialise and start training of ODE runner."""
        pass

    def _perform_network_simulations(self):
        """Initialise and start training of network simulation runner."""
        pass

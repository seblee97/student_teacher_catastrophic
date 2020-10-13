from config_manager import base_configuration


class CoreRunner:
    """Core student-multi-teacher runner."""
    def __init__(self, config: base_configuration.BaseConfiguration) -> None:
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
        self._network_simulation_runner = None
        network_configuration = self._network_simulation_runner.get_network_configuration()

        if self._config.network_simulation:
            self._perform_ode_simulations()
        if self._config.analytic_simulation:
            self._perform_network_simulations()

    def _perform_ode_simulations(self):
        """Initialise and start training of ODE runner.
        """
        raise NotImplementedError

    def _perform_network_simulations(self):
        """Initialise and start training of network simulation runner.
        """
        raise NotImplementedError

from cata.plotters import unified_plotter
from cata.run import student_teacher_config
from cata.runners import network_runner
from cata.runners import ode_runner


class CoreRunner:
    """Core student-multi-teacher runner."""

    def __init__(
        self,
        config: student_teacher_config.StudentTeacherConfig,
        unique_id: str = "",
    ) -> None:
        """Class for orchestrating student teacher framework run.

        Initialise specific ODE simulation and network simulation runners.

        Args:
            config: configuration object specifying experiment setup.
        """
        self._setup_runners(config=config, unique_id=unique_id)
        self._network_run = config.network_simulation
        self._ode_run = config.ode_simulation

        if config.ode_simulation:
            self._ode_log_path = self._ode_simulation_runner.logfile_path
        else:
            self._ode_log_path = None
        if config.network_simulation:
            self._network_log_path = self._network_simulation_runner.logfile_path
        else:
            self._network_log_path = None

        self._log_overlaps = config.log_overlaps
        self._num_steps = config.total_training_steps
        self._save_folder = config.checkpoint_path

    def _setup_runners(
        self, config: student_teacher_config.StudentTeacherConfig, unique_id: str = ""
    ):
        # always generate initial configuration from networks,
        # even if only ODE simulations are performed.
        # This is a. easier and b. ensures correctness
        # e.g. in terms of positive semi-definiteness of covariance matrices.
        if unique_id == "":
            network_id = "network"
            ode_id = "ode"
        else:
            network_id = f"{unique_id}_network"
            ode_id = f"{unique_id}_ode"
        self._network_simulation_runner = network_runner.NetworkRunner(
            config=config, unique_id=network_id
        )
        if config.ode_simulation:
            network_configuration = (
                self._network_simulation_runner.get_network_configuration()
            )
            self._ode_simulation_runner = ode_runner.ODERunner(
                config=config,
                unique_id=ode_id,
                network_configuration=network_configuration,
            )

    def run(self):
        if self._network_run:
            self._network_simulation_runner.train()
        if self._ode_run:
            self._ode_simulation_runner.run()

    def post_process(self) -> None:
        """Solidify any data and make plots."""
        if self._network_run:
            self._network_simulation_runner.post_process()
        if self._ode_run:
            self._ode_simulation_runner.post_process()

        self._plotter = unified_plotter.UnifiedPlotter(
            save_folder=self._save_folder,
            num_steps=self._num_steps,
            log_overlaps=self._log_overlaps,
            ode_log_path=self._ode_log_path,
            network_log_path=self._network_log_path,
        )

        self._plotter.make_plots()

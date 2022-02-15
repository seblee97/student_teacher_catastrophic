import os
import tempfile

from cata import constants
from cata.run import student_teacher_config
from cata.runners import core_runner
from run_modes import single_run
from run_modes import utils


MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
RUNNER_TEST_DIR = os.path.dirname(os.path.realpath(__file__))

TEST_CONFIG_PATH = os.path.join(MAIN_FILE_PATH, "base_test_config.yaml")


results_folder = tempfile.mkdtemp()


class TestRunners:
    """Integration test for runners. Test configs with 10000 steps."""

    def _test_runner(self, changes):

        config_class = student_teacher_config.StudentTeacherConfig
        runner_class = core_runner.CoreRunner

        _, single_checkpoint_path = utils.setup_experiment(
            mode="single", results_folder=results_folder, config_path=TEST_CONFIG_PATH
        )

        single_run.single_run(
            runner_class=runner_class,
            config_class=config_class,
            config_path=TEST_CONFIG_PATH,
            checkpoint_path=single_checkpoint_path,
            run_methods=["run", "post_process"],
            changes=changes,
        )

    def test_base(self):
        self._test_runner(changes=[])

    def test_ode(self):
        for timestep in [0.1, 0.01, 0.001, 0.0001]:
            self._test_runner(
                changes=[
                    {"ode_simulation": True, "network_simulation": False},
                    {"ode_run": {"timestep": timestep}},
                ]
            )

    def test_ode_network(self):
        self._test_runner(changes=[{"ode_simulation": True}])

    def test_meta(self):
        self._test_runner(changes=[{"task": {"learner_configuration": "meta"}}])

    def test_readout_rotation(self):
        self._test_runner(
            changes=[
                {"task": {"teacher_configuration": "readout_rotation"}},
                {"model": {"teachers": {"teacher_hidden_layers": [4]}}},
            ]
        )

    def test_both_rotation(self):
        self._test_runner(
            changes=[
                {"task": {"teacher_configuration": "both_rotation"}},
                {
                    "model": {
                        "input_dimension": 15,
                        "teachers": {"teacher_hidden_layers": [100]},
                        "student": {"student_hidden_layers": [200]},
                    }
                },
                {"logging": {"log_overlaps": False}},
            ]
        )

    def test_classification(self):
        self._test_runner(changes=[{"task": {"loss_type": "classification"}}])

    def test_linear(self):
        self._test_runner(
            changes=[
                {
                    "model": {
                        "student": {"student_nonlinearity": "linear"},
                        "teachers": {"teacher_nonlinearities": ["linear", "linear"]},
                    }
                },
            ]
        )

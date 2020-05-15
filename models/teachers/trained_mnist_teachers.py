from typing import Dict

from .base_teachers import _BaseTeachers

import torch
import os
import hashlib

from utils import train_mnist_classifier, Parameters


class TrainedMNISTTeachers(_BaseTeachers):

    """Teachers are initialised with weights corresponding to
    trained MNIST classifiers.
    """

    def __init__(self, config: Parameters):
        _BaseTeachers.__init__(self, config)

    def _setup_teachers(self, config: Parameters):
        """
        Instantiate all teachers.

        Load weights from specified path
        """
        # get noise configs if applicable
        teacher_noise = config.get(["task", "teacher_noise"])
        if type(teacher_noise) is int:
            teacher_noises = [teacher_noise for _ in range(self._num_teachers)]
        elif type(teacher_noise) is list:
            assert len(teacher_noise) == self._num_teachers, \
                "Provide one noise for each teacher. {} noises given, \
                    {} teachers specified".format(
                        len(teacher_noise), self._num_teachers
                        )
            teacher_noises = teacher_noise

        teacher_specification_components = [
            config.get(["model", "input_dimension"]),
            config.get(["model", "teacher_hidden_layers"]),
            config.get(["model", "teacher_nonlinearities"]),
            config.get(["model", "teacher_bias_parameters"]),
            config.get(["training", "teachers"]),
            config.get(["training", "rotations"]),
            config.get(["trained_mnist", "output_dimension"])
        ]

        # hash contents of above config contents to see if trained teachers
        # of this type exist
        config_hash = hashlib.md5(
            str(teacher_specification_components).encode('utf-8')
            ).hexdigest()

        data_path = config.get(["save_weight_path"])
        os.makedirs(data_path, exist_ok=True)
        saved_weights_path = os.path.join(data_path, config_hash)
        if not os.path.exists(saved_weights_path):

            print(
                "Teachers with given config do not have associated saved \
                weights. Training classifiers..."
                )

            os.makedirs(saved_weights_path)
            for t in range(self._num_teachers):
                mnist_classifier = \
                    train_mnist_classifier.MNISTTrainer(
                        config=config, task_index=t
                        )
                mnist_classifier.train()

                print("Teachers with given config trained. Saving weights...")

                mnist_classifier.save_model_weights(
                    path=os.path.join(
                        saved_weights_path, "teacher_{}.pt".format(t)
                        )
                    )

            print(
                "Teachers with config {} trained and \
                    saved.".format(config_hash)
                )

        self._teachers = []
        for t in range(self._num_teachers):
            teacher = self._init_teacher(config=config, index=t)

            # load teacher weights from save path
            teacher.load_weights(
                weights_path=os.path.join(
                    saved_weights_path, "teacher_{}.pt".format(t)
                    )
                )

            teacher.freeze_weights()
            if teacher_noises[t] != 0:
                teacher_output_std = teacher.get_output_statistics()
                teacher.set_noise_distribution(
                    mean=0, std=teacher_noises[t] * teacher_output_std
                    )
            self._teachers.append(teacher)

    def forward(
        self,
        teacher_index: int,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x = batch['x']
        output = self._teachers[teacher_index](x)
        return output

    def test_set_forward(self, teacher_index, batch) -> torch.Tensor:
        raise NotImplementedError

    def signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def signal_step_boundary_to_teacher(self, step: int, current_task: int):
        pass

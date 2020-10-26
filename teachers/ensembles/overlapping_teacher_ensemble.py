import copy
from typing import Dict
from typing import List

import numpy as np
import torch

from utils import custom_functions
from teachers.ensembles import base_teacher_ensemble


class OverlappingTeacherEnsemble(base_teacher_ensemble.BaseTeacherEnsemble):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: List[int],
        output_dimension: int,
        bias: bool,
        loss_type: str,
        nonlinearity: str,
        unit_norm_teacher_head: bool,
        num_teachers: int,
        initialisation_std: float,
    ):
        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=output_dimension,
            bias=bias,
            loss_type=loss_type,
            nonlinearity=nonlinearity,
            unit_norm_teacher_head=unit_norm_teacher_head,
            num_teachers=num_teachers,
            initialisation_std=initialisation_std,
        )

    def _setup_teachers(self) -> None:
        """Instantiate overlapping teachers.

        Start with 'original' teacher.
        Then instantiate set of teachers by copying specified number
        of weights from original teacher to new teachers.
        Weight duplicates can be in input to hidden,
        hidden to output or both.
        """

        import pdb

        pdb.set_trace()

        original_teacher = self._init_teacher()
        # get noise configs if applicable
        teacher_noise = config.get(["teachers", "teacher_noise"])
        if type(teacher_noise) is int:
            teacher_noises = [teacher_noise for _ in range(self._num_teachers)]
        elif type(teacher_noise) is list:
            assert (
                len(teacher_noise) == self._num_teachers
            ), f"Provide one noise for each teacher. {len(teacher_noise)} noises given, {self._num_teachers} teachers specified"
            teacher_noises = teacher_noise

        original_teacher.freeze_weights()

        hidden_dimensions = config.get(["model", "teacher_hidden_layers"])
        input_dimension = config.get(["model", "input_dimension"])
        output_dimension = config.get(["model", "output_dimension"])

        overlap_types = config.get(["teachers", "overlap_type"])
        overlap_rotations = config.get(["teachers", "overlap_rotations"])
        overlap_copies = config.get(["teachers", "overlap_percentages"])

        unit_norm_teacher_head = config.get(["model", "unit_norm_teacher_head"])

        self._teachers = [
            self._init_teacher(config=config, index=i)
            for i in range(self._num_teachers)
        ]

        # include hidden -> output
        layer_dims = [input_dimension] + hidden_dimensions + [output_dimension]

        for i, (
            overlap_rotation,
            overlap_copy,
            layer_key,
            layer_dim_i,
            layer_dim_j,
        ) in enumerate(
            zip(
                overlap_rotations,
                overlap_copies,
                original_teacher.state_dict().keys(),
                layer_dims[:-1],
                layer_dims[1:],
            )
        ):

            teacher_1_layer = []
            teacher_2_layer = []

            if overlap_types[i] == "rotation":
                assert (
                    self._num_teachers == 2
                ), f"Rotated overlapping teachers only implemented for 2 teachers, {self._num_teachers} requested."
                for _ in range(layer_dim_j):
                    w1, w2 = custom_functions.generate_rotated_vectors(
                        layer_dim_i, overlap_rotation, np.sqrt(layer_dim_i)
                    )
                    teacher_1_layer.append(w1)
                    teacher_2_layer.append(w2)

                with torch.no_grad():
                    teacher_1_layer_tensor = torch.Tensor(np.vstack(teacher_1_layer))
                    teacher_2_layer_tensor = torch.Tensor(np.vstack(teacher_2_layer))
                    if i == len(layer_dims) - 2:
                        # hidden to output.
                        if unit_norm_teacher_head:
                            teacher_1_layer_tensor = (
                                teacher_1_layer_tensor
                                / torch.norm(teacher_1_layer_tensor)
                            )
                            teacher_2_layer_tensor = (
                                teacher_2_layer_tensor
                                / torch.norm(teacher_2_layer_tensor)
                            )

                        self._teachers[
                            0
                        ].output_layer.weight.data = teacher_1_layer_tensor
                        self._teachers[
                            1
                        ].output_layer.weight.data = teacher_2_layer_tensor
                    else:
                        self._teachers[0].layers[i].weight.data = teacher_1_layer_tensor
                        self._teachers[1].layers[i].weight.data = teacher_2_layer_tensor

            elif overlap_types[i] == "copy":
                for t in range(self._num_teachers):
                    layer_shape = self._teachers[t].state_dict()[layer_key].shape
                    assert (
                        len(layer_shape) == 2
                    ), "shape of layer tensor is not 2. \
                        Check consitency of layer construction with task."

                    for row in range(layer_shape[0]):
                        overlapping_dim = round(0.01 * overlap_copy * layer_shape[1])
                        overlapping_weights = copy.deepcopy(
                            original_teacher.state_dict()[layer_key][row][
                                :overlapping_dim
                            ]
                        )
                        self._teachers[t].state_dict()[layer_key][row][
                            :overlapping_dim
                        ] = overlapping_weights

        for t, teacher in enumerate(self._teachers):
            # freeze weights of every teacher
            teacher.freeze_weights()

            # set noise
            if teacher_noises[t] != 0:
                teacher_output_std = teacher.get_output_statistics()
                teacher.set_noise_distribution(
                    mean=0, std=teacher_noises[t] * teacher_output_std
                )

        # if config.get(["teachers", "overlap_type"]) == "rotation":
        #     self._teachers = self._initialise_rotated_teachers(
        #         config=config, teacher_noises=teacher_noises)
        # elif config.get(["teachers", "overlap_type"]) == "copy":
        #     self._teachers = self._initialise_copy_teachers(
        #         config=config, teacher_noises=teacher_noises)

    def _initialise_rotated_teachers(self, teacher_noises: List) -> List:
        assert (
            self._num_teachers == 2
        ), f"Rotated overlapping teachers only implemented for 2 teachers, {self._num_teachers} requested."

        hidden_dimensions = config.get(["model", "teacher_hidden_layers"])
        input_dimension = config.get(["model", "input_dimension"])
        overlap_rotations = config.get(["teachers", "overlap_rotations"])

        teachers = [
            self._init_teacher(config=config, index=i)
            for i in range(self._num_teachers)
        ]

        # exclude hidden -> output
        layer_dims = [input_dimension] + hidden_dimensions

        for i, (overlap, layer_dim_i, layer_dim_j) in enumerate(
            zip(overlap_rotations, layer_dims[:-1], layer_dims[1:])
        ):
            teacher_1_layer = []
            teacher_2_layer = []
            for _ in range(layer_dim_j):
                w1, w2 = custom_functions.generate_rotated_vectors(
                    layer_dim_i, overlap, np.sqrt(layer_dim_i)
                )
                teacher_1_layer.append(w1)
                teacher_2_layer.append(w2)

            with torch.no_grad():
                teachers[0].layers[i].weight.data = torch.Tensor(
                    np.vstack(teacher_1_layer)
                )
                teachers[1].layers[i].weight.data = torch.Tensor(
                    np.vstack(teacher_2_layer)
                )

        for t, teacher in enumerate(teachers):
            # freeze weights of every teacher
            teacher.freeze_weights()

            # set noise
            if teacher_noises[t] != 0:
                teacher_output_std = teacher.get_output_statistics()
                teacher.set_noise_distribution(
                    mean=0, std=teacher_noises[t] * teacher_output_std
                )

        return teachers

    def _initialise_copy_teachers(self, teacher_noises: List) -> List:

        overlap_percentages = config.get(["teachers", "overlap_percentages"])
        teachers = []

        original_teacher = self._init_teacher(config=config, index=0)
        original_teacher.freeze_weights()

        # set noise for 'original' teacher
        if teacher_noises[0] != 0:
            original_teacher_output_std = original_teacher.get_output_statistics()
            original_teacher.set_noise_distribution(
                mean=0, std=teacher_noises[0] * original_teacher_output_std
            )

        teachers.append(original_teacher)

        # setup remaining teachers
        for t in range(self._num_teachers - 1):

            teacher = self._init_teacher(config=config, index=t + 1)

            for layer_index, layer in enumerate(teacher.state_dict()):
                layer_shape = teacher.state_dict()[layer].shape
                assert (
                    len(layer_shape) == 2
                ), "shape of layer tensor is not 2. \
                        Check consitency of layer construction with task."

                for row in range(layer_shape[0]):
                    overlapping_dim = round(
                        0.01 * overlap_percentages[layer_index] * layer_shape[1]
                    )
                    overlapping_weights = copy.deepcopy(
                        original_teacher.state_dict()[layer][row][:overlapping_dim]
                    )
                    teacher.state_dict()[layer][row][
                        :overlapping_dim
                    ] = overlapping_weights

            # freeze weights of every teacher
            teacher.freeze_weights()

            # set noise
            if teacher_noises[t + 1] != 0:
                teacher_output_std = teacher.get_output_statistics()
                teacher.set_noise_distribution(
                    mean=0, std=teacher_noises[t + 1] * teacher_output_std
                )
            teachers.append(teacher)

        return teachers

    def _set_teacher_weights(
        self, teacher, layer: int, row: int, copy_upper_bound: int
    ):
        raise NotImplementedError

    def test_set_forward(self, batch) -> List[torch.Tensor]:
        return [self.forward(t, batch) for t in range(self._num_teachers)]

    def forward(
        self, teacher_index: int, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x = batch["x"]
        output = self._teachers[teacher_index](x)
        return output

    def signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def signal_step_boundary_to_teacher(self, step: int, current_task: int):
        pass

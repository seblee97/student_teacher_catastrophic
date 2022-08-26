from typing import List
from typing import Union

import numpy as np
import torch
from cata.teachers.ensembles import base_teacher_ensemble
from cata.utils import custom_functions


class IdenticalTeacherEnsemble(base_teacher_ensemble.BaseTeacherEnsemble):
    """Teacher ensemble (primarily for mean-field limit regime) in which both feature and
    readout similarities are tuned by rotation. IMPLEMENTED FOR TWO TEACHERS ONLY ATM!!
    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: List[int],
        output_dimension: int,
        bias: bool,
        loss_type: str,
        nonlinearities: str,
        scale_hidden_lr: bool,
        forward_scaling: float,
        unit_norm_teacher_head: bool,
        weight_normalisation: bool,
        noise_stds: Union[int, float],
        num_teachers: int,
        initialisation_std: float,
    ):

        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=output_dimension,
            bias=bias,
            loss_type=loss_type,
            nonlinearities=nonlinearities,
            scale_hidden_lr=scale_hidden_lr,
            forward_scaling=forward_scaling,
            unit_norm_teacher_head=unit_norm_teacher_head,
            weight_normalisation=weight_normalisation,
            noise_stds=noise_stds,
            num_teachers=num_teachers,
            initialisation_std=initialisation_std,
        )

    def _setup_teachers(self) -> None:
        """instantiate teacher network(s) with identical weights"""
        
        assert (
            self._num_teachers
        ) == 2, "Feature rotation teachers currently implemented for 2 teachers only."

        assert (
            len(self._hidden_dimensions) == 1
        ), "Feature rotation teachers currently implemented for 1 hidden layer only."

        # assert (
        #     self._hidden_dimensions[0] == 1
        # ), "Feature rotation teachers implemented for hidden dimension 1 only."

        teachers = [
            self._init_teacher(
                nonlinearity=self._nonlinearities[i],
                noise_std=self._noise_stds[i],
                zero_head=False,
            )
            for i in range(self._num_teachers)
        ]
        with torch.no_grad():
            dimension=self._input_dimension,
            normalisation=np.sqrt(self._input_dimension),
            v = np.random.normal(size=(dimension))
            #v_2 = np.random.normal(size=(dimension))
            normal = normalisation * v / np.linalg.norm(v)
            teacher_tensor = torch.Tensor(normal).reshape(teachers[0].layers[0].weight.data.shape)
            #teacher_1_tensor = torch.Tensor(normal).reshape(teachers[1].layers[0].weight.data.shape)
            for teacher in teachers:
                teacher.layers[0].weight.data = teacher_tensor
            #teachers[1].layers[0].weight.data = teacher_1_tensor
            
            for teacher in teachers:
                teacher.head.weight.data = torch.Tensor([-1]).reshape(teacher.head.weight.data.shape)
            print("-----TEACHER WEIGHT-----")
            #print(teachers[0].head.weight)
            print(teachers[0].head.weight.data)
            print(teachers[1].head.weight.data)

        #print(teachers)
        return teachers



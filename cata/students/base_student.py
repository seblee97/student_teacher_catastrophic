import abc
import copy
import math
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from cata.utils import base_network


class BaseStudent(base_network.BaseNetwork, abc.ABC):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: List[int],
        output_dimension: int,
        bias: bool,
        loss_type: str,
        nonlinearity: str,
        initialise_outputs: bool,
        copy_head_at_switch: bool,
        apply_nonlinearity_on_output: bool,
        soft_committee: bool,
        train_hidden_layers: bool,
        train_head_layer: bool,
        freeze_features: List[int],
        scale_hidden_lr: bool,
        scale_head_lr: bool,
        scale_forward_by_hidden: bool,
        num_teachers: int,
        learning_rate: float,
        symmetric_initialisation: bool = False,
        initialisation_std: Optional[float] = None,
        teacher_features_copy: Optional[torch.Tensor] = None,
    ) -> None:
        self._soft_committee = soft_committee
        self._train_hidden_layers = train_hidden_layers
        self._train_head_layer = train_head_layer
        self._freeze_feature_schedule = iter(freeze_features)
        self._next_freeze_feature_toggle = self._get_next_freeze_feature_toggle()
        self._frozen = False
        self._initialise_outputs = initialise_outputs
        self._copy_head_at_switch = copy_head_at_switch
        self._apply_nonlinearity_on_output = apply_nonlinearity_on_output

        if scale_hidden_lr:
            forward_hidden_scaling = 1 / math.sqrt(input_dimension)
        else:
            forward_hidden_scaling = 1.0
        if scale_head_lr:
            self._head_lr_scaling = 1 / input_dimension
        else:
            self._head_lr_scaling = 1.0

        if scale_forward_by_hidden:
            forward_scaling = 1 / hidden_dimensions[0]
        else:
            forward_scaling = 1.0

        self._num_teachers = num_teachers
        self._learning_rate = learning_rate

        self._num_switches = -1

        # set to 0 by default
        self._current_teacher: int = 0

        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=output_dimension,
            bias=bias,
            loss_type=loss_type,
            nonlinearity=nonlinearity,
            forward_hidden_scaling=forward_hidden_scaling,
            forward_scaling=forward_scaling,
            symmetric_initialisation=symmetric_initialisation,
            initialisation_std=initialisation_std,
        )

        if teacher_features_copy is not None:
            self._manually_initialise_student(teacher_features_copy)

        # for synaptic intelligence
        self._pre_update_shared_parameters = {
            n: copy.deepcopy(param)
            for n, param in self.named_parameters()
            if "head" not in n
        }
        self._path_integral_contributions = []

    @property
    def current_teacher(self) -> int:
        return self._current_teacher

    @property
    def heads(self):
        return self._heads

    @property
    def frozen(self):
        return self._frozen

    @property
    def previous_task_path_integral_contributions(self):
        return self._path_integral_contributions[self._num_switches - 1]

    def save_weights(self, save_path: str) -> None:
        """Save weights of student."""
        torch.save(self.state_dict(), save_path)

    @abc.abstractmethod
    def _construct_output_layers(self):
        """Instantiate the output layer."""
        pass

    @abc.abstractmethod
    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through relevant head of student."""
        pass

    @abc.abstractmethod
    def _signal_task_boundary(self, new_task: int) -> None:
        """Logic for specific students on teacher change."""
        pass

    def signal_task_boundary(self, new_task: int) -> None:
        """Alert student to teacher change."""
        self._num_switches += 1
        self._path_integral_contributions.append(
            {
                n: torch.zeros_like(param)
                for n, param in self.named_parameters()
                if "head" not in n
            }
        )
        self._signal_task_boundary(new_task=new_task)

    def _get_next_freeze_feature_toggle(self) -> Union[int, float]:
        """Get next step index at which to freeze/unfreeze feature weights."""
        try:
            next_freeze_feature_toggle = next(self._freeze_feature_schedule)
        except StopIteration:
            next_freeze_feature_toggle = np.inf
        return next_freeze_feature_toggle

    def signal_step(self, step: int) -> None:
        if step == self._next_freeze_feature_toggle:
            if self._frozen:
                self._unfreeze_hidden_layers()
            else:
                self._freeze_hidden_layers()
            self._next_freeze_feature_toggle = self._get_next_freeze_feature_toggle()

    def get_trainable_parameters(self):  # TODO: return type
        """To instantiate optimiser, returns relevant (trainable) parameters
        of student network.
        """
        trainable_parameters = []
        if self._train_hidden_layers:
            trainable_hidden_parameters = [
                {"params": filter(lambda p: p.requires_grad, layer.parameters())}
                for layer in self._layers
            ]
            trainable_parameters += trainable_hidden_parameters
        if self._train_head_layer:
            trainable_head_parameters = [
                {
                    "params": head.parameters(),
                    "lr": self._learning_rate * self._head_lr_scaling,
                }
                for head in self._heads
            ]
            trainable_parameters += trainable_head_parameters
        return trainable_parameters

    def _freeze_hidden_layers(self) -> None:
        """Freeze weights in all but head weights
        (used for e.g. frozen feature model)."""
        for layer in self._layers:
            for param in layer.parameters():
                param.requires_grad = False
        self._frozen = True

    def _unfreeze_hidden_layers(self) -> None:
        """Unfreeze weights in all but head weights
        (used for e.g. frozen feature model)."""
        for layer in self._layers:
            for param in layer.parameters():
                param.requires_grad = True
        self._frozen = False

    def forward_all(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Makes call to student forward, using all heads (used for evaluation)"""
        for layer in self._layers:
            x = self._nonlinear_function(self._forward_hidden_scaling * layer(x))
        task_outputs = [head(x) for head in self._heads]
        if self._apply_nonlinearity_on_output:
            task_outputs = [self._nonlinear_function(x) for x in task_outputs]
        if self._classification_output:
            task_outputs = [self._threshold(x) for x in task_outputs]
        y = [self._forward_scaling * x for x in task_outputs]
        return y

    def _threshold(self, y: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid threshold."""
        return torch.sigmoid(y)

    def _manually_initialise_student(self, weights: nn.ModuleList):
        student_dim = len(self._layers[0].weight)
        weights_dim = len(weights[0].weight)

        overparameterisation_factor = student_dim / weights_dim

        for i in range(int(overparameterisation_factor)):
            self._layers[0].weight.data[
                i * weights_dim : (i + 1) * weights_dim
            ] = weights[0].weight.data

    def append_to_path_integral_contributions(self):
        grads = {n: p.grad for n, p in self.named_parameters() if "head" not in n}
        diffs = {
            n: (p - self._pre_update_shared_parameters[n]).detach()
            for n, p in self.named_parameters()
            if "head" not in n
        }

        for n in self._path_integral_contributions[self._num_switches].keys():
            self._path_integral_contributions[self._num_switches][n] += (
                diffs[n] * grads[n]
            )

        self._pre_update_shared_parameters = {
            n: copy.deepcopy(param)
            for n, param in self.named_parameters()
            if "head" not in n
        }

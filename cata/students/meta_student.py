import torch
import torch.nn as nn

from cata.students import base_student


class MetaStudent(base_student.BaseStudent):
    def _construct_output_layers(self):
        """Instantiate the output layer."""
        self._heads = nn.ModuleList([])
        output_layer = nn.Linear(
            self._hidden_dimensions[-1], self._output_dimension, bias=self._bias
        )
        if self._soft_committee:
            output_layer.weight.data.fill_(1)
        else:
            if self._initialise_outputs:
                self._initialise_weights(output_layer, 1)
            else:
                self._initialise_weights(output_layer)
                nn.init.zeros_(output_layer.weight)
                if self._bias:
                    nn.init.zeros_(output_layer.bias)
        self._heads.append(output_layer)

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through head of student (only one for meta-learning)."""
        y = self._heads[0](x)
        if self._apply_nonlinearity_on_output:
            y = self._nonlinear_function(y)
        if self._classification_output:
            y = self._threshold(y)
        return y

    def _signal_task_boundary(self, new_task: int) -> None:
        """Alert student to teacher change."""
        pass

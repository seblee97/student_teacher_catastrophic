from students import base_student

import torch
import torch.nn as nn


class ContinualStudent(base_student.BaseStudent):
    def _construct_output_layers(self):
        """Instantiate the output layers."""
        # create one head per teacher
        self._heads = nn.ModuleList([])
        for _ in range(self._num_teachers):
            output_layer = nn.Linear(
                self._hidden_dimensions[-1], self._output_dimension, bias=self._bias
            )
            if self._soft_committee:
                output_layer.weight.data.fill_(1)
            else:
                if self._initialise_outputs:
                    self._initialise_weights(output_layer)
                else:
                    nn.init.zeros_(output_layer.weight)
                    if self._bias:
                        nn.init.zeros_(output_layer.bias)
            # freeze heads (unfrozen when current task)
            for param in output_layer.parameters():
                param.requires_grad = False
            self._heads.append(output_layer)

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through relevant head of student (depending on current teacher)."""
        y = self.heads[self._current_teacher](x)
        return y

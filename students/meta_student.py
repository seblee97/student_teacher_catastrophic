from students import base_student

import torch
import torch.nn as nn


class MetaStudent(base_student.BaseStudent):
    def _construct_output_layers(self):
        """Instantiate the output layer."""
        self._heads = nn.ModuleList([])
        output_layer = nn.Linear(
            self._hidden_dimensions[-1], self.output_dimension, bias=self.bias
        )
        if self._initialise_student_outputs:
            self._initialise_weights(output_layer)
        else:
            nn.init.zeros_(output_layer.weight)
            if self.bias:
                nn.init.zeros_(output_layer.bias)
        if self.soft_committee:
            for param in output_layer.parameters():
                param.requires_grad = False
        self._heads.append(output_layer)

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through head of student (only one for meta-learning)."""
        y = self.heads[0](x)
        return y